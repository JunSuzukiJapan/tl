//! 特殊演算 — Metal GPU シェーダー実装
//! where_cond, tril, cross_entropy, repeat_interleave, index_select

use crate::device::get_device;
use crate::tensor::MetalTensor;
use crate::DType;
use metal::{ComputePipelineState, MTLSize};
use tl_backend::{BackendResult, BackendError};

/// 特殊演算用 Metal シェーダー
const SPECIAL_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// where_cond: out[id] = cond[id] > 0 ? x[id] : y[id]
kernel void where_f32(
    device const float* cond [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* y [[buffer(2)]],
    device float* out [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = (cond[id] > 0.0f) ? x[id] : y[id];
}

// tril: 下三角行列（上三角を 0 に）
kernel void tril_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant int& diagonal [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row_batch = gid.y; // row + batch * rows
    if (col >= cols) return;
    
    uint row = row_batch % rows;
    uint batch = row_batch / rows;
    
    uint idx = batch * rows * cols + row * cols + col;
    output[idx] = (int(col) > int(row) + diagonal) ? 0.0f : input[idx];
}

// cross_entropy element-wise: out[id] = -target[id] * log(pred[id] + eps)
kernel void cross_entropy_elementwise_f32(
    device const float* pred [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = -target[id] * log(pred[id] + 1e-7f);
}

// cross_entropy_backward: fused softmax + gradient computation
// 各スレッドが1バッチ行を処理: grad[n,c] = (softmax(logits[n,c]) - one_hot(targets[n], c)) * grad_output / valid_count
kernel void cross_entropy_backward_f32(
    device const float* logits      [[buffer(0)]],  // [N, C]
    device const float* targets     [[buffer(1)]],  // [N] integer indices
    device const float* grad_output [[buffer(2)]],  // [1] scalar
    device float*       grad_logits [[buffer(3)]],  // [N, C] output
    constant uint&      num_classes [[buffer(4)]],
    constant uint&      batch_size  [[buffer(5)]],
    constant uint&      valid_count [[buffer(6)]],
    uint row_id [[thread_position_in_grid]]
) {
    if (row_id >= batch_size) return;

    uint offset = row_id * num_classes;
    int target_idx = int(targets[row_id]);
    float go = grad_output[0];

    // ignore_index (target < 0): この行の gradient は全て 0
    if (target_idx < 0) {
        for (uint j = 0; j < num_classes; j++) {
            grad_logits[offset + j] = 0.0f;
        }
        return;
    }

    // Softmax pass 1: 数値安定性のため max を求める
    float max_val = logits[offset];
    for (uint j = 1; j < num_classes; j++) {
        max_val = max(max_val, logits[offset + j]);
    }

    // Softmax pass 2: exp の合計
    float sum_exp = 0.0f;
    for (uint j = 0; j < num_classes; j++) {
        sum_exp += exp(logits[offset + j] - max_val);
    }

    // Gradient: (softmax - one_hot) * grad_output / valid_count
    float inv_valid = go / float(valid_count);
    for (uint j = 0; j < num_classes; j++) {
        float softmax_val = exp(logits[offset + j] - max_val) / sum_exp;
        float one_hot = (int(j) == target_idx) ? 1.0f : 0.0f;
        grad_logits[offset + j] = (softmax_val - one_hot) * inv_valid;
    }
}

// cross_entropy_forward: fused softmax + NLL loss
// 各スレッドが1バッチ行を担当: out[n] = -log(softmax(logits[n, target[n]]))
kernel void cross_entropy_forward_f32(
    device const float* logits  [[buffer(0)]],  // [N, C]
    device const float* targets [[buffer(1)]],  // [N] integer indices
    device float*       losses  [[buffer(2)]],  // [N] per-sample loss
    constant uint&      num_classes [[buffer(3)]],
    constant uint&      batch_size  [[buffer(4)]],
    uint row_id [[thread_position_in_grid]]
) {
    if (row_id >= batch_size) return;

    uint offset = row_id * num_classes;
    int target_idx = int(targets[row_id]);

    // ignore_index (target < 0): loss = 0
    if (target_idx < 0 || uint(target_idx) >= num_classes) {
        losses[row_id] = 0.0f;
        return;
    }

    // Softmax pass 1: max for numerical stability
    float max_val = logits[offset];
    for (uint j = 1; j < num_classes; j++) {
        max_val = max(max_val, logits[offset + j]);
    }

    // Softmax pass 2: sum of exp
    float sum_exp = 0.0f;
    for (uint j = 0; j < num_classes; j++) {
        sum_exp += exp(logits[offset + j] - max_val);
    }

    // NLL: -log(softmax[target])
    float log_softmax = (logits[offset + uint(target_idx)] - max_val) - log(sum_exp);
    losses[row_id] = -log_softmax;
}

// repeat_interleave: 要素を repeats 回繰り返す
kernel void repeat_interleave_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& outer_size [[buffer(2)]],
    constant uint& axis_size [[buffer(3)]],
    constant uint& inner_size [[buffer(4)]],
    constant uint& repeats [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // gid.x = flat index within (axis_size * repeats * inner_size)
    // gid.y = outer
    uint flat = gid.x;
    uint outer = gid.y;
    uint total_inner = axis_size * repeats * inner_size;
    if (outer >= outer_size || flat >= total_inner) return;
    
    uint inner = flat % inner_size;
    uint ar = (flat / inner_size);
    uint a = ar / repeats;
    
    uint src_idx = outer * axis_size * inner_size + a * inner_size + inner;
    uint dst_idx = outer * total_inner + flat;
    output[dst_idx] = input[src_idx];
}

// index_select: gather パターン
kernel void index_select_f32(
    device const float* input [[buffer(0)]],
    device const float* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& outer_size [[buffer(3)]],
    constant uint& axis_size [[buffer(4)]],
    constant uint& inner_size [[buffer(5)]],
    constant uint& num_indices [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // gid.x = flat index within (num_indices * inner_size)
    // gid.y = outer
    uint flat = gid.x;
    uint outer = gid.y;
    uint total_inner = num_indices * inner_size;
    if (outer >= outer_size || flat >= total_inner) return;
    
    uint inner = flat % inner_size;
    uint new_a = flat / inner_size;
    uint a = uint(indices[new_a]);
    
    uint src_idx = outer * axis_size * inner_size + a * inner_size + inner;
    uint dst_idx = outer * total_inner + flat;
    output[dst_idx] = input[src_idx];
}
"#;

// パイプラインキャッシュ
static WHERE_PIPELINE: std::sync::OnceLock<Result<ComputePipelineState, String>> = std::sync::OnceLock::new();
static TRIL_PIPELINE: std::sync::OnceLock<Result<ComputePipelineState, String>> = std::sync::OnceLock::new();
#[allow(dead_code)]
static CE_PIPELINE: std::sync::OnceLock<Result<ComputePipelineState, String>> = std::sync::OnceLock::new();
static CE_FORWARD_PIPELINE: std::sync::OnceLock<Result<ComputePipelineState, String>> = std::sync::OnceLock::new();
static CE_BACKWARD_PIPELINE: std::sync::OnceLock<Result<ComputePipelineState, String>> = std::sync::OnceLock::new();
static REPEAT_PIPELINE: std::sync::OnceLock<Result<ComputePipelineState, String>> = std::sync::OnceLock::new();
static INDEX_SEL_PIPELINE: std::sync::OnceLock<Result<ComputePipelineState, String>> = std::sync::OnceLock::new();

fn compile_special_pipeline(function_name: &str) -> Result<ComputePipelineState, String> {
    let device = get_device();
    let options = metal::CompileOptions::new();
    let library = device
        .device()
        .new_library_with_source(SPECIAL_SHADER, &options)
        .map_err(|e| format!("Failed to compile special shader: {}", e))?;
    let function = library
        .get_function(function_name, None)
        .map_err(|e| format!("{} not found: {}", function_name, e))?;
    device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| format!("Failed to create {} pipeline: {}", function_name, e))
}

fn get_where_pipeline() -> BackendResult<&'static ComputePipelineState> {
    WHERE_PIPELINE.get_or_init(|| compile_special_pipeline("where_f32"))
        .as_ref().map_err(|e| BackendError::DeviceError(e.clone()))
}
fn get_tril_pipeline() -> BackendResult<&'static ComputePipelineState> {
    TRIL_PIPELINE.get_or_init(|| compile_special_pipeline("tril_f32"))
        .as_ref().map_err(|e| BackendError::DeviceError(e.clone()))
}
#[allow(dead_code)]
fn get_ce_pipeline() -> BackendResult<&'static ComputePipelineState> {
    CE_PIPELINE.get_or_init(|| compile_special_pipeline("cross_entropy_elementwise_f32"))
        .as_ref().map_err(|e| BackendError::DeviceError(e.clone()))
}
fn get_ce_forward_pipeline() -> BackendResult<&'static ComputePipelineState> {
    CE_FORWARD_PIPELINE.get_or_init(|| compile_special_pipeline("cross_entropy_forward_f32"))
        .as_ref().map_err(|e| BackendError::DeviceError(e.clone()))
}
fn get_ce_backward_pipeline() -> BackendResult<&'static ComputePipelineState> {
    CE_BACKWARD_PIPELINE.get_or_init(|| compile_special_pipeline("cross_entropy_backward_f32"))
        .as_ref().map_err(|e| BackendError::DeviceError(e.clone()))
}
fn get_repeat_pipeline() -> BackendResult<&'static ComputePipelineState> {
    REPEAT_PIPELINE.get_or_init(|| compile_special_pipeline("repeat_interleave_f32"))
        .as_ref().map_err(|e| BackendError::DeviceError(e.clone()))
}
fn get_index_sel_pipeline() -> BackendResult<&'static ComputePipelineState> {
    INDEX_SEL_PIPELINE.get_or_init(|| compile_special_pipeline("index_select_f32"))
        .as_ref().map_err(|e| BackendError::DeviceError(e.clone()))
}

/// u32 パラメータバッファ
fn make_u32_buf(v: u32) -> metal::Buffer {
    let device = get_device();
    device.device().new_buffer_with_data(
        &v as *const u32 as *const _,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    )
}

/// i32 パラメータバッファ
fn make_i32_buf(v: i32) -> metal::Buffer {
    let device = get_device();
    device.device().new_buffer_with_data(
        &v as *const i32 as *const _,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    )
}

impl MetalTensor {
    /// where_cond — Metal GPU シェーダー実装
    pub fn where_cond_impl(condition: &MetalTensor, x: &MetalTensor, y: &MetalTensor) -> BackendResult<MetalTensor> {
        let count = x.elem_count();
        if condition.elem_count() != count || y.elem_count() != count {
             return Err(BackendError::ShapeMismatch(format!(
                 "where_cond: shapes mismatch: cond {:?}, x {:?}, y {:?}",
                 MetalTensor::shape(condition), MetalTensor::shape(x), MetalTensor::shape(y)
             )));
        }

        let result = MetalTensor::uninit(MetalTensor::shape(x), MetalTensor::dtype(x));
        let pipeline = get_where_pipeline()?;

        let cond_buf = condition.buffer() as *const metal::Buffer;
        let x_buf = x.buffer() as *const metal::Buffer;
        let y_buf = y.buffer() as *const metal::Buffer;
        let result_buf = result.buffer() as *const metal::Buffer;

        crate::command_stream::stream_encode(|encoder| {
            encoder.set_compute_pipeline_state(pipeline);
            unsafe {
                encoder.set_buffer(0, Some(&*cond_buf), 0);
                encoder.set_buffer(1, Some(&*x_buf), 0);
                encoder.set_buffer(2, Some(&*y_buf), 0);
                encoder.set_buffer(3, Some(&*result_buf), 0);
            }

            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let threads = count.min(max_threads);
            let tpg = MTLSize::new(threads as u64, 1, 1);
            let grid = MTLSize::new(((count + threads - 1) / threads) as u64, 1, 1);
            encoder.dispatch_thread_groups(grid, tpg);
        });
        Ok(result)
    }

    /// tril — Metal GPU シェーダー実装
    pub fn tril_impl(&self, diagonal: i32) -> BackendResult<MetalTensor> {
        let shape = MetalTensor::shape(self);
        if shape.len() < 2 {
             return Err(BackendError::ShapeMismatch(format!("tril requires at least 2D tensor, got {:?}D", shape.len())));
        }
        
        let rows = shape[shape.len() - 2];
        let cols = shape[shape.len() - 1];
        let batch_size: usize = shape[..shape.len() - 2].iter().product::<usize>().max(1);
        
        let result = MetalTensor::uninit(shape, MetalTensor::dtype(self));
        let pipeline = get_tril_pipeline()?;

        let rows_buf = make_u32_buf(rows as u32);
        let cols_buf = make_u32_buf(cols as u32);
        let diag_buf = make_i32_buf(diagonal);

        let self_buf = self.buffer() as *const metal::Buffer;
        let result_buf = result.buffer() as *const metal::Buffer;
        let rows_buf_ptr = &rows_buf as *const metal::Buffer;
        let cols_buf_ptr = &cols_buf as *const metal::Buffer;
        let diag_buf_ptr = &diag_buf as *const metal::Buffer;

        crate::command_stream::stream_encode(|encoder| {
            encoder.set_compute_pipeline_state(pipeline);
            unsafe {
                encoder.set_buffer(0, Some(&*self_buf), 0);
                encoder.set_buffer(1, Some(&*result_buf), 0);
                encoder.set_buffer(2, Some(&*rows_buf_ptr), 0);
                encoder.set_buffer(3, Some(&*cols_buf_ptr), 0);
                encoder.set_buffer(4, Some(&*diag_buf_ptr), 0);
            }

            let total_rows = batch_size * rows;
            let tpg = MTLSize::new(cols.min(16) as u64, total_rows.min(16) as u64, 1);
            let grid = MTLSize::new(
                ((cols + tpg.width as usize - 1) / tpg.width as usize) as u64,
                ((total_rows + tpg.height as usize - 1) / tpg.height as usize) as u64,
                1,
            );
            encoder.dispatch_thread_groups(grid, tpg);
        });
        Ok(result)
    }

    /// cross_entropy — Metal GPU カーネル実装 (fused softmax + NLL loss)
    pub fn cross_entropy_impl(&self, target: &MetalTensor) -> BackendResult<MetalTensor> {
        let logits_shape = MetalTensor::shape(self);
        let batch_size = if logits_shape.len() >= 2 { logits_shape[0] } else { 1 };
        let num_classes = if logits_shape.len() >= 2 {
            *logits_shape.last().unwrap()
        } else {
            self.elem_count()
        };

        // per-sample loss [N]
        let losses = MetalTensor::uninit(&[batch_size], DType::F32);
        let pipeline = get_ce_forward_pipeline()?;

        let nc = num_classes as u32;
        let bs = batch_size as u32;

        let logits_buf = self.buffer() as *const metal::Buffer;
        let targets_buf = target.buffer() as *const metal::Buffer;
        let losses_buf = losses.buffer() as *const metal::Buffer;

        crate::command_stream::stream_encode(|encoder| {
            encoder.set_compute_pipeline_state(pipeline);
            unsafe {
                encoder.set_buffer(0, Some(&*logits_buf), 0);
                encoder.set_buffer(1, Some(&*targets_buf), 0);
                encoder.set_buffer(2, Some(&*losses_buf), 0);
            }
            encoder.set_bytes(3, 4, &nc as *const u32 as *const _);
            encoder.set_bytes(4, 4, &bs as *const u32 as *const _);

            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let threads = batch_size.min(max_threads);
            let tpg = MTLSize::new(threads as u64, 1, 1);
            let grid = MTLSize::new(
                ((batch_size + threads - 1) / threads) as u64, 1, 1,
            );
            encoder.dispatch_thread_groups(grid, tpg);
        });

        // per-sample losses を直接読み出し → mean
        // GPU カーネル完了を待つ
        crate::command_stream::sync_stream();
        let mut total = 0.0f32;
        unsafe {
            let ptr = losses.buffer().contents() as *const f32;
            for i in 0..batch_size {
                total += *ptr.add(i);
            }
        }
        let mean_loss = total / batch_size as f32;
        Ok(MetalTensor::from_slice(&[mean_loss], &[1], DType::F32))
    }

    /// cross_entropy_backward — Metal GPU カーネル実装
    /// fused softmax + gradient: grad[n,c] = (softmax(logits[n,c]) - one_hot(targets[n], c)) * grad_output / valid_count
    pub fn cross_entropy_backward_impl(
        &self,
        targets: &MetalTensor,
        grad_output: &MetalTensor,
    ) -> BackendResult<MetalTensor> {
        let shape = MetalTensor::shape(self);
        let batch_size = if shape.len() >= 2 { shape[0] } else { 1 };
        let num_classes = if shape.len() >= 2 { shape[1] } else { shape[0] };

        // valid_count を targets バッファから計算 (ignore_index=-1 を除外)
        crate::command_stream::sync_stream();
        let mut valid_count = 0u32;
        unsafe {
            let target_ptr = targets.buffer().contents() as *const f32;
            for i in 0..batch_size {
                if (*target_ptr.add(i) as i64) >= 0 {
                    valid_count += 1;
                }
            }
        }
        if valid_count == 0 { valid_count = 1; } // ゼロ除算防止

        let result = MetalTensor::uninit(shape, MetalTensor::dtype(self));
        let pipeline = get_ce_backward_pipeline()?;

        let nc = num_classes as u32;
        let bs = batch_size as u32;
        let vc = valid_count;

        let logits_buf = self.buffer() as *const metal::Buffer;
        let targets_buf = targets.buffer() as *const metal::Buffer;
        let go_buf = grad_output.buffer() as *const metal::Buffer;
        let result_buf = result.buffer() as *const metal::Buffer;

        crate::command_stream::stream_encode(|encoder| {
            encoder.set_compute_pipeline_state(pipeline);
            unsafe {
                encoder.set_buffer(0, Some(&*logits_buf), 0);
                encoder.set_buffer(1, Some(&*targets_buf), 0);
                encoder.set_buffer(2, Some(&*go_buf), 0);
                encoder.set_buffer(3, Some(&*result_buf), 0);
            }
            encoder.set_bytes(4, 4, &nc as *const u32 as *const _);
            encoder.set_bytes(5, 4, &bs as *const u32 as *const _);
            encoder.set_bytes(6, 4, &vc as *const u32 as *const _);

            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let threads = batch_size.min(max_threads);
            let tpg = MTLSize::new(threads as u64, 1, 1);
            let grid = MTLSize::new(
                ((batch_size + threads - 1) / threads) as u64, 1, 1,
            );
            encoder.dispatch_thread_groups(grid, tpg);
        });
        Ok(result)
    }

    /// repeat_interleave — Metal GPU シェーダー実装
    pub fn repeat_interleave_impl(&self, repeats: usize, axis: usize) -> BackendResult<MetalTensor> {
        let shape = MetalTensor::shape(self);
        if axis >= shape.len() {
             return Err(BackendError::IndexOutOfBounds(format!("repeat_interleave: axis {} out of range", axis)));
        }
        
        let outer_size: usize = shape[..axis].iter().product::<usize>().max(1);
        let axis_size = shape[axis];
        let inner_size: usize = shape[axis + 1..].iter().product::<usize>().max(1);
        
        let mut new_shape = shape.to_vec();
        new_shape[axis] *= repeats;
        
        let result = MetalTensor::uninit(&new_shape, MetalTensor::dtype(self));
        let pipeline = get_repeat_pipeline()?;

        let outer_buf = make_u32_buf(outer_size as u32);
        let axis_buf = make_u32_buf(axis_size as u32);
        let inner_buf = make_u32_buf(inner_size as u32);
        let repeats_buf = make_u32_buf(repeats as u32);

        let self_buf = self.buffer() as *const metal::Buffer;
        let result_buf = result.buffer() as *const metal::Buffer;
        let outer_buf_ptr = &outer_buf as *const metal::Buffer;
        let axis_buf_ptr = &axis_buf as *const metal::Buffer;
        let inner_buf_ptr = &inner_buf as *const metal::Buffer;
        let repeats_buf_ptr = &repeats_buf as *const metal::Buffer;

        crate::command_stream::stream_encode(|encoder| {
            encoder.set_compute_pipeline_state(pipeline);
            unsafe {
                encoder.set_buffer(0, Some(&*self_buf), 0);
                encoder.set_buffer(1, Some(&*result_buf), 0);
                encoder.set_buffer(2, Some(&*outer_buf_ptr), 0);
                encoder.set_buffer(3, Some(&*axis_buf_ptr), 0);
                encoder.set_buffer(4, Some(&*inner_buf_ptr), 0);
                encoder.set_buffer(5, Some(&*repeats_buf_ptr), 0);
            }

            let flat_size = axis_size * repeats * inner_size;
            let tpg = MTLSize::new(flat_size.min(256) as u64, outer_size.min(4) as u64, 1);
            let grid = MTLSize::new(
                ((flat_size + tpg.width as usize - 1) / tpg.width as usize) as u64,
                ((outer_size + tpg.height as usize - 1) / tpg.height as usize) as u64,
                1,
            );
            encoder.dispatch_thread_groups(grid, tpg);
        });
        Ok(result)
    }

    /// to_dtype — データ型変換（GPU 上でクローン）
    pub fn to_dtype(&self, dtype: DType) -> BackendResult<MetalTensor> {
        if MetalTensor::dtype(self) == dtype {
                return Ok(self.clone_data()?);
        }
        
        match (MetalTensor::dtype(self), dtype) {
            (DType::F16, DType::F32) => self.cast_impl(crate::shaders::SHADER_CAST_F16_TO_F32, dtype),
            (DType::F32, DType::F16) => self.cast_impl(crate::shaders::SHADER_CAST_F32_TO_F16, dtype),
            _ => {
                // Fallback: Clone data (limitations)
                Ok(self.clone_data()?)
            }
        }
    }

    fn cast_impl(&self, shader_name: &str, target_dtype: DType) -> BackendResult<MetalTensor> {
        let result = MetalTensor::uninit(MetalTensor::shape(self), target_dtype);
        let device = get_device();
        
        let mut shaders = crate::shaders::get_shaders().lock().unwrap();
        let pipeline = shaders
            .get_pipeline(device.device(), shader_name).map_err(BackendError::InternalError)?;

        let self_buf = self.buffer() as *const metal::Buffer;
        let result_buf = result.buffer() as *const metal::Buffer;
        let elem_count = self.elem_count();

        crate::command_stream::stream_encode(|encoder| {
            encoder.set_compute_pipeline_state(pipeline);
            unsafe {
                encoder.set_buffer(0, Some(&*self_buf), 0);
                encoder.set_buffer(1, Some(&*result_buf), 0);
            }
            
            let (grid_size, threads_per_group) = crate::shaders::compute_thread_groups(elem_count, pipeline);
            encoder.dispatch_thread_groups(grid_size, threads_per_group);
        });
        
        Ok(result)
    }

    /// index_select — Metal GPU シェーダー実装
    pub fn index_select_impl(&self, axis: usize, indices: &MetalTensor) -> BackendResult<MetalTensor> {
        let shape = MetalTensor::shape(self);
        if axis >= shape.len() {
             return Err(BackendError::IndexOutOfBounds(format!("index_select: axis {} out of range", axis)));
        }
        
        let num_indices = indices.elem_count();
        let outer_size: usize = shape[..axis].iter().product::<usize>().max(1);
        let axis_size = shape[axis];
        let inner_size: usize = shape[axis + 1..].iter().product::<usize>().max(1);
        
        let mut new_shape = shape.to_vec();
        new_shape[axis] = num_indices;
        
        let result = MetalTensor::uninit(&new_shape, MetalTensor::dtype(self));
        let pipeline = get_index_sel_pipeline()?;

        let outer_buf = make_u32_buf(outer_size as u32);
        let axis_buf = make_u32_buf(axis_size as u32);
        let inner_buf = make_u32_buf(inner_size as u32);
        let nidx_buf = make_u32_buf(num_indices as u32);

        let self_buf = self.buffer() as *const metal::Buffer;
        let indices_buf = indices.buffer() as *const metal::Buffer;
        let result_buf = result.buffer() as *const metal::Buffer;
        let outer_buf_ptr = &outer_buf as *const metal::Buffer;
        let axis_buf_ptr = &axis_buf as *const metal::Buffer;
        let inner_buf_ptr = &inner_buf as *const metal::Buffer;
        let nidx_buf_ptr = &nidx_buf as *const metal::Buffer;

        crate::command_stream::stream_encode(|encoder| {
            encoder.set_compute_pipeline_state(pipeline);
            unsafe {
                encoder.set_buffer(0, Some(&*self_buf), 0);
                encoder.set_buffer(1, Some(&*indices_buf), 0);
                encoder.set_buffer(2, Some(&*result_buf), 0);
                encoder.set_buffer(3, Some(&*outer_buf_ptr), 0);
                encoder.set_buffer(4, Some(&*axis_buf_ptr), 0);
                encoder.set_buffer(5, Some(&*inner_buf_ptr), 0);
                encoder.set_buffer(6, Some(&*nidx_buf_ptr), 0);
            }

            let flat_size = num_indices * inner_size;
            let tpg = MTLSize::new(flat_size.min(256) as u64, outer_size.min(4) as u64, 1);
            let grid = MTLSize::new(
                ((flat_size + tpg.width as usize - 1) / tpg.width as usize) as u64,
                ((outer_size + tpg.height as usize - 1) / tpg.height as usize) as u64,
                1,
            );
            encoder.dispatch_thread_groups(grid, tpg);
        });
        Ok(result)
    }
}
