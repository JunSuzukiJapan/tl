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
static WHERE_PIPELINE: std::sync::OnceLock<ComputePipelineState> = std::sync::OnceLock::new();
static TRIL_PIPELINE: std::sync::OnceLock<ComputePipelineState> = std::sync::OnceLock::new();
static CE_PIPELINE: std::sync::OnceLock<ComputePipelineState> = std::sync::OnceLock::new();
static REPEAT_PIPELINE: std::sync::OnceLock<ComputePipelineState> = std::sync::OnceLock::new();
static INDEX_SEL_PIPELINE: std::sync::OnceLock<ComputePipelineState> = std::sync::OnceLock::new();

fn compile_special_pipeline(function_name: &str) -> ComputePipelineState {
    let device = get_device();
    let options = metal::CompileOptions::new();
    let library = device
        .device()
        .new_library_with_source(SPECIAL_SHADER, &options)
        .unwrap_or_else(|e| panic!("Failed to compile special shader: {}", e));
    let function = library
        .get_function(function_name, None)
        .unwrap_or_else(|e| panic!("{} not found: {}", function_name, e));
    device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .unwrap_or_else(|e| panic!("Failed to create {} pipeline: {}", function_name, e))
}

fn get_where_pipeline() -> &'static ComputePipelineState {
    WHERE_PIPELINE.get_or_init(|| compile_special_pipeline("where_f32"))
}
fn get_tril_pipeline() -> &'static ComputePipelineState {
    TRIL_PIPELINE.get_or_init(|| compile_special_pipeline("tril_f32"))
}
fn get_ce_pipeline() -> &'static ComputePipelineState {
    CE_PIPELINE.get_or_init(|| compile_special_pipeline("cross_entropy_elementwise_f32"))
}
fn get_repeat_pipeline() -> &'static ComputePipelineState {
    REPEAT_PIPELINE.get_or_init(|| compile_special_pipeline("repeat_interleave_f32"))
}
fn get_index_sel_pipeline() -> &'static ComputePipelineState {
    INDEX_SEL_PIPELINE.get_or_init(|| compile_special_pipeline("index_select_f32"))
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
        let device = get_device();
        let pipeline = get_where_pipeline();

        objc::rc::autoreleasepool(|| {
            let command_buffer = device.command_queue().new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(condition.buffer()), 0);
            encoder.set_buffer(1, Some(x.buffer()), 0);
            encoder.set_buffer(2, Some(y.buffer()), 0);
            encoder.set_buffer(3, Some(result.buffer()), 0);

            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let threads = count.min(max_threads);
            let tpg = MTLSize::new(threads as u64, 1, 1);
            let grid = MTLSize::new(((count + threads - 1) / threads) as u64, 1, 1);
            encoder.dispatch_thread_groups(grid, tpg);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();
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
        let device = get_device();
        let pipeline = get_tril_pipeline();

        objc::rc::autoreleasepool(|| {
            let rows_buf = make_u32_buf(rows as u32);
            let cols_buf = make_u32_buf(cols as u32);
            let diag_buf = make_i32_buf(diagonal);

            let command_buffer = device.command_queue().new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(self.buffer()), 0);
            encoder.set_buffer(1, Some(result.buffer()), 0);
            encoder.set_buffer(2, Some(&rows_buf), 0);
            encoder.set_buffer(3, Some(&cols_buf), 0);
            encoder.set_buffer(4, Some(&diag_buf), 0);

            let total_rows = batch_size * rows;
            let tpg = MTLSize::new(cols.min(16) as u64, total_rows.min(16) as u64, 1);
            let grid = MTLSize::new(
                ((cols + tpg.width as usize - 1) / tpg.width as usize) as u64,
                ((total_rows + tpg.height as usize - 1) / tpg.height as usize) as u64,
                1,
            );
            encoder.dispatch_thread_groups(grid, tpg);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();
        });
        Ok(result)
    }

    /// cross_entropy — GPU element-wise + GPU sumall
    pub fn cross_entropy_impl(&self, target: &MetalTensor) -> BackendResult<MetalTensor> {
        if MetalTensor::shape(self) != MetalTensor::shape(target) {
            return Err(BackendError::ShapeMismatch(format!(
                "cross_entropy: shape mismatch {:?} vs {:?}",
                MetalTensor::shape(self), MetalTensor::shape(target)
            )));
        }

        let count = self.elem_count();
        let temp = MetalTensor::uninit(MetalTensor::shape(self), DType::F32);
        let device = get_device();
        let pipeline = get_ce_pipeline();

        // Step 1: element-wise -t * log(p + eps)
        objc::rc::autoreleasepool(|| {
            let command_buffer = device.command_queue().new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(self.buffer()), 0);
            encoder.set_buffer(1, Some(target.buffer()), 0);
            encoder.set_buffer(2, Some(temp.buffer()), 0);

            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let threads = count.min(max_threads);
            let tpg = MTLSize::new(threads as u64, 1, 1);
            let grid = MTLSize::new(((count + threads - 1) / threads) as u64, 1, 1);
            encoder.dispatch_thread_groups(grid, tpg);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();
        });

        // Step 2: GPU sumall で合計 → スカラーテンソル
        let sum_val = temp.sumall_impl()?;
        Ok(MetalTensor::from_slice(&[sum_val], &[1], DType::F32))
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
        let device = get_device();
        let pipeline = get_repeat_pipeline();

        objc::rc::autoreleasepool(|| {
            let outer_buf = make_u32_buf(outer_size as u32);
            let axis_buf = make_u32_buf(axis_size as u32);
            let inner_buf = make_u32_buf(inner_size as u32);
            let repeats_buf = make_u32_buf(repeats as u32);

            let command_buffer = device.command_queue().new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(self.buffer()), 0);
            encoder.set_buffer(1, Some(result.buffer()), 0);
            encoder.set_buffer(2, Some(&outer_buf), 0);
            encoder.set_buffer(3, Some(&axis_buf), 0);
            encoder.set_buffer(4, Some(&inner_buf), 0);
            encoder.set_buffer(5, Some(&repeats_buf), 0);

            let flat_size = axis_size * repeats * inner_size;
            let tpg = MTLSize::new(flat_size.min(256) as u64, outer_size.min(4) as u64, 1);
            let grid = MTLSize::new(
                ((flat_size + tpg.width as usize - 1) / tpg.width as usize) as u64,
                ((outer_size + tpg.height as usize - 1) / tpg.height as usize) as u64,
                1,
            );
            encoder.dispatch_thread_groups(grid, tpg);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();
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
        let command_queue = device.command_queue();
        
        let mut shaders = crate::shaders::get_shaders().lock().unwrap();
        let pipeline = shaders
            .get_pipeline(device.device(), shader_name).map_err(BackendError::InternalError)?;
            
        objc::rc::autoreleasepool(|| {
            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(self.buffer()), 0);
            encoder.set_buffer(1, Some(result.buffer()), 0);
            
            let (grid_size, threads_per_group) = crate::shaders::compute_thread_groups(self.elem_count(), pipeline);
            encoder.dispatch_thread_groups(grid_size, threads_per_group);
            encoder.end_encoding();
            
            command_buffer.commit();
            command_buffer.wait_until_completed();
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
        let device = get_device();
        let pipeline = get_index_sel_pipeline();

        objc::rc::autoreleasepool(|| {
            let outer_buf = make_u32_buf(outer_size as u32);
            let axis_buf = make_u32_buf(axis_size as u32);
            let inner_buf = make_u32_buf(inner_size as u32);
            let nidx_buf = make_u32_buf(num_indices as u32);

            let command_buffer = device.command_queue().new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(self.buffer()), 0);
            encoder.set_buffer(1, Some(indices.buffer()), 0);
            encoder.set_buffer(2, Some(result.buffer()), 0);
            encoder.set_buffer(3, Some(&outer_buf), 0);
            encoder.set_buffer(4, Some(&axis_buf), 0);
            encoder.set_buffer(5, Some(&inner_buf), 0);
            encoder.set_buffer(6, Some(&nidx_buf), 0);

            let flat_size = num_indices * inner_size;
            let tpg = MTLSize::new(flat_size.min(256) as u64, outer_size.min(4) as u64, 1);
            let grid = MTLSize::new(
                ((flat_size + tpg.width as usize - 1) / tpg.width as usize) as u64,
                ((outer_size + tpg.height as usize - 1) / tpg.height as usize) as u64,
                1,
            );
            encoder.dispatch_thread_groups(grid, tpg);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();
        });
        Ok(result)
    }
}
