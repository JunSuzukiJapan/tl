//! 軸指定 Reduce 演算 — Metal GPU シェーダー実装
//! sum, max, min, argmax, argmin を axis 方向に reduce

use crate::device::get_device;
use crate::tensor::MetalTensor;
use crate::DType;
use metal::{ComputePipelineState, MTLSize};
use tl_backend::{BackendResult, BackendError};

/// 軸指定 Reduce 用 Metal シェーダー
const REDUCE_AXIS_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// sum along axis
kernel void reduce_sum_axis_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& outer_size [[buffer(2)]],
    constant uint& axis_size [[buffer(3)]],
    constant uint& inner_size [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint inner = gid.x;
    uint outer = gid.y;
    if (inner >= inner_size || outer >= outer_size) return;
    
    float sum = 0.0f;
    for (uint a = 0; a < axis_size; a++) {
        sum += input[outer * axis_size * inner_size + a * inner_size + inner];
    }
    output[outer * inner_size + inner] = sum;
}

// max along axis
kernel void reduce_max_axis_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& outer_size [[buffer(2)]],
    constant uint& axis_size [[buffer(3)]],
    constant uint& inner_size [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint inner = gid.x;
    uint outer = gid.y;
    if (inner >= inner_size || outer >= outer_size) return;
    
    float max_val = -INFINITY;
    for (uint a = 0; a < axis_size; a++) {
        max_val = max(max_val, input[outer * axis_size * inner_size + a * inner_size + inner]);
    }
    output[outer * inner_size + inner] = max_val;
}

// min along axis
kernel void reduce_min_axis_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& outer_size [[buffer(2)]],
    constant uint& axis_size [[buffer(3)]],
    constant uint& inner_size [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint inner = gid.x;
    uint outer = gid.y;
    if (inner >= inner_size || outer >= outer_size) return;
    
    float min_val = INFINITY;
    for (uint a = 0; a < axis_size; a++) {
        min_val = min(min_val, input[outer * axis_size * inner_size + a * inner_size + inner]);
    }
    output[outer * inner_size + inner] = min_val;
}

// argmax along axis
kernel void reduce_argmax_axis_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& outer_size [[buffer(2)]],
    constant uint& axis_size [[buffer(3)]],
    constant uint& inner_size [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint inner = gid.x;
    uint outer = gid.y;
    if (inner >= inner_size || outer >= outer_size) return;
    
    float max_val = -INFINITY;
    uint max_idx = 0;
    for (uint a = 0; a < axis_size; a++) {
        float v = input[outer * axis_size * inner_size + a * inner_size + inner];
        if (v > max_val) {
            max_val = v;
            max_idx = a;
        }
    }
    output[outer * inner_size + inner] = float(max_idx);
}

// argmin along axis
kernel void reduce_argmin_axis_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& outer_size [[buffer(2)]],
    constant uint& axis_size [[buffer(3)]],
    constant uint& inner_size [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint inner = gid.x;
    uint outer = gid.y;
    if (inner >= inner_size || outer >= outer_size) return;
    
    float min_val = INFINITY;
    uint min_idx = 0;
    for (uint a = 0; a < axis_size; a++) {
        float v = input[outer * axis_size * inner_size + a * inner_size + inner];
        if (v < min_val) {
            min_val = v;
            min_idx = a;
        }
    }
    output[outer * inner_size + inner] = float(min_idx);
}
"#;

// パイプラインキャッシュ
static REDUCE_SUM_AXIS_PIPELINE: std::sync::OnceLock<ComputePipelineState> = std::sync::OnceLock::new();
static REDUCE_MAX_AXIS_PIPELINE: std::sync::OnceLock<ComputePipelineState> = std::sync::OnceLock::new();
static REDUCE_MIN_AXIS_PIPELINE: std::sync::OnceLock<ComputePipelineState> = std::sync::OnceLock::new();
static REDUCE_ARGMAX_AXIS_PIPELINE: std::sync::OnceLock<ComputePipelineState> = std::sync::OnceLock::new();
static REDUCE_ARGMIN_AXIS_PIPELINE: std::sync::OnceLock<ComputePipelineState> = std::sync::OnceLock::new();

fn compile_reduce_pipeline(function_name: &str) -> ComputePipelineState {
    let device = get_device();
    let options = metal::CompileOptions::new();
    let library = device
        .device()
        .new_library_with_source(REDUCE_AXIS_SHADER, &options)
        .unwrap_or_else(|e| panic!("Failed to compile reduce shader: {}", e));
    let function = library
        .get_function(function_name, None)
        .unwrap_or_else(|e| panic!("{} not found: {}", function_name, e));
    device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .unwrap_or_else(|e| panic!("Failed to create {} pipeline: {}", function_name, e))
}

fn get_reduce_sum_pipeline() -> &'static ComputePipelineState {
    REDUCE_SUM_AXIS_PIPELINE.get_or_init(|| compile_reduce_pipeline("reduce_sum_axis_f32"))
}
fn get_reduce_max_pipeline() -> &'static ComputePipelineState {
    REDUCE_MAX_AXIS_PIPELINE.get_or_init(|| compile_reduce_pipeline("reduce_max_axis_f32"))
}
fn get_reduce_min_pipeline() -> &'static ComputePipelineState {
    REDUCE_MIN_AXIS_PIPELINE.get_or_init(|| compile_reduce_pipeline("reduce_min_axis_f32"))
}
fn get_reduce_argmax_pipeline() -> &'static ComputePipelineState {
    REDUCE_ARGMAX_AXIS_PIPELINE.get_or_init(|| compile_reduce_pipeline("reduce_argmax_axis_f32"))
}
fn get_reduce_argmin_pipeline() -> &'static ComputePipelineState {
    REDUCE_ARGMIN_AXIS_PIPELINE.get_or_init(|| compile_reduce_pipeline("reduce_argmin_axis_f32"))
}

/// u32 パラメータバッファを作成
fn make_u32_buf(v: u32) -> metal::Buffer {
    let device = get_device();
    device.device().new_buffer_with_data(
        &v as *const u32 as *const _,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    )
}

/// 共通の軸パラメータ分解
fn decompose_axis(shape: &[usize], axis: i32) -> BackendResult<(usize, usize, usize, usize, Vec<usize>)> {
    let ndim = shape.len();
    let axis = if axis < 0 { (ndim as i32 + axis) as usize } else { axis as usize };
    if axis >= ndim {
        return Err(BackendError::IndexOutOfBounds(format!("axis {} out of range (ndim={})", axis, ndim)));
    }

    let axis_size = shape[axis];
    let outer_size: usize = shape[..axis].iter().product::<usize>().max(1);
    let inner_size: usize = if axis + 1 < ndim { shape[axis + 1..].iter().product() } else { 1 };

    let mut new_shape: Vec<usize> = shape.to_vec();
    new_shape.remove(axis);
    if new_shape.is_empty() {
        new_shape.push(1);
    }

    Ok((outer_size, axis_size, inner_size, axis, new_shape))
}

/// GPU reduce ディスパッチ（共通）
fn dispatch_reduce(
    input: &MetalTensor,
    pipeline: &ComputePipelineState,
    outer_size: usize,
    axis_size: usize,
    inner_size: usize,
    new_shape: &[usize],
) -> BackendResult<MetalTensor> {
    let result = MetalTensor::uninit(new_shape, DType::F32);
    let device = get_device();
    let command_queue = device.command_queue();

    let outer_buf = make_u32_buf(outer_size as u32);
    let axis_buf = make_u32_buf(axis_size as u32);
    let inner_buf = make_u32_buf(inner_size as u32);

    let command_buffer = command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(input.buffer()), 0);
    encoder.set_buffer(1, Some(result.buffer()), 0);
    encoder.set_buffer(2, Some(&outer_buf), 0);
    encoder.set_buffer(3, Some(&axis_buf), 0);
    encoder.set_buffer(4, Some(&inner_buf), 0);

    // 2D グリッド: [inner_size, outer_size]
    let tpg = MTLSize::new(
        inner_size.min(256) as u64,
        outer_size.min(4) as u64,
        1,
    );
    let grid = MTLSize::new(
        ((inner_size + tpg.width as usize - 1) / tpg.width as usize) as u64,
        ((outer_size + tpg.height as usize - 1) / tpg.height as usize) as u64,
        1,
    );
    encoder.dispatch_thread_groups(grid, tpg);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(result)
}

impl MetalTensor {
    /// 軸指定で合計 (Metal GPU 実装)
    pub fn sum_impl(&self, axis: i32) -> BackendResult<MetalTensor> {
        if MetalTensor::dtype(self) != DType::F32 {
            return Err(BackendError::TypeMismatch(format!("sum_impl supports F32, got {:?}", MetalTensor::dtype(self))));
        }
        let (outer, axis_size, inner, _, new_shape) = decompose_axis(MetalTensor::shape(self), axis)?;
        dispatch_reduce(self, get_reduce_sum_pipeline(), outer, axis_size, inner, &new_shape)
    }

    /// max（軸指定）(Metal GPU 実装)
    pub fn max_impl(&self, axis: i32) -> BackendResult<MetalTensor> {
        if MetalTensor::dtype(self) != DType::F32 {
            return Err(BackendError::TypeMismatch(format!("max_impl supports F32, got {:?}", MetalTensor::dtype(self))));
        }
        let (outer, axis_size, inner, _, new_shape) = decompose_axis(MetalTensor::shape(self), axis)?;
        dispatch_reduce(self, get_reduce_max_pipeline(), outer, axis_size, inner, &new_shape)
    }

    /// min（軸指定）(Metal GPU 実装)
    pub fn min_impl(&self, axis: i32) -> BackendResult<MetalTensor> {
        if MetalTensor::dtype(self) != DType::F32 {
             return Err(BackendError::TypeMismatch(format!("min_impl supports F32, got {:?}", MetalTensor::dtype(self))));
        }
        let (outer, axis_size, inner, _, new_shape) = decompose_axis(MetalTensor::shape(self), axis)?;
        dispatch_reduce(self, get_reduce_min_pipeline(), outer, axis_size, inner, &new_shape)
    }

    /// argmax（軸指定）(Metal GPU 実装)
    pub fn argmax_impl(&self, axis: i32) -> BackendResult<MetalTensor> {
        if MetalTensor::dtype(self) != DType::F32 {
            return Err(BackendError::TypeMismatch(format!("argmax_impl supports F32, got {:?}", MetalTensor::dtype(self))));
        }
        let (outer, axis_size, inner, _, new_shape) = decompose_axis(MetalTensor::shape(self), axis)?;
        dispatch_reduce(self, get_reduce_argmax_pipeline(), outer, axis_size, inner, &new_shape)
    }

    /// argmin（軸指定）(Metal GPU 実装)
    pub fn argmin_impl(&self, axis: i32) -> BackendResult<MetalTensor> {
        if MetalTensor::dtype(self) != DType::F32 {
            return Err(BackendError::TypeMismatch(format!("argmin_impl supports F32, got {:?}", MetalTensor::dtype(self))));
        }
        let (outer, axis_size, inner, _, new_shape) = decompose_axis(MetalTensor::shape(self), axis)?;
        dispatch_reduce(self, get_reduce_argmin_pipeline(), outer, axis_size, inner, &new_shape)
    }

    /// mean（軸指定）— sum / axis_size で GPU 演算を活用
    pub fn mean_impl(&self, axis: i32) -> BackendResult<MetalTensor> {
        let sum = self.sum_impl(axis)?;
        let shape = MetalTensor::shape(self);
        let ndim = shape.len();
        let a = if axis < 0 { (ndim as i32 + axis) as usize } else { axis as usize };
        // Avoid unwrap on shape access if possible, but here we know index is valid if sum_impl succeeded?
        // Actually sum_impl checks index.
        let axis_size = shape[a] as f32;
        sum.div_scalar_impl(axis_size)
    }
}
