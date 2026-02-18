//! 活性化関数
//! Softmax は Metal GPU シェーダーで実装

use crate::device::get_device;
use crate::tensor::MetalTensor;
use crate::DType;
use metal::{ComputePipelineState, MTLSize};
use tl_backend::{BackendResult, BackendError};

/// Softmax 用 Metal シェーダー（汎用 N-D 対応）
/// outer × axis × inner の 3 軸分解パターン
const SOFTMAX_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Softmax: 各 (outer, inner) スライスに対して axis 方向に softmax を計算
// Step 1: max を計算
// Step 2: exp(x - max) を計算し sum を取得
// Step 3: exp(x - max) / sum で正規化
kernel void softmax_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& outer_size [[buffer(2)]],
    constant uint& axis_size [[buffer(3)]],
    constant uint& inner_size [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // gid.x = inner index, gid.y = outer index
    uint inner = gid.x;
    uint outer = gid.y;
    
    if (inner >= inner_size || outer >= outer_size) return;
    
    uint base = outer * axis_size * inner_size + inner;
    uint stride = inner_size;
    
    // Step 1: max
    float max_val = -INFINITY;
    for (uint a = 0; a < axis_size; a++) {
        float v = input[base + a * stride];
        max_val = max(max_val, v);
    }
    
    // Step 2: exp and sum
    float sum = 0.0f;
    for (uint a = 0; a < axis_size; a++) {
        float e = exp(input[base + a * stride] - max_val);
        output[base + a * stride] = e;
        sum += e;
    }
    
    // Step 3: normalize
    float inv_sum = 1.0f / sum;
    for (uint a = 0; a < axis_size; a++) {
        output[base + a * stride] *= inv_sum;
    }
}
"#;

static SOFTMAX_PIPELINE: std::sync::OnceLock<ComputePipelineState> = std::sync::OnceLock::new();

fn get_softmax_pipeline() -> &'static ComputePipelineState {
    SOFTMAX_PIPELINE.get_or_init(|| {
        let device = get_device();
        let options = metal::CompileOptions::new();
        let library = device
            .device()
            .new_library_with_source(SOFTMAX_SHADER, &options)
            .expect("Failed to compile softmax shader");
        let function = library
            .get_function("softmax_f32", None)
            .expect("softmax_f32 not found");
        device
            .device()
            .new_compute_pipeline_state_with_function(&function)
            .expect("Failed to create softmax pipeline")
    })
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

impl MetalTensor {
    /// Softmax（軸指定）— Metal GPU 実装
    /// 汎用 N-D 対応: outer × axis × inner の 3 軸分解
    pub fn softmax_impl(&self, axis: i32) -> BackendResult<MetalTensor> {
        if self.dtype() != DType::F32 {
            return Err(BackendError::TypeMismatch(format!("softmax only supports F32, got {:?}", self.dtype())));
        }
        
        let shape = MetalTensor::shape(self);
        let ndim = shape.len();
        let axis_idx = if axis < 0 { (ndim as i32 + axis) as usize } else { axis as usize };
        
        if axis_idx >= ndim {
            return Err(BackendError::IndexOutOfBounds(format!("softmax axis out of range: {} (ndim={})", axis, ndim)));
        }

        let outer_size: usize = shape[..axis_idx].iter().product::<usize>().max(1);
        let axis_size = shape[axis_idx];
        let inner_size: usize = shape[axis_idx + 1..].iter().product::<usize>().max(1);

        let result = MetalTensor::uninit(shape, DType::F32);
        let device = get_device();
        let command_queue = device.command_queue();
        let pipeline = get_softmax_pipeline();

        objc::rc::autoreleasepool(|| {
            let outer_buf = make_u32_buf(outer_size as u32);
            let axis_buf = make_u32_buf(axis_size as u32);
            let inner_buf = make_u32_buf(inner_size as u32);

            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(self.buffer()), 0);
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
        });

        Ok(result)
    }
}
