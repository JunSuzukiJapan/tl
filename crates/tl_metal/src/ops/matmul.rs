//! 行列積（matmul）

use crate::device::get_device;
use crate::tensor::MetalTensor;
use crate::DType;
use metal::{ComputePipelineState, MTLSize};

/// Matmul 用 Shader ソース
const MATMUL_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// 行列積: C = A * B
// A: [M, K], B: [K, N], C: [M, N]
kernel void matmul_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;  // M 方向
    uint col = gid.x;  // N 方向
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
"#;

/// Matmul 用パイプライン（キャッシュ）
static MATMUL_PIPELINE: std::sync::OnceLock<ComputePipelineState> = std::sync::OnceLock::new();

fn get_matmul_pipeline() -> &'static ComputePipelineState {
    MATMUL_PIPELINE.get_or_init(|| {
        let device = get_device();
        let options = metal::CompileOptions::new();
        let library = device
            .device()
            .new_library_with_source(MATMUL_SHADER, &options)
            .expect("Failed to compile matmul shader");
        let function = library
            .get_function("matmul_f32", None)
            .expect("matmul_f32 not found");
        device
            .device()
            .new_compute_pipeline_state_with_function(&function)
            .expect("Failed to create matmul pipeline")
    })
}

impl MetalTensor {
    /// 行列積: self * other
    /// self: [M, K], other: [K, N] -> result: [M, N]
    pub fn matmul_impl(&self, other: &MetalTensor) -> MetalTensor {
        assert_eq!(MetalTensor::dtype(self), DType::F32, "matmul only supports F32");
        assert_eq!(MetalTensor::dtype(other), DType::F32, "matmul only supports F32");
        
        let self_shape = MetalTensor::shape(self);
        let other_shape = MetalTensor::shape(other);
        
        // 形状チェック
        assert!(self_shape.len() == 2, "self must be 2D");
        assert!(other_shape.len() == 2, "other must be 2D");
        
        let m = self_shape[0];
        let k1 = self_shape[1];
        let k2 = other_shape[0];
        let n = other_shape[1];
        
        assert_eq!(k1, k2, "Inner dimensions must match: {} vs {}", k1, k2);
        
        let result = MetalTensor::uninit(&[m, n], DType::F32);
        let device = get_device();
        let command_queue = device.command_queue();
        let pipeline = get_matmul_pipeline();
        
        // 次元パラメータ用バッファ
        let m_buf = device.device().new_buffer_with_data(
            &(m as u32) as *const u32 as *const _,
            4,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let n_buf = device.device().new_buffer_with_data(
            &(n as u32) as *const u32 as *const _,
            4,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let k_buf = device.device().new_buffer_with_data(
            &(k1 as u32) as *const u32 as *const _,
            4,
            metal::MTLResourceOptions::StorageModeShared,
        );
        
        // GPU 実行
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(self.buffer()), 0);
        encoder.set_buffer(1, Some(other.buffer()), 0);
        encoder.set_buffer(2, Some(result.buffer()), 0);
        encoder.set_buffer(3, Some(&m_buf), 0);
        encoder.set_buffer(4, Some(&n_buf), 0);
        encoder.set_buffer(5, Some(&k_buf), 0);
        
        // 2D グリッド
        let threads_per_group = MTLSize::new(16, 16, 1);
        let grid_size = MTLSize::new(
            ((n + 15) / 16) as u64,
            ((m + 15) / 16) as u64,
            1,
        );
        encoder.dispatch_thread_groups(grid_size, threads_per_group);
        encoder.end_encoding();
        
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        result
    }
}
