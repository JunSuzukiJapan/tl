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
    /// 2D×2D: [M, K] × [K, N] → [M, N]
    /// 3D×2D: [B, M, K] × [K, N] → [B, M, N] (batch matmul)
    /// 3D×3D: [B, M, K] × [B, K, N] → [B, M, N] (batch matmul)
    pub fn matmul_impl(&self, other: &MetalTensor) -> MetalTensor {
        assert_eq!(MetalTensor::dtype(self), DType::F32, "matmul only supports F32");
        assert_eq!(MetalTensor::dtype(other), DType::F32, "matmul only supports F32");

        let self_shape = MetalTensor::shape(self);
        let other_shape = MetalTensor::shape(other);

        // 3D × 2D: batch matmul
        if self_shape.len() == 3 && other_shape.len() == 2 {
            let batch = self_shape[0];
            let m = self_shape[1];
            let k = self_shape[2];
            let k2 = other_shape[0];
            let n = other_shape[1];
            assert_eq!(k, k2, "Inner dimensions must match: {} vs {}", k, k2);

            let result = MetalTensor::uninit(&[batch, m, n], DType::F32);
            let device = get_device();
            let command_queue = device.command_queue();
            let pipeline = get_matmul_pipeline();

            let m_buf = device.device().new_buffer_with_data(
                &(m as u32) as *const u32 as *const _, 4,
                metal::MTLResourceOptions::StorageModeShared,
            );
            let n_buf = device.device().new_buffer_with_data(
                &(n as u32) as *const u32 as *const _, 4,
                metal::MTLResourceOptions::StorageModeShared,
            );
            let k_buf = device.device().new_buffer_with_data(
                &(k as u32) as *const u32 as *const _, 4,
                metal::MTLResourceOptions::StorageModeShared,
            );

            // 各 batch を個別に GPU 実行
            for b in 0..batch {
                let self_offset = (b * m * k * 4) as u64;  // f32 = 4 bytes
                let result_offset = (b * m * n * 4) as u64;

                let command_buffer = command_queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();

                encoder.set_compute_pipeline_state(pipeline);
                encoder.set_buffer(0, Some(self.buffer()), self_offset);
                encoder.set_buffer(1, Some(other.buffer()), 0);
                encoder.set_buffer(2, Some(result.buffer()), result_offset);
                encoder.set_buffer(3, Some(&m_buf), 0);
                encoder.set_buffer(4, Some(&n_buf), 0);
                encoder.set_buffer(5, Some(&k_buf), 0);

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
            }

            return result;
        }

        // 3D × 3D: batch matmul
        if self_shape.len() == 3 && other_shape.len() == 3 {
            let batch = self_shape[0];
            assert_eq!(batch, other_shape[0], "Batch dimensions must match: {} vs {}", batch, other_shape[0]);
            let m = self_shape[1];
            let k = self_shape[2];
            let k2 = other_shape[1];
            let n = other_shape[2];
            assert_eq!(k, k2, "Inner dimensions must match: {} vs {}", k, k2);

            let result = MetalTensor::uninit(&[batch, m, n], DType::F32);
            let device = get_device();
            let command_queue = device.command_queue();
            let pipeline = get_matmul_pipeline();

            let m_buf = device.device().new_buffer_with_data(
                &(m as u32) as *const u32 as *const _, 4,
                metal::MTLResourceOptions::StorageModeShared,
            );
            let n_buf = device.device().new_buffer_with_data(
                &(n as u32) as *const u32 as *const _, 4,
                metal::MTLResourceOptions::StorageModeShared,
            );
            let k_buf = device.device().new_buffer_with_data(
                &(k as u32) as *const u32 as *const _, 4,
                metal::MTLResourceOptions::StorageModeShared,
            );

            for b in 0..batch {
                let self_offset = (b * m * k * 4) as u64;
                let other_offset = (b * k * n * 4) as u64;
                let result_offset = (b * m * n * 4) as u64;

                let command_buffer = command_queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();

                encoder.set_compute_pipeline_state(pipeline);
                encoder.set_buffer(0, Some(self.buffer()), self_offset);
                encoder.set_buffer(1, Some(other.buffer()), other_offset);
                encoder.set_buffer(2, Some(result.buffer()), result_offset);
                encoder.set_buffer(3, Some(&m_buf), 0);
                encoder.set_buffer(4, Some(&n_buf), 0);
                encoder.set_buffer(5, Some(&k_buf), 0);

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
            }

            return result;
        }

        // 4D × 4D: batched head matmul (Attention用)
        // [B, H, M, K] × [B, H, K, N] → [B, H, M, N]
        if self_shape.len() == 4 && other_shape.len() == 4 {
            let b = self_shape[0];
            let h = self_shape[1];
            let m = self_shape[2];
            let k = self_shape[3];
            assert_eq!(b, other_shape[0], "Batch dim mismatch");
            assert_eq!(h, other_shape[1], "Head dim mismatch");
            assert_eq!(k, other_shape[2], "Inner dim mismatch: {} vs {}", k, other_shape[2]);
            let n = other_shape[3];

            // Flatten to 3D: [B*H, M, K] and [B*H, K, N]
            let self_3d = MetalTensor::from_buffer_shared(
                self.buffer_arc().clone(),
                vec![b * h, m, k],
                self.dtype(),
            );
            let other_3d = MetalTensor::from_buffer_shared(
                other.buffer_arc().clone(),
                vec![b * h, k, n],
                other.dtype(),
            );
            let result_3d = self_3d.matmul_impl(&other_3d);
            // Reshape back to 4D: [B, H, M, N]
            return MetalTensor::from_buffer_shared(
                result_3d.buffer_arc().clone(),
                vec![b, h, m, n],
                result_3d.dtype(),
            );
        }

        // 2D × 2D: 標準 matmul
        assert!(self_shape.len() == 2, "self must be 2D, 3D, or 4D, got {}D", self_shape.len());
        assert!(other_shape.len() == 2, "other must be 2D, 3D, or 4D, got {}D", other_shape.len());

        let m = self_shape[0];
        let k1 = self_shape[1];
        let k2 = other_shape[0];
        let n = other_shape[1];

        assert_eq!(k1, k2, "Inner dimensions must match: {} vs {}", k1, k2);

        let result = MetalTensor::uninit(&[m, n], DType::F32);
        let device = get_device();
        let command_queue = device.command_queue();
        let pipeline = get_matmul_pipeline();

        let m_buf = device.device().new_buffer_with_data(
            &(m as u32) as *const u32 as *const _, 4,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let n_buf = device.device().new_buffer_with_data(
            &(n as u32) as *const u32 as *const _, 4,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let k_buf = device.device().new_buffer_with_data(
            &(k1 as u32) as *const u32 as *const _, 4,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(self.buffer()), 0);
        encoder.set_buffer(1, Some(other.buffer()), 0);
        encoder.set_buffer(2, Some(result.buffer()), 0);
        encoder.set_buffer(3, Some(&m_buf), 0);
        encoder.set_buffer(4, Some(&n_buf), 0);
        encoder.set_buffer(5, Some(&k_buf), 0);

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
