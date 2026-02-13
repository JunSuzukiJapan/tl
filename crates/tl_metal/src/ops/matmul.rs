//! 行列積（matmul）

use crate::device::get_device;
use crate::tensor::MetalTensor;
use crate::DType;
use metal::{ComputePipelineState, MTLSize};

/// Matmul 用 Shader ソース
const MATMUL_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// タイル化行列積: C = A * B
// A: [M, K], B: [K, N], C: [M, N]
// 16×16 タイル + threadgroup shared memory
// 各 threadgroup が 16×16 の出力タイルを協調計算
constant uint TILE_SIZE = 16;

kernel void matmul_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    uint row = gid.y;  // M 方向
    uint col = gid.x;  // N 方向
    
    // Shared memory for A and B tiles
    threadgroup float A_tile[TILE_SIZE][TILE_SIZE];
    threadgroup float B_tile[TILE_SIZE][TILE_SIZE];
    
    float sum = 0.0f;
    
    uint local_row = tid.y;
    uint local_col = tid.x;
    
    // K 方向をタイルサイズずつストライド
    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (uint t = 0; t < num_tiles; ++t) {
        // A タイルを shared memory にロード
        uint a_col = t * TILE_SIZE + local_col;
        if (row < M && a_col < K) {
            A_tile[local_row][local_col] = A[row * K + a_col];
        } else {
            A_tile[local_row][local_col] = 0.0f;
        }
        
        // B タイルを shared memory にロード
        uint b_row = t * TILE_SIZE + local_row;
        if (b_row < K && col < N) {
            B_tile[local_row][local_col] = B[b_row * N + col];
        } else {
            B_tile[local_row][local_col] = 0.0f;
        }
        
        // 全スレッドのロード完了を待機
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // タイル内乗算
        for (uint i = 0; i < TILE_SIZE; ++i) {
            sum += A_tile[local_row][i] * B_tile[i][local_col];
        }
        
        // 次のタイルのロード前に現在のタイルの読み取り完了を待機
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ========== 融合量子化 matmul: Q4_K (simdgroup 並列リダクション版) ==========
// out[M, N] = x[M, K] × W_q4k[N, K]^T
// W は [N, K] の行優先格納 (=nn.Linear の weight)、transpose 不要
//
// 1 simdgroup (32 threads) で 1 出力要素を協調計算:
//   - 各スレッドが K 方向の 1/32 を担当
//   - simd_shuffle_down で高速合算
//
// Q4_K ブロック: 256 要素 = 144 bytes
//   d:f16(2), dmin:f16(2), scales:u8×12, qs:u8×128

// Helper: decode scale and min from 12-byte scales array
inline float2 get_scale_min_k4(int j, device const uchar* scales) {
    float sc, m;
    if (j < 4) {
        sc = float(scales[j] & 63);
        m  = float(scales[j + 4] & 63);
    } else {
        sc = float((scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4));
        m  = float((scales[j + 4] >> 4)   | ((scales[j] >> 6) << 4));
    }
    return float2(sc, m);
}

// Threadgroup layout: (32, N_ROWS_PER_TG)
//   - x方向: 32 threads = 1 simdgroup (K方向の分担)
//   - y方向: N_ROWS_PER_TG 個の出力行を同時処理
// Grid layout: (N, ceil(M / N_ROWS_PER_TG))
//   - x方向のthreadgroup数 = N (各出力列に1つ)
//   - y方向のthreadgroup数 = ceil(M / N_ROWS_PER_TG)
constant uint N_ROWS_PER_TG = 4;

kernel void mul_mv_q4_K_f32(
    device const float* x       [[buffer(0)]],  // [M, K]  input activations
    device const uchar* w_raw   [[buffer(1)]],  // Q4_K raw bytes [N * K_blocks * 144]
    device float*       out     [[buffer(2)]],  // [M, N]  output
    constant uint& M            [[buffer(3)]],
    constant uint& N            [[buffer(4)]],
    constant uint& K            [[buffer(5)]],
    uint2 tg_id  [[threadgroup_position_in_grid]],   // (col, row_group)
    uint2 tid    [[thread_position_in_threadgroup]]   // (lane, local_row)
) {
    uint col = tg_id.x;           // 出力列 (N方向)
    uint row = tg_id.y * N_ROWS_PER_TG + tid.y;  // 出力行 (M方向)
    uint lane = tid.x;            // simd lane (0..31), K方向の分担
    
    if (col >= N || row >= M) return;
    
    uint K_blocks = K / 256;  // Q4_K blocks per weight row
    
    float sum = 0.0f;
    
    // x row pointer
    device const float* x_row = x + row * K;
    
    // W row pointer for output neuron 'col'
    device const uchar* w_row = w_raw + col * (K_blocks * 144);
    
    // 各 lane が担当するブロック範囲
    // 32 lanes で K_blocks を分担
    // lane i は block i, i+32, i+64, ... を担当
    for (uint bi = lane; bi < K_blocks; bi += 32) {
        device const uchar* block = w_row + bi * 144;
        uint k_base = bi * 256;
        
        float df    = float(*(device const half*)(block));
        float dminf = float(*(device const half*)(block + 2));
        
        device const uchar* scales = block + 4;
        device const uchar* qs = block + 16;
        
        uint k_off = k_base;
        for (int pair = 0; pair < 4; ++pair) {
            int is0 = 2 * pair;
            int is1 = 2 * pair + 1;
            
            float2 sm0 = get_scale_min_k4(is0, scales);
            float d1   = df * sm0.x;
            float min1 = dminf * sm0.y;
            
            float2 sm1 = get_scale_min_k4(is1, scales);
            float d2   = df * sm1.x;
            float min2 = dminf * sm1.y;
            
            device const uchar* qs_ptr = qs + pair * 32;
            
            for (int l = 0; l < 32; ++l) {
                float w_val = d1 * float(qs_ptr[l] & 0x0F) - min1;
                sum += x_row[k_off + l] * w_val;
            }
            k_off += 32;
            
            for (int l = 0; l < 32; ++l) {
                float w_val = d2 * float(qs_ptr[l] >> 4) - min2;
                sum += x_row[k_off + l] * w_val;
            }
            k_off += 32;
        }
    }
    
    // simdgroup 並列リダクション: 32 lanes の部分和を合算
    // simd_shuffle_down で隣接 lane と合算 (5 ステップ: 16,8,4,2,1)
    sum += simd_shuffle_down(sum, 16);
    sum += simd_shuffle_down(sum, 8);
    sum += simd_shuffle_down(sum, 4);
    sum += simd_shuffle_down(sum, 2);
    sum += simd_shuffle_down(sum, 1);
    
    // lane 0 が結果を書き出し
    if (lane == 0) {
        out[row * N + col] = sum;
    }
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

/// 融合量子化 matmul 用パイプライン（キャッシュ）
static MUL_MV_Q4K_PIPELINE: std::sync::OnceLock<ComputePipelineState> = std::sync::OnceLock::new();

fn get_mul_mv_q4k_pipeline() -> &'static ComputePipelineState {
    MUL_MV_Q4K_PIPELINE.get_or_init(|| {
        let device = get_device();
        let options = metal::CompileOptions::new();
        let library = device
            .device()
            .new_library_with_source(MATMUL_SHADER, &options)
            .expect("Failed to compile matmul shader (q4k)");
        let function = library
            .get_function("mul_mv_q4_K_f32", None)
            .expect("mul_mv_q4_K_f32 not found");
        device
            .device()
            .new_compute_pipeline_state_with_function(&function)
            .expect("Failed to create mul_mv_q4k pipeline")
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

            let command_buffer = command_queue.new_command_buffer();

            for b in 0..batch {
                let self_offset = (b * m * k * 4) as u64;  // f32 = 4 bytes
                let result_offset = (b * m * n * 4) as u64;

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
            }

            command_buffer.commit();
            command_buffer.wait_until_completed();

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

            let command_buffer = command_queue.new_command_buffer();

            for b in 0..batch {
                let self_offset = (b * m * k * 4) as u64;
                let other_offset = (b * k * n * 4) as u64;
                let result_offset = (b * m * n * 4) as u64;

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
            }

            command_buffer.commit();
            command_buffer.wait_until_completed();

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

    /// 融合量子化 matmul: x[M, K] × W_q4k[N, K]^T → out[M, N]
    /// simdgroup 並列リダクション版:
    ///   32 threads (1 simdgroup) で 1 出力要素を協調計算
    pub fn mul_mv_q4_k(&self, w_raw: &MetalTensor, n: usize, k: usize) -> MetalTensor {
        let self_shape = MetalTensor::shape(self);
        let m = if self_shape.len() == 2 {
            self_shape[0]
        } else if self_shape.len() == 1 {
            1
        } else {
            panic!("mul_mv_q4_k: input must be 1D or 2D, got {}D", self_shape.len());
        };

        assert!(k % 256 == 0, "mul_mv_q4_k: K must be multiple of 256, got {}", k);

        let n_rows_per_tg: usize = 4; // MSL 側の N_ROWS_PER_TG と一致

        let result = MetalTensor::uninit(&[m, n], DType::F32);
        let device = get_device();
        let command_queue = device.command_queue();
        let pipeline = get_mul_mv_q4k_pipeline();

        let m_u32 = m as u32;
        let n_u32 = n as u32;
        let k_u32 = k as u32;

        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(self.buffer()), 0);
        encoder.set_buffer(1, Some(w_raw.buffer()), 0);
        encoder.set_buffer(2, Some(result.buffer()), 0);
        encoder.set_bytes(3, 4, &m_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(4, 4, &n_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(5, 4, &k_u32 as *const u32 as *const std::ffi::c_void);

        // Threadgroup: (32, N_ROWS_PER_TG) — 32 = simd 幅
        let threads_per_group = MTLSize::new(32, n_rows_per_tg as u64, 1);
        // Grid: N threadgroups in x (各出力列), ceil(M/N_ROWS_PER_TG) in y
        let grid_size = MTLSize::new(
            n as u64,
            ((m + n_rows_per_tg - 1) / n_rows_per_tg) as u64,
            1,
        );
        encoder.dispatch_thread_groups(grid_size, threads_per_group);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        result
    }

    /// 融合量子化 matmul: x[M, K] × W_q6k[N, K]^T → out[M, N]
    /// simdgroup 並列リダクション版 (Q6_K)
    pub fn mul_mv_q6_k(&self, w_raw: &MetalTensor, n: usize, k: usize) -> MetalTensor {
        let self_shape = MetalTensor::shape(self);
        let m = if self_shape.len() == 2 {
            self_shape[0]
        } else if self_shape.len() == 1 {
            1
        } else {
            panic!("mul_mv_q6_k: input must be 1D or 2D, got {}D", self_shape.len());
        };

        assert!(k % 256 == 0, "mul_mv_q6_k: K must be multiple of 256, got {}", k);

        let n_rows_per_tg: usize = 4;

        let result = MetalTensor::uninit(&[m, n], DType::F32);
        let device = get_device();
        let command_queue = device.command_queue();
        let pipeline = get_mul_mv_q6k_pipeline();

        let m_u32 = m as u32;
        let n_u32 = n as u32;
        let k_u32 = k as u32;

        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(self.buffer()), 0);
        encoder.set_buffer(1, Some(w_raw.buffer()), 0);
        encoder.set_buffer(2, Some(result.buffer()), 0);
        encoder.set_bytes(3, 4, &m_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(4, 4, &n_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(5, 4, &k_u32 as *const u32 as *const std::ffi::c_void);

        let threads_per_group = MTLSize::new(32, n_rows_per_tg as u64, 1);
        let grid_size = MTLSize::new(
            n as u64,
            ((m + n_rows_per_tg - 1) / n_rows_per_tg) as u64,
            1,
        );
        encoder.dispatch_thread_groups(grid_size, threads_per_group);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        result
    }

}

// ========== Q6_K 融合カーネルシェーダ ==========
const MUL_MV_Q6K_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Q6_K ブロック: 256 要素 = 210 bytes
//   ql: 128 bytes (下位4bit)
//   qh: 64 bytes (上位2bit)
//   scales: 16 bytes (i8 × 16)
//   d: 2 bytes (f16)
// dequant: val = d * scale * (q6 - 32)
//   q6 = (ql_nibble) | (qh_2bits << 4)  → 0..63 range

constant uint Q6K_N_ROWS_PER_TG = 4;

kernel void mul_mv_q6_K_f32(
    device const float* x       [[buffer(0)]],
    device const uchar* w_raw   [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint& M            [[buffer(3)]],
    constant uint& N            [[buffer(4)]],
    constant uint& K            [[buffer(5)]],
    uint2 tg_id  [[threadgroup_position_in_grid]],
    uint2 tid    [[thread_position_in_threadgroup]]
) {
    uint col = tg_id.x;
    uint row = tg_id.y * Q6K_N_ROWS_PER_TG + tid.y;
    uint lane = tid.x;
    
    if (col >= N || row >= M) return;
    
    uint K_blocks = K / 256;
    
    float sum = 0.0f;
    
    device const float* x_row = x + row * K;
    device const uchar* w_row = w_raw + col * (K_blocks * 210);
    
    for (uint bi = lane; bi < K_blocks; bi += 32) {
        device const uchar* block = w_row + bi * 210;
        uint k_base = bi * 256;
        
        device const uchar* ql = block;          // 128 bytes
        device const uchar* qh = block + 128;    // 64 bytes
        device const char*  sc = (device const char*)(block + 192);  // 16 bytes (i8)
        float d = float(*(device const half*)(block + 208));
        
        // Process 256 weights in 2 chunks of 128
        for (int n_chunk = 0; n_chunk < 2; ++n_chunk) {
            uint chunk_k = k_base + n_chunk * 128;
            device const uchar* ql_ptr = ql + n_chunk * 64;
            device const uchar* qh_ptr = qh + n_chunk * 32;
            device const char*  sc_ptr = sc + n_chunk * 8;
            
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                
                uchar ql_lo = ql_ptr[l];
                uchar ql_hi = ql_ptr[l + 32];
                uchar qh_val = qh_ptr[l];
                
                int q1 = int((ql_lo & 0x0F) | ((qh_val & 0x03) << 4)) - 32;
                int q2 = int((ql_hi & 0x0F) | (((qh_val >> 2) & 0x03) << 4)) - 32;
                int q3 = int((ql_lo >> 4)    | (((qh_val >> 4) & 0x03) << 4)) - 32;
                int q4 = int((ql_hi >> 4)    | (((qh_val >> 6) & 0x03) << 4)) - 32;
                
                float s1 = d * float(sc_ptr[is + 0]);
                float s2 = d * float(sc_ptr[is + 2]);
                float s3 = d * float(sc_ptr[is + 4]);
                float s4 = d * float(sc_ptr[is + 6]);
                
                sum += x_row[chunk_k + l +  0] * (s1 * float(q1));
                sum += x_row[chunk_k + l + 32] * (s2 * float(q2));
                sum += x_row[chunk_k + l + 64] * (s3 * float(q3));
                sum += x_row[chunk_k + l + 96] * (s4 * float(q4));
            }
        }
    }
    
    // simdgroup 並列リダクション
    sum += simd_shuffle_down(sum, 16);
    sum += simd_shuffle_down(sum, 8);
    sum += simd_shuffle_down(sum, 4);
    sum += simd_shuffle_down(sum, 2);
    sum += simd_shuffle_down(sum, 1);
    
    if (lane == 0) {
        out[row * N + col] = sum;
    }
}
"#;

/// Q6_K 融合 matmul パイプライン
static MUL_MV_Q6K_PIPELINE: std::sync::OnceLock<ComputePipelineState> = std::sync::OnceLock::new();

fn get_mul_mv_q6k_pipeline() -> &'static ComputePipelineState {
    MUL_MV_Q6K_PIPELINE.get_or_init(|| {
        let device = get_device();
        let options = metal::CompileOptions::new();
        let library = device
            .device()
            .new_library_with_source(MUL_MV_Q6K_SHADER, &options)
            .expect("Failed to compile Q6_K matmul shader");
        let function = library
            .get_function("mul_mv_q6_K_f32", None)
            .expect("mul_mv_q6_K_f32 not found");
        device
            .device()
            .new_compute_pipeline_state_with_function(&function)
            .expect("Failed to create mul_mv_q6k pipeline")
    })
}
