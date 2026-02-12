//! Metal Shader パイプライン管理

use metal::{ComputePipelineState, Device, Library, MTLSize};
use std::collections::HashMap;

/// Shader 関数名 - 二項演算
pub const SHADER_ADD_F32: &str = "add_f32";
pub const SHADER_SUB_F32: &str = "sub_f32";
pub const SHADER_MUL_F32: &str = "mul_f32";
pub const SHADER_DIV_F32: &str = "div_f32";
pub const SHADER_POW_F32: &str = "pow_f32";
pub const SHADER_FMOD_F32: &str = "fmod_f32";

/// Shader 関数名 - 単項演算
pub const SHADER_NEG_F32: &str = "neg_f32";
pub const SHADER_ABS_F32: &str = "abs_f32";
pub const SHADER_EXP_F32: &str = "exp_f32";
pub const SHADER_LOG_F32: &str = "log_f32";
pub const SHADER_SQRT_F32: &str = "sqrt_f32";
pub const SHADER_TANH_F32: &str = "tanh_f32";
pub const SHADER_SIGMOID_F32: &str = "sigmoid_f32";
pub const SHADER_RELU_F32: &str = "relu_f32";
pub const SHADER_SIN_F32: &str = "sin_f32";
pub const SHADER_COS_F32: &str = "cos_f32";
pub const SHADER_TAN_F32: &str = "tan_f32";
pub const SHADER_GELU_F32: &str = "gelu_f32";
pub const SHADER_SILU_F32: &str = "silu_f32";

/// Shader 関数名 - スカラー演算
pub const SHADER_ADD_SCALAR_F32: &str = "add_scalar_f32";
pub const SHADER_MUL_SCALAR_F32: &str = "mul_scalar_f32";
pub const SHADER_CLAMP_F32: &str = "clamp_f32";
pub const SHADER_POW_SCALAR_F32: &str = "pow_scalar_f32";
pub const SHADER_FMOD_SCALAR_F32: &str = "fmod_scalar_f32";

/// Shader 関数名 - Reduce
pub const SHADER_SUMALL_F32: &str = "sumall_f32";
pub const SHADER_ARGMAX_F32: &str = "argmax_f32";
pub const SHADER_ARGMIN_F32: &str = "argmin_f32";

/// Shader 関数名 - 比較演算
pub const SHADER_EQ_F32: &str = "eq_f32";
pub const SHADER_NE_F32: &str = "ne_f32";
pub const SHADER_LT_F32: &str = "lt_f32";
pub const SHADER_LE_F32: &str = "le_f32";
pub const SHADER_GT_F32: &str = "gt_f32";
pub const SHADER_GE_F32: &str = "ge_f32";

/// Shader 関数名 - Quantize
pub const SHADER_DEQUANTIZE_Q4_K: &str = "dequantize_q4_k";

/// Metal Shader ソースコード
const SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ========== 二項演算 ==========

kernel void add_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = a[id] + b[id];
}

kernel void sub_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = a[id] - b[id];
}

kernel void mul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = a[id] * b[id];
}

kernel void div_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = a[id] / b[id];
}

kernel void pow_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = pow(a[id], b[id]);
}

kernel void fmod_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = fmod(a[id], b[id]);
}

// ========== 比較演算 ==========

kernel void eq_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = (a[id] == b[id]) ? 1.0f : 0.0f;
}

kernel void ne_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = (a[id] != b[id]) ? 1.0f : 0.0f;
}

kernel void lt_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = (a[id] < b[id]) ? 1.0f : 0.0f;
}

kernel void le_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = (a[id] <= b[id]) ? 1.0f : 0.0f;
}

kernel void gt_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = (a[id] > b[id]) ? 1.0f : 0.0f;
}

kernel void ge_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = (a[id] >= b[id]) ? 1.0f : 0.0f;
}

// ========== 単項演算 ==========

kernel void neg_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = -a[id];
}

kernel void abs_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = abs(a[id]);
}

kernel void exp_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = exp(a[id]);
}

kernel void log_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = log(a[id]);
}

kernel void sqrt_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = sqrt(a[id]);
}

kernel void tanh_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = tanh(a[id]);
}

kernel void sigmoid_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = 1.0f / (1.0f + exp(-a[id]));
}

kernel void relu_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = max(0.0f, a[id]);
}

kernel void sin_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = sin(a[id]);
}

kernel void cos_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = cos(a[id]);
}

kernel void tan_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = tan(a[id]);
}

// GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
kernel void gelu_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    float x = a[id];
    float cdf = 0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    out[id] = x * cdf;
}

// SiLU: x * sigmoid(x) = x / (1 + exp(-x))
kernel void silu_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    float x = a[id];
    out[id] = x / (1.0f + exp(-x));
}

// ========== スカラー演算 ==========

kernel void add_scalar_f32(
    device const float* a [[buffer(0)]],
    constant float& scalar [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = a[id] + scalar;
}

kernel void mul_scalar_f32(
    device const float* a [[buffer(0)]],
    constant float& scalar [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = a[id] * scalar;
}

kernel void clamp_f32(
    device const float* a [[buffer(0)]],
    constant float& min_val [[buffer(1)]],
    constant float& max_val [[buffer(2)]],
    device float* out [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = clamp(a[id], min_val, max_val);
}

kernel void pow_scalar_f32(
    device const float* a [[buffer(0)]],
    constant float& exp [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = pow(a[id], exp);
}

kernel void fmod_scalar_f32(
    device const float* a [[buffer(0)]],
    constant float& s [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = fmod(a[id], s);
}

// ========== Reduce (部分和) ==========

// Threadgroup-local reduction for sumall
kernel void sumall_f32(
    device const float* input [[buffer(0)]],
    device float* partial_sums [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint id [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    threadgroup float shared_data[256];
    
    // Load data
    float val = (id < count) ? input[id] : 0.0f;
    shared_data[tid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction in shared memory
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (tid == 0) {
        partial_sums[tg_id] = shared_data[0];
    }
}

// ========== Argmax/Argmin (部分結果) ==========

// Argmax: 各スレッドグループで最大値とそのインデックスを求める
kernel void argmax_f32(
    device const float* input [[buffer(0)]],
    device float* partial_max [[buffer(1)]],     // 各グループの最大値
    device uint* partial_idx [[buffer(2)]],      // 各グループの最大値インデックス
    constant uint& count [[buffer(3)]],
    uint id [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    threadgroup float shared_val[256];
    threadgroup uint shared_idx[256];
    
    // Load data
    float val = (id < count) ? input[id] : -INFINITY;
    shared_val[tid] = val;
    shared_idx[tid] = id;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction in shared memory
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_val[tid + s] > shared_val[tid]) {
                shared_val[tid] = shared_val[tid + s];
                shared_idx[tid] = shared_idx[tid + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (tid == 0) {
        partial_max[tg_id] = shared_val[0];
        partial_idx[tg_id] = shared_idx[0];
    }
}

// Argmin: 各スレッドグループで最小値とそのインデックスを求める
kernel void argmin_f32(
    device const float* input [[buffer(0)]],
    device float* partial_min [[buffer(1)]],     // 各グループの最小値
    device uint* partial_idx [[buffer(2)]],      // 各グループの最小値インデックス
    constant uint& count [[buffer(3)]],
    uint id [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]
) {
    threadgroup float shared_val[256];
    threadgroup uint shared_idx[256];
    
    // Load data
    float val = (id < count) ? input[id] : INFINITY;
    shared_val[tid] = val;
    shared_idx[tid] = id;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction in shared memory
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_val[tid + s] < shared_val[tid]) {
                shared_val[tid] = shared_val[tid + s];
                shared_idx[tid] = shared_idx[tid + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (tid == 0) {
        partial_min[tg_id] = shared_val[0];
        partial_idx[tg_id] = shared_idx[0];
    }
}

// ========== Quantization (Q4_K) ==========
// Block size: 256, Size: 144 bytes

kernel void dequantize_q4_k(
    device const uchar* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    // 1 thread per block (256 elements)
    uint block_idx = gid;
    uint in_offset = block_idx * 144;
    uint out_offset = block_idx * 256;
    
    device const uchar* b = in + in_offset;
    
    // d, dmin (f16)
    half d = *(device const half*)(b);
    half dmin = *(device const half*)(b + 2);
    
    device const uchar* scales = b + 4;
    device const uchar* qs = b + 16;
    
    float df = (float)d;
    float dminf = (float)dmin;

    // Process 4 pairs of sub-blocks (8 sub-blocks total)
    for (int k = 0; k < 4; ++k) {
        // Pair of sub-blocks: 2*k and 2*k+1
        int is0 = 2 * k;
        int is1 = 2 * k + 1;
        
        // Decode scales/mins for is0
        uchar sc0, m0;
        if (is0 < 4) {
            sc0 = scales[is0] & 63;
            m0  = scales[is0 + 4] & 63;
        } else {
            sc0 = (scales[is0 + 4] & 0x0F) | ((scales[is0 - 4] >> 6) << 4);
            m0  = (scales[is0 + 4] >> 4)   | ((scales[is0] >> 6) << 4);
        }
        float d1 = df * sc0;
        float min1 = dminf * m0;

        // Decode scales/mins for is1
        uchar sc1, m1;
        if (is1 < 4) {
            sc1 = scales[is1] & 63;
            m1  = scales[is1 + 4] & 63;
        } else {
            sc1 = (scales[is1 + 4] & 0x0F) | ((scales[is1 - 4] >> 6) << 4);
            m1  = (scales[is1 + 4] >> 4)   | ((scales[is1] >> 6) << 4);
        }
        float d2 = df * sc1;
        float min2 = dminf * m1;
        
        // qs offset for this pair (32 bytes)
        device const uchar* qs_ptr = qs + k * 32;
        device float* y_ptr = out + out_offset + k * 64;
        
        // First 32 elements (is0) use low nibbles
        for (int l = 0; l < 32; ++l) {
            uchar val = qs_ptr[l] & 0x0F;
            y_ptr[l] = d1 * val - min1;
        }
        
        // Next 32 elements (is1) use high nibbles
        for (int l = 0; l < 32; ++l) {
            uchar val = qs_ptr[l] >> 4;
            y_ptr[32 + l] = d2 * val - min2;
        }
    }
}
"#;

/// Shader パイプラインを管理
pub struct ShaderPipelines {
    library: Library,
    pipelines: HashMap<String, ComputePipelineState>,
}

impl ShaderPipelines {
    /// デバイスから Shader をコンパイルして初期化
    pub fn new(device: &Device) -> Result<Self, String> {
        let options = metal::CompileOptions::new();
        let library = device
            .new_library_with_source(SHADER_SOURCE, &options)
            .map_err(|e| format!("Failed to compile shaders: {}", e))?;

        Ok(ShaderPipelines {
            library,
            pipelines: HashMap::new(),
        })
    }

    /// 指定した関数のパイプラインを取得（キャッシュ済みなら再利用）
    pub fn get_pipeline(
        &mut self,
        device: &Device,
        function_name: &str,
    ) -> Result<&ComputePipelineState, String> {
        if !self.pipelines.contains_key(function_name) {
            let function = self
                .library
                .get_function(function_name, None)
                .map_err(|e| format!("Function {} not found: {}", function_name, e))?;

            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| format!("Failed to create pipeline for {}: {}", function_name, e))?;

            self.pipelines.insert(function_name.to_string(), pipeline);
        }

        Ok(self.pipelines.get(function_name).unwrap())
    }
}

/// スレッドグループサイズを計算
pub fn compute_thread_groups(element_count: usize, pipeline: &ComputePipelineState) -> (MTLSize, MTLSize) {
    let thread_execution_width = pipeline.thread_execution_width() as usize;
    let threads_per_group = MTLSize::new(thread_execution_width as u64, 1, 1);
    let num_groups = (element_count + thread_execution_width - 1) / thread_execution_width;
    let grid_size = MTLSize::new(num_groups as u64, 1, 1);
    (grid_size, threads_per_group)
}

/// グローバル Shader パイプライン
static SHADER_PIPELINES: std::sync::OnceLock<std::sync::Mutex<ShaderPipelines>> = std::sync::OnceLock::new();

/// グローバル Shader パイプラインを取得
pub fn get_shaders() -> &'static std::sync::Mutex<ShaderPipelines> {
    SHADER_PIPELINES.get_or_init(|| {
        let device = crate::device::get_device();
        std::sync::Mutex::new(ShaderPipelines::new(device.device()).expect("Failed to init shaders"))
    })
}
