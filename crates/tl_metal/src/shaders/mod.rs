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

/// Shader 関数名 - Cast
pub const SHADER_CAST_F16_TO_F32: &str = "cast_f16_to_f32";
pub const SHADER_CAST_F32_TO_F16: &str = "cast_f32_to_f16";

/// Shader 関数名 - Quantize
pub const SHADER_DEQUANTIZE_Q4_K: &str = "dequantize_q4_k";

/// Shader 関数名 - 活性化関数 (追加)
pub const SHADER_LEAKY_RELU_F32: &str = "leaky_relu_f32";
pub const SHADER_ELU_F32: &str = "elu_f32";
pub const SHADER_MISH_F32: &str = "mish_f32";
pub const SHADER_HARDSWISH_F32: &str = "hardswish_f32";
pub const SHADER_HARDSIGMOID_F32: &str = "hardsigmoid_f32";

/// Shader 関数名 - 論理演算
pub const SHADER_LOGICAL_AND_F32: &str = "logical_and_f32";
pub const SHADER_LOGICAL_OR_F32: &str = "logical_or_f32";
pub const SHADER_LOGICAL_NOT_F32: &str = "logical_not_f32";

/// Shader 関数名 - Fill
pub const SHADER_FILL_F32: &str = "fill_f32";

/// Shader 関数名 - NN操作
pub const SHADER_CONV1D_F32: &str = "conv1d_f32";
pub const SHADER_CONV_TRANSPOSE2D_F32: &str = "conv_transpose2d_f32";
pub const SHADER_INTERPOLATE_NEAREST_F32: &str = "interpolate_nearest_f32";
pub const SHADER_INTERPOLATE_BILINEAR_F32: &str = "interpolate_bilinear_f32";
pub const SHADER_ADAPTIVE_AVG_POOL2D_F32: &str = "adaptive_avg_pool2d_f32";
pub const SHADER_PAD_F32: &str = "pad_f32";
pub const SHADER_CUMSUM_F32: &str = "cumsum_f32";
pub const SHADER_GROUP_NORM_F32: &str = "group_norm_f32";

/// Shader 関数名 - 融合カーネル
pub const SHADER_FUSED_SILU_MUL_F32: &str = "fused_silu_mul_f32";
pub const SHADER_FUSED_ADD_RELU_F32: &str = "fused_add_relu_f32";
pub const SHADER_FUSED_BIAS_GELU_F32: &str = "fused_bias_gelu_f32";
pub const SHADER_FUSED_RMS_NORM_F32: &str = "fused_rms_norm_f32";
pub const SHADER_FUSED_ADD_RMS_NORM_F32: &str = "fused_add_rms_norm_f32";
pub const SHADER_FUSED_ROTARY_EMB_F32: &str = "fused_rotary_emb_f32";

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

// ========== Cast ==========

kernel void cast_f16_to_f32(
    device const half* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = float(in[id]);
}

kernel void cast_f32_to_f16(
    device const float* in [[buffer(0)]],
    device half* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = half(in[id]);
}

// ========== Quantization (Q4_K) ==========
// Block size: 256, Size: 144 bytes

kernel void dequantize_q4_k(
    device const uchar* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& num_blocks [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_blocks) return;
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

// ========== 活性化関数 (追加) ==========

// LeakyReLU: max(slope*x, x)
kernel void leaky_relu_f32(
    device const float* a [[buffer(0)]],
    constant float& slope [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    float x = a[id];
    out[id] = (x > 0.0f) ? x : slope * x;
}

// ELU: x if x > 0, alpha*(exp(x)-1) otherwise
kernel void elu_f32(
    device const float* a [[buffer(0)]],
    constant float& alpha [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    float x = a[id];
    out[id] = (x > 0.0f) ? x : alpha * (exp(x) - 1.0f);
}

// Mish: x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
kernel void mish_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    float x = a[id];
    out[id] = x * tanh(log(1.0f + exp(x)));
}

// HardSwish: x * clamp(x+3, 0, 6) / 6
kernel void hardswish_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    float x = a[id];
    out[id] = x * clamp(x + 3.0f, 0.0f, 6.0f) / 6.0f;
}

// HardSigmoid: clamp((x+3)/6, 0, 1)
kernel void hardsigmoid_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    float x = a[id];
    out[id] = clamp((x + 3.0f) / 6.0f, 0.0f, 1.0f);
}

// ========== 論理演算 ==========

kernel void logical_and_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = (a[id] != 0.0f && b[id] != 0.0f) ? 1.0f : 0.0f;
}

kernel void logical_or_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = (a[id] != 0.0f || b[id] != 0.0f) ? 1.0f : 0.0f;
}

kernel void logical_not_f32(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = (a[id] == 0.0f) ? 1.0f : 0.0f;
}

// ========== Fill ==========

kernel void fill_f32(
    device float* out [[buffer(0)]],
    constant float& value [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < count) {
        out[id] = value;
    }
}

// ========== 融合カーネル ==========

// fused_silu_mul: silu(gate) * up を1カーネルで (LLaMA SwiGLU FFN)
kernel void fused_silu_mul_f32(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device float* out        [[buffer(2)]],
    constant uint& count     [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < count) {
        float x = gate[id];
        out[id] = (x / (1.0f + exp(-x))) * up[id];
    }
}

// fused_add_relu: relu(a + b) を1カーネルで (CNN 標準)
kernel void fused_add_relu_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out     [[buffer(2)]],
    constant uint& count  [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < count) {
        out[id] = max(a[id] + b[id], 0.0f);
    }
}

// fused_bias_gelu: gelu(x + bias) を1カーネルで (Transformer FFN)
kernel void fused_bias_gelu_f32(
    device const float* x    [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device float* out        [[buffer(2)]],
    constant uint& count     [[buffer(3)]],
    constant uint& bias_len  [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < count) {
        float v = x[id] + bias[id % bias_len];
        float cdf = 0.5f * (1.0f + tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
        out[id] = v * cdf;
    }
}

// fused_rms_norm: RMSNorm 1パス (2パスの統合)
// 各スレッドグループが1行を処理
kernel void fused_rms_norm_f32(
    device const float* input  [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant uint& N           [[buffer(3)]],
    constant uint& D           [[buffer(4)]],
    constant float& eps        [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= N || col >= D) return;
    
    // 行ごとの二乗平均を計算
    float sum_sq = 0.0f;
    for (uint i = 0; i < D; i++) {
        float v = input[row * D + i];
        sum_sq += v * v;
    }
    float rms = rsqrt(sum_sq / float(D) + eps);
    output[row * D + col] = input[row * D + col] * rms * weight[col];
}

// fused_add_rms_norm: residual加算 + RMSNorm を1パスで
kernel void fused_add_rms_norm_f32(
    device const float* input    [[buffer(0)]],
    device const float* residual [[buffer(1)]],
    device const float* weight   [[buffer(2)]],
    device float* output         [[buffer(3)]],
    constant uint& N             [[buffer(4)]],
    constant uint& D             [[buffer(5)]],
    constant float& eps          [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= N || col >= D) return;
    
    // residual 加算 + 二乗平均
    float sum_sq = 0.0f;
    for (uint i = 0; i < D; i++) {
        float v = input[row * D + i] + residual[row * D + i];
        sum_sq += v * v;
    }
    float rms = rsqrt(sum_sq / float(D) + eps);
    float added = input[row * D + col] + residual[row * D + col];
    output[row * D + col] = added * rms * weight[col];
}

// fused_rotary_emb: cos/sin 計算 + RoPE 適用を1パスで
kernel void fused_rotary_emb_f32(
    device const float* input  [[buffer(0)]],
    device const float* freqs  [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant uint& count       [[buffer(3)]],
    constant uint& head_dim    [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;
    uint half_dim = head_dim / 2;
    uint pair_idx = id % half_dim;
    uint base = id - pair_idx;
    
    float freq = freqs[pair_idx];
    float cos_val = cos(freq);
    float sin_val = sin(freq);
    
    float x0 = input[base + pair_idx];
    float x1 = input[base + pair_idx + half_dim];
    
    output[base + pair_idx] = x0 * cos_val - x1 * sin_val;
    output[base + pair_idx + half_dim] = x0 * sin_val + x1 * cos_val;
}

// ========== NN 操作 ==========

// Conv1d: batch×out_ch×out_len の各要素を1スレッドで計算
struct Conv1dParams {
    uint batch;
    uint in_ch;
    uint in_len;
    uint out_ch;
    uint k_len;
    uint stride;
    uint padding;
    uint out_len;
};

kernel void conv1d_f32(
    device const float* input  [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias   [[buffer(2)]],
    device float* output       [[buffer(3)]],
    constant Conv1dParams& p   [[buffer(4)]],
    constant uint& has_bias    [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    uint total = p.batch * p.out_ch * p.out_len;
    if (id >= total) return;
    uint ol = id % p.out_len;
    uint oc = (id / p.out_len) % p.out_ch;
    uint b  = id / (p.out_ch * p.out_len);
    float sum = 0.0f;
    for (uint ic = 0; ic < p.in_ch; ic++) {
        for (uint ki = 0; ki < p.k_len; ki++) {
            uint pos = ol * p.stride + ki;
            if (pos >= p.padding && pos < p.in_len + p.padding) {
                sum += input[b * p.in_ch * p.in_len + ic * p.in_len + (pos - p.padding)]
                     * weight[oc * p.in_ch * p.k_len + ic * p.k_len + ki];
            }
        }
    }
    if (has_bias != 0) sum += bias[oc];
    output[id] = sum;
}

// ConvTranspose2d: batch×out_ch×oh×ow の各要素を計算
struct ConvTranspose2dParams {
    uint batch;
    uint in_ch;
    uint ih;
    uint iw;
    uint out_ch;
    uint kh;
    uint kw;
    uint stride;
    uint padding;
    uint oh;
    uint ow;
};

kernel void conv_transpose2d_f32(
    device const float* input  [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias   [[buffer(2)]],
    device float* output       [[buffer(3)]],
    constant ConvTranspose2dParams& p [[buffer(4)]],
    constant uint& has_bias    [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    uint total = p.batch * p.out_ch * p.oh * p.ow;
    if (id >= total) return;
    uint ox = id % p.ow;
    uint oy = (id / p.ow) % p.oh;
    uint oc = (id / (p.ow * p.oh)) % p.out_ch;
    uint b  = id / (p.out_ch * p.oh * p.ow);
    float sum = 0.0f;
    for (uint ic = 0; ic < p.in_ch; ic++) {
        for (uint ky = 0; ky < p.kh; ky++) {
            for (uint kx = 0; kx < p.kw; kx++) {
                int iy_check = (int)(oy + p.padding) - (int)ky;
                int ix_check = (int)(ox + p.padding) - (int)kx;
                if (iy_check >= 0 && ix_check >= 0 &&
                    iy_check % p.stride == 0 && ix_check % p.stride == 0) {
                    uint iy = (uint)iy_check / p.stride;
                    uint ix = (uint)ix_check / p.stride;
                    if (iy < p.ih && ix < p.iw) {
                        sum += input[b * p.in_ch * p.ih * p.iw + ic * p.ih * p.iw + iy * p.iw + ix]
                             * weight[ic * p.out_ch * p.kh * p.kw + oc * p.kh * p.kw + ky * p.kw + kx];
                    }
                }
            }
        }
    }
    if (has_bias != 0) sum += bias[oc];
    output[id] = sum;
}

// Interpolate Nearest: 各出力ピクセルが最近傍の入力ピクセルをコピー
struct InterpolateParams {
    uint batch;
    uint channels;
    uint in_h;
    uint in_w;
    uint out_h;
    uint out_w;
};

kernel void interpolate_nearest_f32(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant InterpolateParams& p [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint total = p.batch * p.channels * p.out_h * p.out_w;
    if (id >= total) return;
    uint oj = id % p.out_w;
    uint oi = (id / p.out_w) % p.out_h;
    uint c  = (id / (p.out_w * p.out_h)) % p.channels;
    uint b  = id / (p.channels * p.out_h * p.out_w);
    uint si = oi * p.in_h / p.out_h;
    uint sj = oj * p.in_w / p.out_w;
    output[id] = input[b * p.channels * p.in_h * p.in_w + c * p.in_h * p.in_w + si * p.in_w + sj];
}

// Interpolate Bilinear: 双線形補間
kernel void interpolate_bilinear_f32(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant InterpolateParams& p [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint total = p.batch * p.channels * p.out_h * p.out_w;
    if (id >= total) return;
    uint oj = id % p.out_w;
    uint oi = (id / p.out_w) % p.out_h;
    uint c  = (id / (p.out_w * p.out_h)) % p.channels;
    uint b  = id / (p.channels * p.out_h * p.out_w);
    float sy = (float)oi * ((float)p.in_h - 1.0f) / max((float)p.out_h - 1.0f, 1.0f);
    float sx = (float)oj * ((float)p.in_w - 1.0f) / max((float)p.out_w - 1.0f, 1.0f);
    uint y0 = (uint)floor(sy);
    uint x0 = (uint)floor(sx);
    uint y1 = min(y0 + 1, p.in_h - 1);
    uint x1 = min(x0 + 1, p.in_w - 1);
    float fy = sy - (float)y0;
    float fx = sx - (float)x0;
    uint base = b * p.channels * p.in_h * p.in_w + c * p.in_h * p.in_w;
    output[id] = input[base + y0 * p.in_w + x0] * (1.0f - fy) * (1.0f - fx)
               + input[base + y0 * p.in_w + x1] * (1.0f - fy) * fx
               + input[base + y1 * p.in_w + x0] * fy * (1.0f - fx)
               + input[base + y1 * p.in_w + x1] * fy * fx;
}

// Adaptive Average Pooling 2D
kernel void adaptive_avg_pool2d_f32(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant InterpolateParams& p [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint total = p.batch * p.channels * p.out_h * p.out_w;
    if (id >= total) return;
    uint oj = id % p.out_w;
    uint oi = (id / p.out_w) % p.out_h;
    uint c  = (id / (p.out_w * p.out_h)) % p.channels;
    uint b  = id / (p.channels * p.out_h * p.out_w);
    uint hs = oi * p.in_h / p.out_h;
    uint he = (oi + 1) * p.in_h / p.out_h;
    uint ws = oj * p.in_w / p.out_w;
    uint we = (oj + 1) * p.in_w / p.out_w;
    float sum = 0.0f;
    uint cnt = 0;
    uint base = b * p.channels * p.in_h * p.in_w + c * p.in_h * p.in_w;
    for (uint hi = hs; hi < he; hi++) {
        for (uint wi = ws; wi < we; wi++) {
            sum += input[base + hi * p.in_w + wi];
            cnt++;
        }
    }
    output[id] = sum / (float)cnt;
}

// Pad: 最後の次元にパディングを追加
struct PadParams {
    uint outer;
    uint old_last;
    uint new_last;
    uint pad_left;
};

kernel void pad_f32(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant PadParams& p      [[buffer(2)]],
    constant float& pad_value  [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    uint total = p.outer * p.new_last;
    if (id >= total) return;
    uint o = id / p.new_last;
    uint j = id % p.new_last;
    if (j < p.pad_left || j >= p.pad_left + p.old_last) {
        output[id] = pad_value;
    } else {
        output[id] = input[o * p.old_last + (j - p.pad_left)];
    }
}

// Cumsum: 各ローを並列に、ロー内は逐次累積
struct CumsumParams {
    uint outer;
    uint inner;
};

kernel void cumsum_f32(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant CumsumParams& p   [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= p.outer) return;
    uint base = id * p.inner;
    float acc = 0.0f;
    for (uint i = 0; i < p.inner; i++) {
        acc += input[base + i];
        output[base + i] = acc;
    }
}

// GroupNorm: バッチ×グループごとに正規化
struct GroupNormParams {
    uint batch;
    uint channels;
    uint spatial;
    uint num_groups;
    uint group_size;
    float eps;
};

kernel void group_norm_f32(
    device const float* input  [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias   [[buffer(2)]],
    device float* output       [[buffer(3)]],
    constant GroupNormParams& p [[buffer(4)]],
    constant uint& has_weight  [[buffer(5)]],
    constant uint& has_bias    [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
    // id = batch_idx * num_groups + group_idx
    uint total = p.batch * p.num_groups;
    if (id >= total) return;
    uint b = id / p.num_groups;
    uint g = id % p.num_groups;
    uint start_c = g * p.group_size;
    uint end_c = start_c + p.group_size;
    uint group_elems = p.group_size * p.spatial;
    // 平均計算
    float mean = 0.0f;
    for (uint c = start_c; c < end_c; c++) {
        for (uint s = 0; s < p.spatial; s++) {
            mean += input[b * p.channels * p.spatial + c * p.spatial + s];
        }
    }
    mean /= (float)group_elems;
    // 分散計算
    float var = 0.0f;
    for (uint c = start_c; c < end_c; c++) {
        for (uint s = 0; s < p.spatial; s++) {
            float diff = input[b * p.channels * p.spatial + c * p.spatial + s] - mean;
            var += diff * diff;
        }
    }
    var /= (float)group_elems;
    float std_inv = 1.0f / sqrt(var + p.eps);
    // 正規化 + affine
    for (uint c = start_c; c < end_c; c++) {
        for (uint s = 0; s < p.spatial; s++) {
            uint idx = b * p.channels * p.spatial + c * p.spatial + s;
            float v = (input[idx] - mean) * std_inv;
            if (has_weight != 0) v *= weight[c];
            if (has_bias != 0) v += bias[c];
            output[idx] = v;
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
