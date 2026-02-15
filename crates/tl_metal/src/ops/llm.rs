//! LLM 向け演算 — Metal GPU シェーダー実装
//! RMSNorm, RoPE (cos/sin 生成 + 適用), CausalMask

use crate::device::get_device;
use crate::tensor::MetalTensor;
use crate::DType;
use metal::{ComputePipelineState, MTLSize};
use tl_backend::{BackendResult, BackendError};

/// LLM 演算用 Metal シェーダー
const LLM_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ========== RMSNorm ==========

// Pass 1: 各行の mean(x²) を計算
// input: [outer, dim], output: [outer] (mean_sq 配列)
kernel void rms_norm_mean_sq_f32(
    device const float* input [[buffer(0)]],
    device float* mean_sq [[buffer(1)]],
    constant uint& dim [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    float sum = 0.0f;
    for (uint i = 0; i < dim; i++) {
        float v = input[id * dim + i];
        sum += v * v;
    }
    mean_sq[id] = sum / float(dim);
}

// Pass 2: x / sqrt(mean_sq + eps)
kernel void rms_norm_normalize_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* mean_sq [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint d = gid.x;
    uint outer = gid.y;
    if (d >= dim) return;
    
    float rms = sqrt(mean_sq[outer] + eps);
    output[outer * dim + d] = input[outer * dim + d] / rms;
}

// ========== RoPE ==========

// RoPE cos/sin テーブル生成
// output: [seq_len, half_dim]
kernel void rope_cos_sin_f32(
    device float* cos_out [[buffer(0)]],
    device float* sin_out [[buffer(1)]],
    constant uint& half_dim [[buffer(2)]],
    constant float& base [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.x;     // half_dim index
    uint pos = gid.y;   // position index
    if (i >= half_dim) return;
    
    float freq = 1.0f / pow(base, (2.0f * float(i)) / float(head_dim));
    float angle = float(pos) * freq;
    uint idx = pos * half_dim + i;
    cos_out[idx] = cos(angle);
    sin_out[idx] = sin(angle);
}

// RoPE 適用 (4D対応: [batch, seq, heads, head_dim])
// cos/sin: [seq_len, half_dim] (既にnarrowされたスライス)
// 各 (batch, seq, head) の組み合わせに対して独立に回転を適用
kernel void apply_rope_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* cos_table [[buffer(2)]],
    device const float* sin_table [[buffer(3)]],
    constant uint& half_dim [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& seq_len [[buffer(6)]],
    constant uint& n_heads [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.x;        // half_dim index
    uint outer = gid.y;    // batch * seq * head flat index
    
    if (i >= half_dim) return;
    uint total_outer = seq_len * n_heads; // batch=1 assumed
    if (outer >= total_outer) return;
    
    uint seq_pos = outer / n_heads;
    
    uint base = outer * head_dim;
    float x1 = input[base + i];
    float x2 = input[base + i + half_dim];
    float c = cos_table[seq_pos * half_dim + i];
    float s = sin_table[seq_pos * half_dim + i];
    
    output[base + i] = x1 * c - x2 * s;
    output[base + i + half_dim] = x1 * s + x2 * c;
}

// ========== CausalMask ==========

// causal_mask: out[i*n+j] = (j <= i) ? 0 : -inf
kernel void causal_mask_f32(
    device float* output [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint j = gid.x;
    uint i = gid.y;
    if (i >= n || j >= n) return;
    output[i * n + j] = (j <= i) ? 0.0f : -INFINITY;
}
"#;

// パイプラインキャッシュ
static RMS_MEAN_SQ_PIPELINE: std::sync::OnceLock<ComputePipelineState> = std::sync::OnceLock::new();
static RMS_NORMALIZE_PIPELINE: std::sync::OnceLock<ComputePipelineState> = std::sync::OnceLock::new();
static ROPE_COS_SIN_PIPELINE: std::sync::OnceLock<ComputePipelineState> = std::sync::OnceLock::new();
static APPLY_ROPE_PIPELINE: std::sync::OnceLock<ComputePipelineState> = std::sync::OnceLock::new();
static CAUSAL_MASK_PIPELINE: std::sync::OnceLock<ComputePipelineState> = std::sync::OnceLock::new();

fn compile_llm_pipeline(function_name: &str) -> ComputePipelineState {
    let device = get_device();
    let options = metal::CompileOptions::new();
    let library = device
        .device()
        .new_library_with_source(LLM_SHADER, &options)
        .unwrap_or_else(|e| panic!("Failed to compile LLM shader: {}", e));
    let function = library
        .get_function(function_name, None)
        .unwrap_or_else(|e| panic!("{} not found: {}", function_name, e));
    device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .unwrap_or_else(|e| panic!("Failed to create {} pipeline: {}", function_name, e))
}

fn get_rms_mean_sq_pipeline() -> &'static ComputePipelineState {
    RMS_MEAN_SQ_PIPELINE.get_or_init(|| compile_llm_pipeline("rms_norm_mean_sq_f32"))
}
fn get_rms_normalize_pipeline() -> &'static ComputePipelineState {
    RMS_NORMALIZE_PIPELINE.get_or_init(|| compile_llm_pipeline("rms_norm_normalize_f32"))
}
fn get_rope_cos_sin_pipeline() -> &'static ComputePipelineState {
    ROPE_COS_SIN_PIPELINE.get_or_init(|| compile_llm_pipeline("rope_cos_sin_f32"))
}
fn get_apply_rope_pipeline() -> &'static ComputePipelineState {
    APPLY_ROPE_PIPELINE.get_or_init(|| compile_llm_pipeline("apply_rope_f32"))
}
fn get_causal_mask_pipeline() -> &'static ComputePipelineState {
    CAUSAL_MASK_PIPELINE.get_or_init(|| compile_llm_pipeline("causal_mask_f32"))
}

fn make_u32_buf(v: u32) -> metal::Buffer {
    let device = get_device();
    device.device().new_buffer_with_data(
        &v as *const u32 as *const _,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    )
}

fn make_f32_buf(v: f32) -> metal::Buffer {
    let device = get_device();
    device.device().new_buffer_with_data(
        &v as *const f32 as *const _,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    )
}

impl MetalTensor {
    /// RMSNorm — Metal GPU 2-pass 実装
    /// Pass 1: mean(x²) を各行で計算
    /// Pass 2: x / sqrt(mean + eps)
    pub fn rms_norm_impl(&self, eps: f32) -> BackendResult<MetalTensor> {
        let shape = MetalTensor::shape(self);
        if shape.is_empty() {
            return Err(BackendError::ShapeMismatch("rms_norm: input cannot be empty".to_string()));
        }
        let dim = *shape.last().unwrap();
        let outer_size: usize = shape.iter().take(shape.len() - 1).product::<usize>().max(1);
        
        let device = get_device();
        let command_queue = device.command_queue();
        
        // 中間バッファ: mean_sq [outer_size]
        let mean_sq = MetalTensor::uninit(&[outer_size], DType::F32);
        let result = MetalTensor::uninit(shape, DType::F32);
        
        let dim_buf = make_u32_buf(dim as u32);
        let eps_buf = make_f32_buf(eps);
        
        // Pass 1: mean(x²)
        {
            let pipeline = get_rms_mean_sq_pipeline();
            let cb = command_queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipeline);
            enc.set_buffer(0, Some(self.buffer()), 0);
            enc.set_buffer(1, Some(mean_sq.buffer()), 0);
            enc.set_buffer(2, Some(&dim_buf), 0);
            
            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let threads = outer_size.min(max_threads);
            let tpg = MTLSize::new(threads as u64, 1, 1);
            let grid = MTLSize::new(((outer_size + threads - 1) / threads) as u64, 1, 1);
            enc.dispatch_thread_groups(grid, tpg);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        
        // Pass 2: normalize
        {
            let pipeline = get_rms_normalize_pipeline();
            let cb = command_queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipeline);
            enc.set_buffer(0, Some(self.buffer()), 0);
            enc.set_buffer(1, Some(result.buffer()), 0);
            enc.set_buffer(2, Some(mean_sq.buffer()), 0);
            enc.set_buffer(3, Some(&dim_buf), 0);
            enc.set_buffer(4, Some(&eps_buf), 0);
            
            let tpg = MTLSize::new(dim.min(256) as u64, outer_size.min(4) as u64, 1);
            let grid = MTLSize::new(
                ((dim + tpg.width as usize - 1) / tpg.width as usize) as u64,
                ((outer_size + tpg.height as usize - 1) / tpg.height as usize) as u64,
                1,
            );
            enc.dispatch_thread_groups(grid, tpg);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        
        Ok(result)
    }
    
    /// RoPE cos/sin テーブル生成 — Metal GPU 実装
    /// 2 つのテンソル [seq_len, half_dim] を返す (cos, sin)
    pub fn rope_cos_sin_impl(seq_len: usize, head_dim: usize, base: f32) -> BackendResult<(MetalTensor, MetalTensor)> {
        let half_dim = head_dim / 2;
        let cos_tensor = MetalTensor::uninit(&[seq_len, half_dim], DType::F32);
        let sin_tensor = MetalTensor::uninit(&[seq_len, half_dim], DType::F32);
        
        let device = get_device();
        let pipeline = get_rope_cos_sin_pipeline();
        
        let half_dim_buf = make_u32_buf(half_dim as u32);
        let base_buf = make_f32_buf(base);
        let head_dim_buf = make_u32_buf(head_dim as u32);
        
        let cb = device.command_queue().new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(cos_tensor.buffer()), 0);
        enc.set_buffer(1, Some(sin_tensor.buffer()), 0);
        enc.set_buffer(2, Some(&half_dim_buf), 0);
        enc.set_buffer(3, Some(&base_buf), 0);
        enc.set_buffer(4, Some(&head_dim_buf), 0);
        
        let tpg = MTLSize::new(half_dim.min(256) as u64, seq_len.min(4) as u64, 1);
        let grid = MTLSize::new(
            ((half_dim + tpg.width as usize - 1) / tpg.width as usize) as u64,
            ((seq_len + tpg.height as usize - 1) / tpg.height as usize) as u64,
            1,
        );
        enc.dispatch_thread_groups(grid, tpg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        
        Ok((cos_tensor, sin_tensor))
    }
    
    /// RoPE 適用 — Metal GPU 実装 (4D対応)
    /// input: [batch, seq_len, n_heads, head_dim] or [1, seq_len, n_heads, head_dim]
    /// cos/sin: [seq_len, half_dim] (既にnarrowされたスライス)
    pub fn apply_rope_impl(&self, cos_table: &MetalTensor, sin_table: &MetalTensor, _pos: usize) -> BackendResult<MetalTensor> {
        let shape = MetalTensor::shape(self);
        if shape.is_empty() {
             return Err(BackendError::ShapeMismatch("apply_rope: input cannot be empty".to_string()));
        }
        let head_dim = *shape.last().unwrap();
        let half_dim = head_dim / 2;
        
        // 4D: [batch, seq, heads, head_dim]
        // 3D: [seq, heads, head_dim]  
        // 2D: [seq, head_dim] (single head)
        let (seq_len, n_heads) = if shape.len() >= 4 {
            (shape[1], shape[2])
        } else if shape.len() == 3 {
            (shape[0], shape[1])
        } else if shape.len() == 2 {
            (shape[0], 1)
        } else {
            return Err(BackendError::ShapeMismatch(format!("apply_rope: input must be at least 2D, got: {:?}", shape)));
        };
        
        let total_outer = seq_len * n_heads;
        
        let result = MetalTensor::uninit(shape, DType::F32);
        let device = get_device();
        let pipeline = get_apply_rope_pipeline();
        
        let half_dim_buf = make_u32_buf(half_dim as u32);
        let head_dim_buf = make_u32_buf(head_dim as u32);
        let seq_len_buf = make_u32_buf(seq_len as u32);
        let n_heads_buf = make_u32_buf(n_heads as u32);
        
        let cb = device.command_queue().new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(self.buffer()), 0);
        enc.set_buffer(1, Some(result.buffer()), 0);
        enc.set_buffer(2, Some(cos_table.buffer()), 0);
        enc.set_buffer(3, Some(sin_table.buffer()), 0);
        enc.set_buffer(4, Some(&half_dim_buf), 0);
        enc.set_buffer(5, Some(&head_dim_buf), 0);
        enc.set_buffer(6, Some(&seq_len_buf), 0);
        enc.set_buffer(7, Some(&n_heads_buf), 0);
        
        let tpg = MTLSize::new(half_dim.min(256) as u64, total_outer.min(4) as u64, 1);
        let grid = MTLSize::new(
            ((half_dim + tpg.width as usize - 1) / tpg.width as usize) as u64,
            ((total_outer + tpg.height as usize - 1) / tpg.height as usize) as u64,
            1,
        );
        enc.dispatch_thread_groups(grid, tpg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        
        Ok(result)
    }
    
    /// CausalMask 生成 — Metal GPU 実装
    pub fn causal_mask_impl(n: usize) -> BackendResult<MetalTensor> {
        let result = MetalTensor::uninit(&[n, n], DType::F32);
        let device = get_device();
        let pipeline = get_causal_mask_pipeline();
        
        let n_buf = make_u32_buf(n as u32);
        
        let cb = device.command_queue().new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(result.buffer()), 0);
        enc.set_buffer(1, Some(&n_buf), 0);
        
        let tpg = MTLSize::new(n.min(16) as u64, n.min(16) as u64, 1);
        let grid = MTLSize::new(
            ((n + tpg.width as usize - 1) / tpg.width as usize) as u64,
            ((n + tpg.height as usize - 1) / tpg.height as usize) as u64,
            1,
        );
        enc.dispatch_thread_groups(grid, tpg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        
        Ok(result)
    }
}
