//! 深層学習演算（Conv2D, BatchNorm, LayerNorm, MaxPool, AvgPool, Dropout）
//! すべて Metal GPU シェーダーで実装

use crate::device::get_device;
use crate::{MetalTensor, DType};
use metal::{ComputePipelineState, MTLSize};
use tl_backend::{BackendResult, BackendError};

// ========== Metal シェーダーソース ==========

/// Conv2D 用 Metal シェーダー
const CONV2D_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void conv2d_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    constant uint& C_in [[buffer(4)]],
    constant uint& H_in [[buffer(5)]],
    constant uint& W_in [[buffer(6)]],
    constant uint& C_out [[buffer(7)]],
    constant uint& kH [[buffer(8)]],
    constant uint& kW [[buffer(9)]],
    constant uint& H_out [[buffer(10)]],
    constant uint& W_out [[buffer(11)]],
    constant uint& stride_h [[buffer(12)]],
    constant uint& stride_w [[buffer(13)]],
    constant uint& pad_h [[buffer(14)]],
    constant uint& pad_w [[buffer(15)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint ow = gid.x;
    uint oh = gid.y;
    uint batch_oc = gid.z;
    
    if (ow >= W_out || oh >= H_out || batch_oc >= N * C_out) return;
    
    uint batch = batch_oc / C_out;
    uint oc = batch_oc % C_out;
    
    float sum = 0.0f;
    
    for (uint ic = 0; ic < C_in; ic++) {
        for (uint khi = 0; khi < kH; khi++) {
            for (uint kwi = 0; kwi < kW; kwi++) {
                uint ih = oh * stride_h + khi;
                uint iw = ow * stride_w + kwi;
                
                if (ih >= pad_h && ih < H_in + pad_h &&
                    iw >= pad_w && iw < W_in + pad_w) {
                    uint in_idx = batch * C_in * H_in * W_in +
                                  ic * H_in * W_in +
                                  (ih - pad_h) * W_in +
                                  (iw - pad_w);
                    uint k_idx = oc * C_in * kH * kW +
                                ic * kH * kW +
                                khi * kW +
                                kwi;
                    sum += input[in_idx] * weight[k_idx];
                }
            }
        }
    }
    
    uint out_idx = batch * C_out * H_out * W_out +
                   oc * H_out * W_out +
                   oh * W_out +
                   ow;
    output[out_idx] = sum;
}
"#;

/// BatchNorm 用 Metal シェーダー
/// input: [N, C, H, W], gamma/beta/mean/var: [C]
/// 各スレッドが output の1要素を計算
const BATCH_NORM_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void batch_norm_f32(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device const float* running_mean [[buffer(3)]],
    device const float* running_var [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& C [[buffer(7)]],
    constant uint& spatial [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // gid.x = spatial position, gid.y = channel, gid.z = batch
    uint s = gid.x;
    uint ch = gid.y;
    uint batch = gid.z;
    
    if (s >= spatial || ch >= C || batch >= N) return;
    
    uint idx = batch * C * spatial + ch * spatial + s;
    
    float inv_std = 1.0f / sqrt(running_var[ch] + eps);
    output[idx] = (input[idx] - running_mean[ch]) * inv_std * gamma[ch] + beta[ch];
}
"#;

/// MaxPool2D 用 Metal シェーダー
/// input: [N, C, H_in, W_in] → output: [N, C, H_out, W_out]
const MAX_POOL2D_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void max_pool2d_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    constant uint& C [[buffer(3)]],
    constant uint& H_in [[buffer(4)]],
    constant uint& W_in [[buffer(5)]],
    constant uint& kH [[buffer(6)]],
    constant uint& kW [[buffer(7)]],
    constant uint& H_out [[buffer(8)]],
    constant uint& W_out [[buffer(9)]],
    constant uint& stride_h [[buffer(10)]],
    constant uint& stride_w [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // gid.x = output width, gid.y = output height, gid.z = batch * C + channel
    uint ow = gid.x;
    uint oh = gid.y;
    uint batch_ch = gid.z;
    
    if (ow >= W_out || oh >= H_out || batch_ch >= N * C) return;
    
    uint batch = batch_ch / C;
    uint ch = batch_ch % C;
    
    float max_val = -INFINITY;
    
    for (uint khi = 0; khi < kH; khi++) {
        for (uint kwi = 0; kwi < kW; kwi++) {
            uint ih = oh * stride_h + khi;
            uint iw = ow * stride_w + kwi;
            if (ih < H_in && iw < W_in) {
                uint idx = batch * C * H_in * W_in +
                           ch * H_in * W_in +
                           ih * W_in + iw;
                max_val = max(max_val, input[idx]);
            }
        }
    }
    
    uint out_idx = batch * C * H_out * W_out +
                   ch * H_out * W_out +
                   oh * W_out + ow;
    output[out_idx] = max_val;
}
"#;

/// AvgPool2D 用 Metal シェーダー
const AVG_POOL2D_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void avg_pool2d_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    constant uint& C [[buffer(3)]],
    constant uint& H_in [[buffer(4)]],
    constant uint& W_in [[buffer(5)]],
    constant uint& kH [[buffer(6)]],
    constant uint& kW [[buffer(7)]],
    constant uint& H_out [[buffer(8)]],
    constant uint& W_out [[buffer(9)]],
    constant uint& stride_h [[buffer(10)]],
    constant uint& stride_w [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint ow = gid.x;
    uint oh = gid.y;
    uint batch_ch = gid.z;
    
    if (ow >= W_out || oh >= H_out || batch_ch >= N * C) return;
    
    uint batch = batch_ch / C;
    uint ch = batch_ch % C;
    
    float sum = 0.0f;
    uint count = 0;
    
    for (uint khi = 0; khi < kH; khi++) {
        for (uint kwi = 0; kwi < kW; kwi++) {
            uint ih = oh * stride_h + khi;
            uint iw = ow * stride_w + kwi;
            if (ih < H_in && iw < W_in) {
                uint idx = batch * C * H_in * W_in +
                           ch * H_in * W_in +
                           ih * W_in + iw;
                sum += input[idx];
                count += 1;
            }
        }
    }
    
    uint out_idx = batch * C * H_out * W_out +
                   ch * H_out * W_out +
                   oh * W_out + ow;
    output[out_idx] = (count > 0) ? (sum / float(count)) : 0.0f;
}
"#;

/// LayerNorm 用 Metal シェーダー（2パス）
/// パス1: 各行の mean と var を threadgroup reduction で計算
/// パス2: 正規化
const LAYER_NORM_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// パス1: 各行の mean, var を計算
kernel void layer_norm_stats_f32(
    device const float* input [[buffer(0)]],
    device float* means [[buffer(1)]],
    device float* vars [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    constant uint& rest [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= N) return;
    
    uint base = gid * rest;
    
    float sum = 0.0f;
    for (uint j = 0; j < rest; j++) {
        sum += input[base + j];
    }
    float mean = sum / float(rest);
    means[gid] = mean;
    
    float var_sum = 0.0f;
    for (uint j = 0; j < rest; j++) {
        float diff = input[base + j] - mean;
        var_sum += diff * diff;
    }
    vars[gid] = var_sum / float(rest);
}

// パス2: 正規化
kernel void layer_norm_normalize_f32(
    device const float* input [[buffer(0)]],
    device const float* means [[buffer(1)]],
    device const float* vars [[buffer(2)]],
    device const float* gamma [[buffer(3)]],
    device const float* beta [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& rest [[buffer(7)]],
    constant uint& gamma_len [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // gid.x = position within row, gid.y = row index
    uint j = gid.x;
    uint i = gid.y;
    
    if (j >= rest || i >= N) return;
    
    uint idx = i * rest + j;
    float inv_std = 1.0f / sqrt(vars[i] + eps);
    output[idx] = (input[idx] - means[i]) * inv_std * gamma[j % gamma_len] + beta[j % gamma_len];
}
"#;

/// Dropout 用 Metal シェーダー（Philox ハッシュ乱数）
const DROPOUT_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Philox-like hash for pseudo-random number generation
float philox_random(uint seed, uint id) {
    uint h = seed ^ id;
    h ^= h >> 16;
    h *= 0x45d9f3bu;
    h ^= h >> 16;
    h *= 0x45d9f3bu;
    h ^= h >> 16;
    return float(h & 0x00FFFFFFu) / float(0x01000000u);
}

kernel void dropout_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant float& p [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    constant uint& seed [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;
    
    float r = philox_random(seed, id);
    output[id] = (r < p) ? 0.0f : input[id] * scale;
}
"#;

// ========== パイプラインキャッシュ ==========

static CONV2D_PIPELINE: std::sync::OnceLock<ComputePipelineState> = std::sync::OnceLock::new();
static BATCH_NORM_PIPELINE: std::sync::OnceLock<ComputePipelineState> = std::sync::OnceLock::new();
static MAX_POOL2D_PIPELINE: std::sync::OnceLock<ComputePipelineState> = std::sync::OnceLock::new();
static AVG_POOL2D_PIPELINE: std::sync::OnceLock<ComputePipelineState> = std::sync::OnceLock::new();
static LN_STATS_PIPELINE: std::sync::OnceLock<ComputePipelineState> = std::sync::OnceLock::new();
static LN_NORM_PIPELINE: std::sync::OnceLock<ComputePipelineState> = std::sync::OnceLock::new();
static DROPOUT_PIPELINE: std::sync::OnceLock<ComputePipelineState> = std::sync::OnceLock::new();

fn compile_pipeline(shader_source: &str, function_name: &str) -> ComputePipelineState {
    let device = get_device();
    let options = metal::CompileOptions::new();
    let library = device
        .device()
        .new_library_with_source(shader_source, &options)
        .unwrap_or_else(|e| panic!("Failed to compile {} shader: {}", function_name, e));
    let function = library
        .get_function(function_name, None)
        .unwrap_or_else(|e| panic!("{} not found: {}", function_name, e));
    device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .unwrap_or_else(|e| panic!("Failed to create {} pipeline: {}", function_name, e))
}

fn get_conv2d_pipeline() -> &'static ComputePipelineState {
    CONV2D_PIPELINE.get_or_init(|| compile_pipeline(CONV2D_SHADER, "conv2d_f32"))
}

fn get_batch_norm_pipeline() -> &'static ComputePipelineState {
    BATCH_NORM_PIPELINE.get_or_init(|| compile_pipeline(BATCH_NORM_SHADER, "batch_norm_f32"))
}

fn get_max_pool2d_pipeline() -> &'static ComputePipelineState {
    MAX_POOL2D_PIPELINE.get_or_init(|| compile_pipeline(MAX_POOL2D_SHADER, "max_pool2d_f32"))
}

fn get_avg_pool2d_pipeline() -> &'static ComputePipelineState {
    AVG_POOL2D_PIPELINE.get_or_init(|| compile_pipeline(AVG_POOL2D_SHADER, "avg_pool2d_f32"))
}

fn get_ln_stats_pipeline() -> &'static ComputePipelineState {
    LN_STATS_PIPELINE.get_or_init(|| compile_pipeline(LAYER_NORM_SHADER, "layer_norm_stats_f32"))
}

fn get_ln_norm_pipeline() -> &'static ComputePipelineState {
    LN_NORM_PIPELINE.get_or_init(|| compile_pipeline(LAYER_NORM_SHADER, "layer_norm_normalize_f32"))
}

fn get_dropout_pipeline() -> &'static ComputePipelineState {
    DROPOUT_PIPELINE.get_or_init(|| compile_pipeline(DROPOUT_SHADER, "dropout_f32"))
}

// ========== ヘルパー ==========

/// u32 パラメータバッファを作成
fn make_u32_buf(v: u32) -> metal::Buffer {
    let device = get_device();
    device.device().new_buffer_with_data(
        &v as *const u32 as *const _,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    )
}

/// f32 パラメータバッファを作成
fn make_f32_buf(v: f32) -> metal::Buffer {
    let device = get_device();
    device.device().new_buffer_with_data(
        &v as *const f32 as *const _,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    )
}

// ========== 実装 ==========

impl MetalTensor {
    /// Conv2D: 2D 畳み込み演算 (Metal GPU 実装)
    /// input: [N, C_in, H, W]
    /// weight: [C_out, C_in, kH, kW]
    /// output: [N, C_out, H_out, W_out]
    pub fn conv2d_impl(
        &self,
        weight: &MetalTensor,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> BackendResult<MetalTensor> {
        let in_shape = MetalTensor::shape(self);
        let w_shape = weight.shape();
        
        if in_shape.len() != 4 || w_shape.len() != 4 {
             return Err(BackendError::ShapeMismatch(format!("conv2d: inputs must be 4D (got {:?} and {:?})", in_shape, w_shape)));
        }

        let (n, c_in, h_in, w_in) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        let (c_out, w_c_in, kh, kw) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);
        
        if c_in != w_c_in {
             return Err(BackendError::ShapeMismatch(format!("conv2d: channel mismatch: input {} vs weight {}", c_in, w_c_in)));
        }
        
        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;
        
        let h_out = (h_in + 2 * pad_h - kh) / stride_h + 1;
        let w_out = (w_in + 2 * pad_w - kw) / stride_w + 1;
        
        let result = MetalTensor::uninit(&[n, c_out, h_out, w_out], DType::F32);
        let device = get_device();
        let command_queue = device.command_queue();
        let pipeline = get_conv2d_pipeline();
        
        let bufs = [
            make_u32_buf(n as u32), make_u32_buf(c_in as u32),
            make_u32_buf(h_in as u32), make_u32_buf(w_in as u32),
            make_u32_buf(c_out as u32), make_u32_buf(kh as u32),
            make_u32_buf(kw as u32), make_u32_buf(h_out as u32),
            make_u32_buf(w_out as u32), make_u32_buf(stride_h as u32),
            make_u32_buf(stride_w as u32), make_u32_buf(pad_h as u32),
            make_u32_buf(pad_w as u32),
        ];
        
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(self.buffer()), 0);
        encoder.set_buffer(1, Some(weight.buffer()), 0);
        encoder.set_buffer(2, Some(result.buffer()), 0);
        for (i, buf) in bufs.iter().enumerate() {
            encoder.set_buffer((3 + i) as u64, Some(buf), 0);
        }
        
        let threads_per_group = MTLSize::new(8, 8, 4);
        let grid_size = MTLSize::new(
            ((w_out + 7) / 8) as u64,
            ((h_out + 7) / 8) as u64,
            ((n * c_out + 3) / 4) as u64,
        );
        encoder.dispatch_thread_groups(grid_size, threads_per_group);
        encoder.end_encoding();
        
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        Ok(result)
    }
    
    /// BatchNorm: バッチ正規化 (Metal GPU 実装)
    /// input: [N, C, H, W]
    /// gamma/beta: [C]
    /// running_mean/var: [C]
    pub fn batch_norm_impl(
        &self,
        gamma: &MetalTensor,
        beta: &MetalTensor,
        running_mean: &MetalTensor,
        running_var: &MetalTensor,
        eps: f32,
    ) -> BackendResult<MetalTensor> {
        let shape = MetalTensor::shape(self);
        if shape.len() != 4 {
             return Err(BackendError::ShapeMismatch(format!("batch_norm: input must be 4D, got {:?}D", shape.len())));
        }
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let spatial = h * w;
        
        let result = MetalTensor::uninit(&[n, c, h, w], DType::F32);
        let device = get_device();
        let command_queue = device.command_queue();
        let pipeline = get_batch_norm_pipeline();
        
        let n_buf = make_u32_buf(n as u32);
        let c_buf = make_u32_buf(c as u32);
        let spatial_buf = make_u32_buf(spatial as u32);
        let eps_buf = make_f32_buf(eps);
        
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(self.buffer()), 0);
        encoder.set_buffer(1, Some(gamma.buffer()), 0);
        encoder.set_buffer(2, Some(beta.buffer()), 0);
        encoder.set_buffer(3, Some(running_mean.buffer()), 0);
        encoder.set_buffer(4, Some(running_var.buffer()), 0);
        encoder.set_buffer(5, Some(result.buffer()), 0);
        encoder.set_buffer(6, Some(&n_buf), 0);
        encoder.set_buffer(7, Some(&c_buf), 0);
        encoder.set_buffer(8, Some(&spatial_buf), 0);
        encoder.set_buffer(9, Some(&eps_buf), 0);
        
        // 3D グリッド: [spatial, C, N]
        let tpg = MTLSize::new(
            spatial.min(256) as u64,
            c.min(16) as u64,
            n.min(4) as u64,
        );
        let grid = MTLSize::new(
            ((spatial + tpg.width as usize - 1) / tpg.width as usize) as u64,
            ((c + tpg.height as usize - 1) / tpg.height as usize) as u64,
            ((n + tpg.depth as usize - 1) / tpg.depth as usize) as u64,
        );
        encoder.dispatch_thread_groups(grid, tpg);
        encoder.end_encoding();
        
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        Ok(result)
    }
    
    /// LayerNorm: レイヤー正規化 (Metal GPU 実装, 2パス)
    pub fn layer_norm_impl(
        &self,
        gamma: &MetalTensor,
        beta: &MetalTensor,
        eps: f32,
    ) -> BackendResult<MetalTensor> {
        let shape = MetalTensor::shape(self);
        let n = shape[0];
        let rest: usize = shape[1..].iter().product();
        let gamma_len = gamma.shape().iter().product::<usize>();
        
        let result = MetalTensor::uninit(shape, DType::F32);
        let means = MetalTensor::uninit(&[n], DType::F32);
        let vars = MetalTensor::uninit(&[n], DType::F32);
        
        let device = get_device();
        let command_queue = device.command_queue();
        
        let n_buf = make_u32_buf(n as u32);
        let rest_buf = make_u32_buf(rest as u32);
        let gamma_len_buf = make_u32_buf(gamma_len as u32);
        let eps_buf = make_f32_buf(eps);
        
        // パス1: mean, var を計算
        {
            let pipeline = get_ln_stats_pipeline();
            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(self.buffer()), 0);
            encoder.set_buffer(1, Some(means.buffer()), 0);
            encoder.set_buffer(2, Some(vars.buffer()), 0);
            encoder.set_buffer(3, Some(&n_buf), 0);
            encoder.set_buffer(4, Some(&rest_buf), 0);
            
            let tpg = MTLSize::new(n.min(256) as u64, 1, 1);
            let grid = MTLSize::new(
                ((n + tpg.width as usize - 1) / tpg.width as usize) as u64,
                1, 1,
            );
            encoder.dispatch_thread_groups(grid, tpg);
            encoder.end_encoding();
            
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
        
        // パス2: 正規化
        {
            let pipeline = get_ln_norm_pipeline();
            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(self.buffer()), 0);
            encoder.set_buffer(1, Some(means.buffer()), 0);
            encoder.set_buffer(2, Some(vars.buffer()), 0);
            encoder.set_buffer(3, Some(gamma.buffer()), 0);
            encoder.set_buffer(4, Some(beta.buffer()), 0);
            encoder.set_buffer(5, Some(result.buffer()), 0);
            encoder.set_buffer(6, Some(&n_buf), 0);
            encoder.set_buffer(7, Some(&rest_buf), 0);
            encoder.set_buffer(8, Some(&gamma_len_buf), 0);
            encoder.set_buffer(9, Some(&eps_buf), 0);
            
            // 2D グリッド: [rest, N]
            let tpg = MTLSize::new(rest.min(256) as u64, n.min(4) as u64, 1);
            let grid = MTLSize::new(
                ((rest + tpg.width as usize - 1) / tpg.width as usize) as u64,
                ((n + tpg.height as usize - 1) / tpg.height as usize) as u64,
                1,
            );
            encoder.dispatch_thread_groups(grid, tpg);
            encoder.end_encoding();
            
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
        
        Ok(result)
    }
    
    /// MaxPool2D: 最大プーリング (Metal GPU 実装, padding = 0)
    pub fn max_pool2d_impl(
        &self,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> BackendResult<MetalTensor> {
        let shape = MetalTensor::shape(self);
        if shape.len() != 4 {
             return Err(BackendError::ShapeMismatch(format!("max_pool2d: input must be 4D, got {:?}D", shape.len())));
        }
        let (n, c, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
        let (kh, kw) = kernel;
        let (stride_h, stride_w) = stride;
        
        let h_out = (h_in - kh) / stride_h + 1;
        let w_out = (w_in - kw) / stride_w + 1;
        
        let result = MetalTensor::uninit(&[n, c, h_out, w_out], DType::F32);
        let device = get_device();
        let command_queue = device.command_queue();
        let pipeline = get_max_pool2d_pipeline();
        
        let bufs = [
            make_u32_buf(n as u32), make_u32_buf(c as u32),
            make_u32_buf(h_in as u32), make_u32_buf(w_in as u32),
            make_u32_buf(kh as u32), make_u32_buf(kw as u32),
            make_u32_buf(h_out as u32), make_u32_buf(w_out as u32),
            make_u32_buf(stride_h as u32), make_u32_buf(stride_w as u32),
        ];
        
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(self.buffer()), 0);
        encoder.set_buffer(1, Some(result.buffer()), 0);
        for (i, buf) in bufs.iter().enumerate() {
            encoder.set_buffer((2 + i) as u64, Some(buf), 0);
        }
        
        // 3D グリッド: [W_out, H_out, N*C]
        let tpg = MTLSize::new(8, 8, 4);
        let grid = MTLSize::new(
            ((w_out + 7) / 8) as u64,
            ((h_out + 7) / 8) as u64,
            ((n * c + 3) / 4) as u64,
        );
        encoder.dispatch_thread_groups(grid, tpg);
        encoder.end_encoding();
        
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        Ok(result)
    }
    
    /// AvgPool2D: 平均プーリング (Metal GPU 実装, padding = 0)
    pub fn avg_pool2d_impl(
        &self,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> BackendResult<MetalTensor> {
        let shape = MetalTensor::shape(self);
        if shape.len() != 4 {
             return Err(BackendError::ShapeMismatch(format!("avg_pool2d: input must be 4D, got {:?}D", shape.len())));
        }

        let (n, c, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
        let (kh, kw) = kernel;
        let (stride_h, stride_w) = stride;
        
        let h_out = (h_in - kh) / stride_h + 1;
        let w_out = (w_in - kw) / stride_w + 1;
        
        let result = MetalTensor::uninit(&[n, c, h_out, w_out], DType::F32);
        let device = get_device();
        let command_queue = device.command_queue();
        let pipeline = get_avg_pool2d_pipeline();
        
        let bufs = [
            make_u32_buf(n as u32), make_u32_buf(c as u32),
            make_u32_buf(h_in as u32), make_u32_buf(w_in as u32),
            make_u32_buf(kh as u32), make_u32_buf(kw as u32),
            make_u32_buf(h_out as u32), make_u32_buf(w_out as u32),
            make_u32_buf(stride_h as u32), make_u32_buf(stride_w as u32),
        ];
        
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(self.buffer()), 0);
        encoder.set_buffer(1, Some(result.buffer()), 0);
        for (i, buf) in bufs.iter().enumerate() {
            encoder.set_buffer((2 + i) as u64, Some(buf), 0);
        }
        
        let tpg = MTLSize::new(8, 8, 4);
        let grid = MTLSize::new(
            ((w_out + 7) / 8) as u64,
            ((h_out + 7) / 8) as u64,
            ((n * c + 3) / 4) as u64,
        );
        encoder.dispatch_thread_groups(grid, tpg);
        encoder.end_encoding();
        
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        Ok(result)
    }
    
    /// Dropout: ドロップアウト (Metal GPU 実装, Philox ハッシュ乱数)
    pub fn dropout_impl(&self, p: f32, training: bool) -> BackendResult<MetalTensor> {
        if !training || p <= 0.0 {
            return Ok(self.clone());
        }
        
        let count: usize = MetalTensor::shape(self).iter().product();
        let scale = 1.0 / (1.0 - p);
        
        let result = MetalTensor::uninit(MetalTensor::shape(self), DType::F32);
        let device = get_device();
        let command_queue = device.command_queue();
        let pipeline = get_dropout_pipeline();
        
        // ランダムシード生成（CPU 側で軽量に）
        let seed: u32 = {
            use std::time::SystemTime;
            let t = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            (t ^ (t >> 32)) as u32
        };
        
        let count_buf = make_u32_buf(count as u32);
        let p_buf = make_f32_buf(p);
        let scale_buf = make_f32_buf(scale);
        let seed_buf = make_u32_buf(seed);
        
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(self.buffer()), 0);
        encoder.set_buffer(1, Some(result.buffer()), 0);
        encoder.set_buffer(2, Some(&count_buf), 0);
        encoder.set_buffer(3, Some(&p_buf), 0);
        encoder.set_buffer(4, Some(&scale_buf), 0);
        encoder.set_buffer(5, Some(&seed_buf), 0);
        
        let thread_width = pipeline.thread_execution_width() as usize;
        let tpg = MTLSize::new(thread_width as u64, 1, 1);
        let num_groups = (count + thread_width - 1) / thread_width;
        let grid = MTLSize::new(num_groups as u64, 1, 1);
        encoder.dispatch_thread_groups(grid, tpg);
        encoder.end_encoding();
        
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        Ok(result)
    }
}
