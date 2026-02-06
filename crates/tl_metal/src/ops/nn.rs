//! 深層学習演算（Conv2D, BatchNorm, LayerNorm, MaxPool, AvgPool）
//! Conv2D は Metal GPU シェーダーで実装

use crate::device::get_device;
use crate::{MetalTensor, DType};
use metal::{ComputePipelineState, MTLSize};

/// Conv2D 用 Metal シェーダー
const CONV2D_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Conv2D: 2D 畳み込み演算
// input: [N, C_in, H_in, W_in]
// weight: [C_out, C_in, kH, kW]
// output: [N, C_out, H_out, W_out]
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
    // gid.x = output width position
    // gid.y = output height position  
    // gid.z = batch * C_out + output channel
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

/// Conv2D 用パイプライン（キャッシュ）
static CONV2D_PIPELINE: std::sync::OnceLock<ComputePipelineState> = std::sync::OnceLock::new();

fn get_conv2d_pipeline() -> &'static ComputePipelineState {
    CONV2D_PIPELINE.get_or_init(|| {
        let device = get_device();
        let options = metal::CompileOptions::new();
        let library = device
            .device()
            .new_library_with_source(CONV2D_SHADER, &options)
            .expect("Failed to compile conv2d shader");
        let function = library
            .get_function("conv2d_f32", None)
            .expect("conv2d_f32 not found");
        device
            .device()
            .new_compute_pipeline_state_with_function(&function)
            .expect("Failed to create conv2d pipeline")
    })
}

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
    ) -> MetalTensor {
        let in_shape = MetalTensor::shape(self);
        let w_shape = weight.shape();
        
        let (n, c_in, h_in, w_in) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        let (c_out, _, kh, kw) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);
        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;
        
        let h_out = (h_in + 2 * pad_h - kh) / stride_h + 1;
        let w_out = (w_in + 2 * pad_w - kw) / stride_w + 1;
        
        let result = MetalTensor::uninit(&[n, c_out, h_out, w_out], DType::F32);
        let device = get_device();
        let command_queue = device.command_queue();
        let pipeline = get_conv2d_pipeline();
        
        // パラメータバッファ作成
        let make_buf = |v: u32| {
            device.device().new_buffer_with_data(
                &v as *const u32 as *const _,
                4,
                metal::MTLResourceOptions::StorageModeShared,
            )
        };
        
        let n_buf = make_buf(n as u32);
        let c_in_buf = make_buf(c_in as u32);
        let h_in_buf = make_buf(h_in as u32);
        let w_in_buf = make_buf(w_in as u32);
        let c_out_buf = make_buf(c_out as u32);
        let kh_buf = make_buf(kh as u32);
        let kw_buf = make_buf(kw as u32);
        let h_out_buf = make_buf(h_out as u32);
        let w_out_buf = make_buf(w_out as u32);
        let stride_h_buf = make_buf(stride_h as u32);
        let stride_w_buf = make_buf(stride_w as u32);
        let pad_h_buf = make_buf(pad_h as u32);
        let pad_w_buf = make_buf(pad_w as u32);
        
        // GPU 実行
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(self.buffer()), 0);
        encoder.set_buffer(1, Some(weight.buffer()), 0);
        encoder.set_buffer(2, Some(result.buffer()), 0);
        encoder.set_buffer(3, Some(&n_buf), 0);
        encoder.set_buffer(4, Some(&c_in_buf), 0);
        encoder.set_buffer(5, Some(&h_in_buf), 0);
        encoder.set_buffer(6, Some(&w_in_buf), 0);
        encoder.set_buffer(7, Some(&c_out_buf), 0);
        encoder.set_buffer(8, Some(&kh_buf), 0);
        encoder.set_buffer(9, Some(&kw_buf), 0);
        encoder.set_buffer(10, Some(&h_out_buf), 0);
        encoder.set_buffer(11, Some(&w_out_buf), 0);
        encoder.set_buffer(12, Some(&stride_h_buf), 0);
        encoder.set_buffer(13, Some(&stride_w_buf), 0);
        encoder.set_buffer(14, Some(&pad_h_buf), 0);
        encoder.set_buffer(15, Some(&pad_w_buf), 0);
        
        // 3D グリッド: [W_out, H_out, N*C_out]
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
        
        result
    }
    
    /// BatchNorm: バッチ正規化
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
    ) -> MetalTensor {
        let input = self.to_vec::<f32>();
        let gamma_v = gamma.to_vec::<f32>();
        let beta_v = beta.to_vec::<f32>();
        let mean_v = running_mean.to_vec::<f32>();
        let var_v = running_var.to_vec::<f32>();
        
        let shape = MetalTensor::shape(self);
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let spatial = h * w;
        
        let mut output = vec![0.0f32; n * c * spatial];
        
        for batch in 0..n {
            for ch in 0..c {
                let inv_std = 1.0 / (var_v[ch] + eps).sqrt();
                for s in 0..spatial {
                    let idx = batch * c * spatial + ch * spatial + s;
                    output[idx] = (input[idx] - mean_v[ch]) * inv_std * gamma_v[ch] + beta_v[ch];
                }
            }
        }
        
        MetalTensor::from_slice(&output, &[n, c, h, w], DType::F32)
    }
    
    /// LayerNorm: レイヤー正規化
    pub fn layer_norm_impl(
        &self,
        gamma: &MetalTensor,
        beta: &MetalTensor,
        eps: f32,
    ) -> MetalTensor {
        let input = self.to_vec::<f32>();
        let gamma_v = gamma.to_vec::<f32>();
        let beta_v = beta.to_vec::<f32>();
        
        let shape = MetalTensor::shape(self);
        let n = shape[0];
        let rest: usize = shape[1..].iter().product();
        
        let mut output = vec![0.0f32; n * rest];
        
        for i in 0..n {
            let base = i * rest;
            let mean: f32 = input[base..base + rest].iter().sum::<f32>() / rest as f32;
            let var: f32 = input[base..base + rest]
                .iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>() / rest as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            
            for j in 0..rest {
                let idx = base + j;
                output[idx] = (input[idx] - mean) * inv_std * gamma_v[j % gamma_v.len()] + beta_v[j % beta_v.len()];
            }
        }
        
        MetalTensor::from_slice(&output, shape, DType::F32)
    }
    
    /// MaxPool2D: 最大プーリング (padding = 0)
    pub fn max_pool2d_impl(
        &self,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> MetalTensor {
        let input = self.to_vec::<f32>();
        let shape = MetalTensor::shape(self);
        let (n, c, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
        let (kh, kw) = kernel;
        let (stride_h, stride_w) = stride;
        
        let h_out = (h_in - kh) / stride_h + 1;
        let w_out = (w_in - kw) / stride_w + 1;
        
        let mut output = vec![f32::NEG_INFINITY; n * c * h_out * w_out];
        
        for batch in 0..n {
            for ch in 0..c {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut max_val = f32::NEG_INFINITY;
                        for khi in 0..kh {
                            for kwi in 0..kw {
                                let ih = oh * stride_h + khi;
                                let iw = ow * stride_w + kwi;
                                if ih < h_in && iw < w_in {
                                    let idx = batch * c * h_in * w_in +
                                              ch * h_in * w_in +
                                              ih * w_in +
                                              iw;
                                    max_val = max_val.max(input[idx]);
                                }
                            }
                        }
                        let out_idx = batch * c * h_out * w_out +
                                     ch * h_out * w_out +
                                     oh * w_out + ow;
                        output[out_idx] = max_val;
                    }
                }
            }
        }
        
        MetalTensor::from_slice(&output, &[n, c, h_out, w_out], DType::F32)
    }
    
    /// AvgPool2D: 平均プーリング (padding = 0)
    pub fn avg_pool2d_impl(
        &self,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> MetalTensor {
        let input = self.to_vec::<f32>();
        let shape = MetalTensor::shape(self);
        let (n, c, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
        let (kh, kw) = kernel;
        let (stride_h, stride_w) = stride;
        
        let h_out = (h_in - kh) / stride_h + 1;
        let w_out = (w_in - kw) / stride_w + 1;
        
        let mut output = vec![0.0f32; n * c * h_out * w_out];
        
        for batch in 0..n {
            for ch in 0..c {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = 0.0f32;
                        let mut count = 0;
                        for khi in 0..kh {
                            for kwi in 0..kw {
                                let ih = oh * stride_h + khi;
                                let iw = ow * stride_w + kwi;
                                if ih < h_in && iw < w_in {
                                    let idx = batch * c * h_in * w_in +
                                              ch * h_in * w_in +
                                              ih * w_in +
                                              iw;
                                    sum += input[idx];
                                    count += 1;
                                }
                            }
                        }
                        let out_idx = batch * c * h_out * w_out +
                                     ch * h_out * w_out +
                                     oh * w_out + ow;
                        output[out_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
                    }
                }
            }
        }
        
        MetalTensor::from_slice(&output, &[n, c, h_out, w_out], DType::F32)
    }
    
    /// Dropout: ドロップアウト
    pub fn dropout_impl(&self, p: f32, training: bool) -> MetalTensor {
        if !training || p <= 0.0 {
            return self.clone();
        }
        
        let input = self.to_vec::<f32>();
        let mut output = input.clone();
        let scale = 1.0 / (1.0 - p);
        
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for x in output.iter_mut() {
            if rng.gen::<f32>() < p {
                *x = 0.0;
            } else {
                *x *= scale;
            }
        }
        
        MetalTensor::from_slice(&output, MetalTensor::shape(self), DType::F32)
    }
}
