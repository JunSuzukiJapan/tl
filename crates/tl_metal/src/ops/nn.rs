//! 深層学習演算（Conv2D, BatchNorm, LayerNorm, MaxPool, AvgPool）

use crate::{MetalTensor, DType};

impl MetalTensor {
    /// Conv2D: 2D 畳み込み演算
    /// input: [N, C_in, H, W]
    /// weight: [C_out, C_in, kH, kW]
    /// output: [N, C_out, H_out, W_out]
    pub fn conv2d_impl(
        &self,
        weight: &MetalTensor,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> MetalTensor {
        // CPU fallback implementation
        let input = self.to_vec::<f32>();
        let kernel = weight.to_vec::<f32>();
        let in_shape = MetalTensor::shape(self);
        let w_shape = weight.shape();
        
        let (n, c_in, h_in, w_in) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        let (c_out, _, kh, kw) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);
        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;
        
        let h_out = (h_in + 2 * pad_h - kh) / stride_h + 1;
        let w_out = (w_in + 2 * pad_w - kw) / stride_w + 1;
        
        let mut output = vec![0.0f32; n * c_out * h_out * w_out];
        
        for batch in 0..n {
            for oc in 0..c_out {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = 0.0f32;
                        for ic in 0..c_in {
                            for khi in 0..kh {
                                for kwi in 0..kw {
                                    let ih = oh * stride_h + khi;
                                    let iw = ow * stride_w + kwi;
                                    
                                    if ih >= pad_h && ih < h_in + pad_h && 
                                       iw >= pad_w && iw < w_in + pad_w {
                                        let in_idx = batch * c_in * h_in * w_in + 
                                                     ic * h_in * w_in + 
                                                     (ih - pad_h) * w_in + 
                                                     (iw - pad_w);
                                        let k_idx = oc * c_in * kh * kw + 
                                                   ic * kh * kw + 
                                                   khi * kw + 
                                                   kwi;
                                        sum += input[in_idx] * kernel[k_idx];
                                    }
                                }
                            }
                        }
                        let out_idx = batch * c_out * h_out * w_out + 
                                     oc * h_out * w_out + 
                                     oh * w_out + 
                                     ow;
                        output[out_idx] = sum;
                    }
                }
            }
        }
        
        MetalTensor::from_slice(&output, &[n, c_out, h_out, w_out], DType::F32)
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
        
        MetalTensor::from_slice(&output, shape, DType::F32)
    }
    
    /// LayerNorm: レイヤー正規化
    /// input: [*, normalized_shape]
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
        let norm_size = *shape.last().unwrap();
        let batch_size = shape.iter().take(shape.len() - 1).product::<usize>();
        
        let mut output = vec![0.0f32; input.len()];
        
        for b in 0..batch_size {
            let start = b * norm_size;
            let end = start + norm_size;
            
            // 平均計算
            let mean: f32 = input[start..end].iter().sum::<f32>() / norm_size as f32;
            
            // 分散計算
            let var: f32 = input[start..end].iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>() / norm_size as f32;
            
            let inv_std = 1.0 / (var + eps).sqrt();
            
            for i in 0..norm_size {
                output[start + i] = (input[start + i] - mean) * inv_std * gamma_v[i] + beta_v[i];
            }
        }
        
        MetalTensor::from_slice(&output, shape, DType::F32)
    }
    
    /// MaxPool2D: 最大プーリング
    pub fn max_pool2d_impl(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> MetalTensor {
        let input = self.to_vec::<f32>();
        let shape = MetalTensor::shape(self);
        
        let (n, c, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
        let (kh, kw) = kernel_size;
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
                                let in_idx = batch * c * h_in * w_in + 
                                             ch * h_in * w_in + 
                                             ih * w_in + iw;
                                max_val = max_val.max(input[in_idx]);
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
    
    /// AvgPool2D: 平均プーリング
    pub fn avg_pool2d_impl(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> MetalTensor {
        let input = self.to_vec::<f32>();
        let shape = MetalTensor::shape(self);
        
        let (n, c, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
        let (kh, kw) = kernel_size;
        let (stride_h, stride_w) = stride;
        
        let h_out = (h_in - kh) / stride_h + 1;
        let w_out = (w_in - kw) / stride_w + 1;
        let pool_size = (kh * kw) as f32;
        
        let mut output = vec![0.0f32; n * c * h_out * w_out];
        
        for batch in 0..n {
            for ch in 0..c {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = 0.0f32;
                        for khi in 0..kh {
                            for kwi in 0..kw {
                                let ih = oh * stride_h + khi;
                                let iw = ow * stride_w + kwi;
                                let in_idx = batch * c * h_in * w_in + 
                                             ch * h_in * w_in + 
                                             ih * w_in + iw;
                                sum += input[in_idx];
                            }
                        }
                        let out_idx = batch * c * h_out * w_out + 
                                     ch * h_out * w_out + 
                                     oh * w_out + ow;
                        output[out_idx] = sum / pool_size;
                    }
                }
            }
        }
        
        MetalTensor::from_slice(&output, &[n, c, h_out, w_out], DType::F32)
    }
    
    /// Dropout: ドロップアウト（推論時は何もしない）
    pub fn dropout_impl(&self, p: f32, training: bool) -> MetalTensor {
        if !training || p == 0.0 {
            return self.clone();
        }
        
        let input = self.to_vec::<f32>();
        let scale = 1.0 / (1.0 - p);
        
        let mut output = Vec::with_capacity(input.len());
        for val in input {
            let keep = rand::random::<f32>() > p;
            output.push(if keep { val * scale } else { 0.0 });
        }
        
        MetalTensor::from_slice(&output, MetalTensor::shape(self), DType::F32)
    }
}
