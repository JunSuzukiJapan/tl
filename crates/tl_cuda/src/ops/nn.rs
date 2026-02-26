//! ニューラルネットワーク演算

use crate::tensor::CudaTensor;
use crate::DType;
use tl_backend::{BackendError, BackendResult};

impl CudaTensor {
    /// Conv2D: input [N,C_in,H,W] × weight [C_out,C_in,kH,kW] → [N,C_out,H_out,W_out]
    pub fn conv2d_impl(
        &self,
        weight: &CudaTensor,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> BackendResult<CudaTensor> {
        let input_shape = self.shape();
        let weight_shape = weight.shape();

        if input_shape.len() != 4 || weight_shape.len() != 4 {
            return Err(BackendError::ShapeMismatch(
                "conv2d requires 4D input and weight".into(),
            ));
        }

        let (n, c_in, h_in, w_in) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let (c_out, wc_in, kh, kw) = (
            weight_shape[0],
            weight_shape[1],
            weight_shape[2],
            weight_shape[3],
        );

        if c_in != wc_in {
            return Err(BackendError::ShapeMismatch(format!(
                "conv2d channel mismatch: input {} vs weight {}",
                c_in, wc_in
            )));
        }

        let h_out = (h_in + 2 * padding.0 - kh) / stride.0 + 1;
        let w_out = (w_in + 2 * padding.1 - kw) / stride.1 + 1;

        let input_data = self.to_vec::<f32>();
        let weight_data = weight.to_vec::<f32>();
        let mut output = vec![0.0f32; n * c_out * h_out * w_out];

        for batch in 0..n {
            for co in 0..c_out {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = 0.0f32;
                        for ci in 0..c_in {
                            for khi in 0..kh {
                                for kwi in 0..kw {
                                    let ih = oh * stride.0 + khi;
                                    let iw = ow * stride.1 + kwi;
                                    let ih = ih as isize - padding.0 as isize;
                                    let iw = iw as isize - padding.1 as isize;

                                    if ih >= 0
                                        && ih < h_in as isize
                                        && iw >= 0
                                        && iw < w_in as isize
                                    {
                                        let ih = ih as usize;
                                        let iw = iw as usize;
                                        let in_idx = batch * c_in * h_in * w_in
                                            + ci * h_in * w_in
                                            + ih * w_in
                                            + iw;
                                        let w_idx =
                                            co * c_in * kh * kw + ci * kh * kw + khi * kw + kwi;
                                        sum += input_data[in_idx] * weight_data[w_idx];
                                    }
                                }
                            }
                        }
                        let out_idx =
                            batch * c_out * h_out * w_out + co * h_out * w_out + oh * w_out + ow;
                        output[out_idx] = sum;
                    }
                }
            }
        }

        Ok(CudaTensor::from_slice(
            &output,
            &[n, c_out, h_out, w_out],
            DType::F32,
        ))
    }

    /// Batch Normalization
    pub fn batch_norm_impl(
        &self,
        gamma: &CudaTensor,
        beta: &CudaTensor,
        running_mean: &CudaTensor,
        running_var: &CudaTensor,
        eps: f32,
    ) -> BackendResult<CudaTensor> {
        let shape = self.shape();
        if shape.len() < 2 {
            return Err(BackendError::ShapeMismatch(
                "batch_norm requires >= 2D".into(),
            ));
        }
        let c = shape[1];
        let data = self.to_vec::<f32>();
        let gamma_data = gamma.to_vec::<f32>();
        let beta_data = beta.to_vec::<f32>();
        let mean_data = running_mean.to_vec::<f32>();
        let var_data = running_var.to_vec::<f32>();

        let mut result = data.clone();
        let spatial: usize = shape[2..].iter().product();
        let n = shape[0];

        for batch in 0..n {
            for ch in 0..c {
                let g = gamma_data[ch];
                let b = beta_data[ch];
                let m = mean_data[ch];
                let v = var_data[ch];
                let inv_std = 1.0 / (v + eps).sqrt();

                for s in 0..spatial {
                    let idx = batch * c * spatial + ch * spatial + s;
                    result[idx] = g * (result[idx] - m) * inv_std + b;
                }
            }
        }

        Ok(CudaTensor::from_slice(&result, shape, DType::F32))
    }

    /// Layer Normalization
    pub fn layer_norm_impl(
        &self,
        gamma: &CudaTensor,
        beta: &CudaTensor,
        eps: f32,
    ) -> BackendResult<CudaTensor> {
        let shape = self.shape();
        let data = self.to_vec::<f32>();
        let gamma_data = gamma.to_vec::<f32>();
        let beta_data = beta.to_vec::<f32>();

        // 最後の次元で正規化
        let norm_size = *shape.last().unwrap();
        let outer: usize = data.len() / norm_size;
        let mut result = vec![0.0f32; data.len()];

        for o in 0..outer {
            let offset = o * norm_size;
            let slice = &data[offset..offset + norm_size];

            let mean: f32 = slice.iter().sum::<f32>() / norm_size as f32;
            let var: f32 =
                slice.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / norm_size as f32;
            let inv_std = 1.0 / (var + eps).sqrt();

            for i in 0..norm_size {
                result[offset + i] = gamma_data[i] * (slice[i] - mean) * inv_std + beta_data[i];
            }
        }

        Ok(CudaTensor::from_slice(&result, shape, DType::F32))
    }

    /// Max Pooling 2D
    pub fn max_pool2d_impl(
        &self,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> BackendResult<CudaTensor> {
        let shape = self.shape();
        if shape.len() != 4 {
            return Err(BackendError::ShapeMismatch("max_pool2d requires 4D".into()));
        }
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let h_out = (h - kernel.0) / stride.0 + 1;
        let w_out = (w - kernel.1) / stride.1 + 1;

        let data = self.to_vec::<f32>();
        let mut result = vec![0.0f32; n * c * h_out * w_out];

        for batch in 0..n {
            for ch in 0..c {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut max_val = f32::NEG_INFINITY;
                        for kh in 0..kernel.0 {
                            for kw in 0..kernel.1 {
                                let ih = oh * stride.0 + kh;
                                let iw = ow * stride.1 + kw;
                                let idx = batch * c * h * w + ch * h * w + ih * w + iw;
                                max_val = max_val.max(data[idx]);
                            }
                        }
                        let out_idx =
                            batch * c * h_out * w_out + ch * h_out * w_out + oh * w_out + ow;
                        result[out_idx] = max_val;
                    }
                }
            }
        }

        Ok(CudaTensor::from_slice(
            &result,
            &[n, c, h_out, w_out],
            DType::F32,
        ))
    }

    /// Average Pooling 2D
    pub fn avg_pool2d_impl(
        &self,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> BackendResult<CudaTensor> {
        let shape = self.shape();
        if shape.len() != 4 {
            return Err(BackendError::ShapeMismatch("avg_pool2d requires 4D".into()));
        }
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let h_out = (h - kernel.0) / stride.0 + 1;
        let w_out = (w - kernel.1) / stride.1 + 1;
        let k_size = (kernel.0 * kernel.1) as f32;

        let data = self.to_vec::<f32>();
        let mut result = vec![0.0f32; n * c * h_out * w_out];

        for batch in 0..n {
            for ch in 0..c {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = 0.0f32;
                        for kh in 0..kernel.0 {
                            for kw in 0..kernel.1 {
                                let ih = oh * stride.0 + kh;
                                let iw = ow * stride.1 + kw;
                                let idx = batch * c * h * w + ch * h * w + ih * w + iw;
                                sum += data[idx];
                            }
                        }
                        let out_idx =
                            batch * c * h_out * w_out + ch * h_out * w_out + oh * w_out + ow;
                        result[out_idx] = sum / k_size;
                    }
                }
            }
        }

        Ok(CudaTensor::from_slice(
            &result,
            &[n, c, h_out, w_out],
            DType::F32,
        ))
    }

    /// Dropout
    pub fn dropout_impl(&self, p: f32, training: bool) -> BackendResult<CudaTensor> {
        if !training || p == 0.0 {
            return self.clone_data();
        }
        let data = self.to_vec::<f32>();
        let scale = 1.0 / (1.0 - p);
        // 簡易実装: 決定論的ハッシュベースマスク
        let result: Vec<f32> = data
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                // 簡易的な擬似乱数（テスト用）
                let hash = ((i as u64).wrapping_mul(2654435761) >> 16) as f32 / 65536.0;
                if hash < p {
                    0.0
                } else {
                    x * scale
                }
            })
            .collect();
        Ok(CudaTensor::from_slice(&result, self.shape(), self.dtype()))
    }
}
