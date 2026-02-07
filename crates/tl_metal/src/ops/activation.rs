//! 活性化関数

use crate::tensor::MetalTensor;
use crate::DType;

impl MetalTensor {
    /// Softmax（軸指定）
    /// softmax(x)_i = exp(x_i) / sum(exp(x))
    pub fn softmax_impl(&self, axis: i32) -> MetalTensor {
        assert_eq!(MetalTensor::dtype(self), DType::F32, "softmax only supports F32");
        
        let shape = MetalTensor::shape(self);
        let ndim = shape.len();
        let axis = if axis < 0 { (ndim as i32 + axis) as usize } else { axis as usize };
        assert!(axis < ndim, "axis out of range");

        // 現在は 2D テンソルの最後の軸のみ対応
        if ndim == 2 && axis == 1 {
            self.softmax_2d_axis1()
        } else if ndim == 1 {
            self.softmax_1d()
        } else {
            // 汎用的な softmax（CPU fallback）
            self.softmax_generic(axis)
        }
    }

    /// 1D テンソルの softmax
    fn softmax_1d(&self) -> MetalTensor {
        let data: Vec<f32> = self.to_vec();
        let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = data.iter().map(|x| (x - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        let result: Vec<f32> = exp_vals.iter().map(|x| x / sum).collect();
        MetalTensor::from_slice(&result, MetalTensor::shape(self), MetalTensor::dtype(self))
    }

    /// 2D テンソルの axis=1 softmax
    fn softmax_2d_axis1(&self) -> MetalTensor {
        let shape = MetalTensor::shape(self);
        let rows = shape[0];
        let cols = shape[1];
        let data: Vec<f32> = self.to_vec();
        let mut result = vec![0.0f32; data.len()];

        for i in 0..rows {
            let start = i * cols;
            let end = start + cols;
            let row = &data[start..end];
            
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_vals: Vec<f32> = row.iter().map(|x| (x - max_val).exp()).collect();
            let sum: f32 = exp_vals.iter().sum();
            
            for j in 0..cols {
                result[start + j] = exp_vals[j] / sum;
            }
        }

        MetalTensor::from_slice(&result, MetalTensor::shape(self), MetalTensor::dtype(self))
    }

    /// 汎用 softmax（CPU）
    fn softmax_generic(&self, axis: usize) -> MetalTensor {
        // 単純化: flatten して softmax して reshape
        let data: Vec<f32> = self.to_vec();
        let shape = MetalTensor::shape(self);
        
        // 軸のサイズ
        let axis_size = shape[axis];
        let outer_size: usize = shape[..axis].iter().product();
        let inner_size: usize = shape[axis + 1..].iter().product();
        
        let mut result = vec![0.0f32; data.len()];
        
        for outer in 0..outer_size.max(1) {
            for inner in 0..inner_size.max(1) {
                // この slice の max を計算
                let mut max_val = f32::NEG_INFINITY;
                for a in 0..axis_size {
                    let idx = outer * axis_size * inner_size.max(1) + a * inner_size.max(1) + inner;
                    if data[idx] > max_val {
                        max_val = data[idx];
                    }
                }
                
                // exp と sum
                let mut sum = 0.0f32;
                for a in 0..axis_size {
                    let idx = outer * axis_size * inner_size.max(1) + a * inner_size.max(1) + inner;
                    let exp_val = (data[idx] - max_val).exp();
                    result[idx] = exp_val;
                    sum += exp_val;
                }
                
                // normalize
                for a in 0..axis_size {
                    let idx = outer * axis_size * inner_size.max(1) + a * inner_size.max(1) + inner;
                    result[idx] /= sum;
                }
            }
        }

        MetalTensor::from_slice(&result, MetalTensor::shape(self), MetalTensor::dtype(self))
    }
}
