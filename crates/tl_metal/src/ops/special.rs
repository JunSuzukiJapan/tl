//! 特殊演算

use crate::tensor::MetalTensor;
use crate::DType;

impl MetalTensor {
    /// where_cond: condition ? x : y
    pub fn where_cond_impl(condition: &MetalTensor, x: &MetalTensor, y: &MetalTensor) -> MetalTensor {
        let cond: Vec<f32> = condition.to_vec();
        let x_data: Vec<f32> = x.to_vec();
        let y_data: Vec<f32> = y.to_vec();
        
        let result: Vec<f32> = cond.iter()
            .zip(x_data.iter())
            .zip(y_data.iter())
            .map(|((c, x), y)| if *c > 0.0 { *x } else { *y })
            .collect();
        
        MetalTensor::from_slice(&result, MetalTensor::shape(x), MetalTensor::dtype(x))
    }

    /// tril: 下三角行列
    pub fn tril_impl(&self, diagonal: i32) -> MetalTensor {
        let shape = MetalTensor::shape(self);
        assert!(shape.len() >= 2, "tril requires at least 2D tensor");
        
        let rows = shape[shape.len() - 2];
        let cols = shape[shape.len() - 1];
        let batch_size: usize = shape[..shape.len() - 2].iter().product::<usize>().max(1);
        
        let data: Vec<f32> = self.to_vec();
        let mut result = data.clone();
        
        for b in 0..batch_size {
            let offset = b * rows * cols;
            for i in 0..rows {
                for j in 0..cols {
                    if (j as i32) > (i as i32 + diagonal) {
                        result[offset + i * cols + j] = 0.0;
                    }
                }
            }
        }
        
        MetalTensor::from_slice(&result, shape, MetalTensor::dtype(self))
    }

    /// cross_entropy: -sum(target * log(pred))
    pub fn cross_entropy_impl(&self, target: &MetalTensor) -> MetalTensor {
        // self = predictions (after softmax), target = one-hot or probabilities
        let pred: Vec<f32> = self.to_vec();
        let tgt: Vec<f32> = target.to_vec();
        
        let loss: f32 = pred.iter()
            .zip(tgt.iter())
            .map(|(p, t)| -t * (p + 1e-7).ln())
            .sum();
        
        MetalTensor::from_slice(&[loss], &[1], MetalTensor::dtype(self))
    }

    /// repeat_interleave: 要素を繰り返す
    pub fn repeat_interleave_impl(&self, repeats: usize, axis: usize) -> MetalTensor {
        let shape = MetalTensor::shape(self);
        assert!(axis < shape.len(), "axis out of range");
        
        let data: Vec<f32> = self.to_vec();
        let outer_size: usize = shape[..axis].iter().product::<usize>().max(1);
        let axis_size = shape[axis];
        let inner_size: usize = shape[axis + 1..].iter().product::<usize>().max(1);
        
        let mut new_shape = shape.to_vec();
        new_shape[axis] *= repeats;
        
        let mut result = Vec::with_capacity(new_shape.iter().product());
        
        for outer in 0..outer_size {
            for a in 0..axis_size {
                for _ in 0..repeats {
                    for inner in 0..inner_size {
                        let idx = outer * axis_size * inner_size + a * inner_size + inner;
                        result.push(data[idx]);
                    }
                }
            }
        }
        
        MetalTensor::from_slice(&result, &new_shape, MetalTensor::dtype(self))
    }

    /// to_dtype: データ型変換（現在は F32 のみ対応）
    pub fn to_dtype(&self, dtype: DType) -> MetalTensor {
        if MetalTensor::dtype(self) == dtype {
            return self.clone_data();
        }
        // 現在は F32 → F32 のみ
        self.clone_data()
    }

    /// index_select: インデックスで選択
    pub fn index_select_impl(&self, axis: usize, indices: &MetalTensor) -> MetalTensor {
        let shape = MetalTensor::shape(self);
        assert!(axis < shape.len(), "axis out of range");
        
        let idx_data: Vec<f32> = indices.to_vec();
        let num_indices = idx_data.len();
        
        let outer_size: usize = shape[..axis].iter().product::<usize>().max(1);
        let axis_size = shape[axis];
        let inner_size: usize = shape[axis + 1..].iter().product::<usize>().max(1);
        
        let mut new_shape = shape.to_vec();
        new_shape[axis] = num_indices;
        
        let data: Vec<f32> = self.to_vec();
        let mut result = vec![0.0f32; new_shape.iter().product()];
        
        for outer in 0..outer_size {
            for (new_a, &idx) in idx_data.iter().enumerate() {
                let a = idx as usize;
                assert!(a < axis_size, "index {} out of range for axis size {}", a, axis_size);
                for inner in 0..inner_size {
                    let src_idx = outer * axis_size * inner_size + a * inner_size + inner;
                    let dst_idx = outer * num_indices * inner_size + new_a * inner_size + inner;
                    result[dst_idx] = data[src_idx];
                }
            }
        }
        
        MetalTensor::from_slice(&result, &new_shape, MetalTensor::dtype(self))
    }
}
