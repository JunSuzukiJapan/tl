//! インデックス操作

use crate::tensor::MetalTensor;
use crate::DType;

impl MetalTensor {
    /// スライス（1軸のみ、開始位置と長さ指定）
    pub fn slice(&self, axis: usize, start: usize, len: usize) -> MetalTensor {
        let shape = self.shape();
        assert!(axis < shape.len(), "axis out of range");
        assert!(start + len <= shape[axis], "slice out of range");

        // 新しい形状
        let mut new_shape = shape.to_vec();
        new_shape[axis] = len;

        // データを抽出
        let data: Vec<f32> = self.to_vec();
        let outer_size: usize = shape[..axis].iter().product();
        let axis_size = shape[axis];
        let inner_size: usize = if axis + 1 < shape.len() { shape[axis + 1..].iter().product() } else { 1 };
        
        let new_elem_count: usize = new_shape.iter().product();
        let mut result = vec![0.0f32; new_elem_count];
        
        let mut out_idx = 0;
        for outer in 0..outer_size.max(1) {
            for a in start..(start + len) {
                for inner in 0..inner_size.max(1) {
                    let idx = outer * axis_size * inner_size + a * inner_size + inner;
                    result[out_idx] = data[idx];
                    out_idx += 1;
                }
            }
        }

        MetalTensor::from_slice(&result, &new_shape, self.dtype())
    }

    /// embedding lookup
    /// self: [V, D] (埋め込み行列), indices: [T] (インデックス)
    /// → [T, D]
    pub fn embedding(&self, indices: &MetalTensor) -> MetalTensor {
        assert_eq!(self.shape().len(), 2, "embedding matrix must be 2D");
        assert_eq!(indices.shape().len(), 1, "indices must be 1D");
        
        let vocab_size = self.shape()[0];
        let dim = self.shape()[1];
        let seq_len = indices.shape()[0];
        
        let emb_data: Vec<f32> = self.to_vec();
        let idx_data: Vec<f32> = indices.to_vec();
        
        let mut result = vec![0.0f32; seq_len * dim];
        
        for t in 0..seq_len {
            let idx = idx_data[t] as usize;
            assert!(idx < vocab_size, "index {} out of vocabulary size {}", idx, vocab_size);
            for d in 0..dim {
                result[t * dim + d] = emb_data[idx * dim + d];
            }
        }

        MetalTensor::from_slice(&result, &[seq_len, dim], self.dtype())
    }
}
