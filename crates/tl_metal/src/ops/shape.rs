//! 形状操作

use crate::tensor::MetalTensor;

impl MetalTensor {
    /// 形状変更（データコピーなし、参照共有）
    /// 注意: 現在の実装は contiguous なテンソルのみサポート
    pub fn reshape(&self, new_shape: &[usize]) -> MetalTensor {
        // 要素数チェック
        let old_size: usize = self.shape().iter().product();
        let new_size: usize = new_shape.iter().product();
        assert_eq!(old_size, new_size, "reshape: element count mismatch {} vs {}", old_size, new_size);

        MetalTensor::from_buffer_shared(
            self.buffer_arc().clone(),
            new_shape.to_vec(),
            self.dtype(),
        )
    }

    /// squeeze: サイズ1の次元を削除
    pub fn squeeze(&self, dim: usize) -> MetalTensor {
        let shape = self.shape();
        assert!(dim < shape.len(), "dim out of range");
        assert_eq!(shape[dim], 1, "squeeze: dimension {} is not 1", dim);

        let mut new_shape: Vec<usize> = shape.to_vec();
        new_shape.remove(dim);
        
        MetalTensor::from_buffer_shared(
            self.buffer_arc().clone(),
            new_shape,
            self.dtype(),
        )
    }

    /// unsqueeze: サイズ1の次元を追加
    pub fn unsqueeze(&self, dim: usize) -> MetalTensor {
        let mut new_shape: Vec<usize> = self.shape().to_vec();
        assert!(dim <= new_shape.len(), "dim out of range");
        new_shape.insert(dim, 1);
        
        MetalTensor::from_buffer_shared(
            self.buffer_arc().clone(),
            new_shape,
            self.dtype(),
        )
    }

    /// transpose（2Dのみ対応）
    pub fn transpose(&self, dim0: usize, dim1: usize) -> MetalTensor {
        let shape = self.shape();
        assert!(shape.len() == 2, "transpose currently only supports 2D tensors");
        assert!(dim0 < shape.len() && dim1 < shape.len(), "dim out of range");

        let rows = shape[0];
        let cols = shape[1];
        let new_shape = vec![cols, rows];

        let result = MetalTensor::uninit(&new_shape, self.dtype());
        
        // CPU fallback for transpose
        let src: Vec<f32> = self.to_vec();
        let mut dst = vec![0.0f32; src.len()];
        for i in 0..rows {
            for j in 0..cols {
                dst[j * rows + i] = src[i * cols + j];
            }
        }
        
        let ptr = result.buffer().contents() as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(dst.as_ptr(), ptr, dst.len());
        }
        
        result
    }
}
