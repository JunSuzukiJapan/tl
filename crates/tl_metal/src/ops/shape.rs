//! 形状操作

use crate::tensor::MetalTensor;

impl MetalTensor {
    /// 形状変更（データコピーなし、参照共有）
    pub fn reshape_impl(&self, new_shape: &[usize]) -> MetalTensor {
        let old_size: usize = MetalTensor::shape(self).iter().product();
        let new_size: usize = new_shape.iter().product();
        assert_eq!(old_size, new_size, "reshape: element count mismatch {} vs {}", old_size, new_size);

        MetalTensor::from_buffer_shared(
            self.buffer_arc().clone(),
            new_shape.to_vec(),
            MetalTensor::dtype(self),
        )
    }

    /// squeeze: サイズ1の次元を削除
    pub fn squeeze_impl(&self, dim: usize) -> MetalTensor {
        let shape = MetalTensor::shape(self);
        assert!(dim < shape.len(), "dim out of range");
        assert_eq!(shape[dim], 1, "squeeze: dimension {} is not 1", dim);

        let mut new_shape: Vec<usize> = shape.to_vec();
        new_shape.remove(dim);
        
        MetalTensor::from_buffer_shared(
            self.buffer_arc().clone(),
            new_shape,
            MetalTensor::dtype(self),
        )
    }

    /// unsqueeze: サイズ1の次元を追加
    pub fn unsqueeze_impl(&self, dim: usize) -> MetalTensor {
        let mut new_shape: Vec<usize> = MetalTensor::shape(self).to_vec();
        assert!(dim <= new_shape.len(), "dim out of range");
        new_shape.insert(dim, 1);
        
        MetalTensor::from_buffer_shared(
            self.buffer_arc().clone(),
            new_shape,
            MetalTensor::dtype(self),
        )
    }

    /// transpose（2Dのみ対応）
    pub fn transpose_impl(&self, dim0: usize, dim1: usize) -> MetalTensor {
        let shape = MetalTensor::shape(self);
        assert!(shape.len() == 2, "transpose currently only supports 2D tensors");
        assert!(dim0 < shape.len() && dim1 < shape.len(), "dim out of range");

        let rows = shape[0];
        let cols = shape[1];
        let new_shape = vec![cols, rows];

        let result = MetalTensor::uninit(&new_shape, MetalTensor::dtype(self));
        
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
    
    /// broadcast_to
    pub fn broadcast_to_impl(&self, shape: &[usize]) -> MetalTensor {
        // 現時点では単純なコピー（TODO: 実際のブロードキャスト）
        let data: Vec<f32> = self.to_vec();
        let target_size: usize = shape.iter().product();
        
        if data.len() == target_size {
            return MetalTensor::from_slice(&data, shape, MetalTensor::dtype(self));
        }
        
        // 簡易ブロードキャスト
        let mut result = vec![0.0f32; target_size];
        let src_len = data.len();
        for i in 0..target_size {
            result[i] = data[i % src_len];
        }
        MetalTensor::from_slice(&result, shape, MetalTensor::dtype(self))
    }
    
    /// narrow
    pub fn narrow_impl(&self, axis: usize, start: usize, len: usize) -> MetalTensor {
        self.slice_impl(axis, start, len)
    }
    
    /// slice
    pub fn slice_impl(&self, axis: usize, start: usize, len: usize) -> MetalTensor {
        let shape = MetalTensor::shape(self);
        assert!(axis < shape.len(), "axis out of range");
        assert!(start + len <= shape[axis], "slice out of range");
        
        // CPU fallback
        let data: Vec<f32> = self.to_vec();
        let mut new_shape = shape.to_vec();
        new_shape[axis] = len;
        
        let outer: usize = shape[..axis].iter().product();
        let axis_stride: usize = shape[axis..].iter().product();
        let inner: usize = if axis + 1 < shape.len() { shape[axis+1..].iter().product() } else { 1 };
        
        let mut result = Vec::with_capacity(outer * len * inner);
        for o in 0..outer.max(1) {
            for a in start..start+len {
                for i in 0..inner {
                    let idx = o * axis_stride + a * inner + i;
                    result.push(data[idx]);
                }
            }
        }
        
        MetalTensor::from_slice(&result, &new_shape, MetalTensor::dtype(self))
    }
    
    /// contiguous
    pub fn contiguous_impl(&self) -> MetalTensor {
        self.clone_data()
    }
    
    /// cat
    pub fn cat_impl(tensors: &[&MetalTensor], axis: usize) -> MetalTensor {
        assert!(!tensors.is_empty(), "cat: empty tensor list");
        
        let first_shape = MetalTensor::shape(tensors[0]);
        let mut new_shape = first_shape.to_vec();
        let mut total_axis_size = 0usize;
        
        for t in tensors {
            let ts = MetalTensor::shape(*t);
            for (i, (a, b)) in first_shape.iter().zip(ts.iter()).enumerate() {
                if i != axis {
                    assert_eq!(a, b, "cat: shape mismatch at dim {}", i);
                }
            }
            total_axis_size += ts[axis];
        }
        new_shape[axis] = total_axis_size;
        
        let mut all_data = Vec::new();
        for t in tensors {
            all_data.extend(t.to_vec::<f32>());
        }
        
        MetalTensor::from_slice(&all_data, &new_shape, MetalTensor::dtype(tensors[0]))
    }
}
