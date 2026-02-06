//! 軸指定 Reduce 演算

use crate::tensor::MetalTensor;
use crate::DType;

impl MetalTensor {
    /// 軸指定で合計
    pub fn sum(&self, axis: i32) -> MetalTensor {
        assert_eq!(self.dtype(), DType::F32, "sum only supports F32");
        
        let shape = self.shape();
        let ndim = shape.len();
        let axis = if axis < 0 { (ndim as i32 + axis) as usize } else { axis as usize };
        assert!(axis < ndim, "axis out of range");

        let axis_size = shape[axis];
        let mut new_shape: Vec<usize> = shape.to_vec();
        new_shape.remove(axis);
        if new_shape.is_empty() {
            new_shape.push(1); // scalar -> [1]
        }

        let outer_size: usize = shape[..axis].iter().product();
        let inner_size: usize = if axis + 1 < ndim { shape[axis + 1..].iter().product() } else { 1 };
        
        let data: Vec<f32> = self.to_vec();
        let mut result = vec![0.0f32; outer_size.max(1) * inner_size.max(1)];
        
        for outer in 0..outer_size.max(1) {
            for inner in 0..inner_size.max(1) {
                let mut sum = 0.0f32;
                for a in 0..axis_size {
                    let idx = outer * axis_size * inner_size + a * inner_size + inner;
                    sum += data[idx];
                }
                result[outer * inner_size + inner] = sum;
            }
        }

        MetalTensor::from_slice(&result, &new_shape, self.dtype())
    }

    /// argmax（全体）
    pub fn argmax_all(&self) -> usize {
        let data: Vec<f32> = self.to_vec();
        data.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// argmax（軸指定）
    pub fn argmax(&self, axis: i32) -> MetalTensor {
        assert_eq!(self.dtype(), DType::F32, "argmax only supports F32");
        
        let shape = self.shape();
        let ndim = shape.len();
        let axis = if axis < 0 { (ndim as i32 + axis) as usize } else { axis as usize };
        assert!(axis < ndim, "axis out of range");

        let axis_size = shape[axis];
        let mut new_shape: Vec<usize> = shape.to_vec();
        new_shape.remove(axis);
        if new_shape.is_empty() {
            new_shape.push(1);
        }

        let outer_size: usize = shape[..axis].iter().product();
        let inner_size: usize = if axis + 1 < ndim { shape[axis + 1..].iter().product() } else { 1 };
        
        let data: Vec<f32> = self.to_vec();
        let mut result = vec![0.0f32; outer_size.max(1) * inner_size.max(1)];
        
        for outer in 0..outer_size.max(1) {
            for inner in 0..inner_size.max(1) {
                let mut max_val = f32::NEG_INFINITY;
                let mut max_idx = 0usize;
                for a in 0..axis_size {
                    let idx = outer * axis_size * inner_size + a * inner_size + inner;
                    if data[idx] > max_val {
                        max_val = data[idx];
                        max_idx = a;
                    }
                }
                result[outer * inner_size + inner] = max_idx as f32;
            }
        }

        MetalTensor::from_slice(&result, &new_shape, DType::F32)
    }

    /// max（軸指定）
    pub fn max(&self, axis: i32) -> MetalTensor {
        assert_eq!(self.dtype(), DType::F32, "max only supports F32");
        
        let shape = self.shape();
        let ndim = shape.len();
        let axis = if axis < 0 { (ndim as i32 + axis) as usize } else { axis as usize };
        assert!(axis < ndim, "axis out of range");

        let axis_size = shape[axis];
        let mut new_shape: Vec<usize> = shape.to_vec();
        new_shape.remove(axis);
        if new_shape.is_empty() {
            new_shape.push(1);
        }

        let outer_size: usize = shape[..axis].iter().product();
        let inner_size: usize = if axis + 1 < ndim { shape[axis + 1..].iter().product() } else { 1 };
        
        let data: Vec<f32> = self.to_vec();
        let mut result = vec![f32::NEG_INFINITY; outer_size.max(1) * inner_size.max(1)];
        
        for outer in 0..outer_size.max(1) {
            for inner in 0..inner_size.max(1) {
                for a in 0..axis_size {
                    let idx = outer * axis_size * inner_size + a * inner_size + inner;
                    let out_idx = outer * inner_size + inner;
                    if data[idx] > result[out_idx] {
                        result[out_idx] = data[idx];
                    }
                }
            }
        }

        MetalTensor::from_slice(&result, &new_shape, self.dtype())
    }

    /// min（軸指定）
    pub fn min(&self, axis: i32) -> MetalTensor {
        assert_eq!(self.dtype(), DType::F32, "min only supports F32");
        
        let shape = self.shape();
        let ndim = shape.len();
        let axis = if axis < 0 { (ndim as i32 + axis) as usize } else { axis as usize };
        assert!(axis < ndim, "axis out of range");

        let axis_size = shape[axis];
        let mut new_shape: Vec<usize> = shape.to_vec();
        new_shape.remove(axis);
        if new_shape.is_empty() {
            new_shape.push(1);
        }

        let outer_size: usize = shape[..axis].iter().product();
        let inner_size: usize = if axis + 1 < ndim { shape[axis + 1..].iter().product() } else { 1 };
        
        let data: Vec<f32> = self.to_vec();
        let mut result = vec![f32::INFINITY; outer_size.max(1) * inner_size.max(1)];
        
        for outer in 0..outer_size.max(1) {
            for inner in 0..inner_size.max(1) {
                for a in 0..axis_size {
                    let idx = outer * axis_size * inner_size + a * inner_size + inner;
                    let out_idx = outer * inner_size + inner;
                    if data[idx] < result[out_idx] {
                        result[out_idx] = data[idx];
                    }
                }
            }
        }

        MetalTensor::from_slice(&result, &new_shape, self.dtype())
    }

    /// argmin（軸指定）
    pub fn argmin(&self, axis: i32) -> MetalTensor {
        assert_eq!(self.dtype(), DType::F32, "argmin only supports F32");
        
        let shape = self.shape();
        let ndim = shape.len();
        let axis = if axis < 0 { (ndim as i32 + axis) as usize } else { axis as usize };
        assert!(axis < ndim, "axis out of range");

        let axis_size = shape[axis];
        let mut new_shape: Vec<usize> = shape.to_vec();
        new_shape.remove(axis);
        if new_shape.is_empty() {
            new_shape.push(1);
        }

        let outer_size: usize = shape[..axis].iter().product();
        let inner_size: usize = if axis + 1 < ndim { shape[axis + 1..].iter().product() } else { 1 };
        
        let data: Vec<f32> = self.to_vec();
        let mut result = vec![0.0f32; outer_size.max(1) * inner_size.max(1)];
        
        for outer in 0..outer_size.max(1) {
            for inner in 0..inner_size.max(1) {
                let mut min_val = f32::INFINITY;
                let mut min_idx = 0usize;
                for a in 0..axis_size {
                    let idx = outer * axis_size * inner_size + a * inner_size + inner;
                    if data[idx] < min_val {
                        min_val = data[idx];
                        min_idx = a;
                    }
                }
                result[outer * inner_size + inner] = min_idx as f32;
            }
        }

        MetalTensor::from_slice(&result, &new_shape, DType::F32)
    }

    /// mean（軸指定）
    pub fn mean(&self, axis: i32) -> MetalTensor {
        let sum = self.sum(axis);
        let shape = self.shape();
        let ndim = shape.len();
        let axis = if axis < 0 { (ndim as i32 + axis) as usize } else { axis as usize };
        let axis_size = shape[axis] as f32;
        sum.div_scalar(axis_size)
    }
}
