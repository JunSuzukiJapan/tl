//! 二項演算

use crate::tensor::MetalTensor;
use crate::DType;

impl MetalTensor {
    /// 要素ごとの加算
    pub fn add(&self, other: &MetalTensor) -> MetalTensor {
        assert_eq!(self.shape(), other.shape(), "Shape mismatch");
        assert_eq!(self.dtype(), other.dtype(), "DType mismatch");

        let result = MetalTensor::uninit(self.shape(), self.dtype());

        // TODO: Metal Shader で実装
        // 現時点では CPU fallback
        match self.dtype() {
            DType::F32 => {
                let a: Vec<f32> = self.to_vec();
                let b: Vec<f32> = other.to_vec();
                let c: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
                let ptr = result.buffer().contents() as *mut f32;
                unsafe {
                    std::ptr::copy_nonoverlapping(c.as_ptr(), ptr, c.len());
                }
            }
            _ => unimplemented!("add for {:?}", self.dtype()),
        }

        result
    }

    /// 要素ごとの乗算
    pub fn mul(&self, other: &MetalTensor) -> MetalTensor {
        assert_eq!(self.shape(), other.shape(), "Shape mismatch");
        assert_eq!(self.dtype(), other.dtype(), "DType mismatch");

        let result = MetalTensor::uninit(self.shape(), self.dtype());

        match self.dtype() {
            DType::F32 => {
                let a: Vec<f32> = self.to_vec();
                let b: Vec<f32> = other.to_vec();
                let c: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();
                let ptr = result.buffer().contents() as *mut f32;
                unsafe {
                    std::ptr::copy_nonoverlapping(c.as_ptr(), ptr, c.len());
                }
            }
            _ => unimplemented!("mul for {:?}", self.dtype()),
        }

        result
    }
}
