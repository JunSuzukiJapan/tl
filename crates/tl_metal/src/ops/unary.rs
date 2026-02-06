//! 単項演算

use crate::tensor::MetalTensor;
use crate::DType;

impl MetalTensor {
    /// 要素ごとの exp
    pub fn exp(&self) -> MetalTensor {
        let result = MetalTensor::uninit(self.shape(), self.dtype());

        match self.dtype() {
            DType::F32 => {
                let a: Vec<f32> = self.to_vec();
                let c: Vec<f32> = a.iter().map(|x| x.exp()).collect();
                let ptr = result.buffer().contents() as *mut f32;
                unsafe {
                    std::ptr::copy_nonoverlapping(c.as_ptr(), ptr, c.len());
                }
            }
            _ => unimplemented!("exp for {:?}", self.dtype()),
        }

        result
    }

    /// 要素ごとの sqrt
    pub fn sqrt(&self) -> MetalTensor {
        let result = MetalTensor::uninit(self.shape(), self.dtype());

        match self.dtype() {
            DType::F32 => {
                let a: Vec<f32> = self.to_vec();
                let c: Vec<f32> = a.iter().map(|x| x.sqrt()).collect();
                let ptr = result.buffer().contents() as *mut f32;
                unsafe {
                    std::ptr::copy_nonoverlapping(c.as_ptr(), ptr, c.len());
                }
            }
            _ => unimplemented!("sqrt for {:?}", self.dtype()),
        }

        result
    }
}
