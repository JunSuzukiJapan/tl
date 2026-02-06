//! MetalVar の演算メソッド

use super::ops::*;
use super::MetalVar;
use crate::tensor::MetalTensor;

impl MetalVar {
    /// 加算
    pub fn add(&self, other: &MetalVar) -> MetalVar {
        let a_data = self.inner.borrow().data.clone_data();
        let b_data = other.inner.borrow().data.clone_data();
        let result = a_data.add(&b_data);
        
        let requires_grad = self.inner.borrow().requires_grad || other.inner.borrow().requires_grad;
        
        if requires_grad {
            MetalVar::from_op(result, Box::new(AddBackward {
                a: self.inner.clone(),
                b: other.inner.clone(),
            }), true)
        } else {
            MetalVar::new(result, false)
        }
    }

    /// 減算
    pub fn sub(&self, other: &MetalVar) -> MetalVar {
        let a_data = self.inner.borrow().data.clone_data();
        let b_data = other.inner.borrow().data.clone_data();
        let result = a_data.sub(&b_data);
        
        let requires_grad = self.inner.borrow().requires_grad || other.inner.borrow().requires_grad;
        
        if requires_grad {
            MetalVar::from_op(result, Box::new(SubBackward {
                a: self.inner.clone(),
                b: other.inner.clone(),
            }), true)
        } else {
            MetalVar::new(result, false)
        }
    }

    /// 乗算
    pub fn mul(&self, other: &MetalVar) -> MetalVar {
        let a_data = self.inner.borrow().data.clone_data();
        let b_data = other.inner.borrow().data.clone_data();
        let result = a_data.mul(&b_data);
        
        let requires_grad = self.inner.borrow().requires_grad || other.inner.borrow().requires_grad;
        
        if requires_grad {
            MetalVar::from_op(result, Box::new(MulBackward {
                a: self.inner.clone(),
                b: other.inner.clone(),
                a_data: self.inner.borrow().data.clone_data(),
                b_data: other.inner.borrow().data.clone_data(),
            }), true)
        } else {
            MetalVar::new(result, false)
        }
    }

    /// 除算
    pub fn div(&self, other: &MetalVar) -> MetalVar {
        let a_data = self.inner.borrow().data.clone_data();
        let b_data = other.inner.borrow().data.clone_data();
        let result = a_data.div(&b_data);
        
        let requires_grad = self.inner.borrow().requires_grad || other.inner.borrow().requires_grad;
        
        if requires_grad {
            MetalVar::from_op(result, Box::new(DivBackward {
                a: self.inner.clone(),
                b: other.inner.clone(),
                a_data: self.inner.borrow().data.clone_data(),
                b_data: other.inner.borrow().data.clone_data(),
            }), true)
        } else {
            MetalVar::new(result, false)
        }
    }

    /// べき乗
    pub fn pow(&self, other: &MetalVar) -> MetalVar {
        let a_data = self.inner.borrow().data.clone_data();
        let b_data = other.inner.borrow().data.clone_data();
        let result = a_data.pow(&b_data);
        
        let requires_grad = self.inner.borrow().requires_grad || other.inner.borrow().requires_grad;
        
        if requires_grad {
            MetalVar::from_op(result.clone_data(), Box::new(PowBackward {
                a: self.inner.clone(),
                b: other.inner.clone(),
                a_data: self.inner.borrow().data.clone_data(),
                b_data: other.inner.borrow().data.clone_data(),
                output: result,
            }), true)
        } else {
            MetalVar::new(result, false)
        }
    }

    /// 全要素合計
    pub fn sumall(&self) -> MetalVar {
        let data = self.inner.borrow().data.clone_data();
        let sum_val = data.sumall();
        let result = MetalTensor::from_slice(&[sum_val], &[1], data.dtype());
        
        if self.inner.borrow().requires_grad {
            MetalVar::from_op(result, Box::new(SumallBackward {
                a: self.inner.clone(),
                shape: data.shape().to_vec(),
            }), true)
        } else {
            MetalVar::new(result, false)
        }
    }

    /// ReLU
    pub fn relu(&self) -> MetalVar {
        let data = self.inner.borrow().data.clone_data();
        let result = data.relu();
        
        if self.inner.borrow().requires_grad {
            MetalVar::from_op(result, Box::new(ReluBackward {
                a: self.inner.clone(),
                a_data: data,
            }), true)
        } else {
            MetalVar::new(result, false)
        }
    }

    /// Softmax
    pub fn softmax(&self, axis: i32) -> MetalVar {
        let data = self.inner.borrow().data.clone_data();
        let result = data.softmax(axis);
        
        if self.inner.borrow().requires_grad {
            MetalVar::from_op(result.clone_data(), Box::new(SoftmaxBackward {
                a: self.inner.clone(),
                output: result,
                axis,
            }), true)
        } else {
            MetalVar::new(result, false)
        }
    }

    /// 行列積
    pub fn matmul(&self, other: &MetalVar) -> MetalVar {
        let a_data = self.inner.borrow().data.clone_data();
        let b_data = other.inner.borrow().data.clone_data();
        let result = a_data.matmul(&b_data);
        
        let requires_grad = self.inner.borrow().requires_grad || other.inner.borrow().requires_grad;
        
        if requires_grad {
            MetalVar::from_op(result, Box::new(MatmulBackward {
                a: self.inner.clone(),
                b: other.inner.clone(),
                a_data: self.inner.borrow().data.clone_data(),
                b_data: other.inner.borrow().data.clone_data(),
            }), true)
        } else {
            MetalVar::new(result, false)
        }
    }

    /// スカラー加算
    pub fn add_scalar(&self, scalar: f32) -> MetalVar {
        let data = self.inner.borrow().data.clone_data();
        let result = data.add_scalar(scalar);
        // スカラーは定数として扱う
        MetalVar::new(result, self.inner.borrow().requires_grad)
    }

    /// スカラー乗算
    pub fn mul_scalar(&self, scalar: f32) -> MetalVar {
        let data = self.inner.borrow().data.clone_data();
        let result = data.mul_scalar(scalar);
        MetalVar::new(result, self.inner.borrow().requires_grad)
    }
}
