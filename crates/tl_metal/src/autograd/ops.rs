//! 演算の勾配関数

use super::{GradFn, MetalVarInner};
use crate::tensor::MetalTensor;
use std::cell::RefCell;
use std::rc::Rc;

/// 加算の勾配
pub struct AddBackward {
    pub a: Rc<RefCell<MetalVarInner>>,
    pub b: Rc<RefCell<MetalVarInner>>,
}

impl GradFn for AddBackward {
    fn backward(&self, grad_output: &MetalTensor) -> Vec<MetalTensor> {
        // d(a+b)/da = 1, d(a+b)/db = 1
        vec![grad_output.clone_data(), grad_output.clone_data()]
    }
    
    fn inputs(&self) -> Vec<Rc<RefCell<MetalVarInner>>> {
        vec![self.a.clone(), self.b.clone()]
    }
}

/// 減算の勾配
pub struct SubBackward {
    pub a: Rc<RefCell<MetalVarInner>>,
    pub b: Rc<RefCell<MetalVarInner>>,
}

impl GradFn for SubBackward {
    fn backward(&self, grad_output: &MetalTensor) -> Vec<MetalTensor> {
        // d(a-b)/da = 1, d(a-b)/db = -1
        vec![grad_output.clone_data(), grad_output.neg()]
    }
    
    fn inputs(&self) -> Vec<Rc<RefCell<MetalVarInner>>> {
        vec![self.a.clone(), self.b.clone()]
    }
}

/// 乗算の勾配
pub struct MulBackward {
    pub a: Rc<RefCell<MetalVarInner>>,
    pub b: Rc<RefCell<MetalVarInner>>,
    pub a_data: MetalTensor,
    pub b_data: MetalTensor,
}

impl GradFn for MulBackward {
    fn backward(&self, grad_output: &MetalTensor) -> Vec<MetalTensor> {
        // d(a*b)/da = b, d(a*b)/db = a
        vec![
            grad_output.mul(&self.b_data),
            grad_output.mul(&self.a_data),
        ]
    }
    
    fn inputs(&self) -> Vec<Rc<RefCell<MetalVarInner>>> {
        vec![self.a.clone(), self.b.clone()]
    }
}

/// 除算の勾配
pub struct DivBackward {
    pub a: Rc<RefCell<MetalVarInner>>,
    pub b: Rc<RefCell<MetalVarInner>>,
    pub a_data: MetalTensor,
    pub b_data: MetalTensor,
}

impl GradFn for DivBackward {
    fn backward(&self, grad_output: &MetalTensor) -> Vec<MetalTensor> {
        // d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
        let grad_a = grad_output.div(&self.b_data);
        let b_sq = self.b_data.mul(&self.b_data);
        let grad_b = grad_output.mul(&self.a_data).neg().div(&b_sq);
        vec![grad_a, grad_b]
    }
    
    fn inputs(&self) -> Vec<Rc<RefCell<MetalVarInner>>> {
        vec![self.a.clone(), self.b.clone()]
    }
}

/// べき乗の勾配 (a^b)
pub struct PowBackward {
    pub a: Rc<RefCell<MetalVarInner>>,
    pub b: Rc<RefCell<MetalVarInner>>,
    pub a_data: MetalTensor,
    pub b_data: MetalTensor,
    pub output: MetalTensor,
}

impl GradFn for PowBackward {
    fn backward(&self, grad_output: &MetalTensor) -> Vec<MetalTensor> {
        // d(a^b)/da = b * a^(b-1)
        // d(a^b)/db = a^b * log(a)
        let ones = MetalTensor::ones(self.b_data.shape(), self.b_data.dtype());
        let b_minus_1 = self.b_data.sub(&ones);
        let grad_a = grad_output.mul(&self.b_data).mul(&self.a_data.pow(&b_minus_1));
        let grad_b = grad_output.mul(&self.output).mul(&self.a_data.log());
        vec![grad_a, grad_b]
    }
    
    fn inputs(&self) -> Vec<Rc<RefCell<MetalVarInner>>> {
        vec![self.a.clone(), self.b.clone()]
    }
}

/// sumall の勾配
pub struct SumallBackward {
    pub a: Rc<RefCell<MetalVarInner>>,
    pub shape: Vec<usize>,
}

impl GradFn for SumallBackward {
    fn backward(&self, grad_output: &MetalTensor) -> Vec<MetalTensor> {
        // sumall の勾配: すべての要素に同じ勾配
        let grad_val = grad_output.to_vec::<f32>()[0];
        let count: usize = self.shape.iter().product();
        let grads = vec![grad_val; count];
        vec![MetalTensor::from_slice(&grads, &self.shape, grad_output.dtype())]
    }
    
    fn inputs(&self) -> Vec<Rc<RefCell<MetalVarInner>>> {
        vec![self.a.clone()]
    }
}

/// ReLU の勾配
pub struct ReluBackward {
    pub a: Rc<RefCell<MetalVarInner>>,
    pub a_data: MetalTensor,
}

impl GradFn for ReluBackward {
    fn backward(&self, grad_output: &MetalTensor) -> Vec<MetalTensor> {
        // d(relu(a))/da = 1 if a > 0 else 0
        let data: Vec<f32> = self.a_data.to_vec();
        let grad: Vec<f32> = grad_output.to_vec();
        let result: Vec<f32> = data.iter().zip(grad.iter())
            .map(|(a, g)| if *a > 0.0 { *g } else { 0.0 })
            .collect();
        vec![MetalTensor::from_slice(&result, self.a_data.shape(), self.a_data.dtype())]
    }
    
    fn inputs(&self) -> Vec<Rc<RefCell<MetalVarInner>>> {
        vec![self.a.clone()]
    }
}

/// Softmax の勾配
pub struct SoftmaxBackward {
    pub a: Rc<RefCell<MetalVarInner>>,
    pub output: MetalTensor,
    pub axis: i32,
}

impl GradFn for SoftmaxBackward {
    fn backward(&self, grad_output: &MetalTensor) -> Vec<MetalTensor> {
        // softmax の勾配: s_i * (delta_ij - s_j)
        // 簡略化: grad_a = s * (g - sum(s * g))
        let s: Vec<f32> = self.output.to_vec();
        let g: Vec<f32> = grad_output.to_vec();
        
        let shape = self.output.shape();
        if shape.len() == 2 && self.axis == 1 {
            let rows = shape[0];
            let cols = shape[1];
            let mut result = vec![0.0f32; s.len()];
            
            for i in 0..rows {
                let start = i * cols;
                let end = start + cols;
                let s_row = &s[start..end];
                let g_row = &g[start..end];
                
                let sg_sum: f32 = s_row.iter().zip(g_row.iter()).map(|(si, gi)| si * gi).sum();
                
                for j in 0..cols {
                    result[start + j] = s_row[j] * (g_row[j] - sg_sum);
                }
            }
            vec![MetalTensor::from_slice(&result, shape, self.output.dtype())]
        } else {
            // 1D or generic
            let sg_sum: f32 = s.iter().zip(g.iter()).map(|(si, gi)| si * gi).sum();
            let result: Vec<f32> = s.iter().zip(g.iter())
                .map(|(si, gi)| si * (gi - sg_sum))
                .collect();
            vec![MetalTensor::from_slice(&result, shape, self.output.dtype())]
        }
    }
    
    fn inputs(&self) -> Vec<Rc<RefCell<MetalVarInner>>> {
        vec![self.a.clone()]
    }
}

/// matmul の勾配
pub struct MatmulBackward {
    pub a: Rc<RefCell<MetalVarInner>>,
    pub b: Rc<RefCell<MetalVarInner>>,
    pub a_data: MetalTensor,
    pub b_data: MetalTensor,
}

impl GradFn for MatmulBackward {
    fn backward(&self, grad_output: &MetalTensor) -> Vec<MetalTensor> {
        // C = A @ B
        // dL/dA = dL/dC @ B^T
        // dL/dB = A^T @ dL/dC
        let grad_a = grad_output.matmul(&self.b_data.transpose(0, 1));
        let grad_b = self.a_data.transpose(0, 1).matmul(grad_output);
        vec![grad_a, grad_b]
    }
    
    fn inputs(&self) -> Vec<Rc<RefCell<MetalVarInner>>> {
        vec![self.a.clone(), self.b.clone()]
    }
}
