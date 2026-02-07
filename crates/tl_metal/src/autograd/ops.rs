//! 演算の勾配関数

use super::{GradFn, MetalVarInner};
use crate::tensor::MetalTensor;
use std::cell::RefCell;
use std::rc::Rc;
use tl_backend::GpuOps;

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
        let grad_a = grad_output.matmul_impl(&self.b_data.transpose_impl(0, 1));
        let grad_b = self.a_data.transpose_impl(0, 1).matmul_impl(grad_output);
        vec![grad_a, grad_b]
    }
    
    fn inputs(&self) -> Vec<Rc<RefCell<MetalVarInner>>> {
        vec![self.a.clone(), self.b.clone()]
    }
}

/// sigmoid の勾配
pub struct SigmoidBackward {
    pub a: Rc<RefCell<MetalVarInner>>,
    pub output: MetalTensor,
}

impl GradFn for SigmoidBackward {
    fn backward(&self, grad_output: &MetalTensor) -> Vec<MetalTensor> {
        // d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
        let ones = MetalTensor::ones(self.output.shape(), self.output.dtype());
        let one_minus_s = ones.sub(&self.output);
        let grad = grad_output.mul(&self.output).mul(&one_minus_s);
        vec![grad]
    }
    
    fn inputs(&self) -> Vec<Rc<RefCell<MetalVarInner>>> {
        vec![self.a.clone()]
    }
}

/// exp の勾配
pub struct ExpBackward {
    pub a: Rc<RefCell<MetalVarInner>>,
    pub output: MetalTensor,
}

impl GradFn for ExpBackward {
    fn backward(&self, grad_output: &MetalTensor) -> Vec<MetalTensor> {
        // d(exp(x))/dx = exp(x)
        vec![grad_output.mul(&self.output)]
    }
    
    fn inputs(&self) -> Vec<Rc<RefCell<MetalVarInner>>> {
        vec![self.a.clone()]
    }
}

/// log の勾配
pub struct LogBackward {
    pub a: Rc<RefCell<MetalVarInner>>,
    pub a_data: MetalTensor,
}

impl GradFn for LogBackward {
    fn backward(&self, grad_output: &MetalTensor) -> Vec<MetalTensor> {
        // d(log(x))/dx = 1/x
        vec![grad_output.div(&self.a_data)]
    }
    
    fn inputs(&self) -> Vec<Rc<RefCell<MetalVarInner>>> {
        vec![self.a.clone()]
    }
}

/// sum(dim) の勾配
pub struct SumDimBackward {
    pub a: Rc<RefCell<MetalVarInner>>,
    pub input_shape: Vec<usize>,
    pub axis: i32,
}

impl GradFn for SumDimBackward {
    fn backward(&self, grad_output: &MetalTensor) -> Vec<MetalTensor> {
        // sum(dim) の勾配: grad_output を input_shape にブロードキャスト
        let axis = if self.axis < 0 {
            (self.input_shape.len() as i32 + self.axis) as usize
        } else {
            self.axis as usize
        };
        
        let grad_data: Vec<f32> = grad_output.to_vec();
        let numel: usize = self.input_shape.iter().product();
        let mut result = vec![0.0f32; numel];
        
        // axis 方向にブロードキャスト
        let outer_size: usize = self.input_shape[..axis].iter().product();
        let inner_size: usize = self.input_shape[axis + 1..].iter().product();
        let axis_size = self.input_shape[axis];
        
        for i in 0..outer_size {
            for k in 0..inner_size {
                let grad_val = grad_data[i * inner_size + k];
                for j in 0..axis_size {
                    result[i * axis_size * inner_size + j * inner_size + k] = grad_val;
                }
            }
        }
        
        vec![MetalTensor::from_slice(&result, &self.input_shape, grad_output.dtype())]
    }
    
    fn inputs(&self) -> Vec<Rc<RefCell<MetalVarInner>>> {
        vec![self.a.clone()]
    }
}
