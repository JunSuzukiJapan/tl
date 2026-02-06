//! Autograd - 自動微分
//!
//! 計算グラフを構築し、backward で勾配を計算する。

pub mod ops;
pub mod var_ops;

use crate::tensor::MetalTensor;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// ノード ID 生成用カウンタ
static NODE_ID: AtomicUsize = AtomicUsize::new(0);

/// 勾配関数のトレイト
pub trait GradFn {
    /// 出力勾配から入力勾配を計算
    fn backward(&self, grad_output: &MetalTensor) -> Vec<MetalTensor>;
    
    /// 入力ノードへの参照
    fn inputs(&self) -> Vec<Rc<RefCell<MetalVarInner>>>;
}

/// MetalVar の内部データ
pub struct MetalVarInner {
    /// ノード ID
    pub id: usize,
    /// データ
    pub data: MetalTensor,
    /// 勾配（累積）
    pub grad: Option<MetalTensor>,
    /// 勾配が必要か
    pub requires_grad: bool,
    /// 勾配関数（葉ノードは None）
    pub grad_fn: Option<Box<dyn GradFn>>,
}

/// 自動微分対応テンソル
#[derive(Clone)]
pub struct MetalVar {
    inner: Rc<RefCell<MetalVarInner>>,
}

impl MetalVar {
    /// 新規作成
    pub fn new(data: MetalTensor, requires_grad: bool) -> Self {
        let id = NODE_ID.fetch_add(1, Ordering::SeqCst);
        MetalVar {
            inner: Rc::new(RefCell::new(MetalVarInner {
                id,
                data,
                grad: None,
                requires_grad,
                grad_fn: None,
            })),
        }
    }

    /// 演算結果から作成（grad_fn 付き）
    pub fn from_op(data: MetalTensor, grad_fn: Box<dyn GradFn>, requires_grad: bool) -> Self {
        let id = NODE_ID.fetch_add(1, Ordering::SeqCst);
        MetalVar {
            inner: Rc::new(RefCell::new(MetalVarInner {
                id,
                data,
                grad: None,
                requires_grad,
                grad_fn: Some(grad_fn),
            })),
        }
    }

    /// 内部参照を取得
    pub fn inner(&self) -> Rc<RefCell<MetalVarInner>> {
        self.inner.clone()
    }

    /// データへの参照
    pub fn data(&self) -> MetalTensor {
        self.inner.borrow().data.clone_data()
    }

    /// 形状
    pub fn shape(&self) -> Vec<usize> {
        self.inner.borrow().data.shape().to_vec()
    }

    /// 勾配を取得
    pub fn grad(&self) -> Option<MetalTensor> {
        self.inner.borrow().grad.as_ref().map(|g| g.clone_data())
    }

    /// 勾配をゼロクリア
    pub fn zero_grad(&self) {
        self.inner.borrow_mut().grad = None;
    }

    /// 計算グラフから切り離す
    pub fn detach(&self) -> MetalVar {
        MetalVar::new(self.data(), false)
    }

    /// backward（逆伝播）
    pub fn backward(&self) {
        let inner = self.inner.borrow();
        assert!(inner.requires_grad, "backward called on non-requires_grad tensor");
        
        // 初期勾配: すべて 1
        let ones = MetalTensor::ones(inner.data.shape(), inner.data.dtype());
        drop(inner);
        
        self.backward_with_grad(&ones);
    }

    /// 勾配を指定して backward
    pub fn backward_with_grad(&self, grad_output: &MetalTensor) {
        // 勾配を累積
        {
            let mut inner = self.inner.borrow_mut();
            if let Some(ref mut grad) = inner.grad {
                // 累積
                let new_grad = grad.add(grad_output);
                *grad = new_grad;
            } else {
                inner.grad = Some(grad_output.clone_data());
            }
        }

        // 入力ノードに勾配を伝播
        let grad_fn = {
            let inner = self.inner.borrow();
            inner.grad_fn.as_ref().map(|gf| {
                let grads = gf.backward(grad_output);
                let inputs = gf.inputs();
                (grads, inputs)
            })
        };

        if let Some((grads, inputs)) = grad_fn {
            for (input, grad) in inputs.into_iter().zip(grads.into_iter()) {
                let requires_grad = input.borrow().requires_grad;
                if requires_grad {
                    let var = MetalVar { inner: input };
                    var.backward_with_grad(&grad);
                }
            }
        }
    }
}

// MetalTensor に clone_data を追加
impl MetalTensor {
    pub fn clone_data(&self) -> MetalTensor {
        MetalTensor::from_slice(&self.to_vec::<f32>(), self.shape(), self.dtype())
    }
}
