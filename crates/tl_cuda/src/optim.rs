//! CUDA オプティマイザ

use crate::tensor::CudaTensor;

/// SGD optimizer
pub struct SGD {
    pub lr: f64,
    pub momentum: f64,
    pub weight_decay: f64,
}

impl SGD {
    pub fn new(lr: f64, momentum: f64, weight_decay: f64) -> Self {
        SGD { lr, momentum, weight_decay }
    }

    pub fn step(&mut self, _params: &mut [CudaTensor], _grads: &[CudaTensor]) {
        unimplemented!("SGD::step")
    }
}

/// Adam optimizer
pub struct Adam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub step_count: u64,
}

impl Adam {
    pub fn default(lr: f64) -> Self {
        Adam { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, step_count: 0 }
    }

    pub fn step(&mut self, _params: &mut [CudaTensor], _grads: &[CudaTensor]) {
        unimplemented!("Adam::step")
    }
}

/// AdamW optimizer
pub struct AdamW {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub step_count: u64,
}

impl AdamW {
    pub fn default(lr: f64) -> Self {
        AdamW { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.01, step_count: 0 }
    }

    pub fn step(&mut self, _params: &mut [CudaTensor], _grads: &[CudaTensor]) {
        unimplemented!("AdamW::step")
    }
}

/// 勾配ノルムのクリッピング
pub fn clip_grad_norm(_grads: &mut [CudaTensor], _max_norm: f32) -> f32 {
    unimplemented!("clip_grad_norm")
}
