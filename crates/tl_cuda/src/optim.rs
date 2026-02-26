//! CUDA オプティマイザ

use crate::tensor::CudaTensor;
use crate::DType;

/// SGD optimizer
pub struct SGD {
    pub lr: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    velocity: Vec<Vec<f32>>,
}

impl SGD {
    pub fn new(lr: f64, momentum: f64, weight_decay: f64) -> Self {
        SGD {
            lr,
            momentum,
            weight_decay,
            velocity: Vec::new(),
        }
    }

    pub fn step(&mut self, params: &mut [CudaTensor], grads: &[CudaTensor]) {
        // Initialize velocity buffers on first call
        if self.velocity.is_empty() {
            self.velocity = params
                .iter()
                .map(|p| vec![0.0f32; p.elem_count()])
                .collect();
        }

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            let mut p_data = param.to_vec::<f32>();
            let g_data = grad.to_vec::<f32>();
            let shape = param.shape().to_vec();

            for j in 0..p_data.len() {
                let mut g = g_data[j];

                // Weight decay
                if self.weight_decay != 0.0 {
                    g += self.weight_decay as f32 * p_data[j];
                }

                // Momentum
                if self.momentum != 0.0 {
                    self.velocity[i][j] = self.momentum as f32 * self.velocity[i][j] + g;
                    g = self.velocity[i][j];
                }

                p_data[j] -= self.lr as f32 * g;
            }

            *param = CudaTensor::from_slice(&p_data, &shape, DType::F32);
        }
    }
}

/// Adam optimizer
pub struct Adam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub step_count: u64,
    m: Vec<Vec<f32>>, // First moment
    v: Vec<Vec<f32>>, // Second moment
}

impl Adam {
    pub fn default(lr: f64) -> Self {
        Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            step_count: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    pub fn step(&mut self, params: &mut [CudaTensor], grads: &[CudaTensor]) {
        self.step_count += 1;

        // Initialize moment buffers on first call
        if self.m.is_empty() {
            self.m = params
                .iter()
                .map(|p| vec![0.0f32; p.elem_count()])
                .collect();
            self.v = params
                .iter()
                .map(|p| vec![0.0f32; p.elem_count()])
                .collect();
        }

        let beta1 = self.beta1 as f32;
        let beta2 = self.beta2 as f32;
        let lr = self.lr as f32;
        let eps = self.eps as f32;
        let t = self.step_count as f32;

        // Bias correction
        let bc1 = 1.0 - beta1.powf(t);
        let bc2 = 1.0 - beta2.powf(t);

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            let mut p_data = param.to_vec::<f32>();
            let g_data = grad.to_vec::<f32>();
            let shape = param.shape().to_vec();

            for j in 0..p_data.len() {
                let g = g_data[j];

                // Update biased first moment estimate
                self.m[i][j] = beta1 * self.m[i][j] + (1.0 - beta1) * g;
                // Update biased second moment estimate
                self.v[i][j] = beta2 * self.v[i][j] + (1.0 - beta2) * g * g;

                // Bias-corrected estimates
                let m_hat = self.m[i][j] / bc1;
                let v_hat = self.v[i][j] / bc2;

                p_data[j] -= lr * m_hat / (v_hat.sqrt() + eps);
            }

            *param = CudaTensor::from_slice(&p_data, &shape, DType::F32);
        }
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
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl AdamW {
    pub fn default(lr: f64) -> Self {
        AdamW {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            step_count: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    pub fn step(&mut self, params: &mut [CudaTensor], grads: &[CudaTensor]) {
        self.step_count += 1;

        if self.m.is_empty() {
            self.m = params
                .iter()
                .map(|p| vec![0.0f32; p.elem_count()])
                .collect();
            self.v = params
                .iter()
                .map(|p| vec![0.0f32; p.elem_count()])
                .collect();
        }

        let beta1 = self.beta1 as f32;
        let beta2 = self.beta2 as f32;
        let lr = self.lr as f32;
        let eps = self.eps as f32;
        let wd = self.weight_decay as f32;
        let t = self.step_count as f32;

        let bc1 = 1.0 - beta1.powf(t);
        let bc2 = 1.0 - beta2.powf(t);

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            let mut p_data = param.to_vec::<f32>();
            let g_data = grad.to_vec::<f32>();
            let shape = param.shape().to_vec();

            for j in 0..p_data.len() {
                // AdamW: decoupled weight decay (applied before moment update)
                p_data[j] -= lr * wd * p_data[j];

                let g = g_data[j];

                self.m[i][j] = beta1 * self.m[i][j] + (1.0 - beta1) * g;
                self.v[i][j] = beta2 * self.v[i][j] + (1.0 - beta2) * g * g;

                let m_hat = self.m[i][j] / bc1;
                let v_hat = self.v[i][j] / bc2;

                p_data[j] -= lr * m_hat / (v_hat.sqrt() + eps);
            }

            *param = CudaTensor::from_slice(&p_data, &shape, DType::F32);
        }
    }
}

/// 勾配ノルムのクリッピング
pub fn clip_grad_norm(grads: &mut [CudaTensor], max_norm: f32) -> f32 {
    // Compute total L2 norm of all gradients
    let mut total_norm_sq = 0.0f32;
    for grad in grads.iter() {
        let g_data = grad.to_vec::<f32>();
        total_norm_sq += g_data.iter().map(|&x| x * x).sum::<f32>();
    }
    let total_norm = total_norm_sq.sqrt();

    // Clip if needed
    if total_norm > max_norm {
        let clip_coef = max_norm / (total_norm + 1e-6);
        for grad in grads.iter_mut() {
            let g_data = grad.to_vec::<f32>();
            let shape = grad.shape().to_vec();
            let clipped: Vec<f32> = g_data.iter().map(|&x| x * clip_coef).collect();
            *grad = CudaTensor::from_slice(&clipped, &shape, DType::F32);
        }
    }

    total_norm
}
