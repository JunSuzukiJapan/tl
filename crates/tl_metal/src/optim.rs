//! オプティマイザ（SGD, Adam, AdamW）

use crate::{MetalTensor, DType};

/// SGD オプティマイザ状態
pub struct SGD {
    pub learning_rate: f32,
    pub momentum: f32,
    pub weight_decay: f32,
    pub velocities: Vec<MetalTensor>,
}

impl SGD {
    pub fn new(learning_rate: f32, momentum: f32, weight_decay: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            weight_decay,
            velocities: Vec::new(),
        }
    }
    
    /// パラメータを更新
    pub fn step(&mut self, params: &mut [MetalTensor], grads: &[MetalTensor]) {
        // velocities を初期化
        if self.velocities.is_empty() {
            for p in params.iter() {
                self.velocities.push(MetalTensor::zeros(p.shape(), DType::F32));
            }
        }
        
        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            let mut param_data = param.to_vec::<f32>();
            let grad_data = grad.to_vec::<f32>();
            let mut vel_data = self.velocities[i].to_vec::<f32>();
            
            for j in 0..param_data.len() {
                // Weight decay
                let g = grad_data[j] + self.weight_decay * param_data[j];
                
                // Momentum
                vel_data[j] = self.momentum * vel_data[j] + g;
                
                // Update
                param_data[j] -= self.learning_rate * vel_data[j];
            }
            
            *param = MetalTensor::from_slice(&param_data, param.shape(), DType::F32);
            self.velocities[i] = MetalTensor::from_slice(&vel_data, grad.shape(), DType::F32);
        }
    }
}

/// Adam オプティマイザ状態
pub struct Adam {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub m: Vec<MetalTensor>,  // 一次モーメント
    pub v: Vec<MetalTensor>,  // 二次モーメント
    pub t: usize,             // タイムステップ
}

impl Adam {
    pub fn new(learning_rate: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            eps,
            weight_decay,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }
    
    pub fn default(learning_rate: f32) -> Self {
        Self::new(learning_rate, 0.9, 0.999, 1e-8, 0.0)
    }
    
    /// パラメータを更新
    pub fn step(&mut self, params: &mut [MetalTensor], grads: &[MetalTensor]) {
        // 状態を初期化
        if self.m.is_empty() {
            for p in params.iter() {
                self.m.push(MetalTensor::zeros(p.shape(), DType::F32));
                self.v.push(MetalTensor::zeros(p.shape(), DType::F32));
            }
        }
        
        self.t += 1;
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);
        
        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            let mut param_data = param.to_vec::<f32>();
            let grad_data = grad.to_vec::<f32>();
            let mut m_data = self.m[i].to_vec::<f32>();
            let mut v_data = self.v[i].to_vec::<f32>();
            
            for j in 0..param_data.len() {
                // 勾配にweight decayを適用
                let g = grad_data[j] + self.weight_decay * param_data[j];
                
                // 一次モーメント更新
                m_data[j] = self.beta1 * m_data[j] + (1.0 - self.beta1) * g;
                
                // 二次モーメント更新
                v_data[j] = self.beta2 * v_data[j] + (1.0 - self.beta2) * g * g;
                
                // バイアス補正
                let m_hat = m_data[j] / bias_correction1;
                let v_hat = v_data[j] / bias_correction2;
                
                // パラメータ更新
                param_data[j] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.eps);
            }
            
            *param = MetalTensor::from_slice(&param_data, param.shape(), DType::F32);
            self.m[i] = MetalTensor::from_slice(&m_data, grad.shape(), DType::F32);
            self.v[i] = MetalTensor::from_slice(&v_data, grad.shape(), DType::F32);
        }
    }
}

/// AdamW オプティマイザ（分離された weight decay）
pub struct AdamW {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub m: Vec<MetalTensor>,
    pub v: Vec<MetalTensor>,
    pub t: usize,
}

impl AdamW {
    pub fn new(learning_rate: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            eps,
            weight_decay,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }
    
    pub fn default(learning_rate: f32) -> Self {
        Self::new(learning_rate, 0.9, 0.999, 1e-8, 0.01)
    }
    
    /// パラメータを更新
    pub fn step(&mut self, params: &mut [MetalTensor], grads: &[MetalTensor]) {
        if self.m.is_empty() {
            for p in params.iter() {
                self.m.push(MetalTensor::zeros(p.shape(), DType::F32));
                self.v.push(MetalTensor::zeros(p.shape(), DType::F32));
            }
        }
        
        self.t += 1;
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);
        
        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            let mut param_data = param.to_vec::<f32>();
            let grad_data = grad.to_vec::<f32>();
            let mut m_data = self.m[i].to_vec::<f32>();
            let mut v_data = self.v[i].to_vec::<f32>();
            
            for j in 0..param_data.len() {
                // AdamW: weight decay を勾配に含めずに直接パラメータに適用
                param_data[j] *= 1.0 - self.learning_rate * self.weight_decay;
                
                // 一次モーメント更新
                m_data[j] = self.beta1 * m_data[j] + (1.0 - self.beta1) * grad_data[j];
                
                // 二次モーメント更新
                v_data[j] = self.beta2 * v_data[j] + (1.0 - self.beta2) * grad_data[j] * grad_data[j];
                
                // バイアス補正
                let m_hat = m_data[j] / bias_correction1;
                let v_hat = v_data[j] / bias_correction2;
                
                // パラメータ更新
                param_data[j] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.eps);
            }
            
            *param = MetalTensor::from_slice(&param_data, param.shape(), DType::F32);
            self.m[i] = MetalTensor::from_slice(&m_data, grad.shape(), DType::F32);
            self.v[i] = MetalTensor::from_slice(&v_data, grad.shape(), DType::F32);
        }
    }
}

/// 勾配クリッピング
pub fn clip_grad_norm(grads: &mut [MetalTensor], max_norm: f32) -> f32 {
    // 全勾配のL2ノルムを計算
    let mut total_norm_sq = 0.0f32;
    for grad in grads.iter() {
        let data = grad.to_vec::<f32>();
        for val in data {
            total_norm_sq += val * val;
        }
    }
    let total_norm = total_norm_sq.sqrt();
    
    // クリッピング
    if total_norm > max_norm {
        let clip_coef = max_norm / (total_norm + 1e-6);
        for grad in grads.iter_mut() {
            let data = grad.to_vec::<f32>();
            let clipped: Vec<f32> = data.iter().map(|x| x * clip_coef).collect();
            *grad = MetalTensor::from_slice(&clipped, grad.shape(), DType::F32);
        }
    }
    
    total_norm
}
