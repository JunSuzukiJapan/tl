//! オプティマイザ（SGD, Adam, AdamW）
//!
//! sync_stream + contents() バッファ直操作により to_vec()/from_slice() を排除。
//! StorageModeShared バッファの CPU/GPU 共有メモリを活用。

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
        
        crate::command_stream::sync_stream();
        
        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            let n = param.elem_count();
            let shape = param.shape().to_vec();
            
            unsafe {
                let p_ptr = param.buffer().contents() as *mut f32;
                let g_ptr = grad.buffer().contents() as *const f32;
                let v_ptr = self.velocities[i].buffer().contents() as *mut f32;
                
                for j in 0..n {
                    // Weight decay
                    let g = *g_ptr.add(j) + self.weight_decay * *p_ptr.add(j);
                    
                    // Momentum
                    *v_ptr.add(j) = self.momentum * *v_ptr.add(j) + g;
                    
                    // Update
                    *p_ptr.add(j) -= self.learning_rate * *v_ptr.add(j);
                }
            }
            
            // shape が変わらないのでバッファ再確保不要（in-place 更新）
            let _ = shape;
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
        
        crate::command_stream::sync_stream();
        
        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            let n = param.elem_count();
            
            unsafe {
                let p_ptr = param.buffer().contents() as *mut f32;
                let g_ptr = grad.buffer().contents() as *const f32;
                let m_ptr = self.m[i].buffer().contents() as *mut f32;
                let v_ptr = self.v[i].buffer().contents() as *mut f32;
                
                for j in 0..n {
                    // 勾配にweight decayを適用
                    let g = *g_ptr.add(j) + self.weight_decay * *p_ptr.add(j);
                    
                    // 一次モーメント更新
                    *m_ptr.add(j) = self.beta1 * *m_ptr.add(j) + (1.0 - self.beta1) * g;
                    
                    // 二次モーメント更新
                    *v_ptr.add(j) = self.beta2 * *v_ptr.add(j) + (1.0 - self.beta2) * g * g;
                    
                    // バイアス補正
                    let m_hat = *m_ptr.add(j) / bias_correction1;
                    let v_hat = *v_ptr.add(j) / bias_correction2;
                    
                    // パラメータ更新
                    *p_ptr.add(j) -= self.learning_rate * m_hat / (v_hat.sqrt() + self.eps);
                }
            }
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
        
        crate::command_stream::sync_stream();
        
        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            let n = param.elem_count();
            
            unsafe {
                let p_ptr = param.buffer().contents() as *mut f32;
                let g_ptr = grad.buffer().contents() as *const f32;
                let m_ptr = self.m[i].buffer().contents() as *mut f32;
                let v_ptr = self.v[i].buffer().contents() as *mut f32;
                
                for j in 0..n {
                    // AdamW: weight decay を勾配に含めずに直接パラメータに適用
                    *p_ptr.add(j) *= 1.0 - self.learning_rate * self.weight_decay;
                    
                    let g = *g_ptr.add(j);
                    
                    // 一次モーメント更新
                    *m_ptr.add(j) = self.beta1 * *m_ptr.add(j) + (1.0 - self.beta1) * g;
                    
                    // 二次モーメント更新
                    *v_ptr.add(j) = self.beta2 * *v_ptr.add(j) + (1.0 - self.beta2) * g * g;
                    
                    // バイアス補正
                    let m_hat = *m_ptr.add(j) / bias_correction1;
                    let v_hat = *v_ptr.add(j) / bias_correction2;
                    
                    // パラメータ更新
                    *p_ptr.add(j) -= self.learning_rate * m_hat / (v_hat.sqrt() + self.eps);
                }
            }
        }
    }
}

/// 勾配クリッピング
pub fn clip_grad_norm(grads: &mut [MetalTensor], max_norm: f32) -> f32 {
    crate::command_stream::sync_stream();

    // 全勾配のL2ノルムを計算（バッファ直読み）
    let mut total_norm_sq = 0.0f32;
    for grad in grads.iter() {
        let n = grad.elem_count();
        unsafe {
            let ptr = grad.buffer().contents() as *const f32;
            for i in 0..n {
                let val = *ptr.add(i);
                total_norm_sq += val * val;
            }
        }
    }
    let total_norm = total_norm_sq.sqrt();
    
    // クリッピング（バッファ直書き込み）
    if total_norm > max_norm {
        let clip_coef = max_norm / (total_norm + 1e-6);
        for grad in grads.iter_mut() {
            let n = grad.elem_count();
            unsafe {
                let ptr = grad.buffer().contents() as *mut f32;
                for i in 0..n {
                    *ptr.add(i) *= clip_coef;
                }
            }
        }
    }
    
    total_norm
}
