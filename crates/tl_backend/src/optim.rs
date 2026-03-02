//! ジェネリックオプティマイザ（SGD, Adam, AdamW）
//!
//! `GpuTensor` トレイト境界で Metal/CUDA 共通のオプティマイザを提供。
//! 内部ではデータを CPU (Vec<f32>) にコピーして計算し、更新結果を GPU に戻す。

use crate::tensor::GpuTensor;

/// SGD オプティマイザ
pub struct SGD<T: GpuTensor> {
    pub learning_rate: f32,
    pub momentum: f32,
    pub weight_decay: f32,
    velocity: Vec<Vec<f32>>,
    _marker: std::marker::PhantomData<T>,
}

impl<T: GpuTensor> SGD<T> {
    pub fn new(learning_rate: f32, momentum: f32, weight_decay: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            weight_decay,
            velocity: Vec::new(),
            _marker: std::marker::PhantomData,
        }
    }

    /// パラメータを更新
    pub fn step(&mut self, params: &mut [T], grads: &[T]) {
        // velocity を初期化
        if self.velocity.is_empty() {
            self.velocity = params
                .iter()
                .map(|p| vec![0.0f32; p.elem_count()])
                .collect();
        }

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            let mut p_data = param.to_vec_f32();
            let g_data = grad.to_vec_f32();
            let shape = param.shape().to_vec();

            for j in 0..p_data.len() {
                let mut g = g_data[j];

                // Weight decay
                if self.weight_decay != 0.0 {
                    g += self.weight_decay * p_data[j];
                }

                // Momentum
                if self.momentum != 0.0 {
                    self.velocity[i][j] = self.momentum * self.velocity[i][j] + g;
                    g = self.velocity[i][j];
                }

                p_data[j] -= self.learning_rate * g;
            }

            if let Ok(new_param) = T::from_slice_f32(&p_data, &shape) {
                *param = new_param;
            }
        }
    }
}

/// Adam オプティマイザ
pub struct Adam<T: GpuTensor> {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    m: Vec<Vec<f32>>, // 一次モーメント
    v: Vec<Vec<f32>>, // 二次モーメント
    t: usize,         // タイムステップ
    _marker: std::marker::PhantomData<T>,
}

impl<T: GpuTensor> Adam<T> {
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
            _marker: std::marker::PhantomData,
        }
    }

    pub fn default(learning_rate: f32) -> Self {
        Self::new(learning_rate, 0.9, 0.999, 1e-8, 0.0)
    }

    /// パラメータを更新
    pub fn step(&mut self, params: &mut [T], grads: &[T]) {
        // 状態を初期化
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

        self.t += 1;
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            let mut p_data = param.to_vec_f32();
            let g_data = grad.to_vec_f32();
            let shape = param.shape().to_vec();

            for j in 0..p_data.len() {
                // 勾配に weight decay を適用
                let g = g_data[j] + self.weight_decay * p_data[j];

                // 一次モーメント更新
                self.m[i][j] = self.beta1 * self.m[i][j] + (1.0 - self.beta1) * g;

                // 二次モーメント更新
                self.v[i][j] = self.beta2 * self.v[i][j] + (1.0 - self.beta2) * g * g;

                // バイアス補正
                let m_hat = self.m[i][j] / bias_correction1;
                let v_hat = self.v[i][j] / bias_correction2;

                // パラメータ更新
                p_data[j] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.eps);
            }

            if let Ok(new_param) = T::from_slice_f32(&p_data, &shape) {
                *param = new_param;
            }
        }
    }
}

/// AdamW オプティマイザ（分離された weight decay）
pub struct AdamW<T: GpuTensor> {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    t: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T: GpuTensor> AdamW<T> {
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
            _marker: std::marker::PhantomData,
        }
    }

    pub fn default(learning_rate: f32) -> Self {
        Self::new(learning_rate, 0.9, 0.999, 1e-8, 0.01)
    }

    /// パラメータを更新
    pub fn step(&mut self, params: &mut [T], grads: &[T]) {
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

        self.t += 1;
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            let mut p_data = param.to_vec_f32();
            let g_data = grad.to_vec_f32();
            let shape = param.shape().to_vec();

            for j in 0..p_data.len() {
                // AdamW: weight decay をパラメータに直接適用
                p_data[j] *= 1.0 - self.learning_rate * self.weight_decay;

                let g = g_data[j];

                // 一次モーメント更新
                self.m[i][j] = self.beta1 * self.m[i][j] + (1.0 - self.beta1) * g;

                // 二次モーメント更新
                self.v[i][j] = self.beta2 * self.v[i][j] + (1.0 - self.beta2) * g * g;

                // バイアス補正
                let m_hat = self.m[i][j] / bias_correction1;
                let v_hat = self.v[i][j] / bias_correction2;

                // パラメータ更新
                p_data[j] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.eps);
            }

            if let Ok(new_param) = T::from_slice_f32(&p_data, &shape) {
                *param = new_param;
            }
        }
    }
}

/// 勾配ノルムのクリッピング
pub fn clip_grad_norm<T: GpuTensor>(grads: &mut [T], max_norm: f32) -> f32 {
    // 全勾配の L2 ノルムを計算
    let mut total_norm_sq = 0.0f32;
    for grad in grads.iter() {
        let data = grad.to_vec_f32();
        total_norm_sq += data.iter().map(|x| x * x).sum::<f32>();
    }
    let total_norm = total_norm_sq.sqrt();

    // クリッピング
    if total_norm > max_norm {
        let clip_coef = max_norm / (total_norm + 1e-6);
        for grad in grads.iter_mut() {
            let data = grad.to_vec_f32();
            let shape = grad.shape().to_vec();
            let clipped: Vec<f32> = data.iter().map(|x| x * clip_coef).collect();
            if let Ok(new_grad) = T::from_slice_f32(&clipped, &shape) {
                *grad = new_grad;
            }
        }
    }

    total_norm
}
