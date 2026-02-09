//! CPU テンソル

use crate::autograd::GradFn;
use crate::DType;


/// Autograd メタデータ
pub struct AutogradMeta {
    pub grad: Option<CpuTensor>,
    pub grad_fn: Option<Box<dyn GradFn>>,
    pub requires_grad: bool,
}

/// CPU テンソル（Vec<f32> / Vec<i64> ベース）
pub struct CpuTensor {
    pub(crate) data_f32: Vec<f32>,
    pub(crate) data_i64: Option<Vec<i64>>,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: DType,
    pub autograd: Option<Box<AutogradMeta>>,
}

/*
impl CpuTensor {
    /// テンソルの内部データをクリアしてメモリを OS に返却する。
    /// 構造体ポインタ自体は有効なまま残るため、use-after-free を防止できる。
    /// release/free 時に呼ばれる。
    pub(crate) fn clear_data(&mut self) {
        self.data_f32 = Vec::new();
        self.data_i64 = None;
        self.shape = Vec::new();
        self.autograd = None;
    }
}
*/

impl Clone for CpuTensor {
    fn clone(&self) -> Self {
        self.clone_data()
    }
}

impl CpuTensor {
    // ========== コンストラクタ ==========

    fn alloc_from_pool() -> Self {
        // Recycle from pool if available, preserving Vec capacity
        if let Some(boxed) = crate::memory::recycle_tensor() {
            *boxed
        } else {
             CpuTensor {
                data_f32: Vec::new(),
                data_i64: None,
                shape: Vec::new(),
                dtype: DType::F32,
                autograd: None,
            }
        }
    }

    pub fn zeros(shape: &[usize], dtype: DType) -> Self {
        let count: usize = shape.iter().product();
        let mut t = Self::alloc_from_pool();
        
        // Reconfigure
        t.dtype = dtype;
        t.shape = shape.to_vec();
        t.autograd = None;
        t.data_i64 = None;

        // Resize buffer (uses existing capacity)
        t.data_f32.clear();
        t.data_f32.resize(count, 0.0);
        
        t
    }
    
    pub fn ones(shape: &[usize], dtype: DType) -> Self {
        let count: usize = shape.iter().product();
        let mut t = Self::alloc_from_pool();
        
        t.dtype = dtype;
        t.shape = shape.to_vec();
        t.autograd = None;
        t.data_i64 = None;

        t.data_f32.clear();
        t.data_f32.resize(count, 1.0);
        
        t
    }

    pub fn from_slice(data: &[f32], shape: &[usize], dtype: DType) -> Self {
         let mut t = Self::alloc_from_pool();
         
         t.dtype = dtype;
         t.shape = shape.to_vec();
         t.autograd = None;
         t.data_i64 = None;
         
         t.data_f32.clear();
         t.data_f32.extend_from_slice(data);
         
         t
    }

    pub fn from_slice_i64_data(data: &[i64], shape: &[usize], dtype: DType) -> Self {
        let f32_data: Vec<f32> = data.iter().map(|x| *x as f32).collect();
        CpuTensor {
            data_f32: f32_data,
            data_i64: Some(data.to_vec()),
            shape: shape.to_vec(),
            dtype,
            autograd: None,
        }
    }

    pub fn randn(shape: &[usize], dtype: DType) -> Self {
        use rand::Rng;
        let count: usize = shape.iter().product();
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..count)
            .map(|_| {
                let u1: f32 = rng.gen();
                let u2: f32 = rng.gen();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
            })
            .collect();
        CpuTensor {
            data_f32: data,
            data_i64: None,
            shape: shape.to_vec(),
            dtype,
            autograd: None,
        }
    }

    pub fn arange(start: i64, end: i64, dtype: DType) -> Self {
        let data: Vec<f32> = (start..end).map(|x| x as f32).collect();
        let len = data.len();
        CpuTensor {
            data_f32: data,
            data_i64: None,
            shape: vec![len],
            dtype,
            autograd: None,
        }
    }

    // ========== アクセサ ==========

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn elem_count(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn to_vec<T: Copy + Default + 'static>(&self) -> Vec<T> {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            // Safety: T == f32
            let src = &self.data_f32;
            let mut dst = vec![T::default(); src.len()];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr() as *const T,
                    dst.as_mut_ptr(),
                    src.len(),
                );
            }
            dst
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i64>() {
            if let Some(ref i64_data) = self.data_i64 {
                let mut dst = vec![T::default(); i64_data.len()];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        i64_data.as_ptr() as *const T,
                        dst.as_mut_ptr(),
                        i64_data.len(),
                    );
                }
                dst
            } else {
                let i64_data: Vec<i64> = self.data_f32.iter().map(|x| *x as i64).collect();
                let mut dst = vec![T::default(); i64_data.len()];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        i64_data.as_ptr() as *const T,
                        dst.as_mut_ptr(),
                        i64_data.len(),
                    );
                }
                dst
            }
        } else {
            vec![T::default(); self.elem_count()]
        }
    }

    pub fn data_f32(&self) -> &[f32] {
        &self.data_f32
    }

    pub fn data_f32_mut(&mut self) -> &mut [f32] {
        &mut self.data_f32
    }

    /// 浅いクローン（データコピーあり＝CPU なので Arc 不要）
    /// autograd メタデータはコピーしない
    pub fn shallow_clone(&self) -> Self {
        CpuTensor {
            data_f32: self.data_f32.clone(),
            data_i64: self.data_i64.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype,
            autograd: None,
        }
    }

    pub fn clone_data(&self) -> CpuTensor {
        self.shallow_clone()
    }

    // ========== Autograd ==========

    pub fn requires_grad(&self) -> bool {
        self.autograd.as_ref().map_or(false, |a| a.requires_grad)
    }

    pub fn enable_grad(&mut self) {
        if self.autograd.is_none() {
            self.autograd = Some(Box::new(AutogradMeta {
                grad: None,
                grad_fn: None,
                requires_grad: true,
            }));
        } else {
            self.autograd.as_mut().unwrap().requires_grad = true;
        }
    }

    pub fn set_grad_fn(&mut self, grad_fn: Box<dyn GradFn>) {
        self.autograd = Some(Box::new(AutogradMeta {
            grad: None,
            grad_fn: Some(grad_fn),
            requires_grad: true,
        }));
    }

    pub fn get_grad(&self) -> Option<CpuTensor> {
        self.autograd.as_ref().and_then(|a| {
            a.grad.as_ref().map(|g| g.shallow_clone())
        })
    }

    pub fn zero_grad(&mut self) {
        if let Some(ref mut a) = self.autograd {
            a.grad = None;
        }
    }

    pub fn backward(&mut self) {
        if !self.requires_grad() {
            return;
        }
        let self_ptr = self as *mut CpuTensor;

        // 1. DFS でトポロジカル順序を構築
        let mut topo_order: Vec<*mut CpuTensor> = Vec::new();
        let mut visited = std::collections::HashSet::<usize>::new();
        Self::build_topo(self_ptr, &mut visited, &mut topo_order);

        // 2. 出力テンソルの勾配を ones で初期化
        let ones = CpuTensor::ones(self.shape(), self.dtype());
        if let Some(ref mut meta) = self.autograd {
            meta.grad = Some(ones);
        }

        // 3. 逆トポロジカル順序で勾配を伝播
        //    各ノードの蓄積済み勾配を使って backward を呼ぶため、
        //    全パスの勾配が合算されてから伝播される。
        for &ptr in topo_order.iter().rev() {
            let tensor = unsafe { &*ptr };
            let grad = tensor.autograd.as_ref()
                .and_then(|m| m.grad.as_ref())
                .map(|g| g.shallow_clone());

            if let Some(grad_output) = grad {
                let propagation = tensor.autograd.as_ref().and_then(|m| {
                    m.grad_fn.as_ref().map(|gf| {
                        let grads = gf.backward(&grad_output);
                        let inputs = gf.inputs();
                        (grads, inputs)
                    })
                });

                if let Some((grads, inputs)) = propagation {
                    for (input_ptr, grad) in inputs.into_iter().zip(grads.into_iter()) {
                        let input = unsafe { &mut *input_ptr };
                        if input.requires_grad() {
                            if let Some(ref mut meta) = input.autograd {
                                if let Some(ref mut existing) = meta.grad {
                                    *existing = existing.add_impl(&grad);
                                } else {
                                    meta.grad = Some(grad);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// backward 用のトポロジカルソート（DFS）
    fn build_topo(
        ptr: *mut CpuTensor,
        visited: &mut std::collections::HashSet<usize>,
        topo: &mut Vec<*mut CpuTensor>,
    ) {
        let key = ptr as usize;
        if visited.contains(&key) {
            return;
        }
        visited.insert(key);
        let tensor = unsafe { &*ptr };
        if let Some(ref meta) = tensor.autograd {
            if let Some(ref gf) = meta.grad_fn {
                for input in gf.inputs() {
                    let input_tensor = unsafe { &*input };
                    if input_tensor.requires_grad() {
                        Self::build_topo(input, visited, topo);
                    }
                }
            }
        }
        topo.push(ptr);
    }


    pub fn detach(&self) -> CpuTensor {
        self.shallow_clone()
    }

    // ========== 演算実装 (_impl メソッド) ==========

    /// ブロードキャスト用ストライド計算
    /// src_shape を out_shape にブロードキャストする際のストライドを返す。
    /// 次元サイズが1の場合はストライド0（同じ要素を繰り返す）。
    fn broadcast_strides(src_shape: &[usize], out_shape: &[usize]) -> Vec<usize> {
        let out_ndim = out_shape.len();
        let src_ndim = src_shape.len();
        let mut strides = vec![0usize; out_ndim];
        // src_shape を右詰めで out_shape に合わせる
        let offset = out_ndim - src_ndim;
        // まず src_shape のストライドを計算
        let mut src_strides = vec![1usize; src_ndim];
        for i in (0..src_ndim.saturating_sub(1)).rev() {
            src_strides[i] = src_strides[i + 1] * src_shape[i + 1];
        }
        for i in 0..out_ndim {
            if i < offset {
                strides[i] = 0; // src にはこの次元がない → ブロードキャスト
            } else {
                let si = i - offset;
                if src_shape[si] == 1 {
                    strides[i] = 0; // サイズ1 → ブロードキャスト
                } else {
                    strides[i] = src_strides[si];
                }
            }
        }
        strides
    }

    fn elementwise_binop(&self, other: &Self, op: impl Fn(f32, f32) -> f32) -> Self {
        // NumPy互換ブロードキャスト
        let a_shape = &self.shape;
        let b_shape = &other.shape;
        let a = &self.data_f32;
        let b = &other.data_f32;

        // ブロードキャスト結果の shape を計算
        let out_ndim = a_shape.len().max(b_shape.len());
        let mut out_shape = vec![0usize; out_ndim];
        for i in 0..out_ndim {
            let da = if i < out_ndim - a_shape.len() { 1 } else { a_shape[i - (out_ndim - a_shape.len())] };
            let db = if i < out_ndim - b_shape.len() { 1 } else { b_shape[i - (out_ndim - b_shape.len())] };
            out_shape[i] = da.max(db);
        }
        let out_len: usize = out_shape.iter().product();

        // a と b のストライドを計算（ブロードキャスト用）
        let a_strides = Self::broadcast_strides(a_shape, &out_shape);
        let b_strides = Self::broadcast_strides(b_shape, &out_shape);

        let data: Vec<f32> = (0..out_len).map(|flat_idx| {
            let mut a_idx = 0usize;
            let mut b_idx = 0usize;
            let mut remaining = flat_idx;
            for d in 0..out_ndim {
                let stride = out_shape[d+1..].iter().product::<usize>().max(1);
                let coord = remaining / stride;
                remaining %= stride;
                a_idx += coord * a_strides[d];
                b_idx += coord * b_strides[d];
            }
            op(a[a_idx], b[b_idx])
        }).collect();

        CpuTensor {
            data_f32: data,
            data_i64: None,
            shape: out_shape,
            dtype: self.dtype,
            autograd: None,
        }
    }

    pub fn add_impl(&self, other: &Self) -> Self {
        self.elementwise_binop(other, |a, b| a + b)
    }

    pub fn sub_impl(&self, other: &Self) -> Self {
        self.elementwise_binop(other, |a, b| a - b)
    }

    pub fn mul_impl(&self, other: &Self) -> Self {
        self.elementwise_binop(other, |a, b| a * b)
    }

    pub fn div_impl(&self, other: &Self) -> Self {
        self.elementwise_binop(other, |a, b| a / b)
    }

    pub fn pow_impl(&self, other: &Self) -> Self {
        self.elementwise_binop(other, |a, b| a.powf(b))
    }

    pub fn rem_impl(&self, other: &Self) -> Self {
        self.elementwise_binop(other, |a, b| a % b)
    }

    // ========== スカラー演算 ==========

    pub fn add_scalar_impl(&self, scalar: f32) -> Self {
        let data: Vec<f32> = self.data_f32.iter().map(|x| x + scalar).collect();
        CpuTensor { data_f32: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None }
    }

    pub fn sub_scalar_impl(&self, scalar: f32) -> Self {
        let data: Vec<f32> = self.data_f32.iter().map(|x| x - scalar).collect();
        CpuTensor { data_f32: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None }
    }

    pub fn mul_scalar_impl(&self, scalar: f32) -> Self {
        let data: Vec<f32> = self.data_f32.iter().map(|x| x * scalar).collect();
        CpuTensor { data_f32: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None }
    }

    pub fn div_scalar_impl(&self, scalar: f32) -> Self {
        let data: Vec<f32> = self.data_f32.iter().map(|x| x / scalar).collect();
        CpuTensor { data_f32: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None }
    }

    pub fn clamp_impl(&self, min: f32, max: f32) -> Self {
        let data: Vec<f32> = self.data_f32.iter().map(|x| x.clamp(min, max)).collect();
        CpuTensor { data_f32: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None }
    }

    // ========== 単項演算 ==========

    pub fn neg_impl(&self) -> Self {
        let data: Vec<f32> = self.data_f32.iter().map(|x| -x).collect();
        CpuTensor { data_f32: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None }
    }

    pub fn abs_impl(&self) -> Self {
        let data: Vec<f32> = self.data_f32.iter().map(|x| x.abs()).collect();
        CpuTensor { data_f32: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None }
    }

    pub fn exp_impl(&self) -> Self {
        let data: Vec<f32> = self.data_f32.iter().map(|x| x.exp()).collect();
        CpuTensor { data_f32: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None }
    }

    pub fn log_impl(&self) -> Self {
        let data: Vec<f32> = self.data_f32.iter().map(|x| x.ln()).collect();
        CpuTensor { data_f32: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None }
    }

    pub fn sqrt_impl(&self) -> Self {
        let data: Vec<f32> = self.data_f32.iter().map(|x| x.sqrt()).collect();
        CpuTensor { data_f32: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None }
    }

    pub fn sin_impl(&self) -> Self {
        let data: Vec<f32> = self.data_f32.iter().map(|x| x.sin()).collect();
        CpuTensor { data_f32: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None }
    }

    pub fn cos_impl(&self) -> Self {
        let data: Vec<f32> = self.data_f32.iter().map(|x| x.cos()).collect();
        CpuTensor { data_f32: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None }
    }

    pub fn tan_impl(&self) -> Self {
        let data: Vec<f32> = self.data_f32.iter().map(|x| x.tan()).collect();
        CpuTensor { data_f32: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None }
    }

    pub fn tanh_impl(&self) -> Self {
        let data: Vec<f32> = self.data_f32.iter().map(|x| x.tanh()).collect();
        CpuTensor { data_f32: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None }
    }

    pub fn sigmoid_impl(&self) -> Self {
        let data: Vec<f32> = self.data_f32.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
        CpuTensor { data_f32: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None }
    }

    pub fn relu_impl(&self) -> Self {
        let data: Vec<f32> = self.data_f32.iter().map(|x| x.max(0.0)).collect();
        CpuTensor { data_f32: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None }
    }

    pub fn gelu_impl(&self) -> Self {
        let data: Vec<f32> = self.data_f32.iter().map(|x| {
            0.5 * x * (1.0 + (std::f32::consts::FRAC_2_SQRT_PI * (x + 0.044715 * x.powi(3))).tanh())
        }).collect();
        CpuTensor { data_f32: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None }
    }

    // ========== Reduce ==========

    pub fn sumall_impl(&self) -> f32 {
        self.data_f32.iter().sum()
    }

    pub fn mean_all_impl(&self) -> f32 {
        let n = self.data_f32.len() as f32;
        if n > 0.0 { self.sumall_impl() / n } else { 0.0 }
    }

    pub fn sum_impl(&self, axis: i32) -> Self {
        let ndim = self.shape.len();
        let axis = if axis < 0 { (ndim as i32 + axis) as usize } else { axis as usize };
        let outer: usize = self.shape[..axis].iter().product();
        let axis_size = self.shape[axis];
        let inner: usize = self.shape[axis + 1..].iter().product();
        let mut result = vec![0.0f32; outer * inner];
        for i in 0..outer {
            for j in 0..axis_size {
                for k in 0..inner {
                    result[i * inner + k] += self.data_f32[i * axis_size * inner + j * inner + k];
                }
            }
        }
        let mut new_shape: Vec<usize> = self.shape.clone();
        new_shape.remove(axis);
        if new_shape.is_empty() { new_shape.push(1); }
        CpuTensor { data_f32: result, data_i64: None, shape: new_shape, dtype: self.dtype, autograd: None }
    }

    pub fn max_impl(&self, axis: i32) -> Self {
        let ndim = self.shape.len();
        let axis = if axis < 0 { (ndim as i32 + axis) as usize } else { axis as usize };
        let outer: usize = self.shape[..axis].iter().product();
        let axis_size = self.shape[axis];
        let inner: usize = self.shape[axis + 1..].iter().product();
        let mut result = vec![f32::NEG_INFINITY; outer * inner];
        for i in 0..outer {
            for j in 0..axis_size {
                for k in 0..inner {
                    let idx = i * inner + k;
                    let val = self.data_f32[i * axis_size * inner + j * inner + k];
                    if val > result[idx] { result[idx] = val; }
                }
            }
        }
        let mut new_shape: Vec<usize> = self.shape.clone();
        new_shape.remove(axis);
        if new_shape.is_empty() { new_shape.push(1); }
        CpuTensor { data_f32: result, data_i64: None, shape: new_shape, dtype: self.dtype, autograd: None }
    }

    pub fn min_impl(&self, axis: i32) -> Self {
        let ndim = self.shape.len();
        let axis = if axis < 0 { (ndim as i32 + axis) as usize } else { axis as usize };
        let outer: usize = self.shape[..axis].iter().product();
        let axis_size = self.shape[axis];
        let inner: usize = self.shape[axis + 1..].iter().product();
        let mut result = vec![f32::INFINITY; outer * inner];
        for i in 0..outer {
            for j in 0..axis_size {
                for k in 0..inner {
                    let idx = i * inner + k;
                    let val = self.data_f32[i * axis_size * inner + j * inner + k];
                    if val < result[idx] { result[idx] = val; }
                }
            }
        }
        let mut new_shape: Vec<usize> = self.shape.clone();
        new_shape.remove(axis);
        if new_shape.is_empty() { new_shape.push(1); }
        CpuTensor { data_f32: result, data_i64: None, shape: new_shape, dtype: self.dtype, autograd: None }
    }

    pub fn argmax_impl(&self, axis: i32) -> Self {
        let ndim = self.shape.len();
        let axis = if axis < 0 { (ndim as i32 + axis) as usize } else { axis as usize };
        let outer: usize = self.shape[..axis].iter().product();
        let axis_size = self.shape[axis];
        let inner: usize = self.shape[axis + 1..].iter().product();
        let mut result = vec![0.0f32; outer * inner];
        let mut max_vals = vec![f32::NEG_INFINITY; outer * inner];
        for i in 0..outer {
            for j in 0..axis_size {
                for k in 0..inner {
                    let idx = i * inner + k;
                    let val = self.data_f32[i * axis_size * inner + j * inner + k];
                    if val > max_vals[idx] {
                        max_vals[idx] = val;
                        result[idx] = j as f32;
                    }
                }
            }
        }
        let mut new_shape: Vec<usize> = self.shape.clone();
        new_shape.remove(axis);
        if new_shape.is_empty() { new_shape.push(1); }
        CpuTensor { data_f32: result, data_i64: None, shape: new_shape, dtype: self.dtype, autograd: None }
    }

    pub fn argmax_all_impl(&self) -> usize {
        self.data_f32.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    pub fn argmin_impl(&self, axis: i32) -> Self {
        let ndim = self.shape.len();
        let axis = if axis < 0 { (ndim as i32 + axis) as usize } else { axis as usize };
        let outer: usize = self.shape[..axis].iter().product();
        let axis_size = self.shape[axis];
        let inner: usize = self.shape[axis + 1..].iter().product();
        let mut result = vec![0.0f32; outer * inner];
        let mut min_vals = vec![f32::INFINITY; outer * inner];
        for i in 0..outer {
            for j in 0..axis_size {
                for k in 0..inner {
                    let idx = i * inner + k;
                    let val = self.data_f32[i * axis_size * inner + j * inner + k];
                    if val < min_vals[idx] {
                        min_vals[idx] = val;
                        result[idx] = j as f32;
                    }
                }
            }
        }
        let mut new_shape: Vec<usize> = self.shape.clone();
        new_shape.remove(axis);
        if new_shape.is_empty() { new_shape.push(1); }
        CpuTensor { data_f32: result, data_i64: None, shape: new_shape, dtype: self.dtype, autograd: None }
    }

    pub fn mean_impl(&self, axis: i32) -> Self {
        let ndim = self.shape.len();
        let ax = if axis < 0 { (ndim as i32 + axis) as usize } else { axis as usize };
        let axis_size = self.shape[ax] as f32;
        let s = self.sum_impl(axis);
        s.div_scalar_impl(axis_size)
    }

    // ========== 形状操作 ==========

    pub fn reshape_impl(&self, shape: &[usize]) -> Self {
        CpuTensor {
            data_f32: self.data_f32.clone(),
            data_i64: self.data_i64.clone(),
            shape: shape.to_vec(),
            dtype: self.dtype,
            autograd: None,
        }
    }

    pub fn transpose_impl(&self, dim0: usize, dim1: usize) -> Self {
        if self.shape.len() < 2 {
            return self.shallow_clone();
        }
        let mut new_shape = self.shape.clone();
        new_shape.swap(dim0, dim1);
        let ndim = self.shape.len();

        let total = self.elem_count();
        let mut result = vec![0.0f32; total];

        // 汎用転置
        let mut old_indices = vec![0usize; ndim];
        for flat_idx in 0..total {
            // flat_idx → old_indices
            let mut rem = flat_idx;
            for d in (0..ndim).rev() {
                old_indices[d] = rem % self.shape[d];
                rem /= self.shape[d];
            }
            // old_indices → new_indices (swap dim0, dim1)
            let mut new_indices = old_indices.clone();
            new_indices.swap(dim0, dim1);
            // new_indices → new_flat_idx
            let mut new_flat_idx = 0;
            let mut stride = 1;
            for d in (0..ndim).rev() {
                new_flat_idx += new_indices[d] * stride;
                stride *= new_shape[d];
            }
            result[new_flat_idx] = self.data_f32[flat_idx];
        }

        CpuTensor { data_f32: result, data_i64: None, shape: new_shape, dtype: self.dtype, autograd: None }
    }

    pub fn squeeze_impl(&self, dim: usize) -> Self {
        let mut new_shape = self.shape.clone();
        if dim < new_shape.len() && new_shape[dim] == 1 {
            new_shape.remove(dim);
        }
        if new_shape.is_empty() { new_shape.push(1); }
        self.reshape_impl(&new_shape)
    }

    pub fn unsqueeze_impl(&self, dim: usize) -> Self {
        let mut new_shape = self.shape.clone();
        new_shape.insert(dim, 1);
        self.reshape_impl(&new_shape)
    }

    pub fn broadcast_to_impl(&self, shape: &[usize]) -> Self {
        let total: usize = shape.iter().product();
        let mut result = vec![0.0f32; total];
        for i in 0..total {
            result[i] = self.data_f32[i % self.data_f32.len()];
        }
        CpuTensor { data_f32: result, data_i64: None, shape: shape.to_vec(), dtype: self.dtype, autograd: None }
    }

    pub fn narrow_impl(&self, axis: usize, start: usize, len: usize) -> Self {
        self.slice_impl(axis, start, len)
    }

    pub fn slice_impl(&self, axis: usize, start: usize, len: usize) -> Self {
        let outer: usize = self.shape[..axis].iter().product();
        let axis_size = self.shape[axis];
        let inner: usize = self.shape[axis + 1..].iter().product();
        let mut result = Vec::with_capacity(outer * len * inner);
        for i in 0..outer {
            for j in start..start + len {
                for k in 0..inner {
                    result.push(self.data_f32[i * axis_size * inner + j * inner + k]);
                }
            }
        }
        let mut new_shape = self.shape.clone();
        new_shape[axis] = len;
        CpuTensor { data_f32: result, data_i64: None, shape: new_shape, dtype: self.dtype, autograd: None }
    }

    pub fn contiguous_impl(&self) -> Self {
        self.shallow_clone()
    }

    pub fn cat_impl(tensors: &[&Self], axis: usize) -> Self {
        if tensors.is_empty() {
            return CpuTensor::zeros(&[0], DType::F32);
        }
        let _ndim = tensors[0].shape.len();
        let outer: usize = tensors[0].shape[..axis].iter().product();
        let inner: usize = tensors[0].shape[axis + 1..].iter().product();
        let total_axis: usize = tensors.iter().map(|t| t.shape[axis]).sum();
        let mut result = Vec::with_capacity(outer * total_axis * inner);
        for i in 0..outer {
            for t in tensors {
                let axis_size = t.shape[axis];
                for j in 0..axis_size {
                    for k in 0..inner {
                        result.push(t.data_f32[i * axis_size * inner + j * inner + k]);
                    }
                }
            }
        }
        let mut new_shape = tensors[0].shape.clone();
        new_shape[axis] = total_axis;
        CpuTensor { data_f32: result, data_i64: None, shape: new_shape, dtype: tensors[0].dtype, autograd: None }
    }

    // ========== 活性化・特殊演算 ==========

    pub fn softmax_impl(&self, axis: i32) -> Self {
        let ndim = self.shape.len();
        let axis = if axis < 0 { (ndim as i32 + axis) as usize } else { axis as usize };
        let outer: usize = self.shape[..axis].iter().product();
        let axis_size = self.shape[axis];
        let inner: usize = self.shape[axis + 1..].iter().product();
        let mut result = self.data_f32.clone();
        for i in 0..outer {
            for k in 0..inner {
                let mut max_val = f32::NEG_INFINITY;
                for j in 0..axis_size {
                    let idx = i * axis_size * inner + j * inner + k;
                    if result[idx] > max_val { max_val = result[idx]; }
                }
                let mut sum = 0.0f32;
                for j in 0..axis_size {
                    let idx = i * axis_size * inner + j * inner + k;
                    result[idx] = (result[idx] - max_val).exp();
                    sum += result[idx];
                }
                for j in 0..axis_size {
                    let idx = i * axis_size * inner + j * inner + k;
                    result[idx] /= sum;
                }
            }
        }
        CpuTensor { data_f32: result, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None }
    }

    pub fn embedding_impl(&self, indices: &Self) -> Self {
        let _vocab_size = self.shape[0];
        let embed_dim = if self.shape.len() > 1 { self.shape[1] } else { 1 };
        let idx_data: Vec<i64> = indices.to_vec::<f32>().iter().map(|x| *x as i64).collect();
        let mut result = Vec::with_capacity(idx_data.len() * embed_dim);
        for &idx in &idx_data {
            let start = (idx as usize) * embed_dim;
            let end = start + embed_dim;
            result.extend_from_slice(&self.data_f32[start..end.min(self.data_f32.len())]);
        }
        let mut new_shape = indices.shape.to_vec();
        new_shape.push(embed_dim);
        CpuTensor { data_f32: result, data_i64: None, shape: new_shape, dtype: self.dtype, autograd: None }
    }

    pub fn tril_impl(&self, diagonal: i32) -> Self {
        let rows = self.shape[self.shape.len() - 2];
        let cols = self.shape[self.shape.len() - 1];
        let batch: usize = self.shape[..self.shape.len() - 2].iter().product();
        let mut result = self.data_f32.clone();
        for b in 0..batch.max(1) {
            for i in 0..rows {
                for j in 0..cols {
                    if (j as i32) > (i as i32) + diagonal {
                        result[b * rows * cols + i * cols + j] = 0.0;
                    }
                }
            }
        }
        CpuTensor { data_f32: result, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None }
    }

    pub fn cross_entropy_impl(&self, target: &Self) -> Self {
        let batch = self.shape[0];
        let classes = self.shape[1];
        let mut loss = 0.0f32;
        for b in 0..batch {
            let target_idx = target.data_f32[b] as usize;
            let mut max_val = f32::NEG_INFINITY;
            for c in 0..classes {
                let val = self.data_f32[b * classes + c];
                if val > max_val { max_val = val; }
            }
            let mut sum_exp = 0.0f32;
            for c in 0..classes {
                sum_exp += (self.data_f32[b * classes + c] - max_val).exp();
            }
            let log_softmax = self.data_f32[b * classes + target_idx] - max_val - sum_exp.ln();
            loss -= log_softmax;
        }
        loss /= batch as f32;
        CpuTensor { data_f32: vec![loss], data_i64: None, shape: vec![1], dtype: self.dtype, autograd: None }
    }

    pub fn repeat_interleave_impl(&self, repeats: usize, axis: usize) -> Self {
        let outer: usize = self.shape[..axis].iter().product();
        let axis_size = self.shape[axis];
        let inner: usize = self.shape[axis + 1..].iter().product();
        let mut result = Vec::with_capacity(outer * axis_size * repeats * inner);
        for i in 0..outer {
            for j in 0..axis_size {
                for _ in 0..repeats {
                    for k in 0..inner {
                        result.push(self.data_f32[i * axis_size * inner + j * inner + k]);
                    }
                }
            }
        }
        let mut new_shape = self.shape.clone();
        new_shape[axis] *= repeats;
        CpuTensor { data_f32: result, data_i64: None, shape: new_shape, dtype: self.dtype, autograd: None }
    }

    pub fn index_select_impl(&self, axis: usize, indices: &Self) -> Self {
        let idx_list: Vec<usize> = indices.data_f32.iter().map(|x| *x as usize).collect();
        let outer: usize = self.shape[..axis].iter().product();
        let axis_size = self.shape[axis];
        let inner: usize = self.shape[axis + 1..].iter().product();
        let mut result = Vec::with_capacity(outer * idx_list.len() * inner);
        for i in 0..outer {
            for &idx in &idx_list {
                for k in 0..inner {
                    result.push(self.data_f32[i * axis_size * inner + idx * inner + k]);
                }
            }
        }
        let mut new_shape = self.shape.clone();
        new_shape[axis] = idx_list.len();
        CpuTensor { data_f32: result, data_i64: None, shape: new_shape, dtype: self.dtype, autograd: None }
    }

    pub fn where_cond_impl(condition: &Self, x: &Self, y: &Self) -> Self {
        let data: Vec<f32> = condition.data_f32.iter()
            .zip(x.data_f32.iter().zip(y.data_f32.iter()))
            .map(|(c, (xv, yv))| if *c != 0.0 { *xv } else { *yv })
            .collect();
        CpuTensor { data_f32: data, data_i64: None, shape: x.shape.clone(), dtype: x.dtype, autograd: None }
    }

    // ========== Matmul ==========

    pub fn matmul_impl(&self, other: &Self) -> Self {
        let a_shape = &self.shape;
        let b_shape = &other.shape;
        // 2D matmul
        let (m, k) = if a_shape.len() >= 2 {
            (a_shape[a_shape.len() - 2], a_shape[a_shape.len() - 1])
        } else {
            (1, a_shape[0])
        };
        let n = if b_shape.len() >= 2 {
            b_shape[b_shape.len() - 1]
        } else {
            1
        };
        let batch: usize = a_shape[..a_shape.len().saturating_sub(2)].iter().product::<usize>().max(1);
        let mut result = vec![0.0f32; batch * m * n];
        for b in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for p in 0..k {
                        sum += self.data_f32[b * m * k + i * k + p] * other.data_f32[b * k * n + p * n + j];
                    }
                    result[b * m * n + i * n + j] = sum;
                }
            }
        }
        let mut new_shape = a_shape[..a_shape.len().saturating_sub(2)].to_vec();
        new_shape.push(m);
        new_shape.push(n);
        CpuTensor { data_f32: result, data_i64: None, shape: new_shape, dtype: self.dtype, autograd: None }
    }

    // ========== 比較演算 ==========

    pub fn eq_impl(&self, other: &Self) -> Self {
        self.elementwise_binop(other, |a, b| if (a - b).abs() < 1e-6 { 1.0 } else { 0.0 })
    }

    pub fn neq_impl(&self, other: &Self) -> Self {
        self.elementwise_binop(other, |a, b| if (a - b).abs() >= 1e-6 { 1.0 } else { 0.0 })
    }

    pub fn gt_impl(&self, other: &Self) -> Self {
        self.elementwise_binop(other, |a, b| if a > b { 1.0 } else { 0.0 })
    }

    pub fn lt_impl(&self, other: &Self) -> Self {
        self.elementwise_binop(other, |a, b| if a < b { 1.0 } else { 0.0 })
    }

    pub fn ge_impl(&self, other: &Self) -> Self {
        self.elementwise_binop(other, |a, b| if a >= b { 1.0 } else { 0.0 })
    }

    pub fn le_impl(&self, other: &Self) -> Self {
        self.elementwise_binop(other, |a, b| if a <= b { 1.0 } else { 0.0 })
    }

    // ========== 深層学習演算（簡易版） ==========

    pub fn conv2d_impl(&self, weight: &Self, stride: (usize, usize), padding: (usize, usize)) -> Self {
        // self: [batch, in_ch, h, w], weight: [out_ch, in_ch, kh, kw]
        let batch = self.shape[0];
        let in_ch = self.shape[1];
        let h = self.shape[2];
        let w = self.shape[3];
        let out_ch = weight.shape[0];
        let kh = weight.shape[2];
        let kw = weight.shape[3];
        let (sh, sw) = stride;
        let (ph, pw) = padding;
        let out_h = (h + 2 * ph - kh) / sh + 1;
        let out_w = (w + 2 * pw - kw) / sw + 1;

        let mut result = vec![0.0f32; batch * out_ch * out_h * out_w];

        for b in 0..batch {
            for oc in 0..out_ch {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0f32;
                        for ic in 0..in_ch {
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    let ih = oh * sh + ki;
                                    let iw = ow * sw + kj;
                                    let ih = ih as isize - ph as isize;
                                    let iw = iw as isize - pw as isize;
                                    if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                        let in_idx = b * in_ch * h * w + ic * h * w + ih as usize * w + iw as usize;
                                        let w_idx = oc * in_ch * kh * kw + ic * kh * kw + ki * kw + kj;
                                        sum += self.data_f32[in_idx] * weight.data_f32[w_idx];
                                    }
                                }
                            }
                        }
                        let out_idx = b * out_ch * out_h * out_w + oc * out_h * out_w + oh * out_w + ow;
                        result[out_idx] = sum;
                    }
                }
            }
        }

        CpuTensor { data_f32: result, data_i64: None, shape: vec![batch, out_ch, out_h, out_w], dtype: self.dtype, autograd: None }
    }

    pub fn batch_norm_impl(&self, gamma: &Self, beta: &Self, running_mean: &Self, running_var: &Self, eps: f32) -> Self {
        // self: [batch, channels, ...] or [batch, features]
        // running_mean/running_var 使用時はチャネルごと正規化
        if self.shape.len() >= 2 {
            let batch = self.shape[0];
            let channels = self.shape[1];
            let spatial: usize = self.shape[2..].iter().product();
            let spatial = if spatial == 0 { 1 } else { spatial };

            let use_running = running_mean.elem_count() == channels && running_var.elem_count() == channels;

            let mut result = self.data_f32.clone();
            for c in 0..channels {
                let (mean, var) = if use_running {
                    (running_mean.data_f32[c], running_var.data_f32[c])
                } else {
                    // バッチ統計量を計算
                    let n = (batch * spatial) as f32;
                    let mut m = 0.0f32;
                    for b in 0..batch {
                        for s in 0..spatial {
                            m += self.data_f32[b * channels * spatial + c * spatial + s];
                        }
                    }
                    m /= n;
                    let mut v = 0.0f32;
                    for b in 0..batch {
                        for s in 0..spatial {
                            let d = self.data_f32[b * channels * spatial + c * spatial + s] - m;
                            v += d * d;
                        }
                    }
                    v /= n;
                    (m, v)
                };

                let inv_std = 1.0 / (var + eps).sqrt();
                let g = if c < gamma.data_f32.len() { gamma.data_f32[c] } else { 1.0 };
                let bi = if c < beta.data_f32.len() { beta.data_f32[c] } else { 0.0 };

                for b in 0..batch {
                    for s in 0..spatial {
                        let idx = b * channels * spatial + c * spatial + s;
                        result[idx] = (result[idx] - mean) * inv_std * g + bi;
                    }
                }
            }
            CpuTensor { data_f32: result, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None }
        } else {
            // 1D フォールバック
            let mean = self.mean_all_impl();
            let data: Vec<f32> = self.data_f32.iter().map(|x| x - mean).collect();
            let var: f32 = data.iter().map(|x| x * x).sum::<f32>() / data.len() as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            let result: Vec<f32> = data.iter().enumerate().map(|(i, x)| {
                let g = if i < gamma.data_f32.len() { gamma.data_f32[i % gamma.data_f32.len()] } else { 1.0 };
                let b = if i < beta.data_f32.len() { beta.data_f32[i % beta.data_f32.len()] } else { 0.0 };
                x * inv_std * g + b
            }).collect();
            CpuTensor { data_f32: result, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None }
        }
    }

    pub fn layer_norm_impl(&self, gamma: &Self, beta: &Self, eps: f32) -> Self {
        // 最終次元で正規化 (各行独立)
        let ndim = self.shape.len();
        if ndim < 2 {
            // 1D: 全体を正規化
            let mean = self.mean_all_impl();
            let var: f32 = self.data_f32.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / self.data_f32.len() as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            let result: Vec<f32> = self.data_f32.iter().enumerate().map(|(i, x)| {
                let g = gamma.data_f32.get(i).copied().unwrap_or(1.0);
                let b = beta.data_f32.get(i).copied().unwrap_or(0.0);
                (x - mean) * inv_std * g + b
            }).collect();
            return CpuTensor { data_f32: result, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None };
        }

        let last_dim = self.shape[ndim - 1];
        let outer: usize = self.shape[..ndim - 1].iter().product();
        let mut result = self.data_f32.clone();

        for i in 0..outer {
            let start = i * last_dim;
            let end = start + last_dim;
            let slice = &self.data_f32[start..end];

            let mean: f32 = slice.iter().sum::<f32>() / last_dim as f32;
            let var: f32 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / last_dim as f32;
            let inv_std = 1.0 / (var + eps).sqrt();

            for j in 0..last_dim {
                let g = gamma.data_f32.get(j).copied().unwrap_or(1.0);
                let b = beta.data_f32.get(j).copied().unwrap_or(0.0);
                result[start + j] = (slice[j] - mean) * inv_std * g + b;
            }
        }

        CpuTensor { data_f32: result, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None }
    }

    pub fn max_pool2d_impl(&self, kernel_size: (usize, usize), stride: (usize, usize)) -> Self {
        // self: [batch, channels, h, w]
        let batch = self.shape[0];
        let channels = self.shape[1];
        let h = self.shape[2];
        let w = self.shape[3];
        let (kh, kw) = kernel_size;
        let (sh, sw) = stride;
        let out_h = (h - kh) / sh + 1;
        let out_w = (w - kw) / sw + 1;

        let mut result = vec![f32::NEG_INFINITY; batch * channels * out_h * out_w];

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut max_val = f32::NEG_INFINITY;
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let ih = oh * sh + ki;
                                let iw = ow * sw + kj;
                                let idx = b * channels * h * w + c * h * w + ih * w + iw;
                                if self.data_f32[idx] > max_val {
                                    max_val = self.data_f32[idx];
                                }
                            }
                        }
                        let out_idx = b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                        result[out_idx] = max_val;
                    }
                }
            }
        }

        CpuTensor { data_f32: result, data_i64: None, shape: vec![batch, channels, out_h, out_w], dtype: self.dtype, autograd: None }
    }

    pub fn avg_pool2d_impl(&self, kernel_size: (usize, usize), stride: (usize, usize)) -> Self {
        // self: [batch, channels, h, w]
        let batch = self.shape[0];
        let channels = self.shape[1];
        let h = self.shape[2];
        let w = self.shape[3];
        let (kh, kw) = kernel_size;
        let (sh, sw) = stride;
        let out_h = (h - kh) / sh + 1;
        let out_w = (w - kw) / sw + 1;
        let pool_size = (kh * kw) as f32;

        let mut result = vec![0.0f32; batch * channels * out_h * out_w];

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0f32;
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let ih = oh * sh + ki;
                                let iw = ow * sw + kj;
                                let idx = b * channels * h * w + c * h * w + ih * w + iw;
                                sum += self.data_f32[idx];
                            }
                        }
                        let out_idx = b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                        result[out_idx] = sum / pool_size;
                    }
                }
            }
        }

        CpuTensor { data_f32: result, data_i64: None, shape: vec![batch, channels, out_h, out_w], dtype: self.dtype, autograd: None }
    }

    pub fn dropout_impl(&self, _p: f32, _training: bool) -> Self {
        self.shallow_clone()
    }
}

// GpuTensor トレイトが Send + Sync を要求
unsafe impl Send for CpuTensor {}
unsafe impl Sync for CpuTensor {}
