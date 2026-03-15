//! CPU テンソル

use crate::autograd::GradFn;
use crate::scalar::TensorScalar;
use crate::DType;
use std::sync::Arc;
use std::cell::UnsafeCell;

/// テンソルへの共有所有権付きハンドル。
pub type TensorRef<T> = Arc<UnsafeCell<CpuTensor<T>>>;

/// 生ポインタから TensorRef を取得する (RC+1)。
pub unsafe fn tensor_ref_from_ptr<T: TensorScalar>(ptr: *mut CpuTensor<T>) -> TensorRef<T> {
    let arc: TensorRef<T> = Arc::from_raw(ptr as *const UnsafeCell<CpuTensor<T>>);
    let cloned = arc.clone();
    let _ = Arc::into_raw(arc);
    cloned
}

/// TensorRef から内部の CpuTensor への不変参照を取得する。
pub unsafe fn tensor_ref_get<T: TensorScalar>(r: &TensorRef<T>) -> &CpuTensor<T> {
    &*r.get()
}

/// TensorRef から内部の CpuTensor への可変参照を取得する。
pub unsafe fn tensor_ref_get_mut<T: TensorScalar>(r: &TensorRef<T>) -> &mut CpuTensor<T> {
    &mut *r.get()
}

/// Autograd メタデータ
pub struct AutogradMeta<T: TensorScalar> {
    pub grad: Option<CpuTensor<T>>,
    pub grad_fn: Option<Box<dyn GradFn<T>>>,
    pub requires_grad: bool,
}

/// CPU テンソル（ジェネリック: Vec<T> ベース）
pub struct CpuTensor<T: TensorScalar> {
    pub(crate) data: Vec<T>,
    pub(crate) data_i64: Option<Vec<i64>>,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: DType,
    pub autograd: Option<Box<AutogradMeta<T>>>,
}

impl<T: TensorScalar> Drop for CpuTensor<T> {
    fn drop(&mut self) {
        let data_bytes = self.data.capacity() * T::size_in_bytes();
        if data_bytes > 0 {
            crate::memory::track_free(data_bytes);
        }
        if let Some(ref v) = self.data_i64 {
            let i64_bytes = v.capacity() * std::mem::size_of::<i64>();
            if i64_bytes > 0 {
                crate::memory::track_free(i64_bytes);
            }
        }
    }
}

impl<T: TensorScalar> Clone for CpuTensor<T> {
    fn clone(&self) -> Self {
        self.clone_data()
    }
}

impl<T: TensorScalar> CpuTensor<T> {
    // ========== コンストラクタ ==========




    pub fn zeros(shape: &[usize], dtype: DType) -> Self {
        let count: usize = shape.iter().product();
        CpuTensor {
            data: vec![T::zero(); count],
            data_i64: None,
            shape: shape.to_vec(),
            dtype,
            autograd: None,
        }
    }
    
    pub fn ones(shape: &[usize], dtype: DType) -> Self {
        let count: usize = shape.iter().product();
        CpuTensor {
            data: vec![T::one(); count],
            data_i64: None,
            shape: shape.to_vec(),
            dtype,
            autograd: None,
        }
    }

    pub fn from_slice(data: &[T], shape: &[usize], dtype: DType) -> Self {
        CpuTensor {
            data: data.to_vec(),
            data_i64: None,
            shape: shape.to_vec(),
            dtype,
            autograd: None,
        }
    }

    pub fn from_slice_i64_data(data: &[i64], shape: &[usize], dtype: DType) -> Self {
        let t_data: Vec<T> = data.iter().map(|x| T::from_f64(*x as f64)).collect();
        CpuTensor {
            data: t_data,
            data_i64: Some(data.to_vec()),
            shape: shape.to_vec(),
            dtype,
            autograd: None,
        }
    }

    pub fn randn(shape: &[usize], dtype: DType) -> Self {
        let count: usize = shape.iter().product();
        let mut rng = rand::thread_rng();
        let data: Vec<T> = (0..count)
            .map(|_| {
                let u1 = T::gen_uniform(&mut rng);
                let u2 = T::gen_uniform(&mut rng);
                let two = T::from_f64(2.0);
                (T::zero() - two * u1.ln()).sqrt() * (two * T::pi() * u2).cos()
            })
            .collect();
        CpuTensor {
            data,
            data_i64: None,
            shape: shape.to_vec(),
            dtype,
            autograd: None,
        }
    }

    pub fn arange(start: i64, end: i64, dtype: DType) -> Self {
        let data: Vec<T> = (start..end).map(|x| T::from_f64(x as f64)).collect();
        let len = data.len();
        CpuTensor {
            data,
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

    pub fn to_vec_t(&self) -> Vec<T> {
        self.data.clone()
    }

    /// 互換用: 型消去された to_vec（旧コード用）
    pub fn to_vec<U: Copy + Default + 'static>(&self) -> Vec<U> {
        if std::any::TypeId::of::<U>() == std::any::TypeId::of::<T>() {
            let src = &self.data;
            let mut dst = vec![U::default(); src.len()];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr() as *const U,
                    dst.as_mut_ptr(),
                    src.len(),
                );
            }
            dst
        } else if std::any::TypeId::of::<U>() == std::any::TypeId::of::<i64>() {
            if let Some(ref i64_data) = self.data_i64 {
                let mut dst = vec![U::default(); i64_data.len()];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        i64_data.as_ptr() as *const U,
                        dst.as_mut_ptr(),
                        i64_data.len(),
                    );
                }
                dst
            } else {
                let i64_data: Vec<i64> = self.data.iter().map(|x| x.to_f64() as i64).collect();
                let mut dst = vec![U::default(); i64_data.len()];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        i64_data.as_ptr() as *const U,
                        dst.as_mut_ptr(),
                        i64_data.len(),
                    );
                }
                dst
            }
        } else if std::any::TypeId::of::<U>() == std::any::TypeId::of::<f32>() {
            // T -> f32 変換
            let f32_data: Vec<f32> = self.data.iter().map(|x| x.to_f32()).collect();
            let mut dst = vec![U::default(); f32_data.len()];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    f32_data.as_ptr() as *const U,
                    dst.as_mut_ptr(),
                    f32_data.len(),
                );
            }
            dst
        } else if std::any::TypeId::of::<U>() == std::any::TypeId::of::<f64>() {
            // T -> f64 変換
            let f64_data: Vec<f64> = self.data.iter().map(|x| x.to_f64()).collect();
            let mut dst = vec![U::default(); f64_data.len()];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    f64_data.as_ptr() as *const U,
                    dst.as_mut_ptr(),
                    f64_data.len(),
                );
            }
            dst
        } else {
            vec![U::default(); self.elem_count()]
        }
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// 互換用: f32 データへの参照（T=f32 の場合のみ意味がある）
    pub fn data_f32(&self) -> &[T] {
        &self.data
    }

    pub fn data_f32_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// 浅いクローン（データコピーあり＝CPU なので Arc 不要）
    /// autograd メタデータはコピーしない
    pub fn shallow_clone(&self) -> Self {
        CpuTensor {
            data: self.data.clone(),
            data_i64: self.data_i64.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype,
            autograd: None,
        }
    }

    pub fn clone_data(&self) -> Self {
        self.shallow_clone()
    }

    // ========== Autograd ==========

    pub fn requires_grad(&self) -> bool {
        self.autograd.as_ref().map_or(false, |a| a.requires_grad)
    }

    pub fn set_requires_grad(&mut self, req_grad: bool) {
        if req_grad {
            if self.autograd.is_none() {
                self.autograd = Some(Box::new(AutogradMeta {
                    grad: None,
                    grad_fn: None,
                    requires_grad: true,
                }));
            } else {
                if let Some(ref mut meta) = self.autograd {
                    meta.requires_grad = true;
                }
            }
        } else {
            if let Some(ref mut meta) = self.autograd {
                meta.requires_grad = false;
                meta.grad = None;       // 勾配計算を無効化したなら勾配データも消す
                meta.grad_fn = None;    // グラフからも切り離す
            }
        }
    }

    pub fn clip_grad_value(&mut self, min: T, max: T) -> tl_backend::BackendResult<()> {
        if let Some(ref mut meta) = self.autograd {
            if let Some(ref mut g) = meta.grad {
                let clamped: Vec<T> = g.data.iter()
                    .map(|&x| if x < min { min } else if x > max { max } else { x })
                    .collect();
                g.data = clamped;
            }
        }
        Ok(())
    }

    pub fn clip_grad_norm(&mut self, max_norm: T, norm_type: T) -> tl_backend::BackendResult<T> {
        let mut total_norm = T::zero();
        if let Some(ref mut meta) = self.autograd {
            if let Some(ref mut g) = meta.grad {
                if norm_type == T::infinity() {
                    total_norm = g.data.iter().fold(T::zero(), |max, &x| if x.abs() > max { x.abs() } else { max });
                } else {
                    let sum: T = g.data.iter().map(|&x| x.abs().powf(norm_type)).sum();
                    total_norm = sum.powf(T::one() / norm_type);
                }
                
                let clip_coef = max_norm / (total_norm + T::from_f64(1e-6));
                if clip_coef < T::one() {
                    let scaled: Vec<T> = g.data.iter().map(|&x| x * clip_coef).collect();
                    g.data = scaled;
                }
            }
        }
        Ok(total_norm)
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

    pub fn set_grad_fn(&mut self, grad_fn: Box<dyn GradFn<T>>) {
        self.autograd = Some(Box::new(AutogradMeta {
            grad: None,
            grad_fn: Some(grad_fn),
            requires_grad: true,
        }));
    }

    pub fn get_grad(&self) -> Option<Self> {
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
        let self_ptr = self as *mut Self;

        // 1. DFS でトポロジカル順序を構築
        let mut topo_order: Vec<*mut Self> = Vec::new();
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

                if let Some((grads_result, inputs)) = propagation {
                    let grads = match grads_result {
                        Ok(g) => g,
                        Err(e) => {
                            eprintln!("Autograd backward error: {}", e);
                            continue;
                        }
                    };
                    for (input_ref, grad) in inputs.into_iter().zip(grads.into_iter()) {
                        // TensorRef (Arc<UnsafeCell<CpuTensor>>) から可変参照を取得
                        let input = unsafe { &mut *input_ref.get() };
                        if input.requires_grad() {
                            if let Some(ref mut meta) = input.autograd {
                                if let Some(ref mut existing) = meta.grad {
                                    match existing.add_impl(&grad) {
                                        Ok(sum) => *existing = sum,
                                        Err(e) => eprintln!("Autograd accumulation error: {}", e),
                                    }
                                } else {
                                    meta.grad = Some(grad);
                                }
                            }
                        }
                    }
                }
            }
        }

        // 4. 出力テンソルの grad_fn をクリア
        //    grad_fn = None により、GradFn 内の TensorRef (Arc) が drop される。
        //    入力テンソルの Arc RC が下がり、それらの grad_fn も連鎖的に drop
        //    → グラフ全体が自然に解放される。
        //    以前のように全ノードを走査する必要はない。
        if let Some(ref mut meta) = self.autograd {
            meta.grad_fn = None;
        }
    }

    /// backward 用のトポロジカルソート（DFS）
    /// 勾配伝播対象のみ（requires_grad==true）
    fn build_topo(
        ptr: *mut Self,
        visited: &mut std::collections::HashSet<usize>,
        topo: &mut Vec<*mut Self>,
    ) {
        let key = ptr as usize;
        if visited.contains(&key) {
            return;
        }
        visited.insert(key);
        let tensor = unsafe { &*ptr };
        if let Some(ref meta) = tensor.autograd {
            if let Some(ref gf) = meta.grad_fn {
                for input_ref in gf.inputs() {
                    // TensorRef から生ポインタを取得して再帰
                    let input_ptr = input_ref.get() as *mut Self;
                    let input_tensor = unsafe { &*input_ptr };
                    if input_tensor.requires_grad() {
                        Self::build_topo(input_ptr, visited, topo);
                    }
                }
            }
        }
        topo.push(ptr);
    }

    pub fn detach(&self) -> Self {
        self.shallow_clone()
    }

    /// autograd グラフを再帰的に走査し、全中間テンソルのデータバッファをクリアする。
    /// backward 完了後の detach 時に呼ばれる想定。
    /// 自身の autograd もクリアする。
    pub fn clear_autograd_graph(&mut self) {
        use std::collections::HashSet;
        let mut visited = HashSet::new();
        Self::clear_autograd_recursive(self as *mut Self, &mut visited);
    }

    fn clear_autograd_recursive(ptr: *mut Self, visited: &mut std::collections::HashSet<usize>) {
        if ptr.is_null() { return; }
        let key = ptr as usize;
        if visited.contains(&key) { return; }
        visited.insert(key);

        unsafe {
            let tensor = &mut *ptr;
            // 先に GradFn の inputs を再帰的に走査
            if let Some(ref autograd) = tensor.autograd {
                if let Some(ref grad_fn) = autograd.grad_fn {
                    let inputs = grad_fn.inputs();
                    for input_ref in inputs {
                        let input_ptr = input_ref.get() as *mut Self;
                        Self::clear_autograd_recursive(input_ptr, visited);
                    }
                }
            }
            // データバッファをクリア（メモリを OS に返却）
            tensor.data = Vec::new();
            tensor.data_i64 = None;
            tensor.shape = Vec::new();
            // autograd をクリア（Arc の参照カウント管理で安全）
            tensor.autograd = None;
        }
    }

    // ========== 演算実装 (_impl メソッド) ==========

    /// ブロードキャスト用ストライド計算
    /// src_shape を out_shape にブロードキャストする際のストライドを返す。
    /// 次元サイズが1の場合はストライド0（同じ要素を繰り返す）。
    /// ブロードキャスト不可能な場合は Err を返す。
    fn broadcast_strides(src_shape: &[usize], out_shape: &[usize]) -> tl_backend::BackendResult<Vec<usize>> {
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
                } else if src_shape[si] == out_shape[i] {
                    strides[i] = src_strides[si];
                } else {
                    return Err(tl_backend::BackendError::ShapeMismatch(format!(
                        "Cannot broadcast dimension {} with size {} to {}",
                        si, src_shape[si], out_shape[i]
                    )));
                }
            }
        }
        Ok(strides)
    }

    fn elementwise_binop(&self, other: &Self, op: impl Fn(T, T) -> T) -> tl_backend::BackendResult<Self> {
        // 空テンソルの場合は空テンソルを返す
        if self.data.is_empty() || other.data.is_empty() {
            return Ok(CpuTensor {
                data: vec![],
                data_i64: None,
                shape: vec![0],
                dtype: self.dtype,
                autograd: None,
            });
        }
        // NumPy互換ブロードキャスト
        let a_shape = &self.shape;
        let b_shape = &other.shape;
        let a = &self.data;
        let b = &other.data;

        // ブロードキャスト結果の shape を計算
        let out_ndim = a_shape.len().max(b_shape.len());
        let mut out_shape = vec![0usize; out_ndim];
        for i in 0..out_ndim {
            let da = if i < out_ndim - a_shape.len() { 1 } else { a_shape[i - (out_ndim - a_shape.len())] };
            let db = if i < out_ndim - b_shape.len() { 1 } else { b_shape[i - (out_ndim - b_shape.len())] };
            if da != db && da != 1 && db != 1 {
                 return Err(tl_backend::BackendError::ShapeMismatch(format!(
                    "Operands could not be broadcast together with shapes {:?} and {:?}",
                    a_shape, b_shape
                )));
            }
            out_shape[i] = da.max(db);
        }
        let out_len: usize = out_shape.iter().product();

        // a と b のストライドを計算（ブロードキャスト用）
        let a_strides = Self::broadcast_strides(a_shape, &out_shape)?;
        let b_strides = Self::broadcast_strides(b_shape, &out_shape)?;

        let data: Vec<T> = (0..out_len).map(|flat_idx| {
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
            let av = if a_idx < a.len() { a[a_idx] } else { T::zero() };
            let bv = if b_idx < b.len() { b[b_idx] } else { T::zero() };
            op(av, bv)
        }).collect::<Vec<T>>();

        Ok(CpuTensor::<T> {
            data: data,
            data_i64: None,
            shape: out_shape,
            dtype: self.dtype,
            autograd: None,
        })
    }

    pub fn add_impl(&self, other: &Self) -> tl_backend::BackendResult<Self> {
        self.elementwise_binop(other, |a, b| a + b)
    }

    pub fn sub_impl(&self, other: &Self) -> tl_backend::BackendResult<Self> {
        self.elementwise_binop(other, |a, b| a - b)
    }

    pub fn mul_impl(&self, other: &Self) -> tl_backend::BackendResult<Self> {
        self.elementwise_binop(other, |a, b| a * b)
    }

    pub fn div_impl(&self, other: &Self) -> tl_backend::BackendResult<Self> {
        self.elementwise_binop(other, |a, b| a / b)
    }

    pub fn pow_impl(&self, other: &Self) -> tl_backend::BackendResult<Self> {
        self.elementwise_binop(other, |a, b| a.powf(b))
    }

    pub fn rem_impl(&self, other: &Self) -> tl_backend::BackendResult<Self> {
        self.elementwise_binop(other, |a, b| a % b)
    }

    // ========== スカラー演算 ==========

    pub fn add_scalar_impl(&self, scalar: T) -> tl_backend::BackendResult<Self> {
        let data: Vec<T> = self.data.iter().map(|x| *x + scalar).collect();
        Ok(CpuTensor::<T> { data: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    pub fn sub_scalar_impl(&self, scalar: T) -> tl_backend::BackendResult<Self> {
        let data: Vec<T> = self.data.iter().map(|x| *x - scalar).collect();
        Ok(CpuTensor::<T> { data: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    pub fn mul_scalar_impl(&self, scalar: T) -> tl_backend::BackendResult<Self> {
        let data: Vec<T> = self.data.iter().map(|x| *x * scalar).collect();
        Ok(CpuTensor::<T> { data: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    pub fn div_scalar_impl(&self, scalar: T) -> tl_backend::BackendResult<Self> {
        let data: Vec<T> = self.data.iter().map(|x| *x / scalar).collect();
        Ok(CpuTensor::<T> { data: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    pub fn clamp_impl(&self, min: T, max: T) -> tl_backend::BackendResult<Self> {
        let data: Vec<T> = self.data.iter().map(|x| x.clamp(min, max)).collect();
        Ok(CpuTensor::<T> { data: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    // ========== 単項演算 ==========

    // ========== 単項演算 ==========

    pub fn neg_impl(&self) -> tl_backend::BackendResult<Self> {
        let data: Vec<T> = self.data.iter().map(|x| -*x).collect();
        Ok(CpuTensor::<T> { data: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    pub fn abs_impl(&self) -> tl_backend::BackendResult<Self> {
        let data: Vec<T> = self.data.iter().map(|x| x.abs()).collect();
        Ok(CpuTensor::<T> { data: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    pub fn exp_impl(&self) -> tl_backend::BackendResult<Self> {
        let data: Vec<T> = self.data.iter().map(|x| x.exp()).collect();
        Ok(CpuTensor::<T> { data: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    pub fn log_impl(&self) -> tl_backend::BackendResult<Self> {
        let data: Vec<T> = self.data.iter().map(|x| x.ln()).collect();
        Ok(CpuTensor::<T> { data: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    pub fn sqrt_impl(&self) -> tl_backend::BackendResult<Self> {
        let data: Vec<T> = self.data.iter().map(|x| x.sqrt()).collect();
        Ok(CpuTensor::<T> { data: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    pub fn sin_impl(&self) -> tl_backend::BackendResult<Self> {
        let data: Vec<T> = self.data.iter().map(|x| x.sin()).collect();
        Ok(CpuTensor::<T> { data: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    pub fn cos_impl(&self) -> tl_backend::BackendResult<Self> {
        let data: Vec<T> = self.data.iter().map(|x| x.cos()).collect();
        Ok(CpuTensor::<T> { data: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    pub fn tan_impl(&self) -> tl_backend::BackendResult<Self> {
        let data: Vec<T> = self.data.iter().map(|x| x.tan()).collect();
        Ok(CpuTensor::<T> { data: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    pub fn tanh_impl(&self) -> tl_backend::BackendResult<Self> {
        let data: Vec<T> = self.data.iter().map(|x| x.tanh()).collect();
        Ok(CpuTensor::<T> { data: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    pub fn sigmoid_impl(&self) -> tl_backend::BackendResult<Self> {
        let data: Vec<T> = self.data.iter().map(|x| T::one() / (T::one() + (-*x).exp())).collect();
        Ok(CpuTensor::<T> { data: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    pub fn relu_impl(&self) -> tl_backend::BackendResult<Self> {
        let data: Vec<T> = self.data.iter().map(|x| if *x > T::zero() { *x } else { T::zero() }).collect();
        Ok(CpuTensor::<T> { data: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    pub fn gelu_impl(&self) -> tl_backend::BackendResult<Self> {
        let half = T::from_f64(0.5);
        let coeff = T::from_f64(0.044715);
        let sqrt_2_pi = T::frac_2_sqrt_pi();
        let data: Vec<T> = self.data.iter().map(|x| {
            let x = *x;
            half * x * (T::one() + (sqrt_2_pi * (x + coeff * x.powi(3))).tanh())
        }).collect();
        Ok(CpuTensor::<T> { data: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    // ========== Reduce ==========

    pub fn sumall_impl(&self) -> T {
        self.data.iter().copied().sum()
    }

    pub fn mean_all_impl(&self) -> T {
        let n = T::from_f64(self.data.len() as f64);
        if n > T::zero() { self.sumall_impl() / n } else { T::zero() }
    }

    pub fn sum_impl(&self, axis: i32) -> tl_backend::BackendResult<Self> {
        let ndim = self.shape.len();
        if ndim == 0 || self.data.is_empty() {
            return Ok(CpuTensor::<T> { data: vec![T::zero()], data_i64: None, shape: vec![1], dtype: self.dtype, autograd: None });
        }
        let axis = if axis < 0 { (ndim as i32 + axis) as usize } else { axis as usize };
        if axis >= ndim {
            // axis が範囲外の場合は全要素合計
            let s: T = self.data.iter().copied().sum();
            return Ok(CpuTensor::<T> { data: vec![s], data_i64: None, shape: vec![1], dtype: self.dtype, autograd: None });
        }
        let outer: usize = self.shape[..axis].iter().product();
        let axis_size = self.shape[axis];
        let inner: usize = self.shape[axis + 1..].iter().product();
        let mut result = vec![T::zero(); outer * inner];
        for i in 0..outer {
            for j in 0..axis_size {
                for k in 0..inner {
                    result[i * inner + k] += self.data[i * axis_size * inner + j * inner + k];
                }
            }
        }
        let mut new_shape: Vec<usize> = self.shape.clone();
        new_shape.remove(axis);
        if new_shape.is_empty() { new_shape.push(1); }
        Ok(CpuTensor::<T> { data: result, data_i64: None, shape: new_shape, dtype: self.dtype, autograd: None })
    }

    pub fn max_impl(&self, axis: i32) -> tl_backend::BackendResult<Self> {
        let ndim = self.shape.len();
        let axis = if axis < 0 { (ndim as i32 + axis) as usize } else { axis as usize };
        let outer: usize = self.shape[..axis].iter().product();
        let axis_size = self.shape[axis];
        let inner: usize = self.shape[axis + 1..].iter().product();
        let mut result = vec![T::neg_infinity(); outer * inner];
        for i in 0..outer {
            for j in 0..axis_size {
                for k in 0..inner {
                    let idx = i * inner + k;
                    let val = self.data[i * axis_size * inner + j * inner + k];
                    if val > result[idx] { result[idx] = val; }
                }
            }
        }
        let mut new_shape: Vec<usize> = self.shape.clone();
        new_shape.remove(axis);
        if new_shape.is_empty() { new_shape.push(1); }
        Ok(CpuTensor::<T> { data: result, data_i64: None, shape: new_shape, dtype: self.dtype, autograd: None })
    }

    pub fn min_impl(&self, axis: i32) -> tl_backend::BackendResult<Self> {
        let ndim = self.shape.len();
        let axis = if axis < 0 { (ndim as i32 + axis) as usize } else { axis as usize };
        let outer: usize = self.shape[..axis].iter().product();
        let axis_size = self.shape[axis];
        let inner: usize = self.shape[axis + 1..].iter().product();
        let mut result = vec![T::infinity(); outer * inner];
        for i in 0..outer {
            for j in 0..axis_size {
                for k in 0..inner {
                    let idx = i * inner + k;
                    let val = self.data[i * axis_size * inner + j * inner + k];
                    if val < result[idx] { result[idx] = val; }
                }
            }
        }
        let mut new_shape: Vec<usize> = self.shape.clone();
        new_shape.remove(axis);
        if new_shape.is_empty() { new_shape.push(1); }
        Ok(CpuTensor::<T> { data: result, data_i64: None, shape: new_shape, dtype: self.dtype, autograd: None })
    }

    pub fn argmax_impl(&self, axis: i32) -> tl_backend::BackendResult<Self> {
        let ndim = self.shape.len();
        let axis = if axis < 0 { (ndim as i32 + axis) as usize } else { axis as usize };
        let outer: usize = self.shape[..axis].iter().product();
        let axis_size = self.shape[axis];
        let inner: usize = self.shape[axis + 1..].iter().product();
        let mut result = vec![T::zero(); outer * inner];
        let mut max_vals = vec![T::neg_infinity(); outer * inner];
        for i in 0..outer {
            for j in 0..axis_size {
                for k in 0..inner {
                    let idx = i * inner + k;
                    let val = self.data[i * axis_size * inner + j * inner + k];
                    if val > max_vals[idx] {
                        max_vals[idx] = val;
                        result[idx] = T::from_f64(j as f64);
                    }
                }
            }
        }
        let mut new_shape: Vec<usize> = self.shape.clone();
        new_shape.remove(axis);
        if new_shape.is_empty() { new_shape.push(1); }
        Ok(CpuTensor::<T> { data: result, data_i64: None, shape: new_shape, dtype: self.dtype, autograd: None })
    }

    pub fn argmax_all_impl(&self) -> usize {
        self.data.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    pub fn argmin_impl(&self, axis: i32) -> tl_backend::BackendResult<Self> {
        let ndim = self.shape.len();
        let axis = if axis < 0 { (ndim as i32 + axis) as usize } else { axis as usize };
        let outer: usize = self.shape[..axis].iter().product();
        let axis_size = self.shape[axis];
        let inner: usize = self.shape[axis + 1..].iter().product();
        let mut result = vec![T::zero(); outer * inner];
        let mut min_vals = vec![T::infinity(); outer * inner];
        for i in 0..outer {
            for j in 0..axis_size {
                for k in 0..inner {
                    let idx = i * inner + k;
                    let val = self.data[i * axis_size * inner + j * inner + k];
                    if val < min_vals[idx] {
                        min_vals[idx] = val;
                        result[idx] = T::from_f64(j as f64);
                    }
                }
            }
        }
        let mut new_shape: Vec<usize> = self.shape.clone();
        new_shape.remove(axis);
        if new_shape.is_empty() { new_shape.push(1); }
        Ok(CpuTensor::<T> { data: result, data_i64: None, shape: new_shape, dtype: self.dtype, autograd: None })
    }

    pub fn mean_impl(&self, axis: i32) -> tl_backend::BackendResult<Self> {
        let ndim = self.shape.len();
        let ax = if axis < 0 { (ndim as i32 + axis) as usize } else { axis as usize };
        let axis_size = T::from_f64(self.shape[ax] as f64);
        let s = self.sum_impl(axis)?;
        s.div_scalar_impl(axis_size)
    }

    // ========== 形状操作 ==========

    pub fn reshape_impl(&self, shape: &[usize]) -> tl_backend::BackendResult<Self> {
        let current_size: usize = self.shape.iter().product();
        let target_size: usize = shape.iter().product();
        if current_size != target_size {
            return Err(tl_backend::BackendError::ShapeMismatch(format!(
                "Cannot reshape tensor of size {} into shape {:?}",
                current_size, shape
            )));
        }
        Ok(CpuTensor {
            data: self.data.clone(),
            data_i64: self.data_i64.clone(),
            shape: shape.to_vec(),
            dtype: self.dtype,
            autograd: None,
        })
    }

    pub fn transpose_impl(&self, dim0: usize, dim1: usize) -> tl_backend::BackendResult<Self> {
        if self.shape.len() < 2 {
            return Ok(self.shallow_clone());
        }
        let ndim = self.shape.len();
        if dim0 >= ndim || dim1 >= ndim {
            return Err(tl_backend::BackendError::ShapeMismatch(format!(
                "Transpose dims {} and {} are out of bounds for tensor of rank {}",
                dim0, dim1, ndim
            )));
        }
        let mut new_shape = self.shape.clone();
        new_shape.swap(dim0, dim1);

        let total = self.elem_count();
        let mut result = vec![T::zero(); total];

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
            result[new_flat_idx] = self.data[flat_idx];
        }

        Ok(CpuTensor::<T> { data: result, data_i64: None, shape: new_shape, dtype: self.dtype, autograd: None })
    }

    pub fn squeeze_impl(&self, dim: usize) -> tl_backend::BackendResult<Self> {
        let mut new_shape = self.shape.clone();
        if dim < new_shape.len() && new_shape[dim] == 1 {
            new_shape.remove(dim);
        }
        // If dim is not 1 or out of bounds, PyTorch usually returns tensor as is (for squeeze).
        // But if dim is out of bounds, maybe error?
        // PyTorch `squeeze(dim)`: if dim is out of range, it just ignores.
        // Let's stick to current logic but wrap in Ok.
        if new_shape.is_empty() { new_shape.push(1); }
        self.reshape_impl(&new_shape)
    }

    pub fn unsqueeze_impl(&self, dim: usize) -> tl_backend::BackendResult<Self> {
        let mut new_shape = self.shape.clone();
        if dim > new_shape.len() {
             return Err(tl_backend::BackendError::ShapeMismatch(format!(
                "Dimension out of range (expected to be in range of [0, {}], but got {})",
                new_shape.len(), dim
            )));
        }
        new_shape.insert(dim, 1);
        self.reshape_impl(&new_shape)
    }

    pub fn broadcast_to_impl(&self, shape: &[usize]) -> tl_backend::BackendResult<Self> {
        let total: usize = shape.iter().product();
        if self.data.is_empty() {
             if total == 0 {
                return Ok(CpuTensor::<T> { data: vec![], data_i64: None, shape: shape.to_vec(), dtype: self.dtype, autograd: None });
             } else {
                 return Err(tl_backend::BackendError::ShapeMismatch("Cannot broadcast empty tensor to non-empty".to_string()));
             }
        }
        let mut result = vec![T::zero(); total];
        for i in 0..total {
            result[i] = self.data[i % self.data.len()];
        }
        Ok(CpuTensor::<T> { data: result, data_i64: None, shape: shape.to_vec(), dtype: self.dtype, autograd: None })
    }

    pub fn narrow_impl(&self, axis: usize, start: usize, len: usize) -> tl_backend::BackendResult<Self> {
        self.slice_impl(axis, start, len)
    }

    pub fn slice_impl(&self, axis: usize, start: usize, len: usize) -> tl_backend::BackendResult<Self> {
        if axis >= self.shape.len() {
             return Err(tl_backend::BackendError::ShapeMismatch(format!(
                "Slice axis {} out of bounds for tensor of rank {}", axis, self.shape.len()
            )));
        }
        if start + len > self.shape[axis] {
             return Err(tl_backend::BackendError::ShapeMismatch(format!(
                "Slice range {}:{} out of bounds for axis {} of size {}", 
                start, start+len, axis, self.shape[axis]
            )));
        }
        let outer: usize = self.shape[..axis].iter().product();
        let axis_size = self.shape[axis];
        let inner: usize = self.shape[axis + 1..].iter().product();
        let mut result = Vec::with_capacity(outer * len * inner);
        for i in 0..outer {
            for j in start..start + len {
                for k in 0..inner {
                    result.push(self.data[i * axis_size * inner + j * inner + k]);
                }
            }
        }
        let mut new_shape = self.shape.clone();
        new_shape[axis] = len;
        Ok(CpuTensor::<T> { data: result, data_i64: None, shape: new_shape, dtype: self.dtype, autograd: None })
    }

    pub fn contiguous_impl(&self) -> tl_backend::BackendResult<Self> {
        Ok(self.shallow_clone())
    }

    pub fn cat_impl(tensors: &[&Self], axis: usize) -> tl_backend::BackendResult<Self> {
        if tensors.is_empty() {
             return Err(tl_backend::BackendError::ShapeMismatch("Cannot concatenate empty list of tensors".to_string()));
        }
        let first = tensors[0];
        let ndim = first.shape.len();
        if axis >= ndim {
             return Err(tl_backend::BackendError::ShapeMismatch(format!("Cat axis {} out of bounds", axis)));
        }
        for (i, t) in tensors.iter().enumerate().skip(1) {
            if t.shape.len() != ndim {
                 return Err(tl_backend::BackendError::ShapeMismatch(format!("Tensor {} has different rank", i)));
            }
            for (d, &s) in t.shape.iter().enumerate() {
                if d != axis && s != first.shape[d] {
                     return Err(tl_backend::BackendError::ShapeMismatch(format!(
                        "Tensor {} has mismatching shape at dim {} (expected {}, got {})", i, d, first.shape[d], s
                    )));
                }
            }
        }
        let outer: usize = first.shape[..axis].iter().product();
        let inner: usize = first.shape[axis + 1..].iter().product();
        let total_axis: usize = tensors.iter().map(|t| t.shape[axis]).sum();
        let mut result = Vec::with_capacity(outer * total_axis * inner);
        for i in 0..outer {
            for t in tensors {
                let axis_size = t.shape[axis];
                for j in 0..axis_size {
                    for k in 0..inner {
                        result.push(t.data[i * axis_size * inner + j * inner + k]);
                    }
                }
            }
        }
        let mut new_shape = first.shape.clone();
        new_shape[axis] = total_axis;
        Ok(CpuTensor::<T> { data: result, data_i64: None, shape: new_shape, dtype: first.dtype, autograd: None })
    }

    // ========== 活性化・特殊演算 ==========

    // ========== 活性化・特殊演算 ==========

    pub fn softmax_impl(&self, axis: i32) -> tl_backend::BackendResult<Self> {
        let ndim = self.shape.len();
        let axis = if axis < 0 { (ndim as i32 + axis) as usize } else { axis as usize };
        let outer: usize = self.shape[..axis].iter().product();
        let axis_size = self.shape[axis];
        let inner: usize = self.shape[axis + 1..].iter().product();
        let mut result = self.data.clone();
        for i in 0..outer {
            for k in 0..inner {
                let mut max_val = T::neg_infinity();
                for j in 0..axis_size {
                    let idx = i * axis_size * inner + j * inner + k;
                    if result[idx] > max_val { max_val = result[idx]; }
                }
                let mut sum = T::zero();
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
        Ok(CpuTensor::<T> { data: result, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    pub fn embedding_impl(&self, indices: &Self) -> tl_backend::BackendResult<Self> {
        let _vocab_size = self.shape[0];
        let embed_dim = if self.shape.len() > 1 { self.shape[1] } else { 1 };
        let idx_data: Vec<i64> = indices.to_vec::<f32>().iter().map(|x| *x as i64).collect();
        let mut result = Vec::with_capacity(idx_data.len() * embed_dim);
        for &idx in &idx_data {
            if idx < 0 || idx >= _vocab_size as i64 {
                 return Err(tl_backend::BackendError::IndexOutOfBounds(format!(
                    "Embedding index {} out of range for vocab size {}", idx, _vocab_size
                )));
            }
            let start = (idx as usize) * embed_dim;
            let end = start + embed_dim;
            result.extend_from_slice(&self.data[start..end.min(self.data.len())]);
        }
        let mut new_shape = indices.shape.to_vec();
        new_shape.push(embed_dim);
        Ok(CpuTensor::<T> { data: result, data_i64: None, shape: new_shape, dtype: self.dtype, autograd: None })
    }

    pub fn tril_impl(&self, diagonal: i32) -> tl_backend::BackendResult<Self> {
        if self.shape.len() < 2 {
             return Err(tl_backend::BackendError::ShapeMismatch("tril requires at least 2 dimensions".to_string()));
        }
        let rows = self.shape[self.shape.len() - 2];
        let cols = self.shape[self.shape.len() - 1];
        let batch: usize = self.shape[..self.shape.len() - 2].iter().product();
        let mut result = self.data.clone();
        for b in 0..batch.max(1) {
            for i in 0..rows {
                for j in 0..cols {
                    if (j as i32) > (i as i32) + diagonal {
                        result[b * rows * cols + i * cols + j] = T::zero();
                    }
                }
            }
        }
        Ok(CpuTensor::<T> { data: result, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    pub fn cross_entropy_impl(&self, target: &Self) -> tl_backend::BackendResult<Self> {
        let batch = self.shape[0];
        let classes = self.shape[1];
        let mut loss = T::zero();
        for b in 0..batch {
            let target_idx = if b < target.data.len() { target.data[b].to_f64() as usize } else { 0 }; // Handle broadcasting or mismatch?
            if target_idx >= classes {
                 return Err(tl_backend::BackendError::IndexOutOfBounds(format!(
                    "Target index {} out of bounds for {} classes", target_idx, classes
                )));
            }
            let mut max_val = T::neg_infinity();
            for c in 0..classes {
                let val = self.data[b * classes + c];
                if val > max_val { max_val = val; }
            }
            let mut sum_exp = T::zero();
            for c in 0..classes {
                sum_exp += (self.data[b * classes + c] - max_val).exp();
            }
            let log_softmax = self.data[b * classes + target_idx] - max_val - sum_exp.ln();
            loss -= log_softmax;
        }
        loss /= T::from_f64(batch as f64);
        Ok(CpuTensor::<T> { data: vec![loss], data_i64: None, shape: vec![1], dtype: self.dtype, autograd: None })
    }

    pub fn repeat_interleave_impl(&self, repeats: usize, axis: usize) -> tl_backend::BackendResult<Self> {
        if axis >= self.shape.len() {
             return Err(tl_backend::BackendError::ShapeMismatch(format!(
                "Axis {} out of bounds for repeat_interleave (rank {})", axis, self.shape.len()
            )));
        }
        let outer: usize = self.shape[..axis].iter().product();
        let axis_size = self.shape[axis];
        let inner: usize = self.shape[axis + 1..].iter().product();
        let mut result = Vec::with_capacity(outer * axis_size * repeats * inner);
        for i in 0..outer {
            for j in 0..axis_size {
                for _ in 0..repeats {
                    for k in 0..inner {
                        result.push(self.data[i * axis_size * inner + j * inner + k]);
                    }
                }
            }
        }
        let mut new_shape = self.shape.clone();
        new_shape[axis] *= repeats;
        Ok(CpuTensor::<T> { data: result, data_i64: None, shape: new_shape, dtype: self.dtype, autograd: None })
    }

    pub fn index_select_impl(&self, axis: usize, indices: &Self) -> tl_backend::BackendResult<Self> {
        if axis >= self.shape.len() {
             return Err(tl_backend::BackendError::ShapeMismatch(format!(
                "Axis {} out of bounds for index_select (rank {})", axis, self.shape.len()
            )));
        }
        let idx_list: Vec<usize> = indices.data.iter().map(|x| x.to_f64() as usize).collect();
        let outer: usize = self.shape[..axis].iter().product();
        let axis_size = self.shape[axis];
        let inner: usize = self.shape[axis + 1..].iter().product();
        let mut result = Vec::with_capacity(outer * idx_list.len() * inner);
        for i in 0..outer {
            for &idx in &idx_list {
                if idx >= axis_size {
                     return Err(tl_backend::BackendError::IndexOutOfBounds(format!(
                        "Index {} out of bounds for dimension {} size {}", idx, axis, axis_size
                    )));
                }
                for k in 0..inner {
                    result.push(self.data[i * axis_size * inner + idx * inner + k]);
                }
            }
        }
        let mut new_shape = self.shape.clone();
        new_shape[axis] = idx_list.len();
        Ok(CpuTensor::<T> { data: result, data_i64: None, shape: new_shape, dtype: self.dtype, autograd: None })
    }

    pub fn where_cond_impl(condition: &Self, x: &Self, y: &Self) -> tl_backend::BackendResult<Self> {
        if condition.shape != x.shape || x.shape != y.shape {
             return Err(tl_backend::BackendError::ShapeMismatch(format!(
                "Shapes mismatch for where_cond: cond {:?}, x {:?}, y {:?}",
                condition.shape, x.shape, y.shape
            )));
        }
        let data: Vec<T> = condition.data.iter()
            .zip(x.data.iter().zip(y.data.iter()))
            .map(|(c, (xv, yv))| if *c != T::zero() { *xv } else { *yv })
            .collect();
        Ok(CpuTensor::<T> { data: data, data_i64: None, shape: x.shape.clone(), dtype: x.dtype, autograd: None })
    }

    // ========== Matmul ==========

    // ========== Matmul ==========

    pub fn matmul_impl(&self, other: &Self) -> tl_backend::BackendResult<Self> {
        let a_shape = &self.shape;
        let b_shape = &other.shape;
        
        // Basic shape validation for matmul
        if a_shape.len() < 1 || b_shape.len() < 1 {
             return Err(tl_backend::BackendError::ShapeMismatch("Matmul requires at least 1D tensors".to_string()));
        }

        // 2D matmul logic
        let (m, k) = if a_shape.len() >= 2 {
            (a_shape[a_shape.len() - 2], a_shape[a_shape.len() - 1])
        } else {
            (1, a_shape[0])
        };
        let (k2, _n) = if b_shape.len() >= 2 {
            (b_shape[b_shape.len() - 2], b_shape[b_shape.len() - 1])
        } else {
            (b_shape[0], 1)
        };

        if k != k2 {
            return Err(tl_backend::BackendError::ShapeMismatch(format!(
                "Matmul shape mismatch: A={:?}, B={:?} (inner dims {} vs {})", 
                a_shape, b_shape, k, k2
            )));
        }

        let n = if b_shape.len() >= 2 {
            b_shape[b_shape.len() - 1]
        } else {
            // 1D vector: treat as column vector (k, 1), so n=1
            1
        };
        let batch: usize = a_shape[..a_shape.len().saturating_sub(2)].iter().product::<usize>().max(1);
        let mut result = vec![T::zero(); batch * m * n];
        for b in 0..batch {
            let a_offset = b * m * k;
            let b_batch_size = if b_shape.len() >= 2 { k * n } else { b_shape[0] };
            let b_offset = if b_shape.len() > 2 { b * b_batch_size } else { 0 };
            for i in 0..m {
                for j in 0..n {
                    let mut sum = T::zero();
                    for p in 0..k {
                        let a_idx = a_offset + i * k + p;
                        let b_idx = if b_shape.len() >= 2 {
                            b_offset + p * n + j
                        } else {
                            // 1D: treat as column vector, index is just p
                            p
                        };
                        if a_idx < self.data.len() && b_idx < other.data.len() {
                            sum += self.data[a_idx] * other.data[b_idx];
                        }
                    }
                    result[b * m * n + i * n + j] = sum;
                }
            }
        }
        let mut new_shape = a_shape[..a_shape.len().saturating_sub(2)].to_vec();
        new_shape.push(m);
        if b_shape.len() >= 2 {
            new_shape.push(n);
        }
        Ok(CpuTensor::<T> { data: result, data_i64: None, shape: new_shape, dtype: self.dtype, autograd: None })
    }

    // ========== 比較演算 ==========

    pub fn eq_impl(&self, other: &Self) -> tl_backend::BackendResult<Self> {
        let eps = T::from_f64(1e-6);
        self.elementwise_binop(other, move |a, b| if (a - b).abs() < eps { T::one() } else { T::zero() })
    }

    pub fn neq_impl(&self, other: &Self) -> tl_backend::BackendResult<Self> {
        let eps = T::from_f64(1e-6);
        self.elementwise_binop(other, move |a, b| if (a - b).abs() >= eps { T::one() } else { T::zero() })
    }

    pub fn gt_impl(&self, other: &Self) -> tl_backend::BackendResult<Self> {
        self.elementwise_binop(other, |a, b| if a > b { T::one() } else { T::zero() })
    }

    pub fn lt_impl(&self, other: &Self) -> tl_backend::BackendResult<Self> {
        self.elementwise_binop(other, |a, b| if a < b { T::one() } else { T::zero() })
    }

    pub fn ge_impl(&self, other: &Self) -> tl_backend::BackendResult<Self> {
        self.elementwise_binop(other, |a, b| if a >= b { T::one() } else { T::zero() })
    }

    pub fn le_impl(&self, other: &Self) -> tl_backend::BackendResult<Self> {
        self.elementwise_binop(other, |a, b| if a <= b { T::one() } else { T::zero() })
    }

    // ========== 深層学習演算（簡易版） ==========

    pub fn conv2d_impl(&self, weight: &Self, stride: (usize, usize), padding: (usize, usize)) -> tl_backend::BackendResult<Self> {
        // self: [batch, in_ch, h, w], weight: [out_ch, in_ch, kh, kw]
        if self.shape.len() != 4 || weight.shape.len() != 4 {
             return Err(tl_backend::BackendError::ShapeMismatch("conv2d requires 4D tensors".to_string()));
        }
        let batch = self.shape[0];
        let in_ch = self.shape[1];
        let h = self.shape[2];
        let w = self.shape[3];
        let out_ch = weight.shape[0];
        let w_in_ch = weight.shape[1];
        let kh = weight.shape[2];
        let kw = weight.shape[3];

        if in_ch != w_in_ch {
             return Err(tl_backend::BackendError::ShapeMismatch(format!("conv2d channel mismatch: input {} vs weight {}", in_ch, w_in_ch)));
        }

        let (sh, sw) = stride;
        let (ph, pw) = padding;
        if sh == 0 || sw == 0 {
             return Err(tl_backend::BackendError::ShapeMismatch("stride cannot be 0".to_string()));
        }

        let out_h = (h + 2 * ph).saturating_sub(kh) / sh + 1;
        let out_w = (w + 2 * pw).saturating_sub(kw) / sw + 1;

        if out_h == 0 || out_w == 0 {
             // Maybe should be handled gracefully as empty tensor?
             // Or strict error?
             // standard behavior is outputting empty if output size is calculated as 0
             // But let's return error if input is too small for kernel?
             // Actually, out_h calculation above handles saturating_sub, so returns 1 if small? No.
             // If h + 2*ph < kh, then saturating_sub is 0, +1 makes it 1. Wait.
             // (X - K)/S + 1. If X < K, usually invalid or 0.
             // Let's assume standard formula: floor((H + 2*P - K)/S) + 1.
             // If (H + 2*P) < K, then result is usually error or 0? 
             // PyTorch: runtime error if calculated output is too small.
             if (h + 2 * ph) < kh || (w + 2 * pw) < kw {
                 return Err(tl_backend::BackendError::ShapeMismatch("Input too small for kernel".to_string()));
             }
        }

        let mut result = vec![T::zero(); batch * out_ch * out_h * out_w];

        for b in 0..batch {
            for oc in 0..out_ch {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = T::zero();
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
                                        sum += self.data[in_idx] * weight.data[w_idx];
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

        Ok(CpuTensor::<T> { data: result, data_i64: None, shape: vec![batch, out_ch, out_h, out_w], dtype: self.dtype, autograd: None })
    }

    pub fn batch_norm_impl(&self, gamma: &Self, beta: &Self, running_mean: &Self, running_var: &Self, eps: T) -> tl_backend::BackendResult<Self> {
        // self: [batch, channels, ...] or [batch, features]
        // running_mean/running_var 使用時はチャネルごと正規化
        if self.shape.len() >= 2 {
            let batch = self.shape[0];
            let channels = self.shape[1];
            let spatial: usize = self.shape[2..].iter().product();
            let spatial = if spatial == 0 { 1 } else { spatial };

            let use_running = running_mean.elem_count() == channels && running_var.elem_count() == channels;
            // Should check gamma/beta size too? yes.
            // But strict checking might break lax user code if they rely on broadcast or different logic.
            // Assuming strict.

            let mut result = self.data.clone();
            for c in 0..channels {
                let (mean, var) = if use_running {
                    (running_mean.data[c], running_var.data[c])
                } else {
                    // バッチ統計量を計算
                    let n = T::from_f64((batch * spatial) as f64);
                    let mut m = T::zero();
                    for b in 0..batch {
                        for s in 0..spatial {
                            m += self.data[b * channels * spatial + c * spatial + s];
                        }
                    }
                    m /= n;
                    let mut v = T::zero();
                    for b in 0..batch {
                        for s in 0..spatial {
                            let d = self.data[b * channels * spatial + c * spatial + s] - m;
                            v += d * d;
                        }
                    }
                    v /= n;
                    (m, v)
                };

                let inv_std = T::one() / (var + eps).sqrt();
                let g = if c < gamma.data.len() { gamma.data[c] } else { T::one() };
                let bi = if c < beta.data.len() { beta.data[c] } else { T::zero() };

                for b in 0..batch {
                    for s in 0..spatial {
                        let idx = b * channels * spatial + c * spatial + s;
                        result[idx] = (result[idx] - mean) * inv_std * g + bi;
                    }
                }
            }
            Ok(CpuTensor::<T> { data: result, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
        } else {
            // 1D フォールバック
            let mean = self.mean_all_impl();
            let data: Vec<T> = self.data.iter().map(|x| *x - mean).collect();
            let var: T = data.iter().map(|x| (*x) * (*x)).sum::<T>() / T::from_f64(data.len() as f64);
            let inv_std = T::one() / (var + eps).sqrt();
            let result: Vec<T> = data.iter().enumerate().map(|(i, x)| {
                let g = if i < gamma.data.len() { gamma.data[i % gamma.data.len()] } else { T::one() };
                let b = if i < beta.data.len() { beta.data[i % beta.data.len()] } else { T::zero() };
                *x * inv_std * g + b
            }).collect();
            Ok(CpuTensor::<T> { data: result, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
        }
    }

    pub fn layer_norm_impl(&self, gamma: &Self, beta: &Self, eps: T) -> tl_backend::BackendResult<Self> {
        // 最終次元で正規化 (各行独立)
        let ndim = self.shape.len();
        if ndim < 2 {
            // 1D: 全体を正規化
            let mean = self.mean_all_impl();
            let var: T = self.data.iter().map(|x| (*x - mean).powi(2)).fold(T::zero(), |a, b| a + b) / T::from_f64(self.data.len() as f64);
            let inv_std = T::one() / (var + eps).sqrt();
            let result: Vec<T> = self.data.iter().enumerate().map(|(i, x)| {
                let g = gamma.data.get(i).copied().unwrap_or(T::one());
                let b = beta.data.get(i).copied().unwrap_or(T::zero());
                (*x - mean) * inv_std * g + b
            }).collect();
            return Ok(CpuTensor::<T> { data: result, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None });
        }

        let last_dim = self.shape[ndim - 1];
        let outer: usize = self.shape[..ndim - 1].iter().product();
        let mut result = self.data.clone();

        for i in 0..outer {
            let start = i * last_dim;
            let end = start + last_dim;
            let slice = &self.data[start..end];

            let mean: T = slice.iter().copied().sum::<T>() / T::from_f64(last_dim as f64);
            let var: T = slice.iter().map(|x| (*x - mean).powi(2)).sum::<T>() / T::from_f64(last_dim as f64);
            let inv_std = T::one() / (var + eps).sqrt();

            for j in 0..last_dim {
                let g = gamma.data.get(j).copied().unwrap_or(T::one());
                let b = beta.data.get(j).copied().unwrap_or(T::zero());
                result[start + j] = (slice[j] - mean) * inv_std * g + b;
            }
        }

        Ok(CpuTensor::<T> { data: result, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    pub fn max_pool2d_impl(&self, kernel_size: (usize, usize), stride: (usize, usize)) -> tl_backend::BackendResult<Self> {
        // self: [batch, channels, h, w]
        if self.shape.len() != 4 {
             return Err(tl_backend::BackendError::ShapeMismatch("max_pool2d requires 4D tensor".to_string()));
        }
        let batch = self.shape[0];
        let channels = self.shape[1];
        let h = self.shape[2];
        let w = self.shape[3];
        let (kh, kw) = kernel_size;
        let (sh, sw) = stride;
        
        if (h < kh) || (w < kw) {
             return Err(tl_backend::BackendError::ShapeMismatch("Input smaller than kernel size".to_string()));
        }

        let out_h = (h - kh) / sh + 1;
        let out_w = (w - kw) / sw + 1;

        let mut result = vec![T::neg_infinity(); batch * channels * out_h * out_w];

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut max_val = T::neg_infinity();
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let ih = oh * sh + ki;
                                let iw = ow * sw + kj;
                                let idx = b * channels * h * w + c * h * w + ih * w + iw;
                                if self.data[idx] > max_val {
                                    max_val = self.data[idx];
                                }
                            }
                        }
                        let out_idx = b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                        result[out_idx] = max_val;
                    }
                }
            }
        }

        Ok(CpuTensor::<T> { data: result, data_i64: None, shape: vec![batch, channels, out_h, out_w], dtype: self.dtype, autograd: None })
    }

    pub fn avg_pool2d_impl(&self, kernel_size: (usize, usize), stride: (usize, usize)) -> tl_backend::BackendResult<Self> {
        // self: [batch, channels, h, w]
        if self.shape.len() != 4 {
             return Err(tl_backend::BackendError::ShapeMismatch("avg_pool2d requires 4D tensor".to_string()));
        }
        let batch = self.shape[0];
        let channels = self.shape[1];
        let h = self.shape[2];
        let w = self.shape[3];
        let (kh, kw) = kernel_size;
        let (sh, sw) = stride;
        
        if (h < kh) || (w < kw) {
             return Err(tl_backend::BackendError::ShapeMismatch("Input smaller than kernel size".to_string()));
        }
        
        let out_h = (h - kh) / sh + 1;
        let out_w = (w - kw) / sw + 1;
        let pool_size = T::from_f64((kh * kw) as f64);

        let mut result = vec![T::zero(); batch * channels * out_h * out_w];

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = T::zero();
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let ih = oh * sh + ki;
                                let iw = ow * sw + kj;
                                let idx = b * channels * h * w + c * h * w + ih * w + iw;
                                sum += self.data[idx];
                            }
                        }
                        let out_idx = b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                        result[out_idx] = sum / pool_size;
                    }
                }
            }
        }

        Ok(CpuTensor::<T> { data: result, data_i64: None, shape: vec![batch, channels, out_h, out_w], dtype: self.dtype, autograd: None })
    }

    pub fn dropout_impl(&self, _p: T, _training: bool) -> tl_backend::BackendResult<Self> {
        Ok(self.shallow_clone())
    }
}

// GpuTensor トレイトが Send + Sync を要求
unsafe impl<T: TensorScalar> Send for CpuTensor<T> {}
unsafe impl<T: TensorScalar> Sync for CpuTensor<T> {}

impl<T: TensorScalar> CpuTensor<T> {
    pub fn sample_impl(&self, temp: T, top_p: T) -> tl_backend::BackendResult<Self> {
        // Assume 1D logits for now or last dimension
        let logits = self.data_f32();
        if logits.is_empty() {
             return Err(tl_backend::BackendError::InternalError("Empty tensor for sampling".to_string()));
        }

        if temp <= T::zero() {
             // Greedy: argmax
             let max_idx = logits.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i).unwrap_or(0);
             return Ok(CpuTensor::from_slice(&[T::from_f64(max_idx as f64)], &[1], DType::F32));
        }

        // Apply temperature
        let probs: Vec<T> = logits.iter().map(|&x| x / temp).collect();
        // Softmax
        let max_val = probs.iter().copied().fold(T::neg_infinity(), |a, b| if a > b { a } else { b });
        let sum_exp: T = probs.iter().map(|x| (*x - max_val).exp()).sum();
        let probs: Vec<T> = probs.iter().map(|x| (*x - max_val).exp() / sum_exp).collect();

        // Top-p sampling
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut cum_prob = T::zero();
        let mut cutoff_index = indices.len() - 1;
        for (i, &idx) in indices.iter().enumerate() {
            cum_prob += probs[idx];
            if cum_prob > top_p {
                cutoff_index = i;
                break;
            }
        }
        
        let cutoff_prob = cum_prob;
        let mut rng = rand::thread_rng();
        use rand::Rng;
        
        let r: T = T::from_f64(rng.gen::<f64>()) * cutoff_prob;
        let mut acc = T::zero();
        let mut selected_idx = indices[0];
        
        for i in 0..=cutoff_index {
             let idx = indices[i];
             acc += probs[idx];
             if acc >= r {
                 selected_idx = idx;
                 break;
             }
        }
        
        Ok(CpuTensor::from_slice(&[T::from_f64(selected_idx as f64)], &[1], self.dtype))
    }
}
