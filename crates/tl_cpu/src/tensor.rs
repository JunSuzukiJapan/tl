//! CPU テンソル

use crate::autograd::GradFn;
use crate::DType;
use std::sync::Arc;
use std::cell::UnsafeCell;

/// テンソルへの共有所有権付きハンドル。
/// Autograd グラフ内でテンソル間の参照を安全に管理する。
/// CPU 版はシングルスレッドなので UnsafeCell で内部可変性を確保。
pub type TensorRef = Arc<UnsafeCell<CpuTensor>>;

/// 生ポインタから TensorRef を取得する (RC+1)。
/// FFI 境界で `*mut OpaqueTensor` を Autograd 用の `TensorRef` に変換する際に使用。
///
/// # Safety
/// `ptr` は `Arc::into_raw` で作成された有効なポインタでなければならない。
pub unsafe fn tensor_ref_from_ptr(ptr: *mut CpuTensor) -> TensorRef {
    let arc: TensorRef = Arc::from_raw(ptr as *const UnsafeCell<CpuTensor>);
    let cloned = arc.clone();  // RC+1
    let _ = Arc::into_raw(arc);        // 元の参照を戻す（RC を減らさない）
    cloned
}

/// TensorRef から内部の CpuTensor への不変参照を取得する。
///
/// # Safety
/// 同時に可変参照が存在しないことを呼び出し元が保証すること。
pub unsafe fn tensor_ref_get(r: &TensorRef) -> &CpuTensor {
    &*r.get()
}

/// TensorRef から内部の CpuTensor への可変参照を取得する。
///
/// # Safety
/// 同時に他の参照が存在しないことを呼び出し元が保証すること。
pub unsafe fn tensor_ref_get_mut(r: &TensorRef) -> &mut CpuTensor {
    &mut *r.get()
}


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

impl Drop for CpuTensor {
    fn drop(&mut self) {
        // track_alloc で加算された容量分を track_free で減算
        let f32_bytes = self.data_f32.capacity() * std::mem::size_of::<f32>();
        if f32_bytes > 0 {
            crate::memory::track_free(f32_bytes);
        }
        if let Some(ref v) = self.data_i64 {
            let i64_bytes = v.capacity() * std::mem::size_of::<i64>();
            if i64_bytes > 0 {
                crate::memory::track_free(i64_bytes);
            }
        }
    }
}

impl Clone for CpuTensor {
    fn clone(&self) -> Self {
        self.clone_data()
    }
}

impl CpuTensor {
    // ========== コンストラクタ ==========

    fn alloc_from_pool() -> Self {
        CpuTensor {
            data_f32: Vec::new(),
            data_i64: None,
            shape: Vec::new(),
            dtype: DType::F32,
            autograd: None,
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

        // Resize buffer
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
                    for (input_ref, grad) in inputs.into_iter().zip(grads.into_iter()) {
                        // TensorRef (Arc<UnsafeCell<CpuTensor>>) から可変参照を取得
                        let input = unsafe { &mut *input_ref.get() };
                        if input.requires_grad() {
                            if let Some(ref mut meta) = input.autograd {
                                if let Some(ref mut existing) = meta.grad {
                                    *existing = existing.add_impl(&grad).expect("Autograd accumulation failed");
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
                for input_ref in gf.inputs() {
                    // TensorRef から生ポインタを取得して再帰
                    let input_ptr = input_ref.get() as *mut CpuTensor;
                    let input_tensor = unsafe { &*input_ptr };
                    if input_tensor.requires_grad() {
                        Self::build_topo(input_ptr, visited, topo);
                    }
                }
            }
        }
        topo.push(ptr);
    }

    pub fn detach(&self) -> CpuTensor {
        self.shallow_clone()
    }

    /// autograd グラフを再帰的に走査し、全中間テンソルのデータバッファをクリアする。
    /// backward 完了後の detach 時に呼ばれる想定。
    /// 自身の autograd もクリアする。
    pub fn clear_autograd_graph(&mut self) {
        use std::collections::HashSet;
        let mut visited = HashSet::new();
        Self::clear_autograd_recursive(self as *mut CpuTensor, &mut visited);
    }

    fn clear_autograd_recursive(ptr: *mut CpuTensor, visited: &mut std::collections::HashSet<usize>) {
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
                        let input_ptr = input_ref.get() as *mut CpuTensor;
                        Self::clear_autograd_recursive(input_ptr, visited);
                    }
                }
            }
            // データバッファをクリア（メモリを OS に返却）
            tensor.data_f32 = Vec::new();
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

    fn elementwise_binop(&self, other: &Self, op: impl Fn(f32, f32) -> f32) -> tl_backend::BackendResult<Self> {
        // 空テンソルの場合は空テンソルを返す
        if self.data_f32.is_empty() || other.data_f32.is_empty() {
            return Ok(CpuTensor {
                data_f32: vec![],
                data_i64: None,
                shape: vec![0],
                dtype: self.dtype,
                autograd: None,
            });
        }
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
            let av = if a_idx < a.len() { a[a_idx] } else { 0.0 };
            let bv = if b_idx < b.len() { b[b_idx] } else { 0.0 };
            op(av, bv)
        }).collect();

        Ok(CpuTensor {
            data_f32: data,
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

    pub fn add_scalar_impl(&self, scalar: f32) -> tl_backend::BackendResult<Self> {
        let data: Vec<f32> = self.data_f32.iter().map(|x| x + scalar).collect();
        Ok(CpuTensor { data_f32: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    pub fn sub_scalar_impl(&self, scalar: f32) -> tl_backend::BackendResult<Self> {
        let data: Vec<f32> = self.data_f32.iter().map(|x| x - scalar).collect();
        Ok(CpuTensor { data_f32: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    pub fn mul_scalar_impl(&self, scalar: f32) -> tl_backend::BackendResult<Self> {
        let data: Vec<f32> = self.data_f32.iter().map(|x| x * scalar).collect();
        Ok(CpuTensor { data_f32: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    pub fn div_scalar_impl(&self, scalar: f32) -> tl_backend::BackendResult<Self> {
        let data: Vec<f32> = self.data_f32.iter().map(|x| x / scalar).collect();
        Ok(CpuTensor { data_f32: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    pub fn clamp_impl(&self, min: f32, max: f32) -> tl_backend::BackendResult<Self> {
        let data: Vec<f32> = self.data_f32.iter().map(|x| x.clamp(min, max)).collect();
        Ok(CpuTensor { data_f32: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    // ========== 単項演算 ==========

    // ========== 単項演算 ==========

    pub fn neg_impl(&self) -> tl_backend::BackendResult<Self> {
        let data: Vec<f32> = self.data_f32.iter().map(|x| -x).collect();
        Ok(CpuTensor { data_f32: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    pub fn abs_impl(&self) -> tl_backend::BackendResult<Self> {
        let data: Vec<f32> = self.data_f32.iter().map(|x| x.abs()).collect();
        Ok(CpuTensor { data_f32: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    pub fn exp_impl(&self) -> tl_backend::BackendResult<Self> {
        let data: Vec<f32> = self.data_f32.iter().map(|x| x.exp()).collect();
        Ok(CpuTensor { data_f32: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    pub fn log_impl(&self) -> tl_backend::BackendResult<Self> {
        let data: Vec<f32> = self.data_f32.iter().map(|x| x.ln()).collect();
        Ok(CpuTensor { data_f32: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    pub fn sqrt_impl(&self) -> tl_backend::BackendResult<Self> {
        let data: Vec<f32> = self.data_f32.iter().map(|x| x.sqrt()).collect();
        Ok(CpuTensor { data_f32: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    pub fn sin_impl(&self) -> tl_backend::BackendResult<Self> {
        let data: Vec<f32> = self.data_f32.iter().map(|x| x.sin()).collect();
        Ok(CpuTensor { data_f32: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    pub fn cos_impl(&self) -> tl_backend::BackendResult<Self> {
        let data: Vec<f32> = self.data_f32.iter().map(|x| x.cos()).collect();
        Ok(CpuTensor { data_f32: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    pub fn tan_impl(&self) -> tl_backend::BackendResult<Self> {
        let data: Vec<f32> = self.data_f32.iter().map(|x| x.tan()).collect();
        Ok(CpuTensor { data_f32: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    pub fn tanh_impl(&self) -> tl_backend::BackendResult<Self> {
        let data: Vec<f32> = self.data_f32.iter().map(|x| x.tanh()).collect();
        Ok(CpuTensor { data_f32: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    pub fn sigmoid_impl(&self) -> tl_backend::BackendResult<Self> {
        let data: Vec<f32> = self.data_f32.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
        Ok(CpuTensor { data_f32: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    pub fn relu_impl(&self) -> tl_backend::BackendResult<Self> {
        let data: Vec<f32> = self.data_f32.iter().map(|x| x.max(0.0)).collect();
        Ok(CpuTensor { data_f32: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    pub fn gelu_impl(&self) -> tl_backend::BackendResult<Self> {
        let data: Vec<f32> = self.data_f32.iter().map(|x| {
            0.5 * x * (1.0 + (std::f32::consts::FRAC_2_SQRT_PI * (x + 0.044715 * x.powi(3))).tanh())
        }).collect();
        Ok(CpuTensor { data_f32: data, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    // ========== Reduce ==========

    pub fn sumall_impl(&self) -> f32 {
        self.data_f32.iter().sum()
    }

    pub fn mean_all_impl(&self) -> f32 {
        let n = self.data_f32.len() as f32;
        if n > 0.0 { self.sumall_impl() / n } else { 0.0 }
    }

    pub fn sum_impl(&self, axis: i32) -> tl_backend::BackendResult<Self> {
        let ndim = self.shape.len();
        if ndim == 0 || self.data_f32.is_empty() {
            return Ok(CpuTensor { data_f32: vec![0.0], data_i64: None, shape: vec![1], dtype: self.dtype, autograd: None });
        }
        let axis = if axis < 0 { (ndim as i32 + axis) as usize } else { axis as usize };
        if axis >= ndim {
            // axis が範囲外の場合は全要素合計
            let s: f32 = self.data_f32.iter().sum();
            return Ok(CpuTensor { data_f32: vec![s], data_i64: None, shape: vec![1], dtype: self.dtype, autograd: None });
        }
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
        Ok(CpuTensor { data_f32: result, data_i64: None, shape: new_shape, dtype: self.dtype, autograd: None })
    }

    pub fn max_impl(&self, axis: i32) -> tl_backend::BackendResult<Self> {
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
        Ok(CpuTensor { data_f32: result, data_i64: None, shape: new_shape, dtype: self.dtype, autograd: None })
    }

    pub fn min_impl(&self, axis: i32) -> tl_backend::BackendResult<Self> {
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
        Ok(CpuTensor { data_f32: result, data_i64: None, shape: new_shape, dtype: self.dtype, autograd: None })
    }

    pub fn argmax_impl(&self, axis: i32) -> tl_backend::BackendResult<Self> {
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
        Ok(CpuTensor { data_f32: result, data_i64: None, shape: new_shape, dtype: self.dtype, autograd: None })
    }

    pub fn argmax_all_impl(&self) -> usize {
        self.data_f32.iter().enumerate()
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
        Ok(CpuTensor { data_f32: result, data_i64: None, shape: new_shape, dtype: self.dtype, autograd: None })
    }

    pub fn mean_impl(&self, axis: i32) -> tl_backend::BackendResult<Self> {
        let ndim = self.shape.len();
        let ax = if axis < 0 { (ndim as i32 + axis) as usize } else { axis as usize };
        let axis_size = self.shape[ax] as f32;
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
            data_f32: self.data_f32.clone(),
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

        Ok(CpuTensor { data_f32: result, data_i64: None, shape: new_shape, dtype: self.dtype, autograd: None })
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
        if self.data_f32.is_empty() {
             if total == 0 {
                return Ok(CpuTensor { data_f32: vec![], data_i64: None, shape: shape.to_vec(), dtype: self.dtype, autograd: None });
             } else {
                 return Err(tl_backend::BackendError::ShapeMismatch("Cannot broadcast empty tensor to non-empty".to_string()));
             }
        }
        let mut result = vec![0.0f32; total];
        for i in 0..total {
            result[i] = self.data_f32[i % self.data_f32.len()];
        }
        Ok(CpuTensor { data_f32: result, data_i64: None, shape: shape.to_vec(), dtype: self.dtype, autograd: None })
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
                    result.push(self.data_f32[i * axis_size * inner + j * inner + k]);
                }
            }
        }
        let mut new_shape = self.shape.clone();
        new_shape[axis] = len;
        Ok(CpuTensor { data_f32: result, data_i64: None, shape: new_shape, dtype: self.dtype, autograd: None })
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
                        result.push(t.data_f32[i * axis_size * inner + j * inner + k]);
                    }
                }
            }
        }
        let mut new_shape = first.shape.clone();
        new_shape[axis] = total_axis;
        Ok(CpuTensor { data_f32: result, data_i64: None, shape: new_shape, dtype: first.dtype, autograd: None })
    }

    // ========== 活性化・特殊演算 ==========

    // ========== 活性化・特殊演算 ==========

    pub fn softmax_impl(&self, axis: i32) -> tl_backend::BackendResult<Self> {
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
        Ok(CpuTensor { data_f32: result, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
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
            result.extend_from_slice(&self.data_f32[start..end.min(self.data_f32.len())]);
        }
        let mut new_shape = indices.shape.to_vec();
        new_shape.push(embed_dim);
        Ok(CpuTensor { data_f32: result, data_i64: None, shape: new_shape, dtype: self.dtype, autograd: None })
    }

    pub fn tril_impl(&self, diagonal: i32) -> tl_backend::BackendResult<Self> {
        if self.shape.len() < 2 {
             return Err(tl_backend::BackendError::ShapeMismatch("tril requires at least 2 dimensions".to_string()));
        }
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
        Ok(CpuTensor { data_f32: result, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
    }

    pub fn cross_entropy_impl(&self, target: &Self) -> tl_backend::BackendResult<Self> {
        let batch = self.shape[0];
        let classes = self.shape[1];
        let mut loss = 0.0f32;
        for b in 0..batch {
            let target_idx = if b < target.data_f32.len() { target.data_f32[b] as usize } else { 0 }; // Handle broadcasting or mismatch?
            if target_idx >= classes {
                 return Err(tl_backend::BackendError::IndexOutOfBounds(format!(
                    "Target index {} out of bounds for {} classes", target_idx, classes
                )));
            }
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
        Ok(CpuTensor { data_f32: vec![loss], data_i64: None, shape: vec![1], dtype: self.dtype, autograd: None })
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
                        result.push(self.data_f32[i * axis_size * inner + j * inner + k]);
                    }
                }
            }
        }
        let mut new_shape = self.shape.clone();
        new_shape[axis] *= repeats;
        Ok(CpuTensor { data_f32: result, data_i64: None, shape: new_shape, dtype: self.dtype, autograd: None })
    }

    pub fn index_select_impl(&self, axis: usize, indices: &Self) -> tl_backend::BackendResult<Self> {
        if axis >= self.shape.len() {
             return Err(tl_backend::BackendError::ShapeMismatch(format!(
                "Axis {} out of bounds for index_select (rank {})", axis, self.shape.len()
            )));
        }
        let idx_list: Vec<usize> = indices.data_f32.iter().map(|x| *x as usize).collect();
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
                    result.push(self.data_f32[i * axis_size * inner + idx * inner + k]);
                }
            }
        }
        let mut new_shape = self.shape.clone();
        new_shape[axis] = idx_list.len();
        Ok(CpuTensor { data_f32: result, data_i64: None, shape: new_shape, dtype: self.dtype, autograd: None })
    }

    pub fn where_cond_impl(condition: &Self, x: &Self, y: &Self) -> tl_backend::BackendResult<Self> {
        if condition.shape != x.shape || x.shape != y.shape {
             return Err(tl_backend::BackendError::ShapeMismatch(format!(
                "Shapes mismatch for where_cond: cond {:?}, x {:?}, y {:?}",
                condition.shape, x.shape, y.shape
            )));
        }
        let data: Vec<f32> = condition.data_f32.iter()
            .zip(x.data_f32.iter().zip(y.data_f32.iter()))
            .map(|(c, (xv, yv))| if *c != 0.0 { *xv } else { *yv })
            .collect();
        Ok(CpuTensor { data_f32: data, data_i64: None, shape: x.shape.clone(), dtype: x.dtype, autograd: None })
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
        let mut result = vec![0.0f32; batch * m * n];
        for b in 0..batch {
            let a_offset = b * m * k;
            let b_batch_size = if b_shape.len() >= 2 { k * n } else { b_shape[0] };
            let b_offset = if b_shape.len() > 2 { b * b_batch_size } else { 0 };
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for p in 0..k {
                        let a_idx = a_offset + i * k + p;
                        let b_idx = if b_shape.len() >= 2 {
                            b_offset + p * n + j
                        } else {
                            // 1D: treat as column vector, index is just p
                            p
                        };
                        if a_idx < self.data_f32.len() && b_idx < other.data_f32.len() {
                            sum += self.data_f32[a_idx] * other.data_f32[b_idx];
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
        Ok(CpuTensor { data_f32: result, data_i64: None, shape: new_shape, dtype: self.dtype, autograd: None })
    }

    // ========== 比較演算 ==========

    pub fn eq_impl(&self, other: &Self) -> tl_backend::BackendResult<Self> {
        self.elementwise_binop(other, |a, b| if (a - b).abs() < 1e-6 { 1.0 } else { 0.0 })
    }

    pub fn neq_impl(&self, other: &Self) -> tl_backend::BackendResult<Self> {
        self.elementwise_binop(other, |a, b| if (a - b).abs() >= 1e-6 { 1.0 } else { 0.0 })
    }

    pub fn gt_impl(&self, other: &Self) -> tl_backend::BackendResult<Self> {
        self.elementwise_binop(other, |a, b| if a > b { 1.0 } else { 0.0 })
    }

    pub fn lt_impl(&self, other: &Self) -> tl_backend::BackendResult<Self> {
        self.elementwise_binop(other, |a, b| if a < b { 1.0 } else { 0.0 })
    }

    pub fn ge_impl(&self, other: &Self) -> tl_backend::BackendResult<Self> {
        self.elementwise_binop(other, |a, b| if a >= b { 1.0 } else { 0.0 })
    }

    pub fn le_impl(&self, other: &Self) -> tl_backend::BackendResult<Self> {
        self.elementwise_binop(other, |a, b| if a <= b { 1.0 } else { 0.0 })
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

        Ok(CpuTensor { data_f32: result, data_i64: None, shape: vec![batch, out_ch, out_h, out_w], dtype: self.dtype, autograd: None })
    }

    pub fn batch_norm_impl(&self, gamma: &Self, beta: &Self, running_mean: &Self, running_var: &Self, eps: f32) -> tl_backend::BackendResult<Self> {
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
            Ok(CpuTensor { data_f32: result, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
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
            Ok(CpuTensor { data_f32: result, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
        }
    }

    pub fn layer_norm_impl(&self, gamma: &Self, beta: &Self, eps: f32) -> tl_backend::BackendResult<Self> {
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
            return Ok(CpuTensor { data_f32: result, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None });
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

        Ok(CpuTensor { data_f32: result, data_i64: None, shape: self.shape.clone(), dtype: self.dtype, autograd: None })
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

        Ok(CpuTensor { data_f32: result, data_i64: None, shape: vec![batch, channels, out_h, out_w], dtype: self.dtype, autograd: None })
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

        Ok(CpuTensor { data_f32: result, data_i64: None, shape: vec![batch, channels, out_h, out_w], dtype: self.dtype, autograd: None })
    }

    pub fn dropout_impl(&self, _p: f32, _training: bool) -> tl_backend::BackendResult<Self> {
        Ok(self.shallow_clone())
    }
}

// GpuTensor トレイトが Send + Sync を要求
unsafe impl Send for CpuTensor {}
unsafe impl Sync for CpuTensor {}

impl CpuTensor {
    pub fn sample_impl(&self, temp: f32, top_p: f32) -> tl_backend::BackendResult<Self> {
        // Assume 1D logits for now or last dimension
        let logits = self.data_f32();
        if logits.is_empty() {
             return Err(tl_backend::BackendError::InternalError("Empty tensor for sampling".to_string()));
        }

        if temp <= 0.0 {
             // Greedy: argmax
             let max_idx = logits.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i).unwrap_or(0);
             return Ok(CpuTensor::from_slice(&[max_idx as f32], &[1], DType::F32));
        }

        // Apply temperature
        let probs: Vec<f32> = logits.iter().map(|&x| x / temp).collect();
        // Softmax
        let max_val = probs.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let sum_exp: f32 = probs.iter().map(|x| (x - max_val).exp()).sum();
        let probs: Vec<f32> = probs.iter().map(|x| (x - max_val).exp() / sum_exp).collect();

        // Top-p sampling
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut cum_prob = 0.0;
        let mut cutoff_index = indices.len() - 1;
        for (i, &idx) in indices.iter().enumerate() {
            cum_prob += probs[idx];
            if cum_prob > top_p {
                cutoff_index = i;
                break;
            }
        }
        
        // Random sample from top-p
        let cutoff_prob = cum_prob;
        let mut rng = rand::thread_rng();
        use rand::Rng; // Ensure Rng trait is used if available. Wait, need to check if Rng is in scope or just rand::random
        // Or simpler: just use weighted choice from filtered probs
        // Re-normalize top-p probabilities?
        // Usually: sample r ~ [0, top_p_sum] and select
        
        // Simplified sampling for CPU (mostly for debug/small models):
        // Just sample from full distribution if top_p is close to 1.0, else cutoff
        
        let r: f32 = rng.gen::<f32>() * cutoff_prob;
        let mut acc = 0.0;
        let mut selected_idx = indices[0];
        
        for i in 0..=cutoff_index {
             let idx = indices[i];
             acc += probs[idx];
             if acc >= r {
                 selected_idx = idx;
                 break;
             }
        }
        
        Ok(CpuTensor::from_slice(&[selected_idx as f32], &[1], DType::F32))
    }
}
