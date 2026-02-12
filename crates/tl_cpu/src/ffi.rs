//! CPU 版 FFI 関数
//! GPU 版 (tl_runtime/src/lib.rs) と同じ C シグネチャで CpuTensor を操作。
//! JIT の add_global_mapping で同じシンボル名にマッピングされる。

use crate::tensor::{CpuTensor, tensor_ref_from_ptr};
use crate::DType;
use std::sync::Arc;
use std::cell::UnsafeCell;

type OpaqueTensor = CpuTensor;

// ========== テンソル作成 ==========

pub extern "C" fn tl_cpu_tensor_new(
    data: *const f32, rank: usize, shape: *const usize,
) -> *mut OpaqueTensor {
    if data.is_null() || shape.is_null() { return std::ptr::null_mut(); }
    let shape_slice = unsafe { std::slice::from_raw_parts(shape, rank) };
    let numel: usize = shape_slice.iter().product();
    let data_slice = unsafe { std::slice::from_raw_parts(data, numel) };
    make_tensor(CpuTensor::from_slice(data_slice, shape_slice, DType::F32))
}

pub extern "C" fn tl_cpu_tensor_new_i64(
    data: *const i64, rank: usize, shape: *const usize,
) -> *mut OpaqueTensor {
    if data.is_null() || shape.is_null() { return std::ptr::null_mut(); }
    let shape_slice = unsafe { std::slice::from_raw_parts(shape, rank) };
    let numel: usize = shape_slice.iter().product();
    let data_slice = unsafe { std::slice::from_raw_parts(data, numel) };
    let f32_data: Vec<f32> = data_slice.iter().map(|&x| x as f32).collect();
    make_tensor(CpuTensor::from_slice(&f32_data, shape_slice, DType::F32))
}

pub extern "C" fn tl_cpu_tensor_zeros(rank: usize, shape: *const usize, req_grad: bool) -> *mut OpaqueTensor {
    if shape.is_null() { return std::ptr::null_mut(); }
    let shape_slice = unsafe { std::slice::from_raw_parts(shape, rank) };
    let t = CpuTensor::zeros(shape_slice, DType::F32);
    let ptr = make_tensor(t);
    if req_grad { unsafe { (&mut *ptr).enable_grad(); } }
    ptr
}

pub extern "C" fn tl_cpu_tensor_ones(rank: usize, shape: *const usize, req_grad: bool) -> *mut OpaqueTensor {
    if shape.is_null() { return std::ptr::null_mut(); }
    let shape_slice = unsafe { std::slice::from_raw_parts(shape, rank) };
    let mut t = CpuTensor::ones(shape_slice, DType::F32);
    
    if req_grad {
        t.enable_grad();
    }
    
    make_tensor(t)
}

pub extern "C" fn tl_cpu_tensor_randn(rank: usize, shape: *const usize, req_grad: bool) -> *mut OpaqueTensor {
    if shape.is_null() { return std::ptr::null_mut(); }
    let shape_slice = unsafe { std::slice::from_raw_parts(shape, rank) };
    let t = CpuTensor::randn(shape_slice, DType::F32);
    let ptr = make_tensor(t);
    if req_grad { unsafe { (&mut *ptr).enable_grad(); } }
    ptr
}

pub extern "C" fn tl_cpu_tensor_randn_debug(
    rank: usize, shape: *const usize, _seed: u64, req_grad: bool,
) -> *mut OpaqueTensor {
    tl_cpu_tensor_randn(rank, shape, req_grad)
}

pub extern "C" fn tl_cpu_tensor_from_i64(data: *const i64, len: i64) -> *mut OpaqueTensor {
    if data.is_null() { return std::ptr::null_mut(); }
    let len = len as usize;
    let data_slice = unsafe { std::slice::from_raw_parts(data, len) };
    let f32_data: Vec<f32> = data_slice.iter().map(|&x| x as f32).collect();
    make_tensor(CpuTensor::from_slice(&f32_data, &[len], DType::F32))
}

pub extern "C" fn tl_cpu_tensor_from_u8(data: *const u8, len: usize) -> *mut OpaqueTensor {
    if data.is_null() { return std::ptr::null_mut(); }
    let data_slice = unsafe { std::slice::from_raw_parts(data, len) };
    let f32_data: Vec<f32> = data_slice.iter().map(|&x| x as f32).collect();
    make_tensor(CpuTensor::from_slice(&f32_data, &[len], DType::F32))
}


// ========== テンソル解放 (Arc ベース) ==========
// Arc::into_raw で作成されたポインタを Arc::from_raw で復元し、
// 参照カウントを -1 する。RC=0 になれば CpuTensor（autograd グラフ含む）が
// 自然に Drop される。

pub extern "C" fn tl_cpu_tensor_free(t: *mut OpaqueTensor) {
    if t.is_null() { return; }
    crate::memory::promote_tensor(t);
    crate::memory::release_tensor(t);
}

#[no_mangle]
pub extern "C" fn tl_cpu_tensor_acquire(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    // Arc の参照カウントを +1 して同じポインタを返す
    if t.is_null() { return t; }
    if crate::memory::is_mem_log_enabled() {
        eprintln!("[ACQUIRE] Ptr: {:p} (Arc RC+1)", t);
    }
    unsafe {
        let arc = Arc::from_raw(t as *const UnsafeCell<CpuTensor>);
        let _clone = arc.clone();  // RC+1
        let _ = Arc::into_raw(arc);        // 元の参照を戻す
        Arc::into_raw(_clone) as *mut OpaqueTensor
    }
}

pub extern "C" fn tl_cpu_tensor_release(t: *mut OpaqueTensor) {
    if t.is_null() { return; }
    crate::memory::promote_tensor(t);
    crate::memory::release_tensor(t);
}

/// テンソルの内部データをクリアしてメモリを OS に返却する。
/// Arc ベースの所有権管理により、autograd グラフも安全にクリアできる。
/// GradFn 内の TensorRef (Arc) が他のテンソルを保持するため、
/// autograd をクリアしても参照先テンソルは Arc の RC で生存が保証される。
/// `tl_runtime::tl_tensor_release_safe` から呼ばれる。
pub extern "C" fn tl_cpu_tensor_clear_data(t: *mut OpaqueTensor) {
    if t.is_null() { return; }
    unsafe {
        let tensor = &mut *t;
        // データバッファを解放
        tensor.data_f32 = Vec::new();
        tensor.data_i64 = None;
        tensor.shape = Vec::new();
        // autograd も完全クリア（Arc が所有権を管理するため安全）
        tensor.autograd = None;
    }
}

// ========== テンソル情報 ==========

pub extern "C" fn tl_cpu_tensor_dim(t: *mut OpaqueTensor, dim: usize) -> usize {
    if t.is_null() { return 0; }
    unsafe { (&*t).shape().get(dim).cloned().unwrap_or(0) }
}

pub extern "C" fn tl_cpu_tensor_len(t: *mut OpaqueTensor) -> usize {
    if t.is_null() { return 0; }
    let tensor = unsafe { &*t };
    if tensor.shape().is_empty() { return 0; } // clear_data 済みテンソル
    tensor.shape().iter().product()
}

pub extern "C" fn tl_cpu_tensor_shape(t: *mut OpaqueTensor, out: *mut usize) -> usize {
    if t.is_null() || out.is_null() { return 0; }
    let tensor = unsafe { &*t };
    for (i, &dim) in tensor.shape().iter().enumerate() {
        unsafe { *out.add(i) = dim; }
    }
    tensor.shape().len()
}

pub extern "C" fn tl_cpu_tensor_get_shape(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let shape: Vec<f32> = tensor.shape().iter().map(|&d| d as f32).collect();
    make_tensor(CpuTensor::from_slice(&shape, &[shape.len()], DType::F32))
}

// ========== 要素アクセス ==========

pub extern "C" fn tl_cpu_tensor_get_f32(t: *mut OpaqueTensor, indices: *const usize, _rank: usize) -> f32 {
    if t.is_null() || indices.is_null() { return 0.0; }
    let tensor = unsafe { &*t };
    let shape = tensor.shape();
    if shape.is_empty() { return 0.0; } // clear_data 済みテンソル
    let idx_slice = unsafe { std::slice::from_raw_parts(indices, shape.len()) };
    let mut flat_idx = 0;
    let mut stride = 1;
    for i in (0..shape.len()).rev() {
        flat_idx += idx_slice[i] * stride;
        stride *= shape[i];
    }
    tensor.data_f32().get(flat_idx).cloned().unwrap_or(0.0)
}

pub extern "C" fn tl_cpu_tensor_get_i64(t: *mut OpaqueTensor, indices: *const usize, rank: usize) -> i64 {
    if t.is_null() { return 0; }
    let tensor = unsafe { &*t };
    if tensor.shape().is_empty() { return 0; } // clear_data 済みテンソル
    tl_cpu_tensor_get_f32(t, indices, rank) as i64
}

/// runtime::tl_tensor_get_f32_md と同じシグネチャ (多次元インデックスアクセス)
/// LLVM 宣言: (tensor: *mut, indices: *const i64, rank: i64) -> f32
pub extern "C" fn tl_cpu_tensor_get_f32_md(t: *mut OpaqueTensor, indices: *const i64, rank: i64) -> f32 {
    if t.is_null() || indices.is_null() { return 0.0; }
    let tensor = unsafe { &*(t as *mut CpuTensor) };
    let shape = tensor.shape();
    if shape.is_empty() { return 0.0; }
    let data = tensor.data_f32();
    let rank = rank as usize;
    // indices から各次元のインデックスを読み取り、フラットインデックスを計算
    let mut flat_idx = 0usize;
    for d in 0..rank.min(shape.len()) {
        let idx = unsafe { *indices.add(d) } as usize;
        let stride: usize = shape[d+1..].iter().product();
        flat_idx += idx * stride.max(1);
    }
    data.get(flat_idx).cloned().unwrap_or(0.0)
}

/// runtime::tl_tensor_get_i64_md と同じシグネチャ
pub extern "C" fn tl_cpu_tensor_get_i64_md(t: *mut OpaqueTensor, indices: *const i64, rank: i64) -> i64 {
    tl_cpu_tensor_get_f32_md(t, indices, rank) as i64
}

pub extern "C" fn tl_cpu_tensor_set_f32(t: *mut OpaqueTensor, indices: *const usize, _rank: usize, value: f32) {
    if t.is_null() || indices.is_null() { return; }
    let tensor = unsafe { &mut *t };
    let shape = tensor.shape().to_vec();
    if shape.is_empty() { return; } // clear_data 済みテンソル
    let idx_slice = unsafe { std::slice::from_raw_parts(indices, shape.len()) };
    let mut flat_idx = 0;
    let mut stride = 1;
    for i in (0..shape.len()).rev() {
        flat_idx += idx_slice[i] * stride;
        stride *= shape[i];
    }
    let data = tensor.data_f32_mut();
    if flat_idx < data.len() {
        data[flat_idx] = value;
    }
}

/// LLVM 宣言: (t: *mut, indices: *const i64, rank: usize, val: f32) → *mut
/// runtime 版と同じ署名を持つ CPU 版。in-place で書き込み、同じポインタを返す。
pub extern "C" fn tl_cpu_tensor_set_f32_md(
    t: *mut OpaqueTensor,
    indices: *const i64,
    rank: usize,
    value: f32,
) -> *mut OpaqueTensor {
    if t.is_null() || indices.is_null() { return t; }
    let tensor = unsafe { &mut *t };
    let shape = tensor.shape().to_vec();
    if shape.is_empty() { return t; } // clear_data 済みテンソル
    let idx_slice = unsafe { std::slice::from_raw_parts(indices, rank) };
    let mut flat_idx = 0usize;
    let mut stride = 1usize;
    for d in (0..rank).rev() {
        let i = idx_slice[d] as usize;
        if d < shape.len() && i < shape[d] {
            flat_idx += i * stride;
            stride *= shape[d];
        }
    }
    let data = tensor.data_f32_mut();
    if flat_idx < data.len() {
        data[flat_idx] = value;
    }
    t
}

/// for ループ等で使われる flat index アクセス。
/// runtime::tl_tensor_get と同じシグネチャ (t, idx: i64) -> f32
pub extern "C" fn tl_cpu_tensor_get(t: *mut OpaqueTensor, idx: i64) -> f32 {
    if t.is_null() { return 0.0; }
    let tensor = unsafe { &*t };
    tensor.data_f32().get(idx as usize).copied().unwrap_or(0.0)
}

pub extern "C" fn tl_cpu_tensor_item(t: *mut OpaqueTensor) -> f32 {
    if t.is_null() { return 0.0; }
    let tensor = unsafe { &*t };
    tensor.data_f32().first().copied().unwrap_or(0.0)
}

pub extern "C" fn tl_cpu_tensor_item_i64(t: *mut OpaqueTensor) -> i64 {
    if t.is_null() { return 0; }
    let tensor = unsafe { &*t };
    tensor.data_f32().first().map(|&f| f as i64).unwrap_or(0)
}

// ========== 基本演算 ==========

pub extern "C" fn tl_cpu_tensor_add(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    let (ta, tb) = unsafe { (&*a, &*b) };
    let result = ta.add_impl(tb);
    let ptr = make_tensor(result);
    if ta.requires_grad() || tb.requires_grad() {
        use crate::autograd::ops::AddBackward;
        unsafe { (&mut *ptr).set_grad_fn(Box::new(AddBackward {
            a: tensor_ref_from_ptr(a), b: tensor_ref_from_ptr(b),
            a_shape: ta.shape().to_vec(),
            b_shape: tb.shape().to_vec(),
        })); }
    }
    ptr
}

pub extern "C" fn tl_cpu_tensor_sub(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    let (ta, tb) = unsafe { (&*a, &*b) };
    let result = ta.sub_impl(tb);
    let ptr = make_tensor(result);
    if ta.requires_grad() || tb.requires_grad() {
        use crate::autograd::ops::SubBackward;
        unsafe { (&mut *ptr).set_grad_fn(Box::new(SubBackward {
            a: tensor_ref_from_ptr(a), b: tensor_ref_from_ptr(b),
            a_shape: ta.shape().to_vec(),
            b_shape: tb.shape().to_vec(),
        })); }
    }
    ptr
}

pub extern "C" fn tl_cpu_tensor_mul(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    let (ta, tb) = unsafe { (&*a, &*b) };
    let result = ta.mul_impl(tb);
    let ptr = make_tensor(result);
    if ta.requires_grad() || tb.requires_grad() {
        use crate::autograd::ops::MulBackward;
        unsafe { (&mut *ptr).set_grad_fn(Box::new(MulBackward {
            a: tensor_ref_from_ptr(a), b: tensor_ref_from_ptr(b), a_data: ta.shallow_clone(), b_data: tb.shallow_clone(),
            a_shape: ta.shape().to_vec(), b_shape: tb.shape().to_vec(),
        })); }
    }
    ptr
}

pub extern "C" fn tl_cpu_tensor_div(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    let (ta, tb) = unsafe { (&*a, &*b) };
    let result = ta.div_impl(tb);
    let ptr = make_tensor(result);
    if ta.requires_grad() || tb.requires_grad() {
        use crate::autograd::ops::DivBackward;
        unsafe { (&mut *ptr).set_grad_fn(Box::new(DivBackward {
            a: tensor_ref_from_ptr(a), b: tensor_ref_from_ptr(b), a_data: ta.shallow_clone(), b_data: tb.shallow_clone(),
            a_shape: ta.shape().to_vec(), b_shape: tb.shape().to_vec(),
        })); }
    }
    ptr
}

pub extern "C" fn tl_cpu_tensor_rem(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    let (ta, tb) = unsafe { (&*a, &*b) };
    make_tensor(ta.rem_impl(tb))
}

pub extern "C" fn tl_cpu_tensor_neg(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let res = tensor.neg_impl();
    let ptr = make_tensor(res);
    if tensor.requires_grad() {
        use crate::autograd::ops::NegBackward;
        unsafe { (&mut *ptr).set_grad_fn(Box::new(NegBackward { a: tensor_ref_from_ptr(t) })); }
    }
    ptr
}

pub extern "C" fn tl_cpu_tensor_abs(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    make_tensor(unsafe { (&*t).abs_impl() })
}

// ========== スカラー演算 ==========

pub extern "C" fn tl_cpu_tensor_add_scalar(t: *mut OpaqueTensor, s: f64) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let res = tensor.add_scalar_impl(s as f32);
    let ptr = make_tensor(res);
    if tensor.requires_grad() {
        use crate::autograd::ops::AddScalarBackward;
        unsafe { (&mut *ptr).set_grad_fn(Box::new(AddScalarBackward { a: tensor_ref_from_ptr(t) })); }
    }
    ptr
}

pub extern "C" fn tl_cpu_tensor_mul_scalar(t: *mut OpaqueTensor, s: f64) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let res = tensor.mul_scalar_impl(s as f32);
    let ptr = make_tensor(res);
    if tensor.requires_grad() {
        use crate::autograd::ops::MulScalarBackward;
        unsafe { (&mut *ptr).set_grad_fn(Box::new(MulScalarBackward { a: tensor_ref_from_ptr(t), s: s as f32 })); }
    }
    // println!("tl_cpu_tensor_mul_scalar: returning {:p}", ptr);
    ptr
}

pub extern "C" fn tl_cpu_tensor_div_scalar(t: *mut OpaqueTensor, s: f64) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let res = tensor.div_scalar_impl(s as f32);
    let ptr = make_tensor(res);
    if tensor.requires_grad() {
        use crate::autograd::ops::DivScalarBackward;
        unsafe { (&mut *ptr).set_grad_fn(Box::new(DivScalarBackward { a: tensor_ref_from_ptr(t), s: s as f32 })); }
    }
    ptr
}

pub extern "C" fn tl_cpu_tensor_sub_scalar(t: *mut OpaqueTensor, s: f64) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let res = tensor.sub_scalar_impl(s as f32);
    let ptr = make_tensor(res);
    if tensor.requires_grad() {
        use crate::autograd::ops::SubScalarBackward;
        unsafe { (&mut *ptr).set_grad_fn(Box::new(SubScalarBackward { a: tensor_ref_from_ptr(t) })); }
    }
    ptr
}

pub extern "C" fn tl_cpu_tensor_pow_scalar(t: *mut OpaqueTensor, exp: f32) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let exp_tensor = CpuTensor::from_slice(&[exp], &[1], DType::F32);
    let result = tensor.pow_impl(&exp_tensor);
    let ptr = make_tensor(result);
    if tensor.requires_grad() {
        use crate::autograd::ops::PowBackward;
        let result_clone = unsafe { (&*ptr).shallow_clone() };
        unsafe { (&mut *ptr).set_grad_fn(Box::new(PowBackward {
            a: tensor_ref_from_ptr(t), a_data: tensor.shallow_clone(), b_data: exp_tensor, output: result_clone,
        })); }
    }
    ptr
}

/// runtime::tl_tensor_pow と同じシグネチャ (テンソル同士の pow)
pub extern "C" fn tl_cpu_tensor_pow(t: *mut OpaqueTensor, exp: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() || exp.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let exp_tensor = unsafe { &*exp };
    let result = tensor.pow_impl(exp_tensor);
    let ptr = make_tensor(result);
    if tensor.requires_grad() {
        use crate::autograd::ops::PowBackward;
        let result_clone = unsafe { (&*ptr).shallow_clone() };
        unsafe { (&mut *ptr).set_grad_fn(Box::new(PowBackward {
            a: tensor_ref_from_ptr(t), a_data: tensor.shallow_clone(), b_data: exp_tensor.shallow_clone(), output: result_clone,
        })); }
    }
    ptr
}

// ========== インプレース演算 ==========

pub extern "C" fn tl_cpu_tensor_add_assign(a: *mut OpaqueTensor, b: *mut OpaqueTensor) {
    if a.is_null() || b.is_null() { return; }
    unsafe { *a = (&*a).add_impl(&*b); }
}

pub extern "C" fn tl_cpu_tensor_sub_assign(a: *mut OpaqueTensor, b: *mut OpaqueTensor) {
    if a.is_null() || b.is_null() { return; }
    unsafe { *a = (&*a).sub_impl(&*b); }
}

pub extern "C" fn tl_cpu_tensor_mul_assign(a: *mut OpaqueTensor, b: *mut OpaqueTensor) {
    if a.is_null() || b.is_null() { return; }
    unsafe { *a = (&*a).mul_impl(&*b); }
}

pub extern "C" fn tl_cpu_tensor_div_assign(a: *mut OpaqueTensor, b: *mut OpaqueTensor) {
    if a.is_null() || b.is_null() { return; }
    unsafe { *a = (&*a).div_impl(&*b); }
}

pub extern "C" fn tl_cpu_tensor_mod_assign(a: *mut OpaqueTensor, b: *mut OpaqueTensor) {
    if a.is_null() || b.is_null() { return; }
    unsafe { *a = (&*a).rem_impl(&*b); }
}

pub extern "C" fn tl_cpu_tensor_add_assign_scalar_f32(t: *mut OpaqueTensor, s: f32) {
    if t.is_null() { return; }
    unsafe { *t = (&*t).add_scalar_impl(s); }
}

pub extern "C" fn tl_cpu_tensor_sub_assign_scalar_f32(t: *mut OpaqueTensor, s: f32) {
    if t.is_null() { return; }
    unsafe { *t = (&*t).add_scalar_impl(-s); }
}

pub extern "C" fn tl_cpu_tensor_mul_assign_scalar_f32(t: *mut OpaqueTensor, s: f32) {
    if t.is_null() { return; }
    unsafe { *t = (&*t).mul_scalar_impl(s); }
}

pub extern "C" fn tl_cpu_tensor_div_assign_scalar_f32(t: *mut OpaqueTensor, s: f32) {
    if t.is_null() { return; }
    unsafe { *t = (&*t).div_scalar_impl(s); }
}

pub extern "C" fn tl_cpu_tensor_mod_assign_scalar_f32(t: *mut OpaqueTensor, s: f32) {
    if t.is_null() { return; }
    unsafe {
        let a = &*t;
        let b = CpuTensor::from_slice(&[s], &[1], DType::F32);
        *t = a.rem_impl(&b);
    }
}

// ========== 比較演算 ==========

pub extern "C" fn tl_cpu_tensor_eq(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    make_tensor(unsafe { (&*a).eq_impl(&*b) })
}

pub extern "C" fn tl_cpu_tensor_neq(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    make_tensor(unsafe { (&*a).neq_impl(&*b) })
}

pub extern "C" fn tl_cpu_tensor_lt(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    make_tensor(unsafe { (&*a).lt_impl(&*b) })
}

pub extern "C" fn tl_cpu_tensor_le(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    make_tensor(unsafe { (&*a).le_impl(&*b) })
}

pub extern "C" fn tl_cpu_tensor_gt(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    make_tensor(unsafe { (&*a).gt_impl(&*b) })
}

pub extern "C" fn tl_cpu_tensor_ge(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    make_tensor(unsafe { (&*a).ge_impl(&*b) })
}

// ========== 数学関数 ==========

pub extern "C" fn tl_cpu_tensor_exp(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let result = tensor.exp_impl();
    let ptr = make_tensor(result);
    if tensor.requires_grad() {
        use crate::autograd::ops::ExpBackward;
        let output = unsafe { (&*ptr).shallow_clone() };
        unsafe { (&mut *ptr).set_grad_fn(Box::new(ExpBackward { a: tensor_ref_from_ptr(t), output })); }
    }
    ptr
}

pub extern "C" fn tl_cpu_tensor_log(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let result = tensor.log_impl();
    let ptr = make_tensor(result);
    if tensor.requires_grad() {
        use crate::autograd::ops::LogBackward;
        unsafe { (&mut *ptr).set_grad_fn(Box::new(LogBackward { a: tensor_ref_from_ptr(t), a_data: tensor.shallow_clone() })); }
    }
    ptr
}

pub extern "C" fn tl_cpu_tensor_sin(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    make_tensor(unsafe { (&*t).sin_impl() })
}

pub extern "C" fn tl_cpu_tensor_cos(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    make_tensor(unsafe { (&*t).cos_impl() })
}

pub extern "C" fn tl_cpu_tensor_tanh(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    make_tensor(unsafe { (&*t).tanh_impl() })
}

pub extern "C" fn tl_cpu_tensor_relu(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let result = tensor.relu_impl();
    let ptr = make_tensor(result);
    if tensor.requires_grad() {
        use crate::autograd::ops::ReluBackward;
        unsafe { (&mut *ptr).set_grad_fn(Box::new(ReluBackward { a: tensor_ref_from_ptr(t), a_data: tensor.shallow_clone() })); }
    }
    ptr
}

pub extern "C" fn tl_cpu_tensor_softmax(t: *mut OpaqueTensor, dim: i64) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let result = tensor.softmax_impl(dim as i32);
    let ptr = make_tensor(result);
    if tensor.requires_grad() {
        use crate::autograd::ops::SoftmaxBackward;
        let output = unsafe { (&*ptr).shallow_clone() };
        unsafe { (&mut *ptr).set_grad_fn(Box::new(SoftmaxBackward { a: tensor_ref_from_ptr(t), output, axis: dim as i32 })); }
    }
    ptr
}

// ========== Reduction ==========

pub extern "C" fn tl_cpu_tensor_sum(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let sum_val = tensor.sumall_impl();
    let result = CpuTensor::from_slice(&[sum_val], &[1], DType::F32);
    let ptr = make_tensor(result);
    if tensor.requires_grad() {
        use crate::autograd::ops::SumallBackward;
        unsafe { (&mut *ptr).set_grad_fn(Box::new(SumallBackward { a: tensor_ref_from_ptr(t), shape: tensor.shape().to_vec() })); }
    }
    ptr
}

pub extern "C" fn tl_cpu_tensor_mean(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let mean_val = tensor.mean_all_impl();
    make_tensor(CpuTensor::from_slice(&[mean_val], &[1], DType::F32))
}

pub extern "C" fn tl_cpu_tensor_max(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let idx = tensor.argmax_all_impl();
    let max_val = tensor.data_f32()[idx];
    make_tensor(CpuTensor::from_slice(&[max_val], &[1], DType::F32))
}

pub extern "C" fn tl_cpu_tensor_min(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let data = tensor.data_f32();
    let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
    make_tensor(CpuTensor::from_slice(&[min_val], &[1], DType::F32))
}

pub extern "C" fn tl_cpu_tensor_argmax(t: *mut OpaqueTensor, dim: i64, _keep_dim: bool) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let _ = dim; // 現在は全要素の argmax のみ実装
    let max_idx = tensor.argmax_all_impl();
    make_tensor(CpuTensor::from_slice(&[max_idx as f32], &[1], DType::F32))
}

pub extern "C" fn tl_cpu_tensor_argmin(t: *mut OpaqueTensor, dim: i64, _keep_dim: bool) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let _ = dim;
    let data = tensor.data_f32();
    let min_idx = data.iter().enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i).unwrap_or(0);
    make_tensor(CpuTensor::from_slice(&[min_idx as f32], &[1], DType::F32))
}

// ========== Shape 操作 ==========

pub extern "C" fn tl_cpu_tensor_reshape(t: *mut OpaqueTensor, rank: usize, shape: *const usize) -> *mut OpaqueTensor {
    if t.is_null() || shape.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let new_shape = unsafe { std::slice::from_raw_parts(shape, rank) };
    let input_shape = tensor.shape().to_vec();
    let ptr = make_tensor(tensor.reshape_impl(new_shape));
    if tensor.requires_grad() {
        use crate::autograd::ops::ReshapeBackward;
        unsafe { (&mut *ptr).set_grad_fn(Box::new(ReshapeBackward {
            input: tensor_ref_from_ptr(t),
            input_shape,
        })); }
    }
    ptr
}

pub extern "C" fn tl_cpu_tensor_reshape_new(t: *mut OpaqueTensor, new_shape_tensor: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() || new_shape_tensor.is_null() { return std::ptr::null_mut(); }
    let shape_tensor = unsafe { &*new_shape_tensor };
    let new_shape: Vec<usize> = shape_tensor.data_f32().iter().map(|&x| x as usize).collect();
    tl_cpu_tensor_reshape(t, new_shape.len(), new_shape.as_ptr())
}

pub extern "C" fn tl_cpu_tensor_reshape_dims(t: *mut OpaqueTensor, dim1: i64, dim2: i64, dim3: i64, dim4: i64) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let mut new_shape = Vec::new();
    if dim1 > 0 { new_shape.push(dim1 as usize); }
    if dim2 > 0 { new_shape.push(dim2 as usize); }
    if dim3 > 0 { new_shape.push(dim3 as usize); }
    if dim4 > 0 { new_shape.push(dim4 as usize); }
    make_tensor(tensor.reshape_impl(&new_shape))
}

pub extern "C" fn tl_cpu_tensor_transpose(t: *mut OpaqueTensor, dim0: usize, dim1: usize) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    make_tensor(unsafe { (&*t).transpose_impl(dim0, dim1) })
}

pub extern "C" fn tl_cpu_tensor_squeeze(t: *mut OpaqueTensor, dim: i64) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    make_tensor(unsafe { (&*t).squeeze_impl(dim as usize) })
}

pub extern "C" fn tl_cpu_tensor_unsqueeze(t: *mut OpaqueTensor, dim: usize) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    make_tensor(unsafe { (&*t).unsqueeze_impl(dim) })
}

pub extern "C" fn tl_cpu_tensor_contiguous(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    make_tensor(unsafe { (&*t).contiguous_impl() })
}

pub extern "C" fn tl_cpu_tensor_narrow(t: *mut OpaqueTensor, dim: usize, start: usize, len: usize) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    make_tensor(unsafe { (&*t).narrow_impl(dim, start, len) })
}

pub extern "C" fn tl_cpu_tensor_slice(t: *mut OpaqueTensor, dim: i64, start: i64, len: i64) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    make_tensor(unsafe { (&*t).narrow_impl(dim as usize, start as usize, len as usize) })
}

pub extern "C" fn tl_cpu_tensor_cat(a: *mut OpaqueTensor, b: *mut OpaqueTensor, dim: i64) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    let (ta, tb) = unsafe { (&*a, &*b) };
    make_tensor(CpuTensor::cat_impl(&[ta, tb], dim as usize))
}

pub extern "C" fn tl_cpu_tensor_cat_i64(a: *mut OpaqueTensor, b: *mut OpaqueTensor, dim: i64) -> *mut OpaqueTensor {
    tl_cpu_tensor_cat(a, b, dim)
}

// ========== Clone/Print ==========

pub extern "C" fn tl_cpu_tensor_clone(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    make_tensor(unsafe { (&*t).clone() })
}

pub extern "C" fn tl_cpu_tensor_print(t: *mut OpaqueTensor) {
    if t.is_null() { println!("Tensor[null]"); return; }
    let tensor = unsafe { &*t };
    if tensor.shape().is_empty() { println!("Tensor[cleared]"); return; }
    println!("Tensor(shape={:?}, data={:?})", tensor.shape(), tensor.data_f32());
}

pub extern "C" fn tl_cpu_tensor_replace_data(dst: *mut OpaqueTensor, src: *mut OpaqueTensor) {
    if dst.is_null() || src.is_null() { return; }
    unsafe { *dst = (&*src).clone(); }
}

// ========== MatMul ==========

pub extern "C" fn tl_cpu_tensor_matmul(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    let (ta, tb) = unsafe { (&*a, &*b) };
    let result = ta.matmul_impl(tb);
    let ptr = make_tensor(result);
    if ta.requires_grad() || tb.requires_grad() {
        use crate::autograd::ops::MatmulBackward;
        unsafe { (&mut *ptr).set_grad_fn(Box::new(MatmulBackward {
            a: tensor_ref_from_ptr(a), b: tensor_ref_from_ptr(b), a_data: ta.shallow_clone(), b_data: tb.shallow_clone(),
        })); }
    }
    ptr
}

// ========== 高度テンソル関数 ==========

pub extern "C" fn tl_cpu_tensor_sqrt(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    make_tensor(unsafe { (&*t).sqrt_impl() })
}

pub extern "C" fn tl_cpu_tensor_gelu(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    make_tensor(unsafe { (&*t).gelu_impl() })
}

pub extern "C" fn tl_cpu_tensor_tril(t: *mut OpaqueTensor, diagonal: i64) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    if tensor.shape().len() < 2 {
        return make_tensor(tensor.clone());
    }
    make_tensor(tensor.tril_impl(diagonal as i32))
}

pub extern "C" fn tl_cpu_tensor_sum_dim(t: *mut OpaqueTensor, dim: usize, keep_dim: bool) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let mut result = tensor.sum_impl(dim as i32);
    if keep_dim {
        let mut new_shape = tensor.shape().to_vec();
        new_shape[dim] = 1;
        result = result.reshape_impl(&new_shape);
    }
    let ptr = make_tensor(result);
    if tensor.requires_grad() {
        use crate::autograd::ops::SumDimBackward;
        unsafe { (&mut *ptr).set_grad_fn(Box::new(SumDimBackward {
            a: tensor_ref_from_ptr(t),
            input_shape: tensor.shape().to_vec(),
            axis: dim as i32,
        })); }
    }
    ptr
}

pub extern "C" fn tl_cpu_tensor_embedding(
    weight: *mut OpaqueTensor,
    indices: *mut OpaqueTensor,
    _padding_idx: i64,
    _scale_grad_by_freq: bool,
    _sparse: bool,
) -> *mut OpaqueTensor {
    if weight.is_null() || indices.is_null() { return std::ptr::null_mut(); }
    let weights_tensor = unsafe { &*weight };
    let indices_tensor = unsafe { &*indices };
    make_tensor(weights_tensor.embedding_impl(indices_tensor))
}

// ========== Init/Shutdown ==========

pub extern "C" fn tl_cpu_runtime_init() {
    println!("Runtime device initialized: CPU");
}

pub extern "C" fn tl_cpu_runtime_shutdown() {}

// ========== Autograd FFI ==========

pub extern "C" fn tl_cpu_tensor_backward(t: *mut OpaqueTensor) {
    if t.is_null() { return; }
    let tensor = unsafe { &mut *t };
    tensor.backward();
}

pub extern "C" fn tl_cpu_tensor_grad(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    if let Some(grad) = tensor.get_grad() {
        return make_tensor(grad);
    }
    // フォールバック: ゼロテンソル
    make_tensor(CpuTensor::zeros(tensor.shape(), DType::F32))
}

pub extern "C" fn tl_cpu_tensor_detach(t: *mut OpaqueTensor, _req_grad: bool) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let result = make_tensor(tensor.detach());
    
    // Previous optimization: clear_autograd_graph()
    // This was removed because it destroys the source tensor's data. If the source tensor is reused 
    // (due to aggressive release/recycle), this clears the *result* tensor too (if aliased).
    // Cleanup is now handled by standard release_safe (which now supports autograd tensors).
    
    result
}

/// LLVM IR宣言は `(t) -> void`
pub extern "C" fn tl_cpu_tensor_enable_grad(t: *mut OpaqueTensor) {
    if t.is_null() { return; }
    let tensor = unsafe { &mut *t };
    tensor.enable_grad();
}

/// CPU版では noop（CPU autograd はグローバル grad 管理をしないため）
pub extern "C" fn tl_cpu_clear_grads() {
    // noop
}

// ========== Scope Management (V4.5) ==========

#[no_mangle]
pub extern "C" fn tl_cpu_enter_scope() {
    crate::memory::enter_scope();
}





#[no_mangle]
pub extern "C" fn tl_cpu_exit_scope() {
    crate::memory::exit_scope();
}

#[no_mangle]
pub extern "C" fn tl_cpu_tensor_promote(t: *mut OpaqueTensor) {
    crate::memory::promote_tensor(t);
}

#[no_mangle]
pub extern "C" fn tl_cpu_tensor_register(t: *mut OpaqueTensor) {
    crate::memory::register_tensor(t);
}

#[no_mangle]
pub extern "C" fn tl_cpu_tensor_return_to_pool(t: *mut OpaqueTensor) {
    if t.is_null() { return; }
    crate::memory::release_tensor(t);
}

// ========== Phase 2: テスト影響の大きい新規実装 ==========

pub extern "C" fn tl_cpu_tensor_tan(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let data: Vec<f32> = tensor.data_f32.iter().map(|&x| x.tan()).collect();
    make_tensor(CpuTensor { data_f32: data, data_i64: None, shape: tensor.shape.clone(), dtype: tensor.dtype, autograd: None })
}

pub extern "C" fn tl_cpu_tensor_clamp(t: *mut OpaqueTensor, min: f64, max: f64) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let min_f32 = min as f32;
    let max_f32 = max as f32;
    let data: Vec<f32> = tensor.data_f32.iter().map(|&x| x.max(min_f32).min(max_f32)).collect();
    make_tensor(CpuTensor { data_f32: data, data_i64: None, shape: tensor.shape.clone(), dtype: tensor.dtype, autograd: None })
}

pub extern "C" fn tl_cpu_tensor_sigmoid(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let data: Vec<f32> = tensor.data_f32.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
    make_tensor(CpuTensor { data_f32: data, data_i64: None, shape: tensor.shape.clone(), dtype: tensor.dtype, autograd: None })
}

pub extern "C" fn tl_cpu_tensor_scale(t: *mut OpaqueTensor, s: f64) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    make_tensor(tensor.mul_scalar_impl(s as f32))
}

pub extern "C" fn tl_cpu_tensor_silu(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let data: Vec<f32> = tensor.data_f32.iter().map(|&x| x * (1.0 / (1.0 + (-x).exp()))).collect();
    make_tensor(CpuTensor { data_f32: data, data_i64: None, shape: tensor.shape.clone(), dtype: tensor.dtype, autograd: None })
}

pub extern "C" fn tl_cpu_tensor_cross_entropy(
    logits: *mut OpaqueTensor,
    labels: *mut OpaqueTensor,
) -> *mut OpaqueTensor {
    if logits.is_null() || labels.is_null() { return std::ptr::null_mut(); }
    let l = unsafe { &*logits };
    let t = unsafe { &*labels };
    // Cross entropy: -sum(target * log(softmax(logits)))
    // Simple implementation: sum of element-wise -target * log(logit)
    let l_data = &l.data_f32;
    let t_data = &t.data_f32;
    let len = l_data.len().min(t_data.len());
    let mut loss = 0.0f32;
    for i in 0..len {
        let p = l_data[i].max(1e-7); // avoid log(0)
        loss -= t_data[i] * p.ln();
    }
    make_tensor(CpuTensor { data_f32: vec![loss], data_i64: None, shape: vec![1], dtype: DType::F32, autograd: None })
}

pub extern "C" fn tl_cpu_tensor_numel(t: *mut OpaqueTensor) -> i64 {
    if t.is_null() { return 0; }
    let tensor = unsafe { &*t };
    tensor.shape.iter().product::<usize>() as i64
}

pub extern "C" fn tl_cpu_tensor_transpose_2d(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let ndim = tensor.shape.len();
    if ndim < 2 { return make_tensor(tensor.clone_data()); }
    make_tensor(tensor.transpose_impl(ndim - 2, ndim - 1))
}

pub extern "C" fn tl_cpu_tensor_to_f32(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    // Already f32 for CpuTensor, just clone
    make_tensor(tensor.clone_data())
}

pub extern "C" fn tl_cpu_tensor_to_i64(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    // Convert f32 data to i64 representation
    let i64_data: Vec<i64> = tensor.data_f32.iter().map(|&x| x as i64).collect();
    let f32_data: Vec<f32> = i64_data.iter().map(|&x| x as f32).collect();
    make_tensor(CpuTensor { data_f32: f32_data, data_i64: Some(i64_data), shape: tensor.shape.clone(), dtype: DType::I64, autograd: None })
}

pub extern "C" fn tl_cpu_tensor_max_dim(t: *mut OpaqueTensor, dim: usize, _keep_dim: bool) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    make_tensor(tensor.max_impl(dim as i32))
}

pub extern "C" fn tl_cpu_tensor_min_dim(t: *mut OpaqueTensor, dim: usize, _keep_dim: bool) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    make_tensor(tensor.min_impl(dim as i32))
}

pub extern "C" fn tl_cpu_tensor_mean_dim(t: *mut OpaqueTensor, dim: usize, _keep_dim: bool) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    make_tensor(tensor.mean_impl(dim as i32))
}

pub extern "C" fn tl_cpu_tensor_device_id(_t: *mut OpaqueTensor) -> i32 {
    0 // Always CPU
}

pub extern "C" fn tl_cpu_tensor_to_device(t: *mut OpaqueTensor, _device_id: i32) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    make_tensor(tensor.clone_data())
}

pub extern "C" fn tl_cpu_tensor_prepare_return(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    // Identity / passthrough for CPU
    t
}

pub extern "C" fn tl_cpu_tensor_reshape_2d(t: *mut OpaqueTensor, d0: i64, d1: i64) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    make_tensor(tensor.reshape_impl(&[d0 as usize, d1 as usize]))
}

pub extern "C" fn tl_cpu_tensor_reshape_3d_to_2d(t: *mut OpaqueTensor, d0: i64, d1: i64) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    make_tensor(tensor.reshape_impl(&[d0 as usize, d1 as usize]))
}

pub extern "C" fn tl_cpu_tensor_sample(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    // Return argmax as a simple sampling strategy
    let data = &tensor.data_f32;
    let max_idx = data.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).map(|(i, _)| i).unwrap_or(0);
    make_tensor(CpuTensor { data_f32: vec![max_idx as f32], data_i64: None, shape: vec![1], dtype: DType::F32, autograd: None })
}

pub extern "C" fn tl_cpu_tensor_repeat_interleave(
    t: *mut OpaqueTensor,
    _repeats: usize,
    _dim: usize,
) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    make_tensor(tensor.clone_data()) // Placeholder
}

pub extern "C" fn tl_cpu_tensor_data(t: *mut OpaqueTensor) -> *const f32 {
    if t.is_null() { return std::ptr::null(); }
    let tensor = unsafe { &*t };
    tensor.data_f32.as_ptr()
}

pub extern "C" fn tl_cpu_tensor_new_causal_mask(size: usize) -> *mut OpaqueTensor {
    // Lower triangular matrix of 1s (size x size), zeros above diagonal
    let mut data = vec![0.0f32; size * size];
    for i in 0..size {
        for j in 0..=i {
            data[i * size + j] = 1.0;
        }
    }
    make_tensor(CpuTensor { data_f32: data, data_i64: None, shape: vec![size, size], dtype: DType::F32, autograd: None })
}

pub extern "C" fn tl_cpu_tensor_from_vec_u8(data: *mut std::ffi::c_void, len: i64) -> *mut OpaqueTensor {
    if data.is_null() { return std::ptr::null_mut(); }
    let vec = unsafe { &*(data as *mut Vec<u8>) };
    let f32_data: Vec<f32> = vec.iter().map(|&b| b as f32).collect();
    let shape = vec![len as usize];
    make_tensor(CpuTensor { data_f32: f32_data, data_i64: None, shape, dtype: DType::F32, autograd: None })
}

pub extern "C" fn tl_cpu_tensor_from_u8_labels(data: *const u8, len: i64) -> *mut OpaqueTensor {
    if data.is_null() { return std::ptr::null_mut(); }
    let slice = unsafe { std::slice::from_raw_parts(data, len as usize) };
    let f32_data: Vec<f32> = slice.iter().map(|&b| b as f32).collect();
    let shape = vec![len as usize];
    make_tensor(CpuTensor { data_f32: f32_data, data_i64: None, shape, dtype: DType::F32, autograd: None })
}

pub extern "C" fn tl_cpu_tensor_matmul_4d(
    a: *mut OpaqueTensor,
    b: *mut OpaqueTensor,
) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    let a_tensor = unsafe { &*a };
    let b_tensor = unsafe { &*b };
    make_tensor(a_tensor.matmul_impl(b_tensor))
}

pub extern "C" fn tl_cpu_tensor_add_4d(
    a: *mut OpaqueTensor,
    b: *mut OpaqueTensor,
) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    let a_tensor = unsafe { &*a };
    let b_tensor = unsafe { &*b };
    make_tensor(a_tensor.add_impl(b_tensor))
}

pub extern "C" fn tl_cpu_tensor_silu_4d(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    tl_cpu_tensor_silu(t) // Same implementation
}

pub extern "C" fn tl_cpu_tensor_cat2(
    a: *mut OpaqueTensor,
    b: *mut OpaqueTensor,
    dim: i64,
) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    let a_tensor = unsafe { &*a };
    let b_tensor = unsafe { &*b };
    make_tensor(CpuTensor::cat_impl(&[a_tensor, b_tensor], dim as usize))
}

pub extern "C" fn tl_cpu_tensor_cat_4d(
    a: *mut OpaqueTensor,
    b: *mut OpaqueTensor,
    dim: i64,
) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    let a_tensor = unsafe { &*a };
    let b_tensor = unsafe { &*b };
    make_tensor(CpuTensor::cat_impl(&[a_tensor, b_tensor], dim as usize))
}

pub extern "C" fn tl_cpu_tensor_rms_norm(
    input: *mut OpaqueTensor,
    weight: *mut OpaqueTensor,
    eps: f32,
) -> *mut OpaqueTensor {
    if input.is_null() { return std::ptr::null_mut(); }
    let x = unsafe { &*input };
    // RMS Norm: x / sqrt(mean(x^2) + eps)
    let data = &x.data_f32;
    let last_dim = *x.shape.last().unwrap_or(&1);
    let num_groups = data.len() / last_dim;
    let mut result = vec![0.0f32; data.len()];
    for g in 0..num_groups {
        let start = g * last_dim;
        let end = start + last_dim;
        let mean_sq: f32 = data[start..end].iter().map(|&v| v * v).sum::<f32>() / last_dim as f32;
        let rms = (mean_sq + eps).sqrt();
        for i in start..end {
            result[i] = data[i] / rms;
        }
    }
    let normalized = CpuTensor { data_f32: result, data_i64: None, shape: x.shape.clone(), dtype: x.dtype, autograd: None };
    if !weight.is_null() {
        let w = unsafe { &*weight };
        make_tensor(normalized.mul_impl(w))
    } else {
        make_tensor(normalized)
    }
}

pub extern "C" fn tl_cpu_tensor_print_1(t: *mut OpaqueTensor) {
    tl_cpu_tensor_print(t);
}

pub extern "C" fn tl_cpu_tensor_print_2(t: *mut OpaqueTensor) {
    tl_cpu_tensor_print(t);
}

pub extern "C" fn tl_cpu_tensor_print_3(t: *mut OpaqueTensor) {
    tl_cpu_tensor_print(t);
}

pub extern "C" fn tl_cpu_tensor_save(_t: *mut OpaqueTensor, _path: *const i8) {
    // Placeholder: file I/O not yet implemented for CPU backend
}

pub extern "C" fn tl_cpu_tensor_load(_path: *const i8) -> *mut OpaqueTensor {
    // Placeholder
    std::ptr::null_mut()
}

pub extern "C" fn tl_cpu_tensor_rope_new_cos(
    _seq_len: usize, _dim: usize, _base: f32,
) -> *mut OpaqueTensor {
    std::ptr::null_mut() // Placeholder for LLM inference
}

pub extern "C" fn tl_cpu_tensor_rope_new_sin(
    _seq_len: usize, _dim: usize, _base: f32,
) -> *mut OpaqueTensor {
    std::ptr::null_mut()
}

pub extern "C" fn tl_cpu_tensor_apply_rope(
    t: *mut OpaqueTensor,
    _cos: *mut OpaqueTensor,
    _sin: *mut OpaqueTensor,
) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    make_tensor(tensor.clone_data()) // Placeholder
}

#[track_caller]
fn make_tensor(t: CpuTensor) -> *mut OpaqueTensor {
    // track_alloc: テンソルのデータバッファ容量を追跡 (Drop の track_free と対称)
    let f32_bytes = t.data_f32.capacity() * std::mem::size_of::<f32>();
    if f32_bytes > 0 {
        crate::memory::track_alloc(f32_bytes);
    }
    if let Some(ref v) = t.data_i64 {
        let i64_bytes = v.capacity() * std::mem::size_of::<i64>();
        if i64_bytes > 0 {
            crate::memory::track_alloc(i64_bytes);
        }
    }
    let arc = Arc::new(UnsafeCell::new(t));
    let ptr = Arc::into_raw(arc) as *mut CpuTensor;
    let loc = std::panic::Location::caller();
    if crate::memory::is_mem_log_enabled() {
        eprintln!("[ALLOC] Ptr: {:p} at {}:{}", ptr, loc.file(), loc.line());
    }
    crate::memory::register_tensor(ptr);
    ptr as *mut OpaqueTensor
}
#[no_mangle]
pub extern "C" fn tl_cpu_get_pool_count() -> usize {
    crate::memory::get_pool_size()
}

#[no_mangle]
pub extern "C" fn tl_cpu_get_memory_mb() -> f64 {
    let bytes = get_rss_bytes();
    bytes as f64 / 1024.0 / 1024.0
}

#[no_mangle]
pub extern "C" fn tl_cpu_get_memory_bytes() -> usize {
    get_rss_bytes()
}

/// OS レベルの RSS (Resident Set Size) を取得。
/// 内部カウンタではなくプロセス全体の物理メモリ使用量を返す。
fn get_rss_bytes() -> usize {
    #[cfg(target_os = "macos")]
    {
        use std::mem;
        extern "C" {
            fn mach_task_self() -> u32;
            fn task_info(
                target_task: u32,
                flavor: i32,
                task_info_out: *mut std::ffi::c_void,
                task_info_outCnt: *mut u32,
            ) -> i32;
        }
        // MACH_TASK_BASIC_INFO = 20
        #[repr(C)]
        struct MachTaskBasicInfo {
            virtual_size: u64,
            resident_size: u64,
            resident_size_max: u64,
            user_time: [u32; 2],     // time_value_t
            system_time: [u32; 2],   // time_value_t
            policy: i32,
            suspend_count: i32,
        }
        let mut info: MachTaskBasicInfo = unsafe { mem::zeroed() };
        let mut count = (mem::size_of::<MachTaskBasicInfo>() / mem::size_of::<u32>()) as u32;
        let kr = unsafe {
            task_info(
                mach_task_self(),
                20, // MACH_TASK_BASIC_INFO
                &mut info as *mut _ as *mut std::ffi::c_void,
                &mut count,
            )
        };
        if kr == 0 {
            info.resident_size as usize
        } else {
            // Fallback to internal counter
            crate::memory::get_total_allocated()
        }
    }
    #[cfg(target_os = "linux")]
    {
        // /proc/self/statm: columns are in pages
        // Column 2 (resident) * page_size
        if let Ok(statm) = std::fs::read_to_string("/proc/self/statm") {
            let parts: Vec<&str> = statm.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(pages) = parts[1].parse::<usize>() {
                    return pages * 4096; // typical page size
                }
            }
        }
        crate::memory::get_total_allocated()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        crate::memory::get_total_allocated()
    }
}

#[no_mangle]
pub extern "C" fn tl_cpu_get_pool_mb() -> f64 {
    // Estimating pool size... difficult without tracking pool bytes separately.
    // Ideally we should track pool bytes too.
    // For now, let's assume average tensor size or implement pool byte tracking.
    // Given user complaint, let's just return 0.0 or track improved metrics later.
    // But wait, the pool stores allocated Vecs. So their capacity counts towards memory.
    // The `TOTAL_ALLOCATED_BYTES` includes memory held by pool tensors?
    // Yes, because return_to_pool does NOT track_free.
    // track_free should only happen when Vec is dropped.
    // Our implementation:
    //   alloc_from_pool / zeros / ones: track_alloc (only growth)
    //   free -> return_to_pool.
    //   When does track_free happen? Never currently!
    //   We need to implement drop for CpuTensor or handle final free.
    //   Actually, CpuTensor is dropped when displaced from pool? No, pool holds indefinitely.
    //   So memory usage should monotonically increase.
    //   Let's check `tl_cpu_tensor_free`.
    0.0
}

#[no_mangle]
pub extern "C" fn tl_cpu_get_pool_bytes() -> usize {
    0
}

// ========== CPU 版 TensorMap ==========
use std::collections::HashMap;
use std::ffi::CStr;
use std::os::raw::c_char;

#[repr(C)]
pub struct StringStruct {
    pub ptr: *mut c_char,
    pub len: i64,
}

pub struct CpuTensorMap {
    pub map: HashMap<String, *mut OpaqueTensor>,
}

#[no_mangle]
pub extern "C" fn tl_cpu_tensor_map_new() -> *mut CpuTensorMap {
    Box::into_raw(Box::new(CpuTensorMap {
        map: HashMap::new(),
    }))
}

#[no_mangle]
pub extern "C" fn tl_cpu_tensor_map_insert(
    map: *mut CpuTensorMap,
    name: *mut StringStruct,
    tensor: *mut OpaqueTensor,
) {
    unsafe {
        if map.is_null() || name.is_null() || (*name).ptr.is_null() || tensor.is_null() {
            return;
        }
        let map_ref = &mut (*map).map;
        let key = CStr::from_ptr((*name).ptr).to_string_lossy().into_owned();
        // Clone the tensor data
        let t = &*tensor;
        let cloned = Box::into_raw(Box::new(CpuTensor {
            data_f32: t.data_f32.clone(),
            data_i64: t.data_i64.clone(),
            shape: t.shape.clone(),
            dtype: t.dtype,
            autograd: None,
        }));
        map_ref.insert(key, cloned);
    }
}

#[no_mangle]
pub extern "C" fn tl_cpu_tensor_map_get(
    map: *mut CpuTensorMap,
    name: *mut StringStruct,
) -> *mut OpaqueTensor {
    unsafe {
        if map.is_null() || name.is_null() || (*name).ptr.is_null() {
            return std::ptr::null_mut();
        }
        let map_ref = &(*map).map;
        let key = CStr::from_ptr((*name).ptr).to_string_lossy().into_owned();
        match map_ref.get(&key) {
            Some(&ptr) => {
                // Return a clone
                let t = &*ptr;
                Box::into_raw(Box::new(CpuTensor {
                    data_f32: t.data_f32.clone(),
                    data_i64: t.data_i64.clone(),
                    shape: t.shape.clone(),
                    dtype: t.dtype,
                    autograd: None,
                }))
            }
            None => std::ptr::null_mut(),
        }
    }
}

// ========== CPU 版 conv2d ==========
#[no_mangle]
pub extern "C" fn tl_cpu_tensor_conv2d(
    input: *mut OpaqueTensor,
    weight: *mut OpaqueTensor,
    padding: i64,
    stride: i64,
) -> *mut OpaqueTensor {
    if input.is_null() || weight.is_null() {
        return std::ptr::null_mut();
    }
    let padding = padding.max(0) as usize;
    let stride = stride.max(1) as usize;
    let (inp, w) = unsafe { (&*input, &*weight) };

    // Input shape: [N, Cin, H, W] or [Cin, H, W] or [H, W]
    let (batch, in_c, in_h, in_w) = match inp.shape.len() {
        4 => (inp.shape[0], inp.shape[1], inp.shape[2], inp.shape[3]),
        3 => (1, inp.shape[0], inp.shape[1], inp.shape[2]),
        2 => (1, 1, inp.shape[0], inp.shape[1]),
        _ => return std::ptr::null_mut(),
    };
    // Weight shape: [Cout, Cin, Kh, Kw]
    let (out_c, _w_in_c, kh, kw) = match w.shape.len() {
        4 => (w.shape[0], w.shape[1], w.shape[2], w.shape[3]),
        _ => return std::ptr::null_mut(),
    };
    let padded_h = in_h + 2 * padding;
    let padded_w = in_w + 2 * padding;
    if padded_h < kh || padded_w < kw || stride == 0 {
        return make_tensor(CpuTensor {
            data_f32: vec![],
            data_i64: None,
            shape: vec![0],
            dtype: inp.dtype,
            autograd: None,
        });
    }
    let out_h = (padded_h - kh) / stride + 1;
    let out_w = (padded_w - kw) / stride + 1;

    let total = batch.checked_mul(out_c).and_then(|v| v.checked_mul(out_h)).and_then(|v| v.checked_mul(out_w));
    let total = match total {
        Some(v) => v,
        None => return std::ptr::null_mut(),
    };
    let mut output = vec![0.0f32; total];

    for n in 0..batch {
        for oc in 0..out_c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = 0.0f32;
                    for ic in 0..in_c {
                        for khi in 0..kh {
                            for kwi in 0..kw {
                                let ih = oh * stride + khi;
                                let iw = ow * stride + kwi;
                                let ih = ih as isize - padding as isize;
                                let iw = iw as isize - padding as isize;
                                if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                                    let ih = ih as usize;
                                    let iw = iw as usize;
                                    let inp_idx = n * in_c * in_h * in_w + ic * in_h * in_w + ih * in_w + iw;
                                    let w_idx = oc * in_c * kh * kw + ic * kh * kw + khi * kw + kwi;
                                    if inp_idx < inp.data_f32.len() && w_idx < w.data_f32.len() {
                                        sum += inp.data_f32[inp_idx] * w.data_f32[w_idx];
                                    }
                                }
                            }
                        }
                    }
                    let out_idx = n * out_c * out_h * out_w + oc * out_h * out_w + oh * out_w + ow;
                    output[out_idx] = sum;
                }
            }
        }
    }

    let shape = if inp.shape.len() == 4 {
        vec![batch, out_c, out_h, out_w]
    } else if inp.shape.len() == 3 {
        vec![out_c, out_h, out_w]
    } else {
        vec![out_h, out_w]
    };

    make_tensor(CpuTensor {
        data_f32: output,
        data_i64: None,
        shape,
        dtype: inp.dtype,
        autograd: None,
    })
}
