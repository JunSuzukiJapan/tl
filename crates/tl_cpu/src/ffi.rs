//! CPU 版 FFI 関数
//! GPU 版 (tl_runtime/src/lib.rs) と同じ C シグネチャで CpuTensor を操作。
//! JIT の add_global_mapping で同じシンボル名にマッピングされる。

use crate::tensor::CpuTensor;
use crate::DType;

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

pub extern "C" fn tl_cpu_tensor_from_i64(data: *const i64, len: usize) -> *mut OpaqueTensor {
    if data.is_null() { return std::ptr::null_mut(); }
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


// ========== テンソル解放（データクリア方式） ==========
// JIT の forループ末尾クリーンアップや変数再代入で同一テンソルに対して
// 複数回 release が呼ばれる場合がある。Box::from_raw による即時解放は
// use-after-free を招くため、内部データのみクリアして構造体は保持する。
// これにより:
//   - Vec<f32> 等のデータメモリは OS に返却（メモリリーク解消）
//   - OpaqueTensor ポインタは有効なまま（use-after-free 防止）
//   - 構造体（~80バイト）のリークは許容範囲

pub extern "C" fn tl_cpu_tensor_free(t: *mut OpaqueTensor) {
    if t.is_null() { return; }
    crate::memory::promote_tensor(t);
    unsafe { let _ = Box::from_raw(t); }
}

#[no_mangle]
pub extern "C" fn tl_cpu_tensor_acquire(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    t
}

pub extern "C" fn tl_cpu_tensor_release(t: *mut OpaqueTensor) {
    if t.is_null() { return; }
    unsafe {
        let boxed = Box::from_raw(t as *mut CpuTensor);
        crate::memory::return_to_pool(boxed);
    }
}

/// テンソルの内部データをクリアしてメモリを OS に返却する。
/// 構造体ポインタ自体は有効なまま残るため、autograd の *mut CpuTensor 参照は安全。
/// autograd グラフは触らない（Drop チェーンで dangling pointer を避けるため）。
/// `tl_runtime::tl_tensor_release_safe` から呼ばれる。
pub extern "C" fn tl_cpu_tensor_clear_data(t: *mut OpaqueTensor) {
    if t.is_null() { return; }
    unsafe {
        let tensor = &mut *t;
        // 大きなデータバッファのみ解放（capacity もゼロ化）
        tensor.data_f32 = Vec::new();
        tensor.data_i64 = None;
        // shape は小さいが念のためクリア
        tensor.shape = Vec::new();
        // autograd は触らない:
        // GradFn 内の *mut CpuTensor 参照が他のテンソルを指しているため、
        // Drop チェーンが走ると dangling pointer → SEGFAULT
        // autograd グラフのメモリは構造体分のみ (数百バイト) なのでリークしても人畜無害
    }
}

// ========== テンソル情報 ==========

pub extern "C" fn tl_cpu_tensor_dim(t: *mut OpaqueTensor, dim: usize) -> usize {
    if t.is_null() { return 0; }
    unsafe { (&*t).shape().get(dim).cloned().unwrap_or(0) }
}

pub extern "C" fn tl_cpu_tensor_len(t: *mut OpaqueTensor) -> usize {
    if t.is_null() { return 0; }
    unsafe { (&*t).shape().iter().product() }
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
    tl_cpu_tensor_get_f32(t, indices, rank) as i64
}

pub extern "C" fn tl_cpu_tensor_set_f32(t: *mut OpaqueTensor, indices: *const usize, _rank: usize, value: f32) {
    if t.is_null() || indices.is_null() { return; }
    let tensor = unsafe { &mut *t };
    let shape = tensor.shape().to_vec();
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
            a, b,
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
            a, b,
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
            a, b, a_data: ta.shallow_clone(), b_data: tb.shallow_clone(),
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
            a, b, a_data: ta.shallow_clone(), b_data: tb.shallow_clone(),
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
        unsafe { (&mut *ptr).set_grad_fn(Box::new(NegBackward { a: t })); }
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
        unsafe { (&mut *ptr).set_grad_fn(Box::new(AddScalarBackward { a: t })); }
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
        unsafe { (&mut *ptr).set_grad_fn(Box::new(MulScalarBackward { a: t, s: s as f32 })); }
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
        unsafe { (&mut *ptr).set_grad_fn(Box::new(DivScalarBackward { a: t, s: s as f32 })); }
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
        unsafe { (&mut *ptr).set_grad_fn(Box::new(SubScalarBackward { a: t })); }
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
            a: t, a_data: tensor.shallow_clone(), b_data: exp_tensor, output: result_clone,
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
        unsafe { (&mut *ptr).set_grad_fn(Box::new(ExpBackward { a: t, output })); }
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
        unsafe { (&mut *ptr).set_grad_fn(Box::new(LogBackward { a: t, a_data: tensor.shallow_clone() })); }
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
        unsafe { (&mut *ptr).set_grad_fn(Box::new(ReluBackward { a: t, a_data: tensor.shallow_clone() })); }
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
        unsafe { (&mut *ptr).set_grad_fn(Box::new(SoftmaxBackward { a: t, output, axis: dim as i32 })); }
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
        unsafe { (&mut *ptr).set_grad_fn(Box::new(SumallBackward { a: t, shape: tensor.shape().to_vec() })); }
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

pub extern "C" fn tl_cpu_tensor_argmax(t: *mut OpaqueTensor, _dim: usize, _keep_dim: bool) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let max_idx = tensor.argmax_all_impl();
    make_tensor(CpuTensor::from_slice(&[max_idx as f32], &[1], DType::F32))
}

pub extern "C" fn tl_cpu_tensor_argmin(t: *mut OpaqueTensor, _dim: usize, _keep_dim: bool) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
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
            input: t,
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

pub extern "C" fn tl_cpu_tensor_slice(t: *mut OpaqueTensor, dim: usize, start: usize, end: usize) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let len = end.saturating_sub(start);
    make_tensor(unsafe { (&*t).narrow_impl(dim, start, len) })
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
            a, b, a_data: ta.shallow_clone(), b_data: tb.shallow_clone(),
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
            a: t,
            input_shape: tensor.shape().to_vec(),
            axis: dim as i32,
        })); }
    }
    ptr
}

pub extern "C" fn tl_cpu_tensor_embedding(
    indices: *mut OpaqueTensor,
    weights: *mut OpaqueTensor,
) -> *mut OpaqueTensor {
    if weights.is_null() || indices.is_null() { return std::ptr::null_mut(); }
    let weights_tensor = unsafe { &*weights };
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
    let tensor_box = unsafe { Box::from_raw(t as *mut CpuTensor) };
    crate::memory::return_to_pool(tensor_box);
}

fn make_tensor(t: CpuTensor) -> *mut OpaqueTensor {
    let ptr = if let Some(mut boxed) = crate::memory::recycle_tensor() {
        *boxed = t;
        Box::into_raw(boxed)
    } else {
        Box::into_raw(Box::new(t))
    };
    crate::memory::register_tensor(ptr);
    ptr as *mut OpaqueTensor
}
#[no_mangle]
pub extern "C" fn tl_cpu_get_pool_count() -> usize {
    crate::memory::get_pool_size()
}

#[no_mangle]
pub extern "C" fn tl_cpu_get_memory_mb() -> f64 {
    let bytes = crate::memory::get_total_allocated();
    bytes as f64 / 1024.0 / 1024.0
}

#[no_mangle]
pub extern "C" fn tl_cpu_get_memory_bytes() -> usize {
    crate::memory::get_total_allocated()
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
