//! CPU 版 FFI 関数
//! GPU 版 (tl_runtime/src/lib.rs) と同じ C シグネチャで CpuTensor を操作。
//! JIT の add_global_mapping で同じシンボル名にマッピングされる。

use crate::tensor::CpuTensor;
use crate::DType;

type OpaqueTensor = CpuTensor;

fn make_tensor(t: CpuTensor) -> *mut OpaqueTensor {
    Box::into_raw(Box::new(t))
}

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

pub extern "C" fn tl_cpu_tensor_zeros(rank: usize, shape: *const usize) -> *mut OpaqueTensor {
    if shape.is_null() { return std::ptr::null_mut(); }
    let shape_slice = unsafe { std::slice::from_raw_parts(shape, rank) };
    make_tensor(CpuTensor::zeros(shape_slice, DType::F32))
}

pub extern "C" fn tl_cpu_tensor_ones(rank: usize, shape: *const usize, _req_grad: bool) -> *mut OpaqueTensor {
    if shape.is_null() { return std::ptr::null_mut(); }
    let shape_slice = unsafe { std::slice::from_raw_parts(shape, rank) };
    make_tensor(CpuTensor::ones(shape_slice, DType::F32))
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

// ========== テンソル解放 ==========

pub extern "C" fn tl_cpu_tensor_free(t: *mut OpaqueTensor) {
    if !t.is_null() { unsafe { let _ = Box::from_raw(t); } }
}

pub extern "C" fn tl_cpu_tensor_release(t: *mut OpaqueTensor) {
    if !t.is_null() { unsafe { let _ = Box::from_raw(t); } }
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
        unsafe { (&mut *ptr).set_grad_fn(Box::new(AddBackward { a, b })); }
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
        unsafe { (&mut *ptr).set_grad_fn(Box::new(SubBackward { a, b })); }
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
    make_tensor(unsafe { (&*t).neg_impl() })
}

pub extern "C" fn tl_cpu_tensor_abs(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    make_tensor(unsafe { (&*t).abs_impl() })
}

// ========== スカラー演算 ==========

pub extern "C" fn tl_cpu_tensor_add_scalar(t: *mut OpaqueTensor, s: f64) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    make_tensor(unsafe { (&*t).add_scalar_impl(s as f32) })
}

pub extern "C" fn tl_cpu_tensor_mul_scalar(t: *mut OpaqueTensor, s: f64) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    make_tensor(unsafe { (&*t).mul_scalar_impl(s as f32) })
}

pub extern "C" fn tl_cpu_tensor_div_scalar(t: *mut OpaqueTensor, s: f64) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    make_tensor(unsafe { (&*t).div_scalar_impl(s as f32) })
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
    make_tensor(tensor.reshape_impl(new_shape))
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

// ========== Init/Shutdown ==========

pub extern "C" fn tl_cpu_runtime_init() {
    println!("Runtime device initialized: CPU");
}

pub extern "C" fn tl_cpu_runtime_shutdown() {}
