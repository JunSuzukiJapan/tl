//! 拡張テンソル操作 (F32/I64)
//! FFI 関数として公開し、MetalTensor のメソッドを呼び出す。

use crate::OpaqueTensor;
use tl_metal::{MetalTensor, DType};

// 注意: make_tensor は lib.rs で定義されているが、pub(crate) でないためここからは見えない可能性がある。
// もし見えない場合は Box::into_raw(Box::new(t)) を直接使う。
// ここでは安全のため Box::into_raw を使う。

// ========== 型変換 ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_to_f32(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let result = tensor.to_dtype(DType::F32);
    Box::into_raw(Box::new(result))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_to_i64(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let result = tensor.to_dtype(DType::I64); 
    Box::into_raw(Box::new(result))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_to_device(t: *mut OpaqueTensor, _device_id: i32) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let tensor = &*t;
        Box::into_raw(Box::new(tensor.clone_data()))
    }
}

// ========== 活性化関数 ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_tan(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    Box::into_raw(Box::new(tensor.tan_impl()))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_tanh(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    Box::into_raw(Box::new(tensor.tanh_impl()))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_sigmoid(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    Box::into_raw(Box::new(tensor.sigmoid_impl()))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_gelu(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    Box::into_raw(Box::new(tensor.gelu_impl()))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_silu(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    Box::into_raw(Box::new(tensor.silu_impl()))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_relu(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    Box::into_raw(Box::new(tensor.relu_impl()))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_exp(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    Box::into_raw(Box::new(tensor.exp_impl()))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_log(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    Box::into_raw(Box::new(tensor.log_impl()))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_sin(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    Box::into_raw(Box::new(tensor.sin_impl()))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_cos(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    Box::into_raw(Box::new(tensor.cos_impl()))
}

// ========== リダクション ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_max_dim(t: *mut OpaqueTensor, dim: usize, _keep_dim: bool) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    Box::into_raw(Box::new(tensor.max_impl(dim as i32)))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_min_dim(t: *mut OpaqueTensor, dim: usize, _keep_dim: bool) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    Box::into_raw(Box::new(tensor.min_impl(dim as i32)))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_mean_dim(t: *mut OpaqueTensor, dim: usize, _keep_dim: bool) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    Box::into_raw(Box::new(tensor.mean_impl(dim as i32)))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_sum_dim(t: *mut OpaqueTensor, dim: usize, _keep_dim: bool) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    Box::into_raw(Box::new(tensor.sum_impl(dim as i32)))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_max(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    // Flatten and reduce dim 0
    let flat = tensor.reshape_impl(&[tensor.elem_count()]);
    Box::into_raw(Box::new(flat.max_impl(0)))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_min(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let flat = tensor.reshape_impl(&[tensor.elem_count()]);
    Box::into_raw(Box::new(flat.min_impl(0)))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_mean(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let mean_val = tensor.mean_all_impl();
    let result = MetalTensor::from_slice(&[mean_val], &[1], DType::F32);
    Box::into_raw(Box::new(result))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_sum(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let sum_val = tensor.sumall_impl();
    let result = MetalTensor::from_slice(&[sum_val], &[1], DType::F32);
    Box::into_raw(Box::new(result))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_argmin(t: *mut OpaqueTensor, dim: i64, _keepdim: bool) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let idx = if dim < 0 {
        tensor.argmin_all_impl() as f32
    } else {
        0.0
    };
    let result = MetalTensor::from_slice(&[idx], &[1], DType::F32);
    Box::into_raw(Box::new(result))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_softmax(t: *mut OpaqueTensor, dim: i64) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    // Assuming dim is last dim if < 0 or specific dim
    let d = if dim < 0 { (tensor.shape().len() - 1) as i32 } else { dim as i32 };
    Box::into_raw(Box::new(tensor.softmax_impl(d)))
}


// ========== 畳み込み ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_conv2d(
    input: *mut OpaqueTensor,
    weight: *mut OpaqueTensor,
    bias: *mut OpaqueTensor,
    stride: usize,
    padding: usize,
    _dilation: usize,
    _groups: usize,
) -> *mut OpaqueTensor {
    if input.is_null() || weight.is_null() {
        return std::ptr::null_mut();
    }
    let (i, w) = unsafe { (&*input, &*weight) };
    let _b = if bias.is_null() { None } else { unsafe { Some(&*bias) } };
    
    Box::into_raw(Box::new(i.conv2d_impl(w, (stride, stride), (padding, padding))))
}

// ========== NN Layers ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_batch_norm(
    input: *mut OpaqueTensor,
    running_mean: *mut OpaqueTensor,
    running_var: *mut OpaqueTensor,
    weight: *mut OpaqueTensor,
    bias: *mut OpaqueTensor,
    _training: bool,
    _momentum: f64,
    eps: f64,
) -> *mut OpaqueTensor {
    if input.is_null() || running_mean.is_null() || running_var.is_null() || weight.is_null() || bias.is_null() { 
        return std::ptr::null_mut(); 
    }
    let x = unsafe { &*input };
    let mean = unsafe { &*running_mean };
    let var = unsafe { &*running_var };
    let gamma = unsafe { &*weight };
    let beta = unsafe { &*bias };
    
    Box::into_raw(Box::new(x.batch_norm_impl(gamma, beta, mean, var, eps as f32)))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_layer_norm(
    input: *mut OpaqueTensor,
    weight: *mut OpaqueTensor,
    bias: *mut OpaqueTensor,
    eps: f64,
) -> *mut OpaqueTensor {
    if input.is_null() || weight.is_null() || bias.is_null() { return std::ptr::null_mut(); }
    let x = unsafe { &*input };
    let w = unsafe { &*weight };
    let b = unsafe { &*bias };
    
    Box::into_raw(Box::new(x.layer_norm_impl(w, b, eps as f32)))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_max_pool2d(
    input: *mut OpaqueTensor,
    kernel_size: usize,
    stride: usize,
    _padding: usize,
) -> *mut OpaqueTensor {
    if input.is_null() { return std::ptr::null_mut(); }
    let x = unsafe { &*input };
    Box::into_raw(Box::new(x.max_pool2d_impl((kernel_size, kernel_size), (stride, stride))))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_avg_pool2d(
    input: *mut OpaqueTensor,
    kernel_size: usize,
    stride: usize,
    _padding: usize,
) -> *mut OpaqueTensor {
    if input.is_null() { return std::ptr::null_mut(); }
    let x = unsafe { &*input };
    Box::into_raw(Box::new(x.avg_pool2d_impl((kernel_size, kernel_size), (stride, stride))))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_dropout(
    input: *mut OpaqueTensor,
    p: f64,
    training: bool,
) -> *mut OpaqueTensor {
    if input.is_null() { return std::ptr::null_mut(); }
    let x = unsafe { &*input };
    Box::into_raw(Box::new(x.dropout_impl(p as f32, training)))
}

// ========== Embedding ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_embedding(
    weight: *mut OpaqueTensor,
    indices: *mut OpaqueTensor,
    _padding_idx: i64,
    _scale_grad_by_freq: bool,
    _sparse: bool,
) -> *mut OpaqueTensor {
    if weight.is_null() || indices.is_null() { return std::ptr::null_mut(); }
    let w = unsafe { &*weight };
    let idx = unsafe { &*indices };
    
    // CPU gather fallback
    let w_data: Vec<f32> = w.to_vec();
    let idx_data: Vec<f32> = idx.to_vec(); 
    
    let w_shape = w.shape();
    let emb_dim = w_shape[1];
    let num_indices = idx.elem_count();
    let mut out_data = Vec::with_capacity(num_indices * emb_dim);
    let vocab_size = w_shape[0];
    
    for &fd in &idx_data {
        let id = fd as usize;
        if id < vocab_size {
            let start = id * emb_dim;
            out_data.extend_from_slice(&w_data[start..start + emb_dim]);
        } else {
            out_data.extend(std::iter::repeat(0.0).take(emb_dim));
        }
    }
    
    let mut out_shape = idx.shape().to_vec();
    out_shape.push(emb_dim);
    
    let result = MetalTensor::from_slice(&out_data, &out_shape, DType::F32);
    Box::into_raw(Box::new(result))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_cross_entropy(
    logits: *mut OpaqueTensor,
    labels: *mut OpaqueTensor,
) -> *mut OpaqueTensor {
    if logits.is_null() || labels.is_null() { return std::ptr::null_mut(); }
    let l = unsafe { &*logits };
    let _t = unsafe { &*labels };
    // Placeholder
    Box::into_raw(Box::new(l.clone_data()))
}

// ========== Normalization ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_rms_norm(
    input: *mut OpaqueTensor,
    weight: *mut OpaqueTensor,
    eps: f32,
) -> *mut OpaqueTensor {
    if input.is_null() { return std::ptr::null_mut(); }
    let x = unsafe { &*input };
    
    let normalized = x.rms_norm_impl(eps);
    
    if !weight.is_null() {
        let w = unsafe { &*weight };
        Box::into_raw(Box::new(normalized.mul_impl(w)))
    } else {
        Box::into_raw(Box::new(normalized))
    }
}

// ========== Misc ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_repeat_interleave(
    t: *mut OpaqueTensor,
    _repeats: usize,
    _dim: usize,
) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    Box::into_raw(Box::new(tensor.clone_data()))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_sample(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    Box::into_raw(Box::new(tensor.clone_data()))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_scale(t: *mut OpaqueTensor, s: f64) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    Box::into_raw(Box::new(tensor.mul_scalar_impl(s as f32)))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_clamp(
    t: *mut OpaqueTensor,
    min: f64,
    max: f64,
) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let data = tensor.to_vec::<f32>();
    let clamped: Vec<f32> = data.into_iter().map(|v| v.max(min as f32).min(max as f32)).collect();
    let result = MetalTensor::from_slice(&clamped, tensor.shape(), DType::F32);
    Box::into_raw(Box::new(result))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_transpose(t: *mut OpaqueTensor, dim0: usize, dim1: usize) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    Box::into_raw(Box::new(tensor.transpose_impl(dim0, dim1)))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_matmul(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    let (ta, tb) = unsafe { (&*a, &*b) };
    Box::into_raw(Box::new(ta.matmul_impl(tb)))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_contiguous(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    Box::into_raw(Box::new(tensor.contiguous_impl()))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_replace_data(a: *mut OpaqueTensor, b: *mut OpaqueTensor) {
    if a.is_null() || b.is_null() { return; }
    unsafe {
        let replacement = (&*b).clone_data();
        let target = &mut *a;
        *target = replacement;
    }
}

// ========== Device/Grad ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_device_id(t: *mut OpaqueTensor) -> i32 {
    if t.is_null() { return 0; }
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_backward(t: *mut OpaqueTensor) {
    if t.is_null() { return; }
    let tensor = unsafe { &mut *t };
    tensor.backward();
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_grad(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    if let Some(grad) = tensor.get_grad() {
        Box::into_raw(Box::new(grad))
    } else {
        let zeros = MetalTensor::zeros(tensor.shape(), DType::F32);
        Box::into_raw(Box::new(zeros))
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_detach(t: *mut OpaqueTensor, _req_grad: bool) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    Box::into_raw(Box::new(tensor.detach()))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_enable_grad(t: *mut OpaqueTensor) {
    if t.is_null() { return; }
    let tensor = unsafe { &mut *t };
    tensor.enable_grad();
}

// ========== RoPE ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_rope_new_cos(
    head_dim: usize, max_seq: usize, freq_base: f32,
) -> *mut OpaqueTensor {
    let (cos, _) = MetalTensor::rope_cos_sin_impl(max_seq, head_dim, freq_base);
    Box::into_raw(Box::new(cos))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_rope_new_sin(
    head_dim: usize, max_seq: usize, freq_base: f32,
) -> *mut OpaqueTensor {
    let (_, sin) = MetalTensor::rope_cos_sin_impl(max_seq, head_dim, freq_base);
    Box::into_raw(Box::new(sin))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_apply_rope(
    t: *mut OpaqueTensor,
    cos: *mut OpaqueTensor,
    sin: *mut OpaqueTensor,
    pos: usize,
) -> *mut OpaqueTensor {
    if t.is_null() || cos.is_null() || sin.is_null() { return std::ptr::null_mut(); }
    let r = unsafe { &*t };
    let c = unsafe { &*cos };
    let s = unsafe { &*sin };
    Box::into_raw(Box::new(r.apply_rope_impl(c, s, pos)))
}

// ========== Mask ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_new_causal_mask(size: usize) -> *mut OpaqueTensor {
    let mask = MetalTensor::causal_mask_impl(size);
    Box::into_raw(Box::new(mask))
}

// ========== 追加テンソル関数 (Math) ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_sqrt(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    Box::into_raw(Box::new(tensor.pow_scalar_impl(0.5)))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_pow(t: *mut OpaqueTensor, exp: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() || exp.is_null() { return std::ptr::null_mut(); }
    let a = unsafe { &*t };
    let _b = unsafe { &*exp };
    let a_data = a.to_vec::<f32>();
    let b_data = _b.to_vec::<f32>();
    let _count = a.elem_count();
    let res_data: Vec<f32> = a_data.iter().zip(b_data.iter().cycle()).map(|(&base, &e)| base.powf(e)).collect();
    let result = MetalTensor::from_slice(&res_data, a.shape(), DType::F32);
    Box::into_raw(Box::new(result))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_sub_scalar(t: *mut OpaqueTensor, s: f64) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    Box::into_raw(Box::new(tensor.add_scalar_impl(-(s as f32))))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_tril(t: *mut OpaqueTensor, _diagonal: i64) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    Box::into_raw(Box::new(tensor.clone_data()))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_get(t: *mut OpaqueTensor, idx: i64) -> f32 {
    if t.is_null() { return 0.0; }
    let tensor = unsafe { &*t };
    let data: Vec<f32> = tensor.to_vec();
    data.get(idx as usize).copied().unwrap_or(0.0)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_item(t: *mut OpaqueTensor) -> f32 {
    if t.is_null() { return 0.0; }
    let tensor = unsafe { &*t };
    let data: Vec<f32> = tensor.to_vec();
    data.first().copied().unwrap_or(0.0)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_get_f32_md(t: *mut OpaqueTensor, idx0: i64, idx1: i64) -> f32 {
    if t.is_null() { return 0.0; }
    let tensor = unsafe { &*t };
    let shape = tensor.shape();
    let data: Vec<f32> = tensor.to_vec();
    if shape.len() >= 2 {
        let idx = (idx0 as usize) * shape[1] + (idx1 as usize);
        data.get(idx).cloned().unwrap_or(0.0)
    } else {
        data.get(idx0 as usize).cloned().unwrap_or(0.0)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_get_i64_md(t: *mut OpaqueTensor, idx0: i64, idx1: i64) -> i64 {
    tl_tensor_get_f32_md(t, idx0, idx1) as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_set_f32_md(
    t: *mut OpaqueTensor,
    indices: *const i64,
    rank: usize,
    value: f32,
) -> *mut OpaqueTensor {
    if t.is_null() || indices.is_null() { return t; }
    let tensor = unsafe { &*t };
    let shape = MetalTensor::shape(tensor);
    let idx_slice = unsafe { std::slice::from_raw_parts(indices, rank) };
    let mut linear_idx = 0usize;
    let mut stride = 1usize;
    for d in (0..rank).rev() {
        let i = idx_slice[d] as usize;
        if d < shape.len() && i < shape[d] {
            linear_idx += i * stride;
            stride *= shape[d];
        }
    }
    let mut data: Vec<f32> = tensor.to_vec();
    if linear_idx < data.len() {
        data[linear_idx] = value;
    }
    let result = MetalTensor::from_slice(&data, shape, MetalTensor::dtype(tensor));
    Box::into_raw(Box::new(result))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_from_vec_u8(data: *mut Vec<u8>, len: i64) -> *mut OpaqueTensor {
    if data.is_null() { return std::ptr::null_mut(); }
    let vec = unsafe { &*data };
    let f32_data: Vec<f32> = vec.iter().map(|&b| b as f32).collect();
    let shape = vec![len as usize];
    let result = MetalTensor::from_slice(&f32_data, &shape, DType::F32);
    Box::into_raw(Box::new(result))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_from_u8_labels(data: *const u8, len: i64) -> *mut OpaqueTensor {
    if data.is_null() { return std::ptr::null_mut(); }
    let slice = unsafe { std::slice::from_raw_parts(data, len as usize) };
    let f32_data: Vec<f32> = slice.iter().map(|&b| b as f32).collect();
    let shape = vec![len as usize];
    let result = MetalTensor::from_slice(&f32_data, &shape, DType::F32);
    Box::into_raw(Box::new(result))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_from_i64_array(data: *const i64, len: i64) -> *mut OpaqueTensor {
    if data.is_null() { return std::ptr::null_mut(); }
    let slice = unsafe { std::slice::from_raw_parts(data, len as usize) };
    let f32_data: Vec<f32> = slice.iter().map(|&v| v as f32).collect();
    let shape = vec![len as usize];
    let result = MetalTensor::from_slice(&f32_data, &shape, DType::F32);
    Box::into_raw(Box::new(result))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_cat(a: *mut OpaqueTensor, b: *mut OpaqueTensor, dim: i64) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    let (ta, tb) = unsafe { (&*a, &*b) };
    let tensors = vec![ta, tb];
    let result = MetalTensor::cat_impl(&tensors, dim as usize);
    Box::into_raw(Box::new(result))
}

// ========== Image Stubs ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_image_load_grayscale(_path: *const i8) -> *mut OpaqueTensor {
    // Stub
    std::ptr::null_mut()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_image_width(_t: *mut OpaqueTensor) -> i64 {
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_image_height(_t: *mut OpaqueTensor) -> i64 {
    0
}

// ========== IO/System Stubs for Compiler ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_save(path: *mut super::StringStruct, t: *mut OpaqueTensor) {
    if t.is_null() || path.is_null() {
        return;
    }
    let is_cpu = std::env::var("TL_DEVICE").map_or(false, |d| d == "cpu");
    unsafe {
        if (*path).ptr.is_null() { return; }
        let path_str = std::ffi::CStr::from_ptr((*path).ptr).to_string_lossy();
        let path_buf = crate::file_io::expand_path(&path_str);

        let (data, shape) = if is_cpu {
            let tensor = &*(t as *mut tl_cpu::CpuTensor);
            (tensor.data_f32().to_vec(), tensor.shape().to_vec())
        } else {
            let tensor = &*t;
            (tensor.to_vec(), tl_metal::MetalTensor::shape(tensor).to_vec())
        };

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(shape.len() as u64).to_le_bytes());
        for &dim in &shape {
            bytes.extend_from_slice(&(dim as u64).to_le_bytes());
        }
        for &val in &data {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        if let Err(e) = std::fs::write(&path_buf, &bytes) {
            eprintln!("Failed to save tensor: {}", e);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_load(path: *mut super::StringStruct) -> *mut OpaqueTensor {
    let is_cpu = std::env::var("TL_DEVICE").map_or(false, |d| d == "cpu");
    unsafe {
        if path.is_null() || (*path).ptr.is_null() {
            return create_fallback_tensor(is_cpu);
        }
        let path_str = std::ffi::CStr::from_ptr((*path).ptr).to_string_lossy();
        let path_buf = crate::file_io::expand_path(&path_str);

        let bytes = match std::fs::read(&path_buf) {
            Ok(b) => b,
            Err(_) => return create_fallback_tensor(is_cpu),
        };

        if bytes.len() < 8 {
            return create_fallback_tensor(is_cpu);
        }

        let mut offset = 0;
        let rank = u64::from_le_bytes(bytes[offset..offset+8].try_into().unwrap()) as usize;
        offset += 8;

        if bytes.len() < offset + rank * 8 {
            return create_fallback_tensor(is_cpu);
        }

        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            let dim = u64::from_le_bytes(bytes[offset..offset+8].try_into().unwrap()) as usize;
            shape.push(dim);
            offset += 8;
        }

        let numel: usize = shape.iter().product();
        let expected_data_size = numel * 4;
        if bytes.len() < offset + expected_data_size {
            return create_fallback_tensor(is_cpu);
        }

        let mut data = Vec::with_capacity(numel);
        for _ in 0..numel {
            let val = f32::from_le_bytes(bytes[offset..offset+4].try_into().unwrap());
            data.push(val);
            offset += 4;
        }

        if is_cpu {
            let t = tl_cpu::CpuTensor::from_slice(&data, &shape, tl_cpu::DType::F32);
            Box::into_raw(Box::new(t)) as *mut OpaqueTensor
        } else {
            let t = tl_metal::MetalTensor::from_slice(&data, &shape, tl_metal::DType::F32);
            Box::into_raw(Box::new(t))
        }
    }
}

fn create_fallback_tensor(is_cpu: bool) -> *mut OpaqueTensor {
    if is_cpu {
        let t = tl_cpu::CpuTensor::from_slice(&[0.0f32], &[1], tl_cpu::DType::F32);
        Box::into_raw(Box::new(t)) as *mut OpaqueTensor
    } else {
        let t = tl_metal::MetalTensor::zeros(&[1], tl_metal::DType::F32);
        Box::into_raw(Box::new(t))
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_get_memory_bytes() -> i64 {
    let mut usage = std::mem::MaybeUninit::uninit();
    unsafe {
        if libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) == 0 {
            let usage = usage.assume_init();
            // On Mac, ru_maxrss is in bytes. On Linux, it's in KB.
            #[cfg(target_os = "macos")]
            let rss = usage.ru_maxrss as i64;
            #[cfg(target_os = "linux")]
            let rss = (usage.ru_maxrss * 1024) as i64;
            #[cfg(not(any(target_os = "macos", target_os = "linux")))]
            let rss = 0;
            
            return rss;
        } else {
             // eprintln!("[DEBUG] getrusage failed");
        }
    }
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_get_memory_mb() -> f64 {
    tl_get_memory_bytes() as f64 / 1024.0 / 1024.0
}

// ========== Legacy wrappers for compiler ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_cat_i64(a: *mut OpaqueTensor, b: *mut OpaqueTensor, dim: i64) -> *mut OpaqueTensor {
    tl_tensor_cat(a, b, dim)
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_narrow(t: *mut OpaqueTensor, _dim: i64, _start: i64, _len: i64) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    Box::into_raw(Box::new(tensor.clone_data()))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_slice(
   t: *mut OpaqueTensor,
   dim: i64,
   start: i64,
   len: i64,
) -> *mut OpaqueTensor {
   if t.is_null() { return std::ptr::null_mut(); }
   let tensor = unsafe { &*t };
   // assume narrow(dim, start, len)
   Box::into_raw(Box::new(tensor.narrow_impl(dim as usize, start as usize, len as usize)))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_print(t: *mut OpaqueTensor) {
    crate::print_ffi::tl_tensor_print_1(t);
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_reshape_dims(t: *mut OpaqueTensor, shape: *const i64, rank: usize) -> *mut OpaqueTensor {
    if t.is_null() || shape.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let shape_slice = unsafe { std::slice::from_raw_parts(shape, rank) };
    let new_shape: Vec<usize> = shape_slice.iter().map(|&s| s as usize).collect();
    let result = MetalTensor::from_buffer_shared(
        tensor.buffer_arc().clone(), 
        new_shape,
        tensor.dtype()
    );
    Box::into_raw(Box::new(result))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_reshape(t: *mut OpaqueTensor, rank: usize, shape: *const usize) -> *mut OpaqueTensor {
    if t.is_null() || shape.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let shape_slice = unsafe { std::slice::from_raw_parts(shape, rank) };
    let new_shape: Vec<usize> = shape_slice.to_vec();
    let result = MetalTensor::from_buffer_shared(
        tensor.buffer_arc().clone(), 
        new_shape,
        tensor.dtype()
    );
    Box::into_raw(Box::new(result))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_argmax(t: *mut OpaqueTensor, dim: i64, _keepdim: bool) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let mut new_shape = tensor.shape().to_vec();
    if dim >= 0 && (dim as usize) < new_shape.len() {
        new_shape[dim as usize] = 1;
    } else {
        new_shape = vec![1];
    }
    let result = MetalTensor::zeros(&new_shape, DType::F32);
    Box::into_raw(Box::new(result))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_reshape_new(t: *mut OpaqueTensor, _shape: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    Box::into_raw(Box::new(tensor.clone_data()))
}
