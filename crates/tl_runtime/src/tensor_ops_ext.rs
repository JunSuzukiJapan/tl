//! 追加のテンソル演算 FFI 関数

use crate::OpaqueTensor;
use tl_metal::{MetalTensor, DType};

// ========== 型変換 ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_to_f32(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    // MetalTensor は現在 F32 のみをサポート
    let cloned = tensor.clone();
    Box::into_raw(Box::new(cloned))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_to_i64(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    // I64 への変換（データを取得して丸め）
    let data: Vec<f32> = tensor.to_vec();
    let i64_data: Vec<f32> = data.iter().map(|&f| f.round()).collect();
    let result = MetalTensor::from_slice(&i64_data, tensor.shape(), DType::F32);
    Box::into_raw(Box::new(result))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_to_device(t: *mut OpaqueTensor, _device_id: i64) -> *mut OpaqueTensor {
    // Metal バックエンドでは単一デバイスなのでクローンを返す
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let cloned = tensor.clone();
    Box::into_raw(Box::new(cloned))
}

// ========== 三角関数 ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_tan(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    // tan(x) = sin(x) / cos(x)
    let sin = MetalTensor::sin_impl(tensor);
    let cos = MetalTensor::cos_impl(tensor);
    let result = MetalTensor::div_impl(&sin, &cos);
    Box::into_raw(Box::new(result))
}

// ========== 活性化関数 ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_sigmoid(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    // sigmoid(x) = 1 / (1 + exp(-x))
    let neg = MetalTensor::neg_impl(tensor);
    let exp_neg = MetalTensor::exp_impl(&neg);
    let one = MetalTensor::ones(tensor.shape(), DType::F32);
    let denom = MetalTensor::add_impl(&exp_neg, &one);
    let result = MetalTensor::div_impl(&one, &denom);
    Box::into_raw(Box::new(result))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_gelu(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let result = tensor.gelu_impl();
    Box::into_raw(Box::new(result))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_silu(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let result = tensor.silu_impl();
    Box::into_raw(Box::new(result))
}

// ========== Reduction（追加） ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_max_dim(t: *mut OpaqueTensor, dim: usize, keep_dim: bool) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    // GPU accelerated max along dim
    let mut result = MetalTensor::max_impl(tensor, dim as i32);
    if keep_dim {
        // keep_dim フラグがある場合、次元を保持
        let shape = tensor.shape().to_vec();
        let mut new_shape = shape.clone();
        new_shape[dim] = 1;
        result = MetalTensor::reshape_impl(&result, &new_shape);
    }
    Box::into_raw(Box::new(result))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_min_dim(t: *mut OpaqueTensor, dim: usize, keep_dim: bool) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    // GPU accelerated min along dim
    let mut result = MetalTensor::min_impl(tensor, dim as i32);
    if keep_dim {
        let shape = tensor.shape().to_vec();
        let mut new_shape = shape.clone();
        new_shape[dim] = 1;
        result = MetalTensor::reshape_impl(&result, &new_shape);
    }
    Box::into_raw(Box::new(result))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_mean_dim(t: *mut OpaqueTensor, dim: usize, keep_dim: bool) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    // GPU accelerated mean along dim
    let mut result = MetalTensor::mean_impl(tensor, dim as i32);
    if keep_dim {
        let shape = tensor.shape().to_vec();
        let mut new_shape = shape.clone();
        new_shape[dim] = 1;
        result = MetalTensor::reshape_impl(&result, &new_shape);
    }
    Box::into_raw(Box::new(result))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_sum_dim(t: *mut OpaqueTensor, dim: usize, keep_dim: bool) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    // GPU accelerated sum along dim
    let mut result = MetalTensor::sum_impl(tensor, dim as i32);
    if keep_dim {
        let shape = tensor.shape().to_vec();
        let mut new_shape = shape.clone();
        new_shape[dim] = 1;
        result = MetalTensor::reshape_impl(&result, &new_shape);
    }
    Box::into_raw(Box::new(result))
}

// ========== NN 演算 ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_conv2d(
    input: *mut OpaqueTensor,
    weight: *mut OpaqueTensor,
    padding: i64,
    stride: i64,
) -> *mut OpaqueTensor {
    if input.is_null() || weight.is_null() {
        return std::ptr::null_mut();
    }
    let input_tensor = unsafe { &*input };
    let weight_tensor = unsafe { &*weight };
    let result = MetalTensor::conv2d_impl(
        input_tensor,
        weight_tensor,
        (stride as usize, stride as usize),
        (padding as usize, padding as usize),
    );
    Box::into_raw(Box::new(result))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_batch_norm(
    input: *mut OpaqueTensor,
    gamma: *mut OpaqueTensor,
    beta: *mut OpaqueTensor,
    running_mean: *mut OpaqueTensor,
    running_var: *mut OpaqueTensor,
    eps: f64,
) -> *mut OpaqueTensor {
    if input.is_null() || gamma.is_null() || beta.is_null()
        || running_mean.is_null() || running_var.is_null()
    {
        return std::ptr::null_mut();
    }
    let (inp, g, b, rm, rv) = unsafe {
        (&*input, &*gamma, &*beta, &*running_mean, &*running_var)
    };
    let result = inp.batch_norm_impl(g, b, rm, rv, eps as f32);
    Box::into_raw(Box::new(result))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_layer_norm(
    input: *mut OpaqueTensor,
    gamma: *mut OpaqueTensor,
    beta: *mut OpaqueTensor,
    eps: f64,
) -> *mut OpaqueTensor {
    if input.is_null() || gamma.is_null() || beta.is_null() {
        return std::ptr::null_mut();
    }
    let (inp, g, b) = unsafe { (&*input, &*gamma, &*beta) };
    let result = inp.layer_norm_impl(g, b, eps as f32);
    Box::into_raw(Box::new(result))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_max_pool2d(
    input: *mut OpaqueTensor,
    kernel_h: i64,
    kernel_w: i64,
    stride_h: i64,
    stride_w: i64,
) -> *mut OpaqueTensor {
    if input.is_null() {
        return std::ptr::null_mut();
    }
    let inp = unsafe { &*input };
    let result = inp.max_pool2d_impl(
        (kernel_h as usize, kernel_w as usize),
        (stride_h as usize, stride_w as usize),
    );
    Box::into_raw(Box::new(result))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_avg_pool2d(
    input: *mut OpaqueTensor,
    kernel_h: i64,
    kernel_w: i64,
    stride_h: i64,
    stride_w: i64,
) -> *mut OpaqueTensor {
    if input.is_null() {
        return std::ptr::null_mut();
    }
    let inp = unsafe { &*input };
    let result = inp.avg_pool2d_impl(
        (kernel_h as usize, kernel_w as usize),
        (stride_h as usize, stride_w as usize),
    );
    Box::into_raw(Box::new(result))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_dropout(
    input: *mut OpaqueTensor,
    p: f64,
    training: bool,
) -> *mut OpaqueTensor {
    if input.is_null() {
        return std::ptr::null_mut();
    }
    let inp = unsafe { &*input };
    let result = inp.dropout_impl(p as f32, training);
    Box::into_raw(Box::new(result))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_embedding(
    weights: *mut OpaqueTensor,
    indices: *mut OpaqueTensor,
) -> *mut OpaqueTensor {
    if weights.is_null() || indices.is_null() {
        return std::ptr::null_mut();
    }
    let weights_tensor = unsafe { &*weights };
    let indices_tensor = unsafe { &*indices };
    let result = MetalTensor::embedding_impl(weights_tensor, indices_tensor);
    Box::into_raw(Box::new(result))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_cross_entropy(
    input: *mut OpaqueTensor,
    target: *mut OpaqueTensor,
) -> *mut OpaqueTensor {
    if input.is_null() || target.is_null() {
        return std::ptr::null_mut();
    }
    let input_tensor = unsafe { &*input };
    let target_tensor = unsafe { &*target };
    let result = MetalTensor::cross_entropy_impl(input_tensor, target_tensor);
    Box::into_raw(Box::new(result))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_rms_norm(
    t: *mut OpaqueTensor,
    eps: f64,
) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let result = tensor.rms_norm_impl(eps as f32);
    Box::into_raw(Box::new(result))
}

// ========== その他 ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_repeat_interleave(
    t: *mut OpaqueTensor,
    repeats: i64,
    dim: i64,
) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let result = MetalTensor::repeat_interleave_impl(tensor, repeats as usize, dim as usize);
    Box::into_raw(Box::new(result))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_sample(t: *mut OpaqueTensor) -> i64 {
    if t.is_null() {
        return 0;
    }
    let tensor = unsafe { &*t };
    let probs: Vec<f32> = tensor.to_vec();
    
    // Categorical サンプリング
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let total: f32 = probs.iter().sum();
    let r: f32 = rng.r#gen::<f32>() * total;
    
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i as i64;
        }
    }
    (probs.len() - 1) as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_scale(t: *mut OpaqueTensor, scale: f64) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let result = MetalTensor::mul_scalar_impl(tensor, scale as f32);
    Box::into_raw(Box::new(result))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_clamp(
    t: *mut OpaqueTensor,
    min_val: f32,
    max_val: f32,
) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let result = tensor.clamp_impl(min_val, max_val);
    Box::into_raw(Box::new(result))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_device_id(_t: *mut OpaqueTensor) -> i64 {
    // Metal デバイス ID は 0
    0
}

// ========== Autograd（スタブ） ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_backward(_t: *mut OpaqueTensor) {
    // Autograd は未実装
    eprintln!("Warning: Backward not yet supported in Metal backend");
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_grad(_t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    eprintln!("Warning: Gradients not yet supported in Metal backend");
    std::ptr::null_mut()
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_detach(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let cloned = tensor.clone();
    Box::into_raw(Box::new(cloned))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_enable_grad(t: *mut OpaqueTensor, _enable: bool) -> *mut OpaqueTensor {
    // Autograd は未実装、そのまま返す
    t
}

// ========== RoPE 関連 ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_rope_new_cos(
    seq_len: i64,
    head_dim: i64,
    base: f64,
) -> *mut OpaqueTensor {
    let (cos_tensor, _sin_tensor) = MetalTensor::rope_cos_sin_impl(
        seq_len as usize, head_dim as usize, base as f32,
    );
    Box::into_raw(Box::new(cos_tensor))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_rope_new_sin(
    seq_len: i64,
    head_dim: i64,
    base: f64,
) -> *mut OpaqueTensor {
    let (_cos_tensor, sin_tensor) = MetalTensor::rope_cos_sin_impl(
        seq_len as usize, head_dim as usize, base as f32,
    );
    Box::into_raw(Box::new(sin_tensor))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_apply_rope(
    t: *mut OpaqueTensor,
    cos: *mut OpaqueTensor,
    sin: *mut OpaqueTensor,
    pos: i64,
) -> *mut OpaqueTensor {
    if t.is_null() || cos.is_null() || sin.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let cos_tensor = unsafe { &*cos };
    let sin_tensor = unsafe { &*sin };
    let result = tensor.apply_rope_impl(cos_tensor, sin_tensor, pos as usize);
    Box::into_raw(Box::new(result))
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_new_causal_mask(seq_len: i64) -> *mut OpaqueTensor {
    let result = MetalTensor::causal_mask_impl(seq_len as usize);
    Box::into_raw(Box::new(result))
}

// ========== Image 関連 ==========

#[unsafe(no_mangle)]
pub extern "C" fn tl_image_load_grayscale(path: *const std::os::raw::c_char) -> *mut OpaqueTensor {
    if path.is_null() {
        return std::ptr::null_mut();
    }
    let path_str = unsafe { std::ffi::CStr::from_ptr(path).to_string_lossy() };
    let path_buf = crate::file_io::expand_path(&path_str);
    
    match image::open(&path_buf) {
        Ok(img) => {
            let gray = img.to_luma8();
            let (w, h) = gray.dimensions();
            let data: Vec<f32> = gray.into_raw().iter().map(|&p| p as f32 / 255.0).collect();
            let shape = vec![h as usize, w as usize];
            let result = MetalTensor::from_slice(&data, &shape, DType::F32);
            Box::into_raw(Box::new(result))
        }
        Err(e) => {
            eprintln!("Failed to load image: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_image_width(t: *mut OpaqueTensor) -> i64 {
    if t.is_null() {
        return 0;
    }
    let tensor = unsafe { &*t };
    let shape = tensor.shape();
    if shape.len() >= 2 {
        shape[shape.len() - 1] as i64
    } else {
        0
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_image_height(t: *mut OpaqueTensor) -> i64 {
    if t.is_null() {
        return 0;
    }
    let tensor = unsafe { &*t };
    let shape = tensor.shape();
    if shape.len() >= 2 {
        shape[shape.len() - 2] as i64
    } else {
        0
    }
}

// ========== 追加のテンソル演算 ==========

/// テンソルの平方根
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_sqrt(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let result = MetalTensor::sqrt_impl(tensor);
    Box::into_raw(Box::new(result))
}

/// テンソルのべき乗
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_pow(t: *mut OpaqueTensor, exp: f64) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    let result = tensor.pow_scalar_impl(exp as f32);
    Box::into_raw(Box::new(result))
}

/// 下三角行列
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_tril(t: *mut OpaqueTensor, diagonal: i64) -> *mut OpaqueTensor {
    if t.is_null() {
        return std::ptr::null_mut();
    }
    let tensor = unsafe { &*t };
    if tensor.shape().len() < 2 {
        return Box::into_raw(Box::new(tensor.clone()));
    }
    let result = tensor.tril_impl(diagonal as i32);
    Box::into_raw(Box::new(result))
}

/// テンソル要素取得
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_get(t: *mut OpaqueTensor, idx: i64) -> f64 {
    if t.is_null() {
        return 0.0;
    }
    let tensor = unsafe { &*t };
    let data: Vec<f32> = tensor.to_vec();
    data.get(idx as usize).map(|&v| v as f64).unwrap_or(0.0)
}

/// テンソル item（単一要素取得）
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_item(t: *mut OpaqueTensor) -> f64 {
    if t.is_null() {
        return 0.0;
    }
    let tensor = unsafe { &*t };
    let data: Vec<f32> = tensor.to_vec();
    data.first().map(|&v| v as f64).unwrap_or(0.0)
}

/// 多次元インデックスで f32 取得
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_get_f32_md(t: *mut OpaqueTensor, idx0: i64, idx1: i64) -> f32 {
    if t.is_null() {
        return 0.0;
    }
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

/// 多次元インデックスで i64 取得
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_get_i64_md(t: *mut OpaqueTensor, idx0: i64, idx1: i64) -> i64 {
    tl_tensor_get_f32_md(t, idx0, idx1) as i64
}

/// 多次元インデックスで f32 設定
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_set_f32_md(t: *mut OpaqueTensor, idx0: i64, idx1: i64, value: f32) {
    if t.is_null() {
        return;
    }
    let _tensor = unsafe { &mut *t };
    // MetalTensor の直接変更は困難なため警告
    eprintln!("Warning: tl_tensor_set_f32_md modifying tensor at [{}, {}] = {}", idx0, idx1, value);
    // スタブ - 実際の変更は未サポート
}

// tl_tensor_prepare_return は memory_ffi.rs で定義済み

// tl_tensor_acquire と tl_tensor_release は memory_ffi.rs で定義済み

/// Vec<u8> からテンソル作成
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_from_vec_u8(data: *mut Vec<u8>, len: i64) -> *mut OpaqueTensor {
    if data.is_null() {
        return std::ptr::null_mut();
    }
    let vec = unsafe { &*data };
    let f32_data: Vec<f32> = vec.iter().map(|&b| b as f32).collect();
    let shape = vec![len as usize];
    let result = MetalTensor::from_slice(&f32_data, &shape, DType::F32);
    Box::into_raw(Box::new(result))
}

/// u8 ラベル配列からテンソル作成
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_from_u8_labels(data: *const u8, len: i64) -> *mut OpaqueTensor {
    if data.is_null() {
        return std::ptr::null_mut();
    }
    let slice = unsafe { std::slice::from_raw_parts(data, len as usize) };
    let f32_data: Vec<f32> = slice.iter().map(|&b| b as f32).collect();
    let shape = vec![len as usize];
    let result = MetalTensor::from_slice(&f32_data, &shape, DType::F32);
    Box::into_raw(Box::new(result))
}

/// i64 配列からテンソル作成
#[unsafe(no_mangle)]
pub extern "C" fn tl_tensor_from_i64_array(data: *const i64, len: i64) -> *mut OpaqueTensor {
    if data.is_null() {
        return std::ptr::null_mut();
    }
    let slice = unsafe { std::slice::from_raw_parts(data, len as usize) };
    let f32_data: Vec<f32> = slice.iter().map(|&v| v as f32).collect();
    let shape = vec![len as usize];
    let result = MetalTensor::from_slice(&f32_data, &shape, DType::F32);
    Box::into_raw(Box::new(result))
}
