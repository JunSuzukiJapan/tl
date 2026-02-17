//! Metal Backend FFI Operations
//!
//! tl_runtime/src/tensor_ops_ext.rs から移植された Metal 実装。
//! JIT コンパイルされたコードから呼び出される。

use crate::tensor::{MetalTensor, tensor_ref_from_ptr};
use crate::DType;
use std::cell::UnsafeCell;
use std::sync::Arc;
use tl_backend::BackendResult;

// OpaqueTensor は MetalTensor のエイリアス
type OpaqueTensor = MetalTensor;

// use 不要: MetalTensor の演算メソッドは inherent impl で定義

/// 内部ヘルパー: MetalTensor を Arc で包んでポインタを返す（V5.0 メモリ管理）
/// CPU バックエンドの make_tensor と同じパターン。
pub fn make_tensor(t: MetalTensor) -> *mut OpaqueTensor {
    let arc = Arc::new(UnsafeCell::new(t));
    Arc::into_raw(arc) as *mut OpaqueTensor
}

/// Helper to handle BackendResult and return safe pointer or null on error
fn make_tensor_safe(result: BackendResult<MetalTensor>) -> *mut OpaqueTensor {
    match result {
        Ok(t) => make_tensor(t),
        Err(e) => {
            eprintln!("Metal Backend FFI Error: {}", e);
            std::ptr::null_mut()
        }
    }
}

// ========== 型変換 ==========

#[no_mangle]
pub fn tl_metal_to_f32(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    make_tensor_safe(tensor.to_dtype(DType::F32))
}

#[no_mangle]
pub fn tl_metal_to_i64(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    make_tensor_safe(tensor.to_dtype(DType::I64))
}

#[no_mangle]
pub fn tl_metal_to_device(t: *mut OpaqueTensor, _device_id: i32) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    // Metalバックエンド内でのデバイス移動は現状ないため、クローンを返す
    make_tensor_safe(tensor.clone_data())
}

// ========== 活性化関数 ==========

#[no_mangle]
pub fn tl_metal_tan(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    make_tensor_safe(unsafe { (&*t).tan_impl() })
}

#[no_mangle]
pub fn tl_metal_tanh(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    match tensor.tanh_impl() {
        Ok(res) => {
            let ptr = make_tensor(res);
            if tensor.requires_grad() {
                use crate::autograd::ops::TanhBackward;
                let output = unsafe { &*ptr }.shallow_clone();
                unsafe { (&mut *ptr).set_grad_fn(Box::new(TanhBackward { a: tensor_ref_from_ptr(t), output })); }
            }
            ptr
        },
        Err(e) => {
            eprintln!("tanh failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub fn tl_metal_sigmoid(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    match tensor.sigmoid_impl() {
        Ok(res) => {
            let ptr = make_tensor(res);
            if tensor.requires_grad() {
                use crate::autograd::ops::SigmoidBackward;
                let output = unsafe { &*ptr }.shallow_clone();
                unsafe { (&mut *ptr).set_grad_fn(Box::new(SigmoidBackward { a: tensor_ref_from_ptr(t), output })); }
            }
            ptr
        },
        Err(e) => {
            eprintln!("sigmoid failed: {}", e);
            std::ptr::null_mut()
        }
    }
}


#[no_mangle]
pub fn tl_metal_argmax(t: *mut OpaqueTensor, dim: i64, _keepdim: bool) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    make_tensor_safe(unsafe { (&*t).argmax_impl((dim as usize).try_into().unwrap()) })
}

#[no_mangle]
pub fn tl_metal_gelu(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    match tensor.gelu_impl() {
        Ok(res) => {
            let ptr = make_tensor(res);
            if tensor.requires_grad() {
                use crate::autograd::ops::GeluBackward;
                unsafe { (&mut *ptr).set_grad_fn(Box::new(GeluBackward { a: tensor_ref_from_ptr(t) })); }
            }
            ptr
        },
        Err(e) => {
            eprintln!("gelu failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub fn tl_metal_silu(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    // silu = x * sigmoid(x)
    let tensor = unsafe { &*t };
    let res = (|| -> BackendResult<MetalTensor> {
        let sig = tensor.sigmoid_impl()?;
        tensor.mul_impl(&sig)
    })();
    
    match res {
        Ok(r) => {
            let ptr = make_tensor(r);
            if tensor.requires_grad() {
                 use crate::autograd::ops::SiluBackward;
                 unsafe { (&mut *ptr).set_grad_fn(Box::new(SiluBackward { a: tensor_ref_from_ptr(t) })); }
            }
            ptr
        },
        Err(e) => {
            eprintln!("silu failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub fn tl_metal_relu(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    match tensor.relu_impl() {
        Ok(res) => {
            let ptr = make_tensor(res);
            if tensor.requires_grad() {
                use crate::autograd::ops::ReluBackward;
                let data = tensor.shallow_clone(); 
                unsafe { (&mut *ptr).set_grad_fn(Box::new(ReluBackward { a: tensor_ref_from_ptr(t), a_data: data })); }
            }
            ptr
        },
        Err(e) => {
             eprintln!("relu failed: {}", e);
             std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub fn tl_metal_exp(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    match tensor.exp_impl() {
        Ok(res) => {
            let ptr = make_tensor(res);
            if tensor.requires_grad() {
                use crate::autograd::ops::ExpBackward;
                let output = unsafe { &*ptr }.shallow_clone();
                unsafe { (&mut *ptr).set_grad_fn(Box::new(ExpBackward { a: tensor_ref_from_ptr(t), output })); }
            }
            ptr
        },
        Err(e) => {
            eprintln!("exp failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub fn tl_metal_log(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    match tensor.log_impl() {
        Ok(res) => {
            let ptr = make_tensor(res);
            if tensor.requires_grad() {
                use crate::autograd::ops::LogBackward;
                unsafe { (&mut *ptr).set_grad_fn(Box::new(LogBackward { a: tensor_ref_from_ptr(t), a_data: tensor.shallow_clone() })); }
            }
            ptr
        },
        Err(e) => {
             eprintln!("log failed: {}", e);
             std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub fn tl_metal_sin(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    make_tensor_safe(unsafe { (&*t).sin_impl() })
}

#[no_mangle]
pub fn tl_metal_cos(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    make_tensor_safe(unsafe { (&*t).cos_impl() })
}

// ========== リダクション ==========

#[no_mangle]
pub fn tl_metal_max_dim(t: *mut OpaqueTensor, dim: usize, _keep_dim: bool) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    make_tensor_safe(unsafe { (&*t).max_impl((dim as usize).try_into().unwrap()) })
}

#[no_mangle]
pub fn tl_metal_min_dim(t: *mut OpaqueTensor, dim: usize, _keep_dim: bool) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    make_tensor_safe(unsafe { (&*t).min_impl((dim as usize).try_into().unwrap()) })
}

#[no_mangle]
pub fn tl_metal_mean_dim(t: *mut OpaqueTensor, dim: usize, _keep_dim: bool) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let input_shape = tensor.shape().to_vec();
    match tensor.mean_impl((dim as usize).try_into().unwrap()) {
        Ok(res) => {
            let ptr = make_tensor(res);
            if tensor.requires_grad() {
                use crate::autograd::ops::MeanDimBackward;
                unsafe { (&mut *ptr).set_grad_fn(Box::new(MeanDimBackward { a: tensor_ref_from_ptr(t), dim, input_shape })); }
            }
            ptr
        },
        Err(e) => {
            eprintln!("mean_dim failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub fn tl_metal_sum_dim(t: *mut OpaqueTensor, dim: usize, _keep_dim: bool) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let input_shape = tensor.shape().to_vec();
    match tensor.sum_impl((dim as usize).try_into().unwrap()) {
        Ok(res) => {
            let ptr = make_tensor(res);
            if tensor.requires_grad() {
                 use crate::autograd::ops::SumDimBackward;
                 unsafe { (&mut *ptr).set_grad_fn(Box::new(SumDimBackward { a: tensor_ref_from_ptr(t), input_shape, axis: dim as i32 })); }
            }
            ptr
        },
        Err(e) => {
            eprintln!("sum_dim failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub fn tl_metal_max(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let t_ref = unsafe { &*t };
    let res = (|| -> BackendResult<MetalTensor> {
        let flat = t_ref.reshape_impl(&[t_ref.elem_count()])?;
        flat.max_impl(0)
    })();
    make_tensor_safe(res)
}

#[no_mangle]
pub fn tl_metal_min(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let t_ref = unsafe { &*t };
    let res = (|| -> BackendResult<MetalTensor> {
        let flat = t_ref.reshape_impl(&[t_ref.elem_count()])?;
        flat.min_impl(0)
    })();
    make_tensor_safe(res)
}

#[no_mangle]
pub fn tl_metal_mean(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let res = (|| -> BackendResult<MetalTensor> {
        let val = tensor.mean_all_impl()?;
        Ok(MetalTensor::from_slice(&[val], &[1], DType::F32))
    })();
    
    match res {
        Ok(r) => {
            let ptr = make_tensor(r);
            if tensor.requires_grad() {
                use crate::autograd::ops::MeanAllBackward;
                unsafe { (&mut *ptr).set_grad_fn(Box::new(MeanAllBackward { a: tensor_ref_from_ptr(t), shape: tensor.shape().to_vec() })); }
            }
            ptr
        },
        Err(e) => {
            eprintln!("mean failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub fn tl_metal_sum(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let res = (|| -> BackendResult<MetalTensor> {
        let val = tensor.sumall_impl()?;
        Ok(MetalTensor::from_slice(&[val], &[1], DType::F32))
    })();

    match res {
        Ok(r) => {
             let ptr = make_tensor(r);
             if tensor.requires_grad() {
                 use crate::autograd::ops::SumallBackward;
                 unsafe { (&mut *ptr).set_grad_fn(Box::new(SumallBackward { a: tensor_ref_from_ptr(t), shape: tensor.shape().to_vec() })); }
             }
             ptr
        },
        Err(e) => {
             eprintln!("sum failed: {}", e);
             std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub fn tl_metal_argmin(t: *mut OpaqueTensor, dim: i64, _keepdim: bool) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    make_tensor_safe(unsafe { (&*t).argmin_impl((dim as usize).try_into().unwrap()) })
}

#[no_mangle]
pub fn tl_metal_softmax(t: *mut OpaqueTensor, dim: i64) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    match tensor.softmax_impl((dim as usize).try_into().unwrap()) {
        Ok(res) => {
            let ptr = make_tensor(res);
            if tensor.requires_grad() {
                 use crate::autograd::ops::SoftmaxBackward;
                 let output = unsafe { &*ptr }.shallow_clone();
                 unsafe { (&mut *ptr).set_grad_fn(Box::new(SoftmaxBackward { a: tensor_ref_from_ptr(t), output, axis: dim as i32 })); }
            }
            ptr
        },
        Err(e) => {
            eprintln!("softmax failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

// ========== 畳み込み / NN ==========

#[no_mangle]
pub fn tl_metal_conv2d(
    input: *mut OpaqueTensor,
    weight: *mut OpaqueTensor,
    bias: *mut OpaqueTensor,
    stride: usize,
    padding: usize,
    _dilation: usize,
    _groups: usize,
) -> *mut OpaqueTensor {
    if input.is_null() || weight.is_null() { return std::ptr::null_mut(); }
    let (i, w) = unsafe { (&*input, &*weight) };
    let b = if bias.is_null() { None } else { unsafe { Some(&*bias) } };
    
    let res = (|| -> BackendResult<MetalTensor> {
        let conv = i.conv2d_impl(w, (stride, stride), (padding, padding))?;
        if let Some(bias) = b {
            conv.add_impl(bias)
        } else {
            Ok(conv)
        }
    })();
    
    match res {
        Ok(r) => {
            let ptr = make_tensor(r);
            if i.requires_grad() || w.requires_grad() {
                 use crate::autograd::ops::Conv2dBackward;
                 unsafe { (&mut *ptr).set_grad_fn(Box::new(Conv2dBackward {
                     input: tensor_ref_from_ptr(input),
                     weight: tensor_ref_from_ptr(weight),
                     stride: (stride, stride),
                     padding: (padding, padding),
                 })); }
            }
            ptr
        },
        Err(e) => {
            eprintln!("conv2d failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub fn tl_metal_batch_norm(
    input: *mut OpaqueTensor,
    running_mean: *mut OpaqueTensor,
    running_var: *mut OpaqueTensor,
    weight: *mut OpaqueTensor,
    bias: *mut OpaqueTensor,
    _training: bool,
    _momentum: f64,
    eps: f64,
) -> *mut OpaqueTensor {
    if input.is_null() { return std::ptr::null_mut(); }
    let x = unsafe { &*input };
    let mean = if running_mean.is_null() { None } else { unsafe { Some(&*running_mean) } };
    let var = if running_var.is_null() { None } else { unsafe { Some(&*running_var) } };
    let w = if weight.is_null() { None } else { unsafe { Some(&*weight) } };
    let b = if bias.is_null() { None } else { unsafe { Some(&*bias) } };
    
    // Default values if None
    let w_default_res = if w.is_none() { Some(MetalTensor::ones(x.shape(), DType::F32)) } else { None };
    let b_default_res = if b.is_none() { Some(MetalTensor::zeros(x.shape(), DType::F32)) } else { None };
    let mean_default_res = if mean.is_none() { Some(MetalTensor::zeros(x.shape(), DType::F32)) } else { None };
    let var_default_res = if var.is_none() { Some(MetalTensor::ones(x.shape(), DType::F32)) } else { None };

    // MetalTensor::zeros/ones return Self (as per inherent definition).
    let w_ref = w.or(w_default_res.as_ref()).unwrap();
    let b_ref = b.or(b_default_res.as_ref()).unwrap();
    let mean_ref = mean.or(mean_default_res.as_ref()).unwrap();
    let var_ref = var.or(var_default_res.as_ref()).unwrap();
    
    match x.batch_norm_impl(w_ref, b_ref, mean_ref, var_ref, eps as f32) {
        Ok(res) => {
            let ptr = make_tensor(res);
            if x.requires_grad() {
                 use crate::autograd::ops::BatchNormBackward;
                 let weight_ref = if !weight.is_null() { unsafe { tensor_ref_from_ptr(weight) } } else {
                     unsafe { tensor_ref_from_ptr(input) } // dummy
                 };
                 let mean_r = if !running_mean.is_null() { unsafe { tensor_ref_from_ptr(running_mean) } } else {
                     unsafe { tensor_ref_from_ptr(input) }
                 };
                 let var_r = if !running_var.is_null() { unsafe { tensor_ref_from_ptr(running_var) } } else {
                     unsafe { tensor_ref_from_ptr(input) }
                 };
                 unsafe { (&mut *ptr).set_grad_fn(Box::new(BatchNormBackward {
                     input: tensor_ref_from_ptr(input),
                     weight: weight_ref,
                     running_mean: mean_r,
                     running_var: var_r,
                     eps: eps as f32,
                 })); }
            }
            ptr
        },
        Err(e) => {
            eprintln!("batch_norm failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub fn tl_metal_layer_norm(
    input: *mut OpaqueTensor,
    weight: *mut OpaqueTensor,
    bias: *mut OpaqueTensor,
    eps: f64,
) -> *mut OpaqueTensor {
    if input.is_null() { return std::ptr::null_mut(); }
    let x = unsafe { &*input };
    let w = if weight.is_null() { None } else { unsafe { Some(&*weight) } };
    let b = if bias.is_null() { None } else { unsafe { Some(&*bias) } };
    
    let w_default = if w.is_none() { Some(MetalTensor::ones(x.shape(), DType::F32)) } else { None };
    let b_default = if b.is_none() { Some(MetalTensor::zeros(x.shape(), DType::F32)) } else { None };

    let w_ref = w.or(w_default.as_ref()).unwrap();
    let b_ref = b.or(b_default.as_ref()).unwrap();
    
    match x.layer_norm_impl(w_ref, b_ref, eps as f32) {
        Ok(res) => {
            let ptr = make_tensor(res);
            if x.requires_grad() {
                use crate::autograd::ops::LayerNormBackward;
                let weight_ref = if !weight.is_null() { Some(unsafe { tensor_ref_from_ptr(weight) }) } else { None };
                unsafe { (&mut *ptr).set_grad_fn(Box::new(LayerNormBackward {
                    input: tensor_ref_from_ptr(input),
                    weight: weight_ref,
                    eps: eps as f32,
                })); }
            }
            ptr
        },
        Err(e) => {
             eprintln!("layer_norm failed: {}", e);
             std::ptr::null_mut()
        }
    }
}


#[no_mangle]
pub fn tl_metal_max_pool2d(
    input: *mut OpaqueTensor,
    kernel_size: usize,
    stride: usize,
    _padding: usize,
) -> *mut OpaqueTensor {
    if input.is_null() { return std::ptr::null_mut(); }
    make_tensor_safe(unsafe { (&*input).max_pool2d_impl((kernel_size, kernel_size), (stride, stride)) })
}

#[no_mangle]
pub fn tl_metal_avg_pool2d(
    input: *mut OpaqueTensor,
    kernel_size: usize,
    stride: usize,
    _padding: usize,
) -> *mut OpaqueTensor {
    if input.is_null() { return std::ptr::null_mut(); }
    make_tensor_safe(unsafe { (&*input).avg_pool2d_impl((kernel_size, kernel_size), (stride, stride)) })
}

#[no_mangle]
pub fn tl_metal_dropout(
    input: *mut OpaqueTensor,
    p: f64,
    training: bool,
) -> *mut OpaqueTensor {
    if input.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*input };
    match tensor.dropout_impl(p as f32, training) {
        Ok(res) => {
            let ptr = make_tensor(res);
            if training && tensor.requires_grad() {
                use crate::autograd::ops::DropoutBackward;
                let output = unsafe { &*ptr }.shallow_clone();
                unsafe { (&mut *ptr).set_grad_fn(Box::new(DropoutBackward {
                    a: tensor_ref_from_ptr(input),
                    output,
                    p: p as f32,
                })); }
            }
            ptr
        },
        Err(e) => {
            eprintln!("dropout failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub fn tl_metal_embedding(
    weight: *mut OpaqueTensor,
    indices: *mut OpaqueTensor,
    _padding_idx: i64,
    _scale_grad_by_freq: bool,
    _sparse: bool,
) -> *mut OpaqueTensor {
    if weight.is_null() || indices.is_null() { return std::ptr::null_mut(); }
    let (w, i) = unsafe { (&*weight, &*indices) };
//    eprintln!("[DEBUG] tl_metal_embedding w={:p} i={:p}", weight, indices);
    match w.embedding_impl(i) {
        Ok(res) => {
            let ptr = make_tensor(res);
            if w.requires_grad() {
                use crate::autograd::ops::EmbeddingBackward;
                let ws = w.shape();
                let num_embeddings = ws[0];
                let embed_dim = if ws.len() > 1 { ws[1] } else { 1 };
                unsafe { (&mut *ptr).set_grad_fn(Box::new(EmbeddingBackward {
                    weight: tensor_ref_from_ptr(weight),
                    indices: tensor_ref_from_ptr(indices),
                    num_embeddings, embed_dim,
                })); }
            }
            ptr
        },
        Err(e) => {
            eprintln!("embedding failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub fn tl_metal_cross_entropy(
    logits: *mut OpaqueTensor,
    labels: *mut OpaqueTensor,
) -> *mut OpaqueTensor {
    if logits.is_null() || labels.is_null() { return std::ptr::null_mut(); }
    let (l, t) = unsafe { (&*logits, &*labels) };
    match l.cross_entropy_impl(t) {
        Ok(res) => {
            let ptr = make_tensor(res);
            if l.requires_grad() {
                use crate::autograd::ops::CrossEntropyBackward;
                unsafe { (&mut *ptr).set_grad_fn(Box::new(CrossEntropyBackward {
                    logits: tensor_ref_from_ptr(logits),
                    labels: tensor_ref_from_ptr(labels),
                })); }
            }
            ptr
        },
        Err(e) => {
            eprintln!("cross_entropy failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub fn tl_metal_rms_norm(
    input: *mut OpaqueTensor,
    weight: *mut OpaqueTensor,
    eps: f32,
) -> *mut OpaqueTensor {
    if input.is_null() { return std::ptr::null_mut(); }
    let x = unsafe { &*input };
    
    let res = (|| -> BackendResult<MetalTensor> {
        let norm = x.rms_norm_impl(eps as f32)?;
        if !weight.is_null() {
            let w = unsafe { &*weight };
            // eprintln!("[DEBUG] rms_norm w.dtype={:?} norm.dtype={:?}", w.dtype(), norm.dtype());
            if w.dtype() != norm.dtype() {
                let w_casted = w.to_dtype(norm.dtype())?;
                norm.mul_impl(&w_casted)
            } else {
                norm.mul_impl(w)
            }
        } else {
            Ok(norm)
        }
    })();
    make_tensor_safe(res)
}

// ========== Math / Misc ==========

#[no_mangle]
pub fn tl_metal_sqrt(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    match tensor.sqrt_impl() {
        Ok(res) => {
            let ptr = make_tensor(res);
            if tensor.requires_grad() {
                use crate::autograd::ops::SqrtBackward;
                let output = unsafe { &*ptr }.shallow_clone();
                unsafe { (&mut *ptr).set_grad_fn(Box::new(SqrtBackward { a: tensor_ref_from_ptr(t), output })); }
            }
            ptr
        },
        Err(e) => {
            eprintln!("sqrt failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub fn tl_metal_pow(t: *mut OpaqueTensor, exp: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() || exp.is_null() { return std::ptr::null_mut(); }
    let (a, b) = unsafe { (&*t, &*exp) };
    match a.pow_impl(b) {
        Ok(res) => {
            let ptr = make_tensor(res);
            if a.requires_grad() {
                use crate::autograd::ops::PowBackward;
                let result_clone = unsafe { &*ptr }.shallow_clone();
                unsafe { (&mut *ptr).set_grad_fn(Box::new(PowBackward {
                    a: tensor_ref_from_ptr(t), a_data: a.shallow_clone(), b_data: b.shallow_clone(), output: result_clone,
                })); }
            }
            ptr
        },
        Err(e) => {
            eprintln!("pow failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub fn tl_metal_sub_scalar(t: *mut OpaqueTensor, s: f64) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    match tensor.add_scalar_impl(-(s as f32)) {
        Ok(res) => {
            let ptr = make_tensor(res);
            if tensor.requires_grad() {
                 use crate::autograd::ops::SubScalarBackward;
                 unsafe { (&mut *ptr).set_grad_fn(Box::new(SubScalarBackward { a: tensor_ref_from_ptr(t) })); }
            }
            ptr
        },
        Err(e) => {
            eprintln!("sub_scalar failed: {}", e);
            std::ptr::null_mut()
        }
    }
}


#[no_mangle]
pub fn tl_metal_tril(t: *mut OpaqueTensor, diagonal: i64) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    make_tensor_safe(unsafe { (&*t).tril_impl(diagonal as i32) })
}

#[no_mangle]
pub fn tl_metal_get(t: *mut OpaqueTensor, idx: i64) -> f32 {
    if t.is_null() { return 0.0; }
    let tensor = unsafe { &*t };
    let vec: Vec<f32> = tensor.to_vec();
    vec.get(idx as usize).copied().unwrap_or(0.0)
}

#[no_mangle]
pub fn tl_metal_item(t: *mut OpaqueTensor) -> f32 {
    if t.is_null() { return 0.0; }
    let tensor = unsafe { &*t };
    let vec: Vec<f32> = tensor.to_vec();
    vec.first().copied().unwrap_or(0.0)
}

#[no_mangle]
pub fn tl_metal_get_f32_md(t: *mut OpaqueTensor, idx0: i64, idx1: i64) -> f32 {
    if t.is_null() { return 0.0; }
    let tensor = unsafe { &*t };
    let shape = tensor.shape();
    let vec: Vec<f32> = tensor.to_vec();
    if shape.len() >= 2 {
        let idx = (idx0 as usize) * shape[1] + (idx1 as usize);
        vec.get(idx).copied().unwrap_or(0.0)
    } else {
        vec.get(idx0 as usize).copied().unwrap_or(0.0)
    }
}

#[no_mangle]
pub fn tl_metal_get_i64_md(t: *mut OpaqueTensor, idx0: i64, idx1: i64) -> i64 {
    tl_metal_get_f32_md(t, idx0, idx1) as i64
}

#[no_mangle]
pub fn tl_metal_set_f32_md(
    t: *mut OpaqueTensor,
    indices: *const i64,
    rank: usize,
    value: f32,
) -> *mut OpaqueTensor {
    if t.is_null() || indices.is_null() { return t; }
    let tensor = unsafe { &*t };
    let shape = tensor.shape();
    let idx_slice = unsafe { std::slice::from_raw_parts(indices, rank) };
    
    let mut linear_idx = 0usize;
    // let _stride = 1usize;
    // stride 計算 (row-major)
    if !shape.is_empty() {
        let mut strides = vec![1; rank];
        let mut s = 1;
        for i in (0..rank).rev() {
            if i < shape.len() {
                strides[i] = s;
                s *= shape[i];
            }
        }
        for i in 0..rank {
             if i < shape.len() {
                 linear_idx += (idx_slice[i] as usize) * strides[i];
             }
        }
    }

    let mut data: Vec<f32> = tensor.to_vec();
    if linear_idx < data.len() {
        data[linear_idx] = value;
    }
    
    let res = MetalTensor::from_slice(&data, shape, tensor.dtype());
    make_tensor(res)
}

#[no_mangle]
pub fn tl_metal_from_vec_u8(data: *mut std::ffi::c_void, len: i64) -> *mut OpaqueTensor {
     if data.is_null() { return std::ptr::null_mut(); }
     // data は *mut Vec<u8> と仮定
     let vec = unsafe { &*(data as *mut Vec<u8>) };
     let shape = vec![len as usize];
     make_tensor(MetalTensor::from_slice(vec, &shape, DType::U8))
}

#[no_mangle]
pub fn tl_metal_from_u8_labels(data: *const u8, len: i64) -> *mut OpaqueTensor {
    if data.is_null() { return std::ptr::null_mut(); }
    let slice = unsafe { std::slice::from_raw_parts(data, len as usize) };
    let f32_data: Vec<f32> = slice.iter().map(|&b| b as f32).collect();
    let shape = vec![len as usize];
    make_tensor(MetalTensor::from_slice(&f32_data, &shape, DType::F32))
}

#[no_mangle]
pub fn tl_metal_from_i64_array(data: *const i64, len: i64) -> *mut OpaqueTensor {
    if data.is_null() { return std::ptr::null_mut(); }
    let slice = unsafe { std::slice::from_raw_parts(data, len as usize) };
    let f32_data: Vec<f32> = slice.iter().map(|&v| v as f32).collect();
    let shape = vec![len as usize];
    make_tensor(MetalTensor::from_slice(&f32_data, &shape, DType::F32))
}

#[no_mangle]
pub fn tl_metal_cat(a: *mut OpaqueTensor, b: *mut OpaqueTensor, dim: i64) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    let (ta, tb) = unsafe { (&*a, &*b) };
    make_tensor_safe(MetalTensor::cat_impl(&[ta, tb], dim as usize))
}


#[no_mangle]
pub fn tl_metal_repeat_interleave(t: *mut OpaqueTensor, repeats: usize, dim: usize) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    make_tensor_safe(unsafe { (&*t).repeat_interleave_impl(repeats, dim) })
}

#[no_mangle]
pub fn tl_metal_sample(t: *mut OpaqueTensor, temp: f32, top_p: f32) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let logits: Vec<f32> = tensor.to_vec();
    
    if logits.is_empty() {
        return make_tensor(MetalTensor::from_slice(&[0.0f32], &[1], DType::F32));
    }

    // 温度適用 + softmax
    let temp = if temp <= 0.0 { 1e-8 } else { temp };
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temp).collect();
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&x| x / sum).collect();

    // Top-p (nucleus) sampling
    let mut sorted_indices: Vec<usize> = (0..probs.len()).collect();
    sorted_indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumsum = 0.0f32;
    let mut cutoff_idx = sorted_indices.len();
    for (i, &idx) in sorted_indices.iter().enumerate() {
        cumsum += probs[idx];
        if cumsum >= top_p {
            cutoff_idx = i + 1;
            break;
        }
    }

    // Renormalize
    let top_indices = &sorted_indices[..cutoff_idx];
    let top_sum: f32 = top_indices.iter().map(|&i| probs[i]).sum();
    
    // Multinomial sampling from top-p distribution
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();
    // let mut acc = 0.0f32;
    let mut chosen = top_indices[0];
    let mut acc = 0.0f32;
    for &idx in top_indices {
        acc += probs[idx] / top_sum;
        if r < acc {
            chosen = idx;
            break;
        }
    }

    // Return token ID as 1-element i64-compatible tensor
    make_tensor(MetalTensor::from_slice(&[chosen as f32], &[1], DType::F32))
}

#[no_mangle]
pub fn tl_metal_scale(t: *mut OpaqueTensor, s: f32) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    match tensor.mul_scalar_impl(s) {
        Ok(res) => {
            let ptr = make_tensor(res);
            if tensor.requires_grad() {
                 use crate::autograd::ops::ScaleBackward;
                 unsafe { (&mut *ptr).set_grad_fn(Box::new(ScaleBackward { a: tensor_ref_from_ptr(t), s })); }
            }
            ptr
        },
        Err(e) => {
            eprintln!("scale failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub fn tl_metal_clamp(t: *mut OpaqueTensor, min: f64, max: f64) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    make_tensor_safe(unsafe { (&*t).clamp_impl(min as f32, max as f32) })
}

#[no_mangle]
pub fn tl_metal_transpose(t: *mut OpaqueTensor, dim0: usize, dim1: usize) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    match tensor.transpose_impl(dim0, dim1) {
        Ok(res) => {
            let ptr = make_tensor(res);
            if tensor.requires_grad() {
                use crate::autograd::ops::TransposeBackward;
                unsafe { (&mut *ptr).set_grad_fn(Box::new(TransposeBackward { a: tensor_ref_from_ptr(t), dim0, dim1 })); }
            }
            ptr
        },
        Err(e) => {
             eprintln!("transpose failed: {}", e);
             std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub fn tl_metal_matmul(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    let (ta, tb) = unsafe { (&*a, &*b) };
    match ta.matmul_impl(tb) {
        Ok(res) => {
            let ptr = make_tensor(res);
            if ta.requires_grad() || tb.requires_grad() {
                use crate::autograd::ops::MatmulBackward;
                unsafe { (&mut *ptr).set_grad_fn(Box::new(MatmulBackward {
                    a: tensor_ref_from_ptr(a), b: tensor_ref_from_ptr(b), a_data: ta.shallow_clone(), b_data: tb.shallow_clone(),
                })); }
            }
            ptr
        },
        Err(e) => {
            eprintln!("matmul failed: {}", e);
            std::ptr::null_mut()
        }
    }
}


#[no_mangle]
pub fn tl_metal_contiguous(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    match unsafe { (&*t).clone_data() } {
        Ok(res) => make_tensor(res),
        Err(e) => {
            eprintln!("contiguous failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub fn tl_metal_replace_data(a: *mut OpaqueTensor, b: *mut OpaqueTensor) {
    if a.is_null() || b.is_null() { return; }
    unsafe {
        match (&*b).clone_data() {
            Ok(cloned) => *a = cloned,
            Err(e) => eprintln!("replace_data failed: {}", e),
        }
    }
}

// ========== Device/Grad ==========

#[no_mangle]
pub fn tl_metal_device_id(_t: *mut OpaqueTensor) -> i32 {
    0 // GPU
}

#[no_mangle]
pub fn tl_metal_backward(t: *mut OpaqueTensor) {
    if t.is_null() { return; }
    match unsafe { (&mut *t).backward() } {
        Ok(_) => {},
        Err(e) => eprintln!("backward failed: {}", e),
    }
}

#[no_mangle]
pub fn tl_metal_grad(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    if let Some(grad) = tensor.get_grad() {
        make_tensor(grad)
    } else {
        std::ptr::null_mut()
    }
}

#[no_mangle]
pub fn tl_metal_detach(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    unsafe { make_tensor((&*t).detach()) }
}

#[no_mangle]
pub fn tl_metal_enable_grad(t: *mut OpaqueTensor) {
    if t.is_null() { return; }
    unsafe { (&mut *t).enable_grad(); }
}

// ========== RoPE ==========

#[no_mangle]
pub fn tl_metal_rope_new_cos(
    dim: usize, seq_len: usize, freq_base: f32,
) -> *mut OpaqueTensor {
    match MetalTensor::rope_cos_sin_impl(seq_len, dim, freq_base) {
        Ok((cos, _)) => make_tensor(cos),
        Err(e) => {
            eprintln!("rope_new_cos failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub fn tl_metal_rope_new_sin(
    dim: usize, seq_len: usize, freq_base: f32,
) -> *mut OpaqueTensor {
    match MetalTensor::rope_cos_sin_impl(seq_len, dim, freq_base) {
        Ok((_, sin)) => make_tensor(sin),
        Err(e) => {
            eprintln!("rope_new_sin failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub fn tl_metal_apply_rope(
    q: *mut OpaqueTensor,
    k: *mut OpaqueTensor,
    cos: *mut OpaqueTensor,
    sin: *mut OpaqueTensor,
    pos: usize,
) {
    if q.is_null() || k.is_null() || cos.is_null() || sin.is_null() { return; }
    let (q_ref, k_ref) = unsafe { (&mut *q, &mut *k) };
    let c = unsafe { &*cos };
    let s = unsafe { &*sin };
    
    // MetalTensor::apply_rope は (new_q, new_k) を返すが、
    // Rust 側で *q, *k を書き換える必要がある。
    // apply_rope_impl は &self (Tensor) に対する操作で、新しい Tensor を返す。
    // q と k それぞれに適用する。
    match q_ref.apply_rope_impl(c, s, pos) {
        Ok(new_q) => *q_ref = new_q,
        Err(e) => eprintln!("apply_rope (q) failed: {}", e),
    }
    match k_ref.apply_rope_impl(c, s, pos) {
        Ok(new_k) => *k_ref = new_k,
        Err(e) => eprintln!("apply_rope (k) failed: {}", e),
    }
}

// ========== Mask ==========

#[no_mangle]
pub fn tl_metal_new_causal_mask(size: usize) -> *mut OpaqueTensor {
    make_tensor_safe(MetalTensor::causal_mask_impl(size))
}

// ========== Narrow / Slice ==========

#[no_mangle]
pub fn tl_metal_narrow(t: *mut OpaqueTensor, dim: usize, start: usize, len: usize) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    make_tensor_safe(unsafe { (&*t).narrow(dim, start, len) })
}

#[no_mangle]
pub fn tl_metal_slice(t: *mut OpaqueTensor, dim: usize, start: usize, len: usize) -> *mut OpaqueTensor {
    tl_metal_narrow(t, dim, start, len)
}

#[no_mangle]
pub fn tl_metal_reshape(t: *mut OpaqueTensor, dims: *const i64, num_dims: usize) -> *mut OpaqueTensor {
    if t.is_null() || dims.is_null() { return std::ptr::null_mut(); }
    if num_dims == 0 || num_dims > 8 {
        eprintln!("Warning: tl_metal_reshape: invalid num_dims={}", num_dims);
        return std::ptr::null_mut();
    }
    let dims_slice = unsafe { std::slice::from_raw_parts(dims, num_dims) };
    // 負のdims値やゼロを検出 — 不正値が usize に変換されると
    // 巨大サイズになりOOM → abort を引き起こすため
    if dims_slice.iter().any(|&d| d <= 0) {
        eprintln!("Warning: tl_metal_reshape: invalid dims {:?}", dims_slice);
        return std::ptr::null_mut();
    }
    let dims_usize: Vec<usize> = dims_slice.iter().map(|&d| d as usize).collect();
    let old_count = unsafe { (&*t).elem_count() };
    let new_count: usize = dims_usize.iter().product();
    if old_count != new_count {
        eprintln!("Warning: tl_metal_reshape: element count mismatch {} vs {}", old_count, new_count);
        return std::ptr::null_mut();
    }
    
    // reshape_impl returns BackendResult
    match unsafe { (&*t).reshape_impl(&dims_usize) } {
        Ok(result) => {
            let ptr = make_tensor(result);
            let tensor = unsafe { &*t };
            if tensor.requires_grad() {
                use crate::autograd::ops::ReshapeBackward;
                let input_shape = tensor.shape().to_vec();
                unsafe { (&mut *ptr).set_grad_fn(Box::new(ReshapeBackward {
                    input: tensor_ref_from_ptr(t),
                    input_shape,
                })); }
            }
            ptr
        },
        Err(e) => {
            eprintln!("reshape failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

// ========== テンソル作成 ==========

#[no_mangle]
pub fn tl_metal_new(
    data: *const f32,
    rank: usize,
    shape: *const usize,
) -> *mut OpaqueTensor {
    if data.is_null() || shape.is_null() {
        return std::ptr::null_mut();
    }
    let shape_slice = unsafe { std::slice::from_raw_parts(shape, rank) };
    let numel: usize = shape_slice.iter().product();
    let data_slice = unsafe { std::slice::from_raw_parts(data, numel) };
    
    let tensor = MetalTensor::from_slice(data_slice, shape_slice, DType::F32);
    make_tensor(tensor)
}

#[no_mangle]
pub fn tl_metal_new_i64(
    data: *const i64,
    rank: usize,
    shape: *const usize,
) -> *mut OpaqueTensor {
    if data.is_null() || shape.is_null() {
        return std::ptr::null_mut();
    }
    let shape_slice = unsafe { std::slice::from_raw_parts(shape, rank) };
    let numel: usize = shape_slice.iter().product();
    let data_slice = unsafe { std::slice::from_raw_parts(data, numel) };
    let f32_data: Vec<f32> = data_slice.iter().map(|&x| x as f32).collect();
    
    let tensor = MetalTensor::from_slice(&f32_data, shape_slice, DType::F32);
    make_tensor(tensor)
}

#[no_mangle]
pub fn tl_metal_zeros(rank: usize, shape: *const usize, req_grad: bool) -> *mut OpaqueTensor {
    if shape.is_null() {
        return std::ptr::null_mut();
    }
    let shape_slice = unsafe { std::slice::from_raw_parts(shape, rank) };
    let tensor = MetalTensor::zeros(shape_slice, DType::F32);
    let ptr = make_tensor(tensor);
    if req_grad {
        let t = unsafe { &mut *ptr };
        t.enable_grad();
    }
    ptr
}

#[no_mangle]
pub fn tl_metal_ones(rank: usize, shape: *const usize, req_grad: bool) -> *mut OpaqueTensor {
    if shape.is_null() {
        return std::ptr::null_mut();
    }
    let shape_slice = unsafe { std::slice::from_raw_parts(shape, rank) };
    let tensor = MetalTensor::ones(shape_slice, DType::F32);
    let ptr = make_tensor(tensor);
    if req_grad {
        let t = unsafe { &mut *ptr };
        t.enable_grad();
    }
    ptr
}

#[no_mangle]
pub fn tl_metal_randn(rank: usize, shape: *const usize, req_grad: bool) -> *mut OpaqueTensor {
    if shape.is_null() {
        return std::ptr::null_mut();
    }
    let shape_slice = unsafe { std::slice::from_raw_parts(shape, rank) };
    let tensor = MetalTensor::randn(shape_slice, DType::F32);
    let ptr = make_tensor(tensor);
    if req_grad {
        let t = unsafe { &mut *ptr };
        t.enable_grad();
    }
    ptr
}

#[no_mangle]
pub fn tl_metal_randn_debug(
    rank: usize, 
    shape: *const usize, 
    _seed: u64,
    _req_grad: bool,
) -> *mut OpaqueTensor {
    tl_metal_randn(rank, shape, _req_grad)
}

#[no_mangle]
pub fn tl_metal_from_i64(data: *const i64, len: usize) -> *mut OpaqueTensor {
    if data.is_null() {
        return std::ptr::null_mut();
    }
    let data_slice = unsafe { std::slice::from_raw_parts(data, len) };
    let f32_data: Vec<f32> = data_slice.iter().map(|&x| x as f32).collect();
    let tensor = MetalTensor::from_slice(&f32_data, &[len], DType::F32);
    make_tensor(tensor)
}

#[no_mangle]
pub fn tl_metal_from_u8(data: *const u8, len: usize) -> *mut OpaqueTensor {
    if data.is_null() {
        return std::ptr::null_mut();
    }
    let data_slice = unsafe { std::slice::from_raw_parts(data, len) };
    let f32_data: Vec<f32> = data_slice.iter().map(|&x| x as f32).collect();
    let tensor = MetalTensor::from_slice(&f32_data, &[len], DType::F32);
    make_tensor(tensor)
}

// ========== テンソル解放 ==========

#[no_mangle]
pub fn tl_metal_free(t: *mut OpaqueTensor) {
    if !t.is_null() {
        // Arc::from_raw で復元し drop。RC-1、RC=0 で MetalTensor が Drop される。
        unsafe {
            let _ = Arc::from_raw(t as *const UnsafeCell<MetalTensor>);
        }
    }
}

// ========== 後方互換用エイリアス ==========

// ========== テンソル情報取得 ==========

#[no_mangle]
pub fn tl_metal_dim(t: *mut OpaqueTensor, dim: usize) -> usize {
    if t.is_null() { return 0; }
    let tensor = unsafe { &*t };
    let dims = tensor.shape();
    if dim < dims.len() {
        dims[dim]
    } else {
        0
    }
}

#[no_mangle]
pub fn tl_metal_len(t: *mut OpaqueTensor) -> usize {
    if t.is_null() { return 0; }
    let tensor = unsafe { &*t };
    tensor.shape().iter().product()
}

#[no_mangle]
pub fn tl_metal_shape(t: *mut OpaqueTensor, dim: usize) -> i64 {
    tl_metal_dim(t, dim) as i64
}

#[no_mangle]
pub fn tl_metal_get_shape(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let dims: Vec<f32> = tensor.shape().iter().map(|&d| d as f32).collect();
    let shape_tensor = MetalTensor::from_slice(&dims, &[dims.len()], DType::F32);
    make_tensor(shape_tensor)
}

// ========== テンソルデータアクセス ==========

#[no_mangle]
pub fn tl_metal_get_f32(t: *mut OpaqueTensor, idx: usize) -> f32 {
    if t.is_null() { return 0.0; }
    let tensor = unsafe { &*t };
    let data = tensor.to_vec();
    if idx < data.len() {
        data[idx]
    } else {
        0.0
    }
}

#[no_mangle]
pub fn tl_metal_get_i64(t: *mut OpaqueTensor, idx: usize) -> i64 {
    tl_metal_get_f32(t, idx) as i64
}

#[no_mangle]
pub fn tl_metal_set_f32(t: *mut OpaqueTensor, idx: usize, val: f32) {
    if t.is_null() { return; }
    let tensor = unsafe { &mut *t };
    let mut data = tensor.to_vec();
    if idx < data.len() {
        data[idx] = val;
        *tensor = MetalTensor::from_slice(&data, tensor.shape(), DType::F32);
    }
}

#[no_mangle]
pub fn tl_metal_item_i64(t: *mut OpaqueTensor) -> i64 {
    if t.is_null() { return 0; }
    let tensor = unsafe { &*t };
    let data: Vec<f32> = tensor.to_vec();
    if !data.is_empty() {
        data[0] as i64
    } else {
        0
    }
}

// ========== 基本演算 ==========

#[no_mangle]
pub fn tl_metal_add(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    let (ta, tb) = unsafe { (&*a, &*b) };
    match ta.add_impl(tb) {
        Ok(result) => {
            let ptr = make_tensor(result);
            if ta.requires_grad() || tb.requires_grad() {
                use crate::autograd::ops::AddBackward;
                unsafe { (&mut *ptr).set_grad_fn(Box::new(AddBackward {
                    a: tensor_ref_from_ptr(a), b: tensor_ref_from_ptr(b),
                    a_shape: ta.shape().to_vec(), b_shape: tb.shape().to_vec(),
                })); }
            }
            ptr
        },
        Err(e) => {
            eprintln!("add failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub fn tl_metal_sub(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    let (ta, tb) = unsafe { (&*a, &*b) };
    match ta.sub_impl(tb) {
        Ok(result) => {
            let ptr = make_tensor(result);
            if ta.requires_grad() || tb.requires_grad() {
                use crate::autograd::ops::SubBackward;
                unsafe { (&mut *ptr).set_grad_fn(Box::new(SubBackward {
                    a: tensor_ref_from_ptr(a), b: tensor_ref_from_ptr(b),
                    a_shape: ta.shape().to_vec(), b_shape: tb.shape().to_vec(),
                })); }
            }
            ptr
        },
        Err(e) => {
            eprintln!("sub failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub fn tl_metal_mul(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    let (ta, tb) = unsafe { (&*a, &*b) };
    match ta.mul_impl(tb) {
        Ok(result) => {
            let ptr = make_tensor(result);
            if ta.requires_grad() || tb.requires_grad() {
                use crate::autograd::ops::MulBackward;
                unsafe { (&mut *ptr).set_grad_fn(Box::new(MulBackward {
                    a: tensor_ref_from_ptr(a), b: tensor_ref_from_ptr(b),
                    a_data: ta.shallow_clone(), b_data: tb.shallow_clone(),
                    a_shape: ta.shape().to_vec(), b_shape: tb.shape().to_vec(),
                })); }
            }
            ptr
        },
        Err(e) => {
            eprintln!("mul failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub fn tl_metal_div(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    let (ta, tb) = unsafe { (&*a, &*b) };
    match ta.div_impl(tb) {
        Ok(result) => {
            let ptr = make_tensor(result);
            if ta.requires_grad() || tb.requires_grad() {
                use crate::autograd::ops::DivBackward;
                unsafe { (&mut *ptr).set_grad_fn(Box::new(DivBackward {
                    a: tensor_ref_from_ptr(a), b: tensor_ref_from_ptr(b),
                    a_data: ta.shallow_clone(), b_data: tb.shallow_clone(),
                    a_shape: ta.shape().to_vec(), b_shape: tb.shape().to_vec(),
                })); }
            }
            ptr
        },
        Err(e) => {
            eprintln!("div failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub fn tl_metal_rem(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    let (ta, tb) = unsafe { (&*a, &*b) };
    make_tensor_safe(ta.rem_impl(tb))
}

#[no_mangle]
pub fn tl_metal_neg(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    match tensor.neg_impl() {
        Ok(res) => {
            let ptr = make_tensor(res);
            if tensor.requires_grad() {
                use crate::autograd::ops::NegBackward;
                unsafe { (&mut *ptr).set_grad_fn(Box::new(NegBackward { a: tensor_ref_from_ptr(t) })); }
            }
            ptr
        },
        Err(e) => {
            eprintln!("neg failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub fn tl_metal_abs(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    match tensor.abs_impl() {
        Ok(result) => {
            let ptr = make_tensor(result);
            if tensor.requires_grad() {
                use crate::autograd::ops::AbsBackward;
                unsafe { (&mut *ptr).set_grad_fn(Box::new(AbsBackward { a: tensor_ref_from_ptr(t) })); }
            }
            ptr
        },
        Err(e) => {
            eprintln!("abs failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

// ========== スカラー演算 ==========

#[no_mangle]
pub fn tl_metal_add_scalar(t: *mut OpaqueTensor, s: f64) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    match tensor.add_scalar_impl(s as f32) {
        Ok(res) => {
            let ptr = make_tensor(res);
            if tensor.requires_grad() {
                use crate::autograd::ops::AddScalarBackward;
                unsafe { (&mut *ptr).set_grad_fn(Box::new(AddScalarBackward { a: tensor_ref_from_ptr(t) })); }
            }
            ptr
        },
        Err(e) => {
            eprintln!("add_scalar failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub fn tl_metal_mul_scalar(t: *mut OpaqueTensor, s: f64) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    match tensor.mul_scalar_impl(s as f32) {
        Ok(res) => {
            let ptr = make_tensor(res);
            if tensor.requires_grad() {
                use crate::autograd::ops::MulScalarBackward;
                unsafe { (&mut *ptr).set_grad_fn(Box::new(MulScalarBackward { a: tensor_ref_from_ptr(t), s: s as f32 })); }
            }
            ptr
        },
        Err(e) => {
            eprintln!("mul_scalar failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub fn tl_metal_div_scalar(t: *mut OpaqueTensor, s: f64) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    match tensor.div_scalar_impl(s as f32) {
        Ok(res) => {
            let ptr = make_tensor(res);
            if tensor.requires_grad() {
                use crate::autograd::ops::DivScalarBackward;
                unsafe { (&mut *ptr).set_grad_fn(Box::new(DivScalarBackward { a: tensor_ref_from_ptr(t), s: s as f32 })); }
            }
            ptr
        },
        Err(e) => {
            eprintln!("div_scalar failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub fn tl_metal_pow_scalar(t: *mut OpaqueTensor, s: f64) -> *mut OpaqueTensor {
    if t.is_null() { return std::ptr::null_mut(); }
    let tensor = unsafe { &*t };
    let exp_tensor = MetalTensor::from_slice(&[s as f32], &[1], DType::F32);
    match tensor.pow_impl(&exp_tensor) {
        Ok(result) => {
            let ptr = make_tensor(result);
            if tensor.requires_grad() {
                use crate::autograd::ops::PowBackward;
                let result_clone = unsafe { &*ptr }.shallow_clone();
                unsafe { (&mut *ptr).set_grad_fn(Box::new(PowBackward {
                    a: tensor_ref_from_ptr(t), a_data: tensor.shallow_clone(), b_data: exp_tensor, output: result_clone,
                })); }
            }
            ptr
        },
        Err(e) => {
            eprintln!("pow_scalar failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

// ========== インプレース演算 ==========

#[no_mangle]
pub fn tl_metal_add_assign(a: *mut OpaqueTensor, b: *mut OpaqueTensor) {
    if a.is_null() || b.is_null() { return; }
    let (ta, tb) = unsafe { (&mut *a, &*b) };
    match ta.add_impl(tb) {
        Ok(res) => *ta = res,
        Err(e) => eprintln!("add_assign failed: {}", e),
    }
}

#[no_mangle]
pub fn tl_metal_sub_assign(a: *mut OpaqueTensor, b: *mut OpaqueTensor) {
    if a.is_null() || b.is_null() { return; }
    let (ta, tb) = unsafe { (&mut *a, &*b) };
    match ta.sub_impl(tb) {
        Ok(res) => *ta = res,
        Err(e) => eprintln!("sub_assign failed: {}", e),
    }
}

#[no_mangle]
pub fn tl_metal_mul_assign(a: *mut OpaqueTensor, b: *mut OpaqueTensor) {
    if a.is_null() || b.is_null() { return; }
    let (ta, tb) = unsafe { (&mut *a, &*b) };
    match ta.mul_impl(tb) {
        Ok(res) => *ta = res,
        Err(e) => eprintln!("mul_assign failed: {}", e),
    }
}

#[no_mangle]
pub fn tl_metal_div_assign(a: *mut OpaqueTensor, b: *mut OpaqueTensor) {
    if a.is_null() || b.is_null() { return; }
    let (ta, tb) = unsafe { (&mut *a, &*b) };
    match ta.div_impl(tb) {
        Ok(res) => *ta = res,
        Err(e) => eprintln!("div_assign failed: {}", e),
    }
}

// ========== 比較演算 ==========

#[no_mangle]
pub fn tl_metal_eq(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    let (ta, tb) = unsafe { (&*a, &*b) };
    make_tensor_safe(ta.eq_impl(tb))
}

#[no_mangle]
pub fn tl_metal_neq(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    let (ta, tb) = unsafe { (&*a, &*b) };
    make_tensor_safe(ta.ne_impl(tb))
}

#[no_mangle]
pub fn tl_metal_lt(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    let (ta, tb) = unsafe { (&*a, &*b) };
    make_tensor_safe(ta.lt_impl(tb))
}

#[no_mangle]
pub fn tl_metal_le(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    let (ta, tb) = unsafe { (&*a, &*b) };
    make_tensor_safe(ta.le_impl(tb))
}

#[no_mangle]
pub fn tl_metal_gt(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    let (ta, tb) = unsafe { (&*a, &*b) };
    make_tensor_safe(ta.gt_impl(tb))
}

#[no_mangle]
pub fn tl_metal_ge(a: *mut OpaqueTensor, b: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    let (ta, tb) = unsafe { (&*a, &*b) };
    make_tensor_safe(ta.ge_impl(tb))
}

// ========== スカラー In-place 演算 ==========

#[no_mangle]
pub fn tl_metal_add_assign_scalar_f32(t: *mut OpaqueTensor, s: f32) {
    if t.is_null() { return; }
    let tensor = unsafe { &mut *t };
    match tensor.add_scalar_impl(s) {
        Ok(res) => *tensor = res,
        Err(e) => eprintln!("add_assign_scalar failed: {}", e),
    }
}

#[no_mangle]
pub fn tl_metal_sub_assign_scalar_f32(t: *mut OpaqueTensor, s: f32) {
    if t.is_null() { return; }
    let tensor = unsafe { &mut *t };
    match tensor.add_scalar_impl(-s) {
        Ok(res) => *tensor = res,
        Err(e) => eprintln!("sub_assign_scalar failed: {}", e),
    }
}

#[no_mangle]
pub fn tl_metal_mul_assign_scalar_f32(t: *mut OpaqueTensor, s: f32) {
    if t.is_null() { return; }
    let tensor = unsafe { &mut *t };
    match tensor.mul_scalar_impl(s) {
        Ok(res) => *tensor = res,
        Err(e) => eprintln!("mul_assign_scalar failed: {}", e),
    }
}

#[no_mangle]
pub fn tl_metal_div_assign_scalar_f32(t: *mut OpaqueTensor, s: f32) {
    if t.is_null() { return; }
    let tensor = unsafe { &mut *t };
    match tensor.div_scalar_impl(s) {
        Ok(res) => *tensor = res,
        Err(e) => eprintln!("div_assign_scalar failed: {}", e),
    }
}

#[no_mangle]
pub fn tl_metal_mod_assign_scalar_f32(t: *mut OpaqueTensor, s: f32) {
    if t.is_null() { return; }
    let tensor = unsafe { &mut *t };
    match tensor.fmod_scalar_impl(s) {
        Ok(res) => *tensor = res,
        Err(e) => eprintln!("mod_assign_scalar failed: {}", e),
    }
}

#[no_mangle]
pub fn tl_metal_mod_assign(a: *mut OpaqueTensor, b: *mut OpaqueTensor) {
    if a.is_null() || b.is_null() { return; }
    let (ta, tb) = unsafe { (&mut *a, &*b) };
    match ta.rem_impl(tb) {
        Ok(res) => *ta = res,
        Err(e) => eprintln!("mod_assign failed: {}", e),
    }
}

#[no_mangle]
pub fn tl_metal_reshape_new(t: *mut OpaqueTensor, dims: *const i64, num_dims: usize) -> *mut OpaqueTensor {
    tl_metal_reshape(t, dims, num_dims)
}

#[no_mangle]
pub fn tl_metal_reshape_dims(t: *mut OpaqueTensor, dims: *const i64, num_dims: usize) -> *mut OpaqueTensor {
    tl_metal_reshape(t, dims, num_dims)
}

#[no_mangle]
pub fn tl_metal_cat_i64(a: *mut OpaqueTensor, b: *mut OpaqueTensor, dim: i64) -> *mut OpaqueTensor {
    tl_metal_cat(a, b, dim)
}
