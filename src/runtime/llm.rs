// use crate::runtime::memory_manager; // Used implicitly? No, but keep to match other files if needed. Actually it was warned used.
// use crate::runtime::mod::{make_tensor, OpaqueTensor}; // Fixed in previous step to `use crate::runtime::{make_tensor, OpaqueTensor};`
use crate::runtime::{device::get_device, make_tensor, OpaqueTensor};
use candle_core::{DType, Device, Tensor};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use tokenizers::Tokenizer;

// Opaque wrapper for Tokenizer
pub struct OpaqueTokenizer(pub Tokenizer);

#[no_mangle]
pub extern "C" fn tl_tokenizer_new(path: *const c_char) -> i64 {
    unsafe {
        let path_str = CStr::from_ptr(path).to_str().unwrap();
        match Tokenizer::from_file(path_str) {
            Ok(tokenizer) => {
                let ptr = Box::into_raw(Box::new(OpaqueTokenizer(tokenizer)));
                println!("DEBUG: Tokenizer New Ptr: {:p}", ptr);
                ptr as i64
            }
            Err(e) => {
                eprintln!("Failed to load tokenizer from {}: {}", path_str, e);
                0
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn tl_tokenizer_encode(tokenizer: i64, prompt: *const c_char) -> *mut OpaqueTensor {
    unsafe {
        let tokenizer_ptr = tokenizer as *mut OpaqueTokenizer;
        println!("DEBUG: Tokenizer Encode Ptr: {:p}", tokenizer_ptr);
        let prompt_str = CStr::from_ptr(prompt).to_str().unwrap();
        println!("DEBUG: Tokenizer encode input: '{}'", prompt_str);
        let t = &(*tokenizer_ptr).0;
        match t.encode(prompt_str, true) {
            Ok(encoding) => {
                let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
                let len = ids.len();
                let device = get_device();
                let t_cpu = Tensor::from_vec(ids, (len,), &Device::Cpu).unwrap(); // [Seq]
                let tensor = if device.is_metal() || device.is_cuda() {
                    t_cpu.to_device(&device).unwrap()
                } else {
                    t_cpu
                };
                make_tensor(tensor)
            }
            Err(e) => {
                eprintln!("Tokenizer error: {}", e);
                std::ptr::null_mut()
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn tl_tokenizer_decode(tokenizer: i64, ids: *mut OpaqueTensor) -> *mut c_char {
    unsafe {
        eprintln!(
            "DEBUG: tl_tokenizer_decode called with tokenizer={}, ids={:p}",
            tokenizer, ids
        );

        let t = &(*(tokenizer as *mut OpaqueTokenizer)).0;
        eprintln!("DEBUG: Tokenizer ref obtained");

        if ids.is_null() {
            eprintln!("DEBUG: ids is null!");
            return CString::new("").unwrap().into_raw();
        }

        let tensor_wrapper = &(*ids);
        eprintln!("DEBUG: Tensor wrapper obtained");

        // Check tensor validity if possible or just access
        let tensor = &tensor_wrapper.0;
        eprintln!("DEBUG: Tensor ref obtained: {:?}", tensor);

        // Flatten to 1D and get values
        // Expecting [1, 1] or [N] tensor of I64
        let tensor_i64 = match tensor.flatten_all() {
            Ok(t) => t,
            Err(e) => {
                eprintln!("DEBUG: Flatten failed: {}", e);
                return CString::new("").unwrap().into_raw();
            }
        };
        eprintln!("DEBUG: Tensor flattened");

        let vec_i64 = tensor_i64.to_vec1::<i64>().unwrap_or_else(|_| {
            eprintln!("DEBUG: Fallback to float conversion");
            // Fallback if float
            tensor_i64
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .map(|&x| x as i64)
                .collect()
        });
        eprintln!("DEBUG: Vec i64 obtained: {:?}", vec_i64);

        // Convert i64 -> u32
        let vec_u32: Vec<u32> = vec_i64.iter().map(|&x| x as u32).collect();
        eprintln!("DEBUG: Vec u32 ready: {:?}", vec_u32);

        match t.decode(&vec_u32, true) {
            Ok(s) => {
                eprintln!("DEBUG: Decode success: \'{}\'", s);
                let c_string = CString::new(s).unwrap();
                c_string.into_raw()
            }
            Err(e) => {
                eprintln!("Tokenizer decode error: {}", e);
                CString::new("").unwrap().into_raw()
            }
        }
    }
}

// --- GGUF Handling ---
use crate::runtime::OpaqueTensorMap;

#[no_mangle]
pub extern "C" fn tl_gguf_load(path: *const c_char) -> i64 {
    unsafe {
        let path_str = CStr::from_ptr(path).to_str().unwrap();

        let mut file = std::fs::File::open(path_str).expect("failed to open file");
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .expect("failed to read gguf");

        let map_ptr = Box::into_raw(Box::new(OpaqueTensorMap(std::collections::HashMap::new())));
        {
            let map = &mut *map_ptr;
            println!("DEBUG: GGUF Load Map Ptr: {:p}", map_ptr);

            // Print keys to debug
            // for tensor_name in content.tensor_infos.keys() {
            //    println!("GGUF Tensor: {}", tensor_name);
            // }

            let device = get_device();
            for (tensor_name, qtensor_info) in content.tensor_infos.iter() {
                let qtensor = qtensor_info
                    .read(&mut file, content.tensor_data_offset, &Device::Cpu)
                    .expect("failed to read qtensor");
                let tensor = qtensor.dequantize(&device).unwrap();

                // Insert Tensor (not OpaqueTensor)
                map.0.insert(tensor_name.clone(), tensor);
            }
        }

        map_ptr as i64
    }
}

// --- Missing Ops ---

#[no_mangle]
pub extern "C" fn tl_tensor_cat(
    tensors: *mut Vec<*mut OpaqueTensor>,
    dim: i64,
) -> *mut OpaqueTensor {
    unsafe {
        let vec = &*tensors;
        let mut candle_tensors = Vec::new();
        for &t_ptr in vec {
            candle_tensors.push((*t_ptr).0.clone());
        }

        let result = Tensor::cat(&candle_tensors, dim as usize).unwrap();
        make_tensor(result)
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_silu(t: *mut OpaqueTensor) -> *mut OpaqueTensor {
    unsafe {
        let tensor = &(*t).0;
        let result = candle_nn::ops::silu(tensor).unwrap();
        make_tensor(result)
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_rms_norm(
    x: *mut OpaqueTensor,
    w: *mut OpaqueTensor,
    eps: f32,
) -> *mut OpaqueTensor {
    unsafe {
        let x_t = &(*x).0;
        let w_t = &(*w).0;

        // RMSNorm: x * rsqrt(mean(x^2) + eps) * w
        let x_dtype = x_t.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };

        let hidden_size = x_t.dim(x_t.rank() - 1).unwrap();
        let x_f32 = x_t.to_dtype(internal_dtype).unwrap();
        let sum_sq = x_f32.sqr().unwrap().sum_keepdim(x_t.rank() - 1).unwrap();
        let mean_sq = (sum_sq / (hidden_size as f64)).unwrap();
        let rsqrt = (mean_sq + (eps as f64))
            .unwrap()
            .sqrt()
            .unwrap()
            .recip()
            .unwrap();

        let norm_x = x_f32
            .broadcast_mul(&rsqrt)
            .unwrap()
            .to_dtype(x_dtype)
            .unwrap();
        let result = norm_x.broadcast_mul(w_t).unwrap();

        make_tensor(result)
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_cat2(
    a: *mut OpaqueTensor,
    b: *mut OpaqueTensor,
    dim: i64,
) -> *mut OpaqueTensor {
    unsafe {
        let a_t = &(*a).0;
        let b_t = &(*b).0;
        let result = Tensor::cat(&[a_t, b_t], dim as usize).unwrap();
        make_tensor(result)
    }
}

#[no_mangle]
pub extern "C" fn tl_tensor_apply_rope(
    x: *mut OpaqueTensor,
    cos: *mut OpaqueTensor,
    sin: *mut OpaqueTensor,
) -> *mut OpaqueTensor {
    unsafe {
        let x_t = &(*x).0; // [B, S, H, D]
        let cos_t = &(*cos).0; // [S, D/2]
        let sin_t = &(*sin).0; // [S, D/2]

        // x is [B, S, H, D]. We split into two halves along D.
        let d_m2 = x_t.dim(x_t.rank() - 1).unwrap();
        // x1 = x[..., :D/2], x2 = x[..., D/2:]
        let x1 = x_t.narrow(x_t.rank() - 1, 0, d_m2 / 2).unwrap();
        let x2 = x_t.narrow(x_t.rank() - 1, d_m2 / 2, d_m2 / 2).unwrap();

        // rotate_half(x) = [-x2, x1]
        let x2_neg = x2.neg().unwrap();
        let rotated_cat = Tensor::cat(&[&x2_neg, &x1], x_t.rank() - 1).unwrap();

        // (x * cos) + (rotate_half(x) * sin)
        // x1/x2: [B, S, H, D/2]
        // cos/sin: [S, D/2] -> reshape to [1, S, 1, D/2] for broadcast
        let cos_4d = cos_t
            .reshape((1, cos_t.dim(0).unwrap(), 1, cos_t.dim(1).unwrap()))
            .unwrap();
        let sin_4d = sin_t
            .reshape((1, sin_t.dim(0).unwrap(), 1, sin_t.dim(1).unwrap()))
            .unwrap();

        // Need to broadcast cos/sin to match x shape - they apply to both halves
        // For RoPE: x1 * cos + (-x2) * sin and x2 * cos + x1 * sin
        // Simplified as: x * cos + rotated * sin (where rotated has same shape as x)
        // But our rotated_cat is [-x2, x1] so the math is slightly different
        // Standard RoPE: out = x * cos + rotate_half(x) * sin
        // rotate_half swaps and negates: [..., -x2, x1]

        // Since cos/sin are [1, S, 1, D/2] and x is [B, S, H, D]:
        // We need cos/sin repeated for D dimension
        // cat cos with cos to get [1, S, 1, D]
        let cos_full = Tensor::cat(&[&cos_4d, &cos_4d], 3).unwrap();
        let sin_full = Tensor::cat(&[&sin_4d, &sin_4d], 3).unwrap();

        let x_cos = x_t.broadcast_mul(&cos_full).unwrap();
        let rot_sin = rotated_cat.broadcast_mul(&sin_full).unwrap();

        let result = (x_cos + rot_sin).unwrap();
        make_tensor(result)
    }
}

// Helper to get raw pointer from String for FFI
#[no_mangle]
pub extern "C" fn tl_string_as_ptr(s: *const c_char) -> *const c_char {
    s
}

#[no_mangle]
pub extern "C" fn tl_tensor_rope_new_cos(dim: i64, len: i64, theta: f32) -> *mut OpaqueTensor {
    let dim = dim as usize;
    let len = len as usize;
    let theta = theta as f64;

    let inv_freq: Vec<f32> = (0..dim)
        .step_by(2)
        .map(|i| 1.0 / (theta.powf(i as f64 / dim as f64) as f32))
        .collect();
    let device = get_device();
    let inv_freq_t = Tensor::from_vec(inv_freq, (1, dim / 2), &device).unwrap();

    let t: Vec<f32> = (0..len).map(|i| i as f32).collect();
    let t_tensor = Tensor::from_vec(t, (len, 1), &device).unwrap();

    let freqs = t_tensor.matmul(&inv_freq_t).unwrap();
    let cos = freqs.cos().unwrap();

    make_tensor(cos)
}

#[no_mangle]
pub extern "C" fn tl_tensor_rope_new_sin(dim: i64, len: i64, theta: f32) -> *mut OpaqueTensor {
    let dim = dim as usize;
    let len = len as usize;
    let theta = theta as f64;

    let inv_freq: Vec<f32> = (0..dim)
        .step_by(2)
        .map(|i| 1.0 / (theta.powf(i as f64 / dim as f64) as f32))
        .collect();
    let device = get_device();
    let inv_freq_t = Tensor::from_vec(inv_freq, (1, dim / 2), &device).unwrap();

    let t: Vec<f32> = (0..len).map(|i| i as f32).collect();
    let t_tensor = Tensor::from_vec(t, (len, 1), &device).unwrap();

    let freqs = t_tensor.matmul(&inv_freq_t).unwrap();
    let sin = freqs.sin().unwrap();

    make_tensor(sin)
}
