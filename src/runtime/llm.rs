use crate::runtime::memory_manager; // Used implicitly? No, but keep to match other files if needed. Actually it was warned used.
                                    // use crate::runtime::mod::{make_tensor, OpaqueTensor}; // Fixed in previous step to `use crate::runtime::{make_tensor, OpaqueTensor};`
use crate::runtime::{make_tensor, OpaqueTensor};
use candle_core::{DType, Device, Tensor};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use tokenizers::Tokenizer;

// Opaque wrapper for Tokenizer
pub struct OpaqueTokenizer(pub Tokenizer);

#[no_mangle]
pub extern "C" fn tl_tokenizer_new(path: *const c_char) -> *mut OpaqueTokenizer {
    unsafe {
        let path_str = CStr::from_ptr(path).to_str().unwrap();
        match Tokenizer::from_file(path_str) {
            Ok(tokenizer) => Box::into_raw(Box::new(OpaqueTokenizer(tokenizer))),
            Err(e) => {
                eprintln!("Failed to load tokenizer from {}: {}", path_str, e);
                std::ptr::null_mut()
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn tl_tokenizer_encode(
    tokenizer: *mut OpaqueTokenizer,
    prompt: *const c_char,
) -> *mut OpaqueTensor {
    unsafe {
        let prompt_str = CStr::from_ptr(prompt).to_str().unwrap();
        println!("DEBUG: Tokenizer encode input: '{}'", prompt_str);
        let t = &(*tokenizer).0;
        match t.encode(prompt_str, true) {
            Ok(encoding) => {
                let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
                let len = ids.len();
                let tensor = Tensor::from_vec(ids, (1, len), &Device::Cpu).unwrap(); // [1, Seq]
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
pub extern "C" fn tl_tokenizer_decode(
    tokenizer: *mut OpaqueTokenizer,
    ids: *mut OpaqueTensor,
) -> *mut c_char {
    unsafe {
        let t = &(*tokenizer).0;
        let tensor = &(*ids).0;

        // Flatten to 1D and get values
        // Expecting [1, 1] or [N] tensor of I64
        let tensor_i64 = tensor.flatten_all().unwrap();
        let vec_i64 = tensor_i64.to_vec1::<i64>().unwrap_or_else(|_| {
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

        // Convert i64 -> u32
        let vec_u32: Vec<u32> = vec_i64.iter().map(|&x| x as u32).collect();

        match t.decode(&vec_u32, true) {
            Ok(s) => {
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
pub extern "C" fn tl_gguf_load(path: *const c_char) -> *mut OpaqueTensorMap {
    unsafe {
        let path_str = CStr::from_ptr(path).to_str().unwrap();

        let mut file = std::fs::File::open(path_str).expect("failed to open file");
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .expect("failed to read gguf");

        let map_ptr = Box::into_raw(Box::new(OpaqueTensorMap(std::collections::HashMap::new())));
        let map = &mut *map_ptr;

        for (tensor_name, qtensor_info) in content.tensor_infos.iter() {
            let qtensor = qtensor_info
                .read(&mut file, content.tensor_data_offset, &Device::Cpu)
                .expect("failed to read qtensor");
            let tensor = qtensor.dequantize(&Device::Cpu).unwrap();

            // Insert Tensor (not OpaqueTensor)
            map.0.insert(tensor_name.clone(), tensor);
        }

        map_ptr
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
pub extern "C" fn tl_tensor_apply_rope(
    x: *mut OpaqueTensor,
    cos: *mut OpaqueTensor,
    sin: *mut OpaqueTensor,
) -> *mut OpaqueTensor {
    unsafe {
        let x_t = &(*x).0; // [..., D]
        let cos_t = &(*cos).0; // [..., D/2]
        let sin_t = &(*sin).0; // [..., D/2]

        // candle_nn::rotary_emb::rope(x, cos, sin)
        // We need to match shapes.
        // Assuming x is [B, S, H, D] or similar.
        // cos/sin usually [B, S, 1, D/2] or Broadcastable.

        let d_m2 = x_t.dim(x_t.rank() - 1).unwrap();
        // x1 = x[..., :D/2], x2 = x[..., D/2:]
        let x1 = x_t.narrow(x_t.rank() - 1, 0, d_m2 / 2).unwrap();
        let x2 = x_t.narrow(x_t.rank() - 1, d_m2 / 2, d_m2 / 2).unwrap();

        // rotate_half(x) = [-x2, x1]
        let x2_neg = x2.neg().unwrap();
        let rotated_cat = Tensor::cat(&[&x2_neg, &x1], x_t.rank() - 1).unwrap();

        // (x * cos) + (rotate_half(x) * sin)
        // We need to broadcast cos/sin to x's shape.
        // x: [B, S, H, D]
        // cos: [S, D] or [1, S, 1, D]

        let x_cos = x_t.broadcast_mul(cos_t).unwrap();
        let rot_sin = rotated_cat.broadcast_mul(sin_t).unwrap();

        let result = (x_cos + rot_sin).unwrap();
        make_tensor(result)
    }
}

// Helper to get raw pointer from String for FFI
#[no_mangle]
pub extern "C" fn tl_string_as_ptr(s: *const c_char) -> *const c_char {
    s
}
