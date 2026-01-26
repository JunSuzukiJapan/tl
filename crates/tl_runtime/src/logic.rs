use crate::knowledge_base::{perform_kb_query, Constant};
use crate::{make_tensor, OpaqueTensor};
use candle_core::Tensor;
use std::ffi::CStr;
use std::os::raw::c_char;

#[unsafe(no_mangle)]
pub extern "C" fn tl_query(
    name: *const c_char,
    mask: i64,
    args: *const OpaqueTensor,
    tags: *const u8,
) -> *mut OpaqueTensor {
    if name.is_null() {
        return std::ptr::null_mut();
    }
    let c_str = unsafe { CStr::from_ptr(name) };
    let r_str = c_str.to_str().unwrap_or("?");

    let mut args_vec: Vec<Constant> = Vec::new();
    if !args.is_null() && !tags.is_null() {
        unsafe {
            let args_tensor = &(*args).0;
            // Get actual number of arguments
            let shape = args_tensor.shape();
            let arity = if shape.rank() > 0 { shape.dims()[0] } else { 0 };
            
            if arity > 0 {
                let tags_slice = std::slice::from_raw_parts(tags, arity);
                if let Ok(data) = args_tensor.flatten_all().and_then(|t| t.to_vec1::<i64>()) {
                    for (i, &tag_val) in tags_slice.iter().enumerate() {
                        if i >= data.len() { break; }
                        let bits = data[i];
                        
                        let c = match tag_val {
                            0 => Constant::Int(bits), // ConstantTag::Int
                            1 => Constant::Float(f64::from_bits(bits as u64)), // ConstantTag::Float
                            2 => Constant::Bool(bits != 0), // ConstantTag::Bool
                            3 => Constant::Entity(bits), // ConstantTag::Entity
                            4 => Constant::String(String::new()), // ConstantTag::String
                            _ => Constant::Int(bits),
                        };
                        args_vec.push(c);
                    }
                }
            }
        }
    }

    // Perform Query
    let results = perform_kb_query(r_str, &args_vec, mask);

    let mut has_float = false;
    for row in &results {
        for val in row {
            if let Constant::Float(_) = val {
                has_float = true;
                break;
            }
        }
        if has_float { break; }
    }

    let device = crate::device::get_device();

    // Construct Result Tensor
    let t_res = if mask == 0 {
        // Boolean result: [1.0] or [0.0]
        let val = if results.is_empty() { 0.0f32 } else { 1.0f32 };
        Tensor::from_slice(&[val], (1,), &device)
    } else {
        // Variable binding result: Vec<Vec<Constant>>
        let rows = results.len();
        let cols = if rows > 0 {
            results[0].len()
        } else {
            mask.count_ones() as usize
        };

        if has_float {
            let mut flat: Vec<f32> = Vec::with_capacity(rows * cols);
            for row in results {
                for val in row {
                    let v = match val {
                        Constant::Int(i) => i as f32,
                        Constant::Float(f) => f as f32,
                        Constant::Bool(b) => if b { 1.0 } else { 0.0 },
                        Constant::Entity(e) => e as f32,
                        _ => 0.0,
                    };
                    flat.push(v);
                }
            }
            if flat.is_empty() {
                Tensor::zeros((rows, cols), candle_core::DType::F32, &device)
            } else {
                Tensor::from_slice(&flat, (rows, cols), &device)
            }
        } else {
            // Use I64 for Entities/Ints/Bools
            let mut flat: Vec<i64> = Vec::with_capacity(rows * cols);
            for row in results {
                for val in row {
                    let v = match val {
                        Constant::Int(i) => i,
                        Constant::Bool(b) => if b { 1 } else { 0 },
                        Constant::Entity(e) => e,
                        Constant::Char(c) => c as i64,
                        _ => 0,
                    };
                    flat.push(v);
                }
            }
            if flat.is_empty() {
                Tensor::zeros((rows, cols), candle_core::DType::I64, &device)
            } else {
                Tensor::from_slice(&flat, (rows, cols), &device)
            }
        }
    };

    match t_res {
        Ok(t) => make_tensor(t),
        Err(e) => {
            eprintln!("Runtime Error: tl_query failed to create tensor: {}", e);
            std::ptr::null_mut()
        }
    }
}
