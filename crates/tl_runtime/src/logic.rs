use crate::knowledge_base::perform_kb_query;
use crate::{make_tensor, OpaqueTensor};
use candle_core::Tensor;
use std::ffi::CStr;
use std::os::raw::c_char;

#[no_mangle]
pub extern "C" fn tl_query(
    name: *const c_char,
    mask: i64,
    args: *const OpaqueTensor,
) -> *mut OpaqueTensor {
    if name.is_null() {
        return std::ptr::null_mut();
    }
    let c_str = unsafe { CStr::from_ptr(name) };
    let r_str = c_str.to_str().unwrap_or("?");

    let mut args_vec: Vec<i64> = Vec::new();
    if !args.is_null() {
        unsafe {
            let args_tensor = &(*args).0;
            // args tensor from codegen is I64.
            // We expect it to be 1D or scalar effectively.
            match args_tensor.flatten_all().and_then(|t| t.to_vec1::<i64>()) {
                Ok(v) => args_vec = v,
                Err(e) => {
                    // In case it's passed as F32 or something else by error, we might log here
                    eprintln!("Runtime Warning: tl_query args tensor error: {}", e);
                    // Try casting?
                    if let Ok(casted) = args_tensor.to_dtype(candle_core::DType::I64) {
                        if let Ok(v) = casted.flatten_all().and_then(|t| t.to_vec1::<i64>()) {
                            args_vec = v;
                        }
                    }
                }
            }
        }
    }

    // Perform Query
    let results = perform_kb_query(r_str, &args_vec, mask);

    let device = crate::device::get_device();

    // Construct Result Tensor
    let t_res = if mask == 0 {
        // Boolean result: [1.0] or [0.0]
        let val = if results.is_empty() { 0.0f32 } else { 1.0f32 };
        Tensor::from_slice(&[val], (1,), &device)
    } else {
        // Variable binding result: Vec<Vec<i64>>
        // Return as I64 tensor of shape (matches, vars)
        let rows = results.len();
        let cols = if rows > 0 {
            results[0].len()
        } else {
            // count bits in mask
            mask.count_ones() as usize
        };

        let mut flat: Vec<i64> = Vec::with_capacity(rows * cols);
        for row in results {
            flat.extend(row);
        }

        if flat.is_empty() {
            Tensor::zeros((0, cols), candle_core::DType::I64, &device)
        } else {
            Tensor::from_slice(&flat, (rows, cols), &device)
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
