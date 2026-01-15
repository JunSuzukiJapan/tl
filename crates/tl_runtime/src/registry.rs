use crate::context::{TensorContext, TensorValue};
use crate::OpaqueTensor;
use lazy_static::lazy_static;
use std::ffi::CStr;
use std::os::raw::c_char;
use std::sync::Mutex;

lazy_static! {
    // Global tensor context to store tensors created by JIT execution
    pub static ref GLOBAL_CONTEXT: Mutex<TensorContext> = Mutex::new(TensorContext::new());
}

/// Reset the global context (call before running a new program)
pub fn reset_global_context() {
    let mut ctx = GLOBAL_CONTEXT.lock().unwrap();
    // TensorContext doesn't have a clear method, so let's replace it
    *ctx = TensorContext::new();
}

/// Helper to expose the context to main.rs
pub fn get_global_context() -> TensorContext {
    let ctx = GLOBAL_CONTEXT.lock().unwrap();
    ctx.clone()
}

/// Register a tensor from JIT code.
/// This function is called by the compiled code for every 'let' statement.
#[no_mangle]
pub extern "C" fn tl_register_tensor(name_ptr: *const c_char, tensor: *mut OpaqueTensor) {
    unsafe {
        if name_ptr.is_null() || tensor.is_null() {
            return;
        }

        let c_str = CStr::from_ptr(name_ptr);
        let name = match c_str.to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return,
        };

        let t_opaque = &*tensor;
        let t_candle = &t_opaque.0;

        // Convert Candle tensor to TensorValue (Vec<f64>)
        // Flatten and to_vec1
        // Note: converting f32 to f64 as TensorValue uses f64
        let data_f32: Vec<f32> = t_candle.flatten_all().unwrap().to_vec1().unwrap();
        let data_f64: Vec<f64> = data_f32.into_iter().map(|x| x as f64).collect();
        let shape = t_candle.dims().to_vec();

        let tensor_value = TensorValue {
            data: data_f64,
            shape,
        };

        let mut ctx = GLOBAL_CONTEXT.lock().unwrap();
        ctx.insert(name, tensor_value);
        // println!("Registered tensor: {}", name);
    }
}
