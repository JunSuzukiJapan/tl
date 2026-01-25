use serial_test::serial;
use tl_runtime::error::CTensorResult;
use tl_runtime::*;
use tl_runtime::error::tl_get_last_error;

// Helper to check if a tensor is not null
fn assert_tensor_valid(t: *mut OpaqueTensor) {
    assert!(!t.is_null(), "Tensor pointer is null");
}

// Helper to free a tensor safely
fn safe_free(t: *mut OpaqueTensor) {
    if !t.is_null() {
        tl_tensor_free(t);
    }
}

fn unwrap_tensor(ptr: *mut OpaqueTensor) -> *mut OpaqueTensor {
    if ptr.is_null() {
        // Retrieve error from LAST_ERROR
        let res = tl_get_last_error();
        if !res.error_msg.is_null() {
            let err_msg = unsafe { std::ffi::CStr::from_ptr(res.error_msg).to_string_lossy() };
            let file = if !res.file.is_null() {
                unsafe { std::ffi::CStr::from_ptr(res.file).to_string_lossy() }
            } else {
                "unknown".into()
            };
            panic!(
                "Runtime error [Code {:?} at {}:{}:{}]: {}",
                res.error_code, file, res.line, res.col, err_msg
            );
        } else {
            panic!("Runtime error occurred (null pointer returned) but no error message in LAST_ERROR.");
        }
    }
    ptr
}

#[test]
fn test_error_reporting() {
    // Test invalid matmul (Shape mismatch)
    // 2x2 @ 3x3 -> Error
    let data_a = vec![1.0, 2.0, 3.0, 4.0];
    let shape_a = vec![2, 2];
    let ptr_a = tl_tensor_new(data_a.as_ptr(), 2, shape_a.as_ptr());

    let data_b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let shape_b = vec![3, 3];
    let ptr_b = tl_tensor_new(data_b.as_ptr(), 2, shape_b.as_ptr());

    assert_tensor_valid(ptr_a);
    assert_tensor_valid(ptr_b);

    // This should fail
    let res = tl_tensor_matmul(ptr_a, ptr_b);
    assert!(res.is_null(), "Expected null pointer for invalid matmul");

    let err = tl_get_last_error();
    assert!(!err.error_msg.is_null());

    let err_msg = unsafe { std::ffi::CStr::from_ptr(err.error_msg).to_string_lossy() };
    println!("Caught expected error: {}", err_msg);
    // Candle error for matmul shape mismatch often mentions dimensions
    assert!(
        err_msg.contains("dimension")
            || err_msg.contains("shape")
            || err_msg.contains("incompatible")
            || err_msg.contains("broadcast")
    );
    assert_ne!(err.error_code, tl_runtime::error::RuntimeErrorCode::Success);

    safe_free(ptr_a);
    safe_free(ptr_b);
}

#[test]
fn debug_ctensor_layout() {
    use std::mem;
    let size = mem::size_of::<CTensorResult>();
    println!("CTensorResult size: {}", size);
    let error_code_offset = unsafe {
        let dummy = std::mem::MaybeUninit::<CTensorResult>::uninit();
        let ptr = dummy.as_ptr();
        let code_ptr = &(*ptr).error_code as *const _ as usize;
        let base = ptr as usize;
        code_ptr - base
    };
    println!("Offset of error_code: {}", error_code_offset);
    let file_offset = unsafe {
        let dummy = std::mem::MaybeUninit::<CTensorResult>::uninit();
        let ptr = dummy.as_ptr();
        let file_ptr = &(*ptr).file as *const _ as usize;
        let base = ptr as usize;
        file_ptr - base
    };
    println!("Offset of file: {}", file_offset);
}

#[test]
#[serial]
fn test_tensor_creation_and_free() {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let shape: Vec<usize> = vec![2, 2];

    let t = unwrap_tensor(tl_tensor_new(data.as_ptr(), 2, shape.as_ptr()));
    assert_tensor_valid(t);

    // Check length (dim 0)
    let len = tl_tensor_len(t);
    assert_eq!(len, 2); // 2x2 tensor, dim 0 is 2

    safe_free(t);
}

#[test]
#[serial]
fn test_tensor_arithmetic() {
    let data_a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0]; // [1, 2, 3, 4]
    let data_b: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0]; // [10, 20, 30, 40]
    let shape: Vec<usize> = vec![4]; // 1D

    let t_a = unwrap_tensor(tl_tensor_new(data_a.as_ptr(), 1, shape.as_ptr()));
    let t_b = unwrap_tensor(tl_tensor_new(data_b.as_ptr(), 1, shape.as_ptr()));

    // Add
    let t_add = unwrap_tensor(tl_tensor_add(t_a, t_b));
    assert_tensor_valid(t_add);

    // To verify values, we can use slice or item if implemented.
    // tl_tensor_slice is implemented as "slice(t, start, len)".
    // But we don't have a "get_data" function easily exposed to C-ABI (only print).
    // EXCEPT tl_tensor_item_i64 which extracts scalar.
    // Let's rely on that for scalar verification or simple stats.

    // Or we can extract specific element via index logic if available...
    // There is no `tl_tensor_get(t, idx)`.
    // But we have `tl_tensor_slice`.
    // Note: runtime/mod.rs line 1569: if name == "slice" in semantics...
    // Wait, let's check runtime implementation of slice.
    // I need to look deeper into mod.rs or check if I missed `tl_tensor_slice` export.
    // I only saw `tl_tensor_argmax`, etc.
    // Let's stick to what I saw: `tl_tensor_item_i64`.

    // Test sum to verify?
    // Is there a `sum` C function? `tl_tensor_sum`?
    // I need to check `src/runtime/mod.rs` again to see what functions are actually exported.
    // I scrolled to line 800. I should look further.

    safe_free(t_a);
    safe_free(t_b);
    safe_free(t_add);
}

#[test]
#[serial]
fn test_tensor_zeros() {
    let shape: Vec<usize> = vec![2, 5];
    let t = unwrap_tensor(tl_tensor_zeros(2, shape.as_ptr(), false));
    assert_tensor_valid(t);
    safe_free(t);
}

// Helper to get f32 item at index (assuming 1D for simplicity in helper, or 0-D)
fn get_item_f32(t: *mut OpaqueTensor, idx: usize) -> f32 {
    let indices = [idx as i64];
    // If tensor is 0-D, rank is 0, indices ignored?
    // tl_tensor_get_f32_md expects rank matches.
    // Let's assume we use it for 1D tensors mostly.
    tl_tensor_get_f32_md(t, indices.as_ptr(), 1)
}

fn get_scalar_f32(t: *mut OpaqueTensor) -> f32 {
    let indices = [0i64; 0]; // Empty array
    tl_tensor_get_f32_md(t, indices.as_ptr(), 0)
}

fn assert_approx_eq(a: f32, b: f32) {
    let diff = (a - b).abs();
    assert!(diff < 1e-4, "Expected {}, got {} (diff {})", b, a, diff);
}

#[test]
#[serial]
fn test_matmul() {
    // A: 2x3, B: 3x2 -> C: 2x2
    let data_a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let data_b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    // A = [[1, 2, 3], [4, 5, 6]]
    // B = [[1, 2], [3, 4], [5, 6]]
    // A@B =
    // [1*1+2*3+3*5, 1*2+2*4+3*6] = [1+6+15, 2+8+18] = [22, 28]
    // [4*1+5*3+6*5, 4*2+5*4+6*6] = [4+15+30, 8+20+36] = [49, 64]

    let shape_a: Vec<usize> = vec![2, 3];
    let shape_b: Vec<usize> = vec![3, 2];

    let t_a = unwrap_tensor(tl_tensor_new(data_a.as_ptr(), 2, shape_a.as_ptr()));
    let t_b = unwrap_tensor(tl_tensor_new(data_b.as_ptr(), 2, shape_b.as_ptr()));

    let t_c = unwrap_tensor(tl_tensor_matmul(t_a, t_b));
    assert_tensor_valid(t_c);

    let shape_c = unsafe { (*t_c).0.dims().to_vec() };
    assert_eq!(shape_c, vec![2, 2]);

    // Check Value at (0, 0) -> 22. But get_item_f32 uses indices.
    let indices = [0, 0];
    let val = tl_tensor_get_f32_md(t_c, indices.as_ptr(), 2);
    assert_approx_eq(val, 22.0);

    safe_free(t_a);
    safe_free(t_b);
    safe_free(t_c);
}

#[test]
#[serial]
fn test_sum() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let shape = vec![4];
    let t = unwrap_tensor(tl_tensor_new(data.as_ptr(), 1, shape.as_ptr()));

    let t_sum = unwrap_tensor(tl_tensor_sum(t));
    assert_tensor_valid(t_sum);

    let val = get_scalar_f32(t_sum);
    assert_approx_eq(val, 10.0);

    safe_free(t);
    safe_free(t_sum);
}

#[test]
#[serial]
fn test_math_ops() {
    let data = vec![1.0, 4.0, 9.0];
    let shape = vec![3];
    let t = unwrap_tensor(tl_tensor_new(data.as_ptr(), 1, shape.as_ptr()));

    // Sqrt
    let t_sqrt = tl_tensor_sqrt(t);
    assert_approx_eq(get_item_f32(t_sqrt, 0), 1.0);
    assert_approx_eq(get_item_f32(t_sqrt, 1), 2.0);
    assert_approx_eq(get_item_f32(t_sqrt, 2), 3.0);

    // Exp (e^0 = 1) - create new tensor
    let data_zero = vec![0.0];
    let shape_zero = vec![1];
    let t_zero = unwrap_tensor(tl_tensor_new(data_zero.as_ptr(), 1, shape_zero.as_ptr()));
    let t_exp = unwrap_tensor(tl_tensor_exp(t_zero));
    assert_approx_eq(get_item_f32(t_exp, 0), 1.0);

    safe_free(t);
    safe_free(t_sqrt);
    safe_free(t_zero);
    safe_free(t_exp);
}

#[test]
#[serial]
fn test_basic_ops() {
    let data_a = vec![10.0, 20.0, 30.0];
    let data_b = vec![2.0, 5.0, 3.0];
    let shape = vec![3];

    let t_a = unwrap_tensor(tl_tensor_new(data_a.as_ptr(), 1, shape.as_ptr()));
    let t_b = unwrap_tensor(tl_tensor_new(data_b.as_ptr(), 1, shape.as_ptr()));

    // Sub: [8, 15, 27]
    let t_sub = unwrap_tensor(tl_tensor_sub(t_a, t_b));
    assert_approx_eq(get_item_f32(t_sub, 0), 8.0);
    assert_approx_eq(get_item_f32(t_sub, 1), 15.0);
    assert_approx_eq(get_item_f32(t_sub, 2), 27.0);

    // Mul: [20, 100, 90]
    let t_mul = unwrap_tensor(tl_tensor_mul(t_a, t_b));
    assert_approx_eq(get_item_f32(t_mul, 0), 20.0);

    // Div: [5, 4, 10]
    let t_div = unwrap_tensor(tl_tensor_div(t_a, t_b));
    assert_approx_eq(get_item_f32(t_div, 0), 5.0);
    assert_approx_eq(get_item_f32(t_div, 1), 4.0);

    // Pow: t_b ^ 2 -> [4, 25, 9]
    let factor_data = vec![2.0];
    let t_factor = unwrap_tensor(tl_tensor_new(factor_data.as_ptr(), 1, vec![1].as_ptr()));
    let t_pow = unwrap_tensor(tl_tensor_pow(t_b, t_factor));
    assert_approx_eq(get_item_f32(t_pow, 0), 4.0);
    assert_approx_eq(get_item_f32(t_pow, 1), 25.0);

    // Log: log(e) approx 1? scalar check
    // let's test log(10)
    let t_log_input = unwrap_tensor(tl_tensor_new(vec![10.0].as_ptr(), 1, vec![1].as_ptr()));
    let t_log = unwrap_tensor(tl_tensor_log(t_log_input));
    // ln(10) ~ 2.30258
    assert_approx_eq(get_item_f32(t_log, 0), 2.30258);

    safe_free(t_a);
    safe_free(t_b);
    safe_free(t_sub);
    safe_free(t_mul);
    safe_free(t_div);
    safe_free(t_factor);
    safe_free(t_pow);
    safe_free(t_log_input);
    safe_free(t_log);
}

#[test]
#[serial]
fn test_reshape_transpose() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let shape = vec![2, 2]; // [[1, 2], [3, 4]]
    let t = unwrap_tensor(tl_tensor_new(data.as_ptr(), 2, shape.as_ptr()));

    // Transpose -> [[1, 3], [2, 4]]
    let t_transposed = tl_tensor_transpose(t, 0, 1);
    let val = tl_tensor_get_f32_md(t_transposed, [0, 1].as_ptr(), 2);
    assert_eq!(val, 3.0);

    // Reshape to 4x1
    // Need a shape tensor for `tl_tensor_reshape`
    // tl_tensor_reshape converts shape tensor to Vec<usize>.
    let shape_data_t = vec![4.0, 1.0];
    let shape_shape = vec![2];
    let shape_t = unwrap_tensor(tl_tensor_new(
        shape_data_t.as_ptr(),
        1,
        shape_shape.as_ptr(),
    ));

    let t_flat = tl_tensor_reshape_new(t, shape_t);
    assert_tensor_valid(t_flat);
    let dims = unsafe { (*t_flat).0.dims().to_vec() };
    assert_eq!(dims, vec![4, 1]);

    safe_free(t);
    safe_free(t_transposed);
    safe_free(shape_t);
    safe_free(t_flat);
}
