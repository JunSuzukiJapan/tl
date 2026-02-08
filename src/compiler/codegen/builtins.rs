use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::module::Module as InkwellModule;
use inkwell::AddressSpace;

use crate::compiler::ast::Type;
use tl_runtime as runtime;
use tl_cpu::ffi as cpu_ffi;

pub fn declare_runtime_functions<'ctx>(
    context: &'ctx Context,
    module: &InkwellModule<'ctx>,
    execution_engine: &ExecutionEngine<'ctx>,
) {
    // ゼロオーバーヘッド CPU/GPU 切替: JIT リンク時にシンボルを差し替え
    let is_cpu = std::env::var("TL_DEVICE").map_or(false, |d| d == "cpu");

    // テンソル FFI のマッピングマクロ
    // is_cpu なら cpu_ffi の関数を、そうでなければ runtime の関数をマッピング
    macro_rules! map_tensor_fn {
        ($name:expr, $gpu:expr, $cpu:expr) => {
            if let Some(f) = module.get_function($name) {
                if is_cpu {
                    execution_engine.add_global_mapping(&f, $cpu as usize);
                } else {
                    execution_engine.add_global_mapping(&f, $gpu as usize);
                }
            }
        };
    }
    let i64_type = context.i64_type(); // usize
    let i32_type = context.i32_type();
    // let i8_type = context.i8_type();
    let f32_type = context.f32_type();
    let f32_ptr = context.ptr_type(AddressSpace::default());
    let usize_ptr = context.ptr_type(AddressSpace::default());
    let i64_ptr = context.ptr_type(AddressSpace::default());
    let void_ptr = context.ptr_type(AddressSpace::default()); // OpaqueTensor*
    let void_type = context.void_type();
    let i8_ptr = context.ptr_type(AddressSpace::default());

    // FFI Result struct: { tensor, error_msg, error_code, file, line, col }
    let c_tensor_result_type = context.struct_type(
        &[
            void_ptr.into(), // tensor (0)
            i8_ptr.into(),   // error_msg (1)
            i32_type.into(), // error_code (2)
            i8_ptr.into(),   // file (3)
            i32_type.into(), // line (4)
            i32_type.into(), // col (5)
        ],
        false,
    );

    // Helper to add function only if not exists

    let add_fn = |name: &str, ty: inkwell::types::FunctionType<'ctx>| {
        if module.get_function(name).is_none() {
            module.add_function(name, ty, None);
        }
    };

    let print_i64_type = void_type.fn_type(&[i64_type.into()], false);
    add_fn("tl_print_i64", print_i64_type.clone());
    add_fn("tl_display_i64", print_i64_type);

    let print_i32_type = void_type.fn_type(&[i32_type.into()], false);
    add_fn("tl_print_i32", print_i32_type.clone());
    add_fn("tl_display_i32", print_i32_type);

    // tl_print_char(c: i32)
    let print_char_type = void_type.fn_type(&[i32_type.into()], false);
    add_fn("tl_print_char", print_char_type.clone());
    add_fn("tl_display_char", print_char_type);

    let print_f32_type = void_type.fn_type(&[f32_type.into()], false);
    add_fn("tl_print_f32", print_f32_type.clone());
    add_fn("tl_display_f32", print_f32_type);

    let f64_type = context.f64_type();
    let print_f64_type = void_type.fn_type(&[f64_type.into()], false);
    add_fn("tl_print_f64", print_f64_type.clone());
    add_fn("tl_display_f64", print_f64_type);

    let print_bool_type = void_type.fn_type(&[context.bool_type().into()], false);
    add_fn("tl_print_bool", print_bool_type.clone());
    add_fn("tl_display_bool", print_bool_type);

    let f32_unary_type = f32_type.fn_type(&[f32_type.into()], false);
    let f32_binary_type = f32_type.fn_type(&[f32_type.into(), f32_type.into()], false);
    let f32_powi_type = f32_type.fn_type(&[f32_type.into(), i64_type.into()], false);
    let f64_unary_type = f64_type.fn_type(&[f64_type.into()], false);
    let f64_binary_type = f64_type.fn_type(&[f64_type.into(), f64_type.into()], false);
    let f64_powi_type = f64_type.fn_type(&[f64_type.into(), i64_type.into()], false);
    let i64_unary_type = i64_type.fn_type(&[i64_type.into()], false);
    let i64_binary_type = i64_type.fn_type(&[i64_type.into(), i64_type.into()], false);
    let i32_unary_type = i32_type.fn_type(&[i32_type.into()], false);
    let i32_binary_type = i32_type.fn_type(&[i32_type.into(), i32_type.into()], false);
    let i64_bool_unary_type = context.bool_type().fn_type(&[i64_type.into()], false);
    let i32_bool_unary_type = context.bool_type().fn_type(&[i32_type.into()], false);

    let f32_unary_methods = [
        "abs",
        "acos",
        "acosh",
        "asin",
        "asinh",
        "atan",
        "atanh",
        "cbrt",
        "ceil",
        "cos",
        "cosh",
        "exp",
        "exp2",
        "exp_m1",
        "floor",
        "fract",
        "ln",
        "ln_1p",
        "log10",
        "log2",
        "recip",
        "round",
        "signum",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
        "to_degrees",
        "to_radians",
        "trunc",
    ];
    for name in f32_unary_methods {
        add_fn(&format!("tl_f32_{}", name), f32_unary_type);
    }

    let f32_binary_methods = ["atan2", "copysign", "hypot", "log", "powf"];
    for name in f32_binary_methods {
        add_fn(&format!("tl_f32_{}", name), f32_binary_type);
    }
    add_fn("tl_f32_powi", f32_powi_type);

    let f64_unary_methods = [
        "abs",
        "acos",
        "acosh",
        "asin",
        "asinh",
        "atan",
        "atanh",
        "cbrt",
        "ceil",
        "cos",
        "cosh",
        "exp",
        "exp2",
        "exp_m1",
        "floor",
        "fract",
        "ln",
        "ln_1p",
        "log10",
        "log2",
        "recip",
        "round",
        "signum",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
        "to_degrees",
        "to_radians",
        "trunc",
    ];
    for name in f64_unary_methods {
        add_fn(&format!("tl_f64_{}", name), f64_unary_type);
    }
    let f64_binary_methods = ["atan2", "copysign", "hypot", "log", "powf"];
    for name in f64_binary_methods {
        add_fn(&format!("tl_f64_{}", name), f64_binary_type);
    }
    add_fn("tl_f64_powi", f64_powi_type);

    let i64_unary_methods = ["abs", "signum"];
    for name in i64_unary_methods {
        add_fn(&format!("tl_i64_{}", name), i64_unary_type);
    }
    let i64_binary_methods = ["div_euclid", "rem_euclid", "pow"];
    for name in i64_binary_methods {
        add_fn(&format!("tl_i64_{}", name), i64_binary_type);
    }
    add_fn("tl_i64_is_positive", i64_bool_unary_type);
    add_fn("tl_i64_is_negative", i64_bool_unary_type);

    let i32_unary_methods = ["abs", "signum"];
    for name in i32_unary_methods {
        add_fn(&format!("tl_i32_{}", name), i32_unary_type);
    }
    let i32_binary_methods = ["div_euclid", "rem_euclid", "pow"];
    for name in i32_binary_methods {
        add_fn(&format!("tl_i32_{}", name), i32_binary_type);
    }
    add_fn("tl_i32_is_positive", i32_bool_unary_type);
    add_fn("tl_i32_is_negative", i32_bool_unary_type);

    let print_str_type = void_type.fn_type(&[void_ptr.into()], false);
    add_fn("tl_print_string", print_str_type.clone());
    add_fn("tl_display_string", print_str_type);

    // tl_file_exists(path: *const c_char) -> bool
    let file_exists_type = context.bool_type().fn_type(&[i8_ptr.into()], false);
    add_fn("tl_file_exists", file_exists_type);

    // Workaround: i64 return type
    let file_exists_i64_type = i64_type.fn_type(&[i8_ptr.into()], false);
    add_fn("tl_file_exists_i64", file_exists_i64_type);

    // tl_download_file(url: *const c_char, path: *const c_char) -> i64
    let download_file_type = i64_type.fn_type(&[i8_ptr.into(), i8_ptr.into()], false);
    add_fn("tl_download_file", download_file_type);

    // tl_read_file(path: *const c_char) -> *const c_char
    let read_file_type = i8_ptr.fn_type(&[i8_ptr.into()], false);
    add_fn("tl_read_file", read_file_type);

    // tl_write_file(path: *const c_char, content: *const c_char) -> i64
    let write_file_type = i64_type.fn_type(&[i8_ptr.into(), i8_ptr.into()], false);
    add_fn("tl_write_file", write_file_type);

    // tl_print_ptr for debugging tensor pointers
    let print_ptr_type = void_type.fn_type(&[void_ptr.into()], false);
    add_fn("tl_print_ptr", print_ptr_type);

    // tl_report_runtime_error(msg: *const c_char) -> void
    let report_err_type = void_type.fn_type(&[i8_ptr.into()], false);
    add_fn("tl_report_runtime_error", report_err_type);

    // tl_handle_runtime_error(code: u32, msg: *const i8, file: *const i8, line: u32, col: u32) -> void
    let handle_err_type = void_type.fn_type(
        &[
            i32_type.into(),
            i8_ptr.into(),
            i8_ptr.into(),
            i32_type.into(),
            i32_type.into(),
        ],
        false,
    );
    add_fn("tl_handle_runtime_error", handle_err_type.clone());

    // tl_amend_error_loc(file: *const i8, line: u32, col: u32) -> void
    let amend_err_type =
        void_type.fn_type(&[i8_ptr.into(), i32_type.into(), i32_type.into()], false);
    add_fn("tl_amend_error_loc", amend_err_type);

    if let Some(f) = module.get_function("tl_handle_runtime_error") {
        execution_engine.add_global_mapping(&f, runtime::tl_handle_runtime_error as usize);
    }
    if let Some(f) = module.get_function("tl_amend_error_loc") {
        execution_engine.add_global_mapping(&f, runtime::tl_amend_error_loc as usize);
    }

    // tl_get_last_error() -> CTensorResult
    let get_last_error_type = c_tensor_result_type.fn_type(&[], false);
    add_fn("tl_get_last_error", get_last_error_type);

    // tl_set_device(name: *const i8) -> void
    let set_dev_type = void_type.fn_type(&[void_ptr.into()], false);
    add_fn("tl_set_device", set_dev_type);

    // tl_tensor_enable_grad(t: *mut OpaqueTensor) -> void
    let enable_grad_type = void_type.fn_type(&[void_ptr.into()], false);
    add_fn("tl_tensor_enable_grad", enable_grad_type);

    // tl_tensor_to_device(tensor: *mut Opaque, name: *const i8) -> *mut Opaque
    let to_dev_type = void_ptr.fn_type(&[void_ptr.into(), void_ptr.into()], false);
    add_fn("tl_tensor_to_device", to_dev_type);

    // malloc(size: i64) -> *u8
    let malloc_type = void_ptr.fn_type(&[i64_type.into()], false);
    add_fn("malloc", malloc_type);

    // calloc(num: i64, size: i64) -> *u8
    let calloc_type = void_ptr.fn_type(&[i64_type.into(), i64_type.into()], false);
    add_fn("calloc", calloc_type);

    // free(ptr: *u8) -> void
    let free_type = void_type.fn_type(&[void_ptr.into()], false);
    add_fn("free", free_type);

    // realloc(ptr: *u8, size: i64) -> *u8
    let realloc_type = void_ptr.fn_type(&[void_ptr.into(), i64_type.into()], false);
    add_fn("realloc", realloc_type);

    // abort() -> void (for panic! support)
    let abort_type = void_type.fn_type(&[], false);
    add_fn("abort", abort_type);

    // tl_tensor_dim(t: *mut OpaqueTensor, dim_idx: usize) -> i64
    let dim_type = i64_type.fn_type(&[void_ptr.into(), i64_type.into()], false);
    add_fn("tl_tensor_dim", dim_type);

    // tl_tensor_get_f32_md(t: *mut OpaqueTensor, indices: *const i64, rank: usize) -> f32
    let get_md_type = f32_type.fn_type(&[void_ptr.into(), i64_ptr.into(), i64_type.into()], false);
    add_fn("tl_tensor_get_f32_md", get_md_type);
    // tl_tensor_get_i64_md(t: *mut OpaqueTensor, indices: *const i64, rank: usize) -> i64
    let get_md_i64_type =
        i64_type.fn_type(&[void_ptr.into(), i64_ptr.into(), i64_type.into()], false);
    add_fn("tl_tensor_get_i64_md", get_md_i64_type);

    // tl_tensor_set_f32_md(t: *mut OpaqueTensor, indices: *const i64, rank: usize, val: f32) -> *mut OpaqueTensor
    let set_md_type = void_ptr.fn_type(
        &[
            void_ptr.into(),
            i64_ptr.into(),
            i64_type.into(),
            f32_type.into(),
        ],
        false,
    );
    add_fn("tl_tensor_set_f32_md", set_md_type);

    // tl_tensor_new(data: *const f32, rank: usize, shape: *const usize) -> *mut OpaqueTensor
    let new_type = void_ptr.fn_type(&[f32_ptr.into(), i64_type.into(), usize_ptr.into()], false);
    add_fn("tl_tensor_new", new_type);

    // tl_tensor_from_i64_array(data: *const i64, len: usize) -> *mut OpaqueTensor
    let from_i64_type = void_ptr.fn_type(&[i64_ptr.into(), i64_type.into()], false);
    add_fn("tl_tensor_from_i64_array", from_i64_type);

    // tl_tensor_new_i64(data: *const i64, rank: usize, shape: *const usize) -> *mut OpaqueTensor
    let new_i64_type =
        void_ptr.fn_type(&[i64_ptr.into(), i64_type.into(), usize_ptr.into()], false);
    add_fn("tl_tensor_new_i64", new_i64_type);

    let binop_type = void_ptr.fn_type(&[void_ptr.into(), void_ptr.into()], false);
    add_fn("tl_tensor_sub", binop_type);

    // tl_tensor_free(t: *mut) -> void
    let free_type = void_type.fn_type(&[void_ptr.into()], false);
    add_fn("tl_tensor_free", free_type);

    // tl_tensor_clone(t: *mut) -> *mut OpaqueTensor
    let clone_type = void_ptr.fn_type(&[void_ptr.into()], false);
    add_fn("tl_tensor_clone", clone_type);

    // tl_tensor_acquire(t: *mut) -> void
    add_fn("tl_tensor_acquire", free_type);

    // tl_tensor_release(t: *mut) -> void
    add_fn("tl_tensor_release", free_type);

    // tl_tensor_data(t: *mut) -> *mut
    let data_type = void_ptr.fn_type(&[void_ptr.into()], false);
    add_fn("tl_tensor_data", data_type);

    // tl_tensor_numel(t: *mut) -> i64
    let numel_type = i64_type.fn_type(&[void_ptr.into()], false);
    add_fn("tl_tensor_numel", numel_type);

    // tl_ptr_dec_ref(ptr: *mut) -> i32
    let dec_ref_type = i32_type.fn_type(&[void_ptr.into()], false);
    add_fn("tl_ptr_dec_ref", dec_ref_type);

    // tl_ptr_inc_ref(ptr: *mut) -> void
    let inc_ref_type = void_type.fn_type(&[void_ptr.into()], false);
    add_fn("tl_ptr_inc_ref", inc_ref_type);

    // tl_ptr_inc_ref_i64(addr: i64) -> void
    let inc_ref_i64_type = void_type.fn_type(&[i64_type.into()], false);
    add_fn("tl_ptr_inc_ref_i64", inc_ref_i64_type);

    // tl_mem_free(ptr: *mut) -> void
    add_fn("tl_mem_free", free_type);


    // tl_tensor_add(a: *mut, b: *mut) -> *mut OpaqueTensor
    let bin_type = void_ptr.fn_type(&[void_ptr.into(), void_ptr.into()], false);
    add_fn("tl_tensor_add", bin_type);
    add_fn("tl_tensor_mul", bin_type);

    let print_type = void_type.fn_type(&[void_ptr.into()], false);
    add_fn("tl_print_string", print_type.clone());
    add_fn("tl_string_print", print_type.clone());
    add_fn("tl_display_string", print_type.clone());
    add_fn("tl_tensor_print", print_type.clone());
    add_fn("tl_tensor_display", print_type.clone());
    add_fn("tl_tensor_print_1", print_type.clone());
    add_fn("tl_tensor_print_2", print_type.clone());
    add_fn("tl_tensor_print_3", print_type.clone());

    // tl_tensor_get(t: *mut, index: usize) -> f32 (Simplification: 1D get)
    let get_type = f32_type.fn_type(&[void_ptr.into(), i64_type.into()], false);
    add_fn("tl_tensor_get", get_type);

    // tl_vec_slice(t: *mut, start: usize, len: usize) -> *mut
    let slice_type = void_ptr.fn_type(&[void_ptr.into(), i64_type.into(), i64_type.into()], false);
    add_fn("tl_tensor_slice", slice_type);




    // tl_string_len(s: *const c_char) -> i64
    let len_str_type = i64_type.fn_type(&[i8_ptr.into()], false);
    add_fn("tl_string_len", len_str_type);
    
    // tl_string_from_char(c: i32) -> *mut c_char
    let from_char_type = i8_ptr.fn_type(&[i32_type.into()], false);
    add_fn("tl_string_from_char", from_char_type);

    // tl_string_char_at(s: *mut StringStruct, index: i64) -> i64 (char as i64)
    let char_at_type = i64_type.fn_type(&[void_ptr.into(), i64_type.into()], false);
    add_fn("tl_string_char_at", char_at_type);

    // tl_string_eq(s1: *mut StringStruct, s2: *mut StringStruct) -> bool
    let string_eq_type = context.bool_type().fn_type(&[void_ptr.into(), void_ptr.into()], false);
    add_fn("tl_string_eq", string_eq_type);

    // tl_string_new(s: *const c_char) -> *mut StringStruct
    // StringStruct is { ptr, len }, pointer to it is void* (or specific struct type?)
    // In LLVM IR we invoke it as returning pointer to { i8*, i64 }.
    // So we can declare it returning ptr.
    // The struct type definition is local in expr.rs usually, but we can treat as void* here for declaration?
    // No, expr.rs casts return value anyway? 
    // Wait, expr.rs uses build_call. The return type of the FUNCTION matches build_call expectation.
    // If function returns void*, then build_call returns void*.
    // But expr.rs expects pointer to StringStruct. void* is fine.
    let new_string_type = void_ptr.fn_type(&[i8_ptr.into()], false);
    add_fn("tl_string_new", new_string_type);
    
    // tl_string_free(s: *mut StringStruct) -> void
    add_fn("tl_string_free", free_type);

    // tl_tensor_len(t: *mut) -> i64
    let len_type = i64_type.fn_type(&[void_ptr.into()], false);
    add_fn("tl_tensor_len", len_type);

    // tl_tensor_neg(t: *mut) -> *mut
    let neg_type = void_ptr.fn_type(&[void_ptr.into()], false);
    add_fn("tl_tensor_neg", neg_type);

    // tl_tensor_transpose(t: *mut, d0: usize, d1: usize) -> *mut
    let transpose_type =
        void_ptr.fn_type(&[void_ptr.into(), i64_type.into(), i64_type.into()], false);
    add_fn("tl_tensor_transpose", transpose_type);

    // tl_tensor_pow(t: *mut Tensor, exponent: *mut Tensor) -> *mut OpaqueTensor
    let pow_type = void_ptr.fn_type(&[void_ptr.into(), void_ptr.into()], false);
    add_fn("tl_tensor_pow", pow_type);

    // tl_tensor_pow_scalar(t: *mut Tensor, exponent: f32) -> *mut OpaqueTensor
    let pow_scalar_type = void_ptr.fn_type(&[void_ptr.into(), f32_type.into()], false);
    add_fn("tl_tensor_pow_scalar", pow_scalar_type);

    // tl_tensor_sqrt(t: *mut Tensor) -> *mut Tensor
    let unary_type = void_ptr.fn_type(&[void_ptr.into()], false);
    add_fn("tl_tensor_sqrt", unary_type);

    // Transformer Ops
    // tl_tensor_sin(t: *mut Tensor) -> *mut Tensor
    add_fn("tl_tensor_sin", unary_type); // Same signature as sqrt

    // tl_tensor_cos(t: *mut Tensor) -> *mut Tensor
    add_fn("tl_tensor_cos", unary_type);

    // tl_tensor_relu(t: *mut Tensor) -> *mut Tensor
    add_fn("tl_tensor_relu", unary_type);

    // tl_tensor_gelu(t: *mut Tensor) -> *mut Tensor
    add_fn("tl_tensor_gelu", unary_type);

    // tl_tensor_tril(t: *mut Tensor, diagonal: i32) -> *mut Tensor
    let i32_type = context.i32_type();
    let tril_type = void_ptr.fn_type(&[void_ptr.into(), i32_type.into()], false);
    add_fn("tl_tensor_tril", tril_type);

    // Explicitly add tl_clear_grads
    let clear_grads_type = void_type.fn_type(&[], false);
    add_fn("tl_clear_grads", clear_grads_type);

    // tl_tensor_set_f32_md(t: *mut, indices: *const i64, rank: usize, val: f32) -> *mut
    let set_md_type = void_ptr.fn_type(
        &[
            void_ptr.into(),
            void_ptr.into(), // indices ptr
            i64_type.into(), // rank
            f32_type.into(), // val
        ],
        false,
    );
    add_fn("tl_tensor_set_f32_md", set_md_type);

    // tl_tensor_get_f32_md(t: *mut, indices: *const i64, rank: usize) -> f32
    let get_md_type = f32_type.fn_type(
        &[
            void_ptr.into(),
            void_ptr.into(), // indices ptr
            i64_type.into(), // rank
        ],
        false,
    );
    add_fn("tl_tensor_get_f32_md", get_md_type);

    // Manual mapping for tl_clear_grads to avoid dlsym issues
    map_tensor_fn!("tl_clear_grads", runtime::tl_clear_grads, cpu_ffi::tl_cpu_clear_grads);
    if let Some(f) = module.get_function("tl_file_exists") {
        execution_engine.add_global_mapping(&f, runtime::tl_file_exists as usize);
    }
    if let Some(f) = module.get_function("tl_file_exists_i64") {
        execution_engine.add_global_mapping(&f, runtime::tl_file_exists_i64 as usize);
    }
    if let Some(f) = module.get_function("tl_download_file") {
        execution_engine.add_global_mapping(&f, runtime::tl_download_file as usize);
    }
    if let Some(f) = module.get_function("tl_read_file") {
        execution_engine.add_global_mapping(&f, runtime::tl_read_file as usize);
    }
    if let Some(f) = module.get_function("tl_write_file") {
        execution_engine.add_global_mapping(&f, runtime::tl_write_file as usize);
    }

    // String mappings
    if let Some(f) = module.get_function("tl_string_new") {
        execution_engine.add_global_mapping(&f, runtime::tl_string_new as usize);
    }
    if let Some(f) = module.get_function("tl_print_string") {
        execution_engine.add_global_mapping(&f, runtime::tl_print_string as usize);
    }
    // Add mapping for legacy/extern default name
    if let Some(f) = module.get_function("tl_string_print") {
        execution_engine.add_global_mapping(&f, runtime::tl_string_print as usize);
    }
    if let Some(f) = module.get_function("tl_display_string") {
        execution_engine.add_global_mapping(&f, runtime::tl_display_string as usize);
    }
    if let Some(f) = module.get_function("tl_string_concat") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_string_concat as usize);
    }
    if let Some(f) = module.get_function("tl_string_len") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_string_len as usize);
    }
    if let Some(f) = module.get_function("tl_string_from_char") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_string_from_char as usize);
    }
    if let Some(f) = module.get_function("tl_string_to_i64") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_string_to_i64 as usize);
    }
    if let Some(f) = module.get_function("tl_string_contains") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_string_contains as usize);
    }
    if let Some(f) = module.get_function("tl_string_char_at") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_string_char_at as usize);
    }
    if let Some(f) = module.get_function("tl_string_eq") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_string_eq as usize);
    }
    
    // tl_hash_string(s: *mut StringStruct) -> i64
    let hash_string_type = i64_type.fn_type(&[void_ptr.into()], false);
    add_fn("tl_hash_string", hash_string_type);
    
    if let Some(f) = module.get_function("tl_hash_string") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_hash_string as usize);
    }

    // tl_checkpoint(ctx: *mut, func: *mut, input: *mut) -> *mut
    let checkpoint_type =
        void_ptr.fn_type(&[void_ptr.into(), void_ptr.into(), void_ptr.into()], false);
    add_fn("tl_checkpoint", checkpoint_type);

    if let Some(f) = module.get_function("tl_checkpoint") {
        execution_engine.add_global_mapping(&f, runtime::checkpoint::tl_checkpoint as usize);
    }

    if let Some(f) = module.get_function("tl_set_device") {
        execution_engine.add_global_mapping(&f, runtime::tl_set_device as usize);
    }

    if let Some(f) = module.get_function("tl_tensor_to_device") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_to_device as usize);
    }

    if let Some(f) = module.get_function("tl_report_runtime_error") {
        execution_engine.add_global_mapping(&f, runtime::tl_report_runtime_error as usize);
    }

    if let Some(f) = module.get_function("tl_tensor_replace_data") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_replace_data as usize);
    }

    // tl_tensor_sum_dim(t: *mut Tensor, dim: usize, keep: bool) -> *mut OpaqueTensor
    // usize -> i64 on 64-bit
    let sum_dim_type = void_ptr.fn_type(
        &[void_ptr.into(), i64_type.into(), context.bool_type().into()],
        false,
    );
    add_fn("tl_tensor_sum_dim", sum_dim_type);

    // tl_tensor_embedding(indices: *mut Tensor, weights: *mut Tensor) -> *mut Tensor
    let embedding_type = void_ptr.fn_type(&[void_ptr.into(), void_ptr.into()], false);
    add_fn("tl_tensor_embedding", embedding_type);

    // tl_tensor_sum(t: *mut) -> *mut OpaqueTensor
    let unary_res_type = void_ptr.fn_type(&[void_ptr.into()], false);
    add_fn("tl_tensor_sum", unary_res_type);

    // tl_tensor_div(a: *mut, b: *mut) -> *mut
    add_fn("tl_tensor_div", bin_type);

    // Comparisons
    add_fn("tl_tensor_eq", bin_type);
    add_fn("tl_tensor_neq", bin_type);
    add_fn("tl_tensor_gt", bin_type);
    add_fn("tl_tensor_lt", bin_type);
    add_fn("tl_tensor_ge", bin_type);
    add_fn("tl_tensor_le", bin_type);


    // tl_tensor_matmul(a: *mut, b: *mut) -> *mut
    add_fn("tl_tensor_matmul", bin_type);

    // tl_tensor_rem(a: *mut, b: *mut) -> *mut
    add_fn("tl_tensor_rem", bin_type);


    // Unary ops: exp, log
    add_fn("tl_tensor_exp", unary_res_type);
    add_fn("tl_tensor_log", unary_res_type);

    // Assign ops: add_assign, sub_assign, mul_assign, div_assign (return void)
    let assign_type = void_type.fn_type(&[void_ptr.into(), void_ptr.into()], false);
    add_fn("tl_tensor_add_assign", assign_type);
    add_fn("tl_tensor_sub_assign", assign_type);
    add_fn("tl_tensor_mul_assign", assign_type);
    add_fn("tl_tensor_div_assign", assign_type);

    // Scalar assign ops: *_scalar_f32(tensor, f32) -> void
    let scalar_assign_type =
        void_type.fn_type(&[void_ptr.into(), context.f32_type().into()], false);
    add_fn("tl_tensor_add_assign_scalar_f32", scalar_assign_type);
    add_fn("tl_tensor_sub_assign_scalar_f32", scalar_assign_type);
    add_fn("tl_tensor_mul_assign_scalar_f32", scalar_assign_type);
    add_fn("tl_tensor_div_assign_scalar_f32", scalar_assign_type);
    add_fn("tl_tensor_mod_assign_scalar_f32", scalar_assign_type);

    // Mod assign (tensor, tensor) -> void
    add_fn("tl_tensor_mod_assign", assign_type);

    let i8_ptr = context.ptr_type(AddressSpace::default());
    let register_type = void_type.fn_type(&[i8_ptr.into(), void_ptr.into()], false);
    add_fn("tl_register_tensor", register_type);

    // strcmp(s1: *const i8, s2: *const i8) -> i32
    let strcmp_type = context
        .i32_type()
        .fn_type(&[i8_ptr.into(), i8_ptr.into()], false);
    add_fn("strcmp", strcmp_type);

    // tl_tensor_save(t: *mut Tensor, path: *const i8) -> void
    let save_type = void_type.fn_type(&[void_ptr.into(), i8_ptr.into()], false);
    add_fn("tl_tensor_save", save_type);

    // tl_tensor_load(path: *const i8) -> *mut Tensor
    let load_type = void_ptr.fn_type(&[i8_ptr.into()], false);
    add_fn("tl_tensor_load", load_type);

    // tl_tensor_replace_data(dest: *mut Tensor, src: *mut Tensor) -> void
    let replace_data_type = void_type.fn_type(&[void_ptr.into(), void_ptr.into()], false);
    add_fn("tl_tensor_replace_data", replace_data_type);

    // --- Logic / Knowledge Base Builtins ---

    // tl_kb_add_entity(name: *const i8) -> i64
    let add_entity_type = context.i64_type().fn_type(&[i8_ptr.into()], false);
    add_fn("tl_kb_add_entity", add_entity_type);

    // tl_kb_add_fact_serialized(relation: *const i8) -> void
    let add_fact_serialized_type = void_type.fn_type(&[i8_ptr.into()], false);
    add_fn("tl_kb_add_fact_serialized", add_fact_serialized_type);

    // tl_kb_fact_args_clear()
    add_fn("tl_kb_fact_args_clear", void_type.fn_type(&[], false));

    // tl_kb_fact_args_add_*
    add_fn("tl_kb_fact_args_add_int", void_type.fn_type(&[context.i64_type().into()], false));
    add_fn("tl_kb_fact_args_add_float", void_type.fn_type(&[context.f64_type().into()], false));
    add_fn("tl_kb_fact_args_add_bool", void_type.fn_type(&[context.bool_type().into()], false));
    add_fn("tl_kb_fact_args_add_entity", void_type.fn_type(&[context.i64_type().into()], false));
    add_fn("tl_kb_fact_args_add_string", void_type.fn_type(&[i8_ptr.into()], false));

    // Old API for compatibility
    let args_ptr_type = context.ptr_type(AddressSpace::default());
    let add_fact_type = void_type.fn_type(
        &[
            i8_ptr.into(),
            args_ptr_type.into(),
            context.i64_type().into(), // arity
        ],
        false,
    );
    add_fn("tl_kb_add_fact", add_fact_type);

    // tl_kb_infer() -> void
    let infer_type = void_type.fn_type(&[], false);
    add_fn("tl_kb_infer", infer_type);

    // Rule Builder API
    // tl_kb_rule_start(head_rel: *const i8)
    let rule_start_type = void_type.fn_type(&[i8_ptr.into()], false);
    add_fn("tl_kb_rule_start", rule_start_type);

    // tl_kb_rule_add_head_arg_var(index: i64)
    let arg_i64_type = void_type.fn_type(&[context.i64_type().into()], false);
    add_fn("tl_kb_rule_add_head_arg_var", arg_i64_type);
    add_fn("tl_kb_rule_add_head_arg_const_int", arg_i64_type);
    let arg_f64_type = void_type.fn_type(&[context.f64_type().into()], false);
    add_fn("tl_kb_rule_add_head_arg_const_float", arg_f64_type);
    add_fn("tl_kb_rule_add_head_arg_const_entity", arg_i64_type);

    add_fn("tl_kb_rule_add_body_atom", rule_start_type);
    add_fn("tl_kb_rule_add_body_atom_neg", rule_start_type);

    // tl_kb_rule_add_body_arg_var(index: i64)
    add_fn("tl_kb_rule_add_body_arg_var", arg_i64_type);
    add_fn("tl_kb_rule_add_body_arg_const_int", arg_i64_type);
    add_fn("tl_kb_rule_add_body_arg_const_float", arg_f64_type);
    add_fn("tl_kb_rule_add_body_arg_const_entity", arg_i64_type);

    // tl_kb_rule_finish()
    add_fn("tl_kb_rule_finish", infer_type);

    // --- Map Support ---
    // tl_tensor_map_new() -> *mut Map
    let map_new_type = void_ptr.fn_type(&[], false);
    add_fn("tl_tensor_map_new", map_new_type);

    // tl_tensor_map_insert(map, name, tensor)
    let map_insert_type =
        void_type.fn_type(&[void_ptr.into(), i8_ptr.into(), void_ptr.into()], false);
    add_fn("tl_tensor_map_insert", map_insert_type);

    // tl_tensor_map_save(map, path)
    let map_save_type = void_type.fn_type(&[void_ptr.into(), i8_ptr.into()], false);
    add_fn("tl_tensor_map_save", map_save_type);

    // tl_tensor_map_load(path) -> *mut Map
    let map_load_type = void_ptr.fn_type(&[i8_ptr.into()], false);
    add_fn("tl_tensor_map_load", map_load_type);

    // tl_tensor_map_free(map)
    let map_free_type = void_type.fn_type(&[void_ptr.into()], false);
    add_fn("tl_tensor_map_free", map_free_type);




    // Reshape
    let reshape_dims_type =
        void_ptr.fn_type(&[void_ptr.into(), i64_ptr.into(), i64_type.into()], false);
    add_fn("tl_tensor_reshape_dims", reshape_dims_type);

    let reshape_tensor_type = void_ptr.fn_type(&[void_ptr.into(), void_ptr.into()], false);
    add_fn("tl_tensor_reshape_new", reshape_tensor_type);

    // Randn
    // tl_tensor_randn(rank: usize, shape: *const usize, req_grad: bool) -> *mut OpaqueTensor
    let randn_type = void_ptr.fn_type(
        &[
            i64_type.into(),            // Rank
            usize_ptr.into(),           // Shape Ptr
            context.bool_type().into(), // Req Grad
        ],
        false,
    );
    add_fn("tl_tensor_randn_debug", randn_type);

    // tl_tensor_zeros(rank, shape_ptr, req_grad) -> *mut OpaqueTensor
    let zeros_type = void_ptr.fn_type(
        &[
            i64_type.into(),            // rank
            usize_ptr.into(),           // shape pointer
            context.bool_type().into(), // req_grad
        ],
        false,
    );

    add_fn("tl_tensor_zeros", zeros_type);

    // VarBuilder
    // tl_varbuilder_get(name: *const c_char, rank: usize, shape: *const usize) -> *mut OpaqueTensor
    let varbuilder_get_type =
        void_ptr.fn_type(&[i8_ptr.into(), i64_type.into(), usize_ptr.into()], false);
    add_fn("tl_varbuilder_get", varbuilder_get_type);

    // tl_varbuilder_get_from_tensor(name: *const c_char, shape_tensor: *mut OpaqueTensor) -> *mut OpaqueTensor
    let varbuilder_get_tensor_type = void_ptr.fn_type(&[i8_ptr.into(), void_ptr.into()], false);
    add_fn("tl_varbuilder_get_from_tensor", varbuilder_get_tensor_type);

    // update_all_params(lr: f32)
    let update_type = void_type.fn_type(&[f32_type.into()], false);
    add_fn("tl_update_all_params", update_type);

    // grad(name: *const c_char) -> *mut OpaqueTensor
    let grad_type = void_ptr.fn_type(&[i8_ptr.into()], false);
    add_fn("tl_varbuilder_grad", grad_type);

    // Autograd helpers
    let backward_type = void_type.fn_type(&[void_ptr.into()], false);
    add_fn("tl_tensor_backward", backward_type);

    let grad_fn_type = void_ptr.fn_type(&[void_ptr.into()], false);
    add_fn("tl_tensor_grad", grad_fn_type);

    let detach_type = void_ptr.fn_type(&[void_ptr.into(), context.bool_type().into()], false);
    add_fn("tl_tensor_detach", detach_type);

    // Contiguous (メモリレイアウト連続化)
    let contiguous_type = void_ptr.fn_type(&[void_ptr.into()], false);
    add_fn("tl_tensor_contiguous", contiguous_type);

    // Softmax / CrossEntropy
    let softmax_type = void_ptr.fn_type(&[void_ptr.into(), i64_type.into()], false);
    add_fn("tl_tensor_softmax", softmax_type);

    let cross_entropy_type = void_ptr.fn_type(&[void_ptr.into(), void_ptr.into()], false);
    add_fn("tl_tensor_cross_entropy", cross_entropy_type);

    // Checkpointing: save_all_params(dir), load_all...
    let params_io_type = void_type.fn_type(&[i8_ptr.into()], false);
    add_fn("tl_save_all_params", params_io_type);
    let add_param_type = void_type.fn_type(&[i8_ptr.into(), void_ptr.into()], false);
    add_fn("tl_add_parameter", add_param_type);
    add_fn("tl_load_all_params", params_io_type);

    // Parameter Registration
    let register_param_type = void_ptr.fn_type(&[void_ptr.into()], false);
    add_fn("tl_register_parameter", register_param_type);

    // String ops
    let str_new_type = i8_ptr.fn_type(&[i8_ptr.into()], false);
    add_fn("tl_string_new", str_new_type);

    let str_concat_type = i8_ptr.fn_type(&[i8_ptr.into(), i8_ptr.into()], false);
    add_fn("tl_string_concat", str_concat_type);

    // File ops
    let file_open_type = void_ptr.fn_type(&[i8_ptr.into(), i8_ptr.into()], false);
    add_fn("tl_file_open", file_open_type);

    let file_read_type = i8_ptr.fn_type(&[void_ptr.into()], false); // Returns String (i8*)
    add_fn("tl_file_read_string", file_read_type);

    let file_write_type = void_type.fn_type(&[void_ptr.into(), i8_ptr.into()], false);
    add_fn("tl_file_write_string", file_write_type);

    let file_close_type = void_type.fn_type(&[void_ptr.into()], false);
    add_fn("tl_file_close", file_close_type);

    // Path
    let path_new_type = void_ptr.fn_type(&[i8_ptr.into()], false);
    add_fn("tl_path_new", path_new_type);

    let path_join_type = void_ptr.fn_type(&[void_ptr.into(), i8_ptr.into()], false);
    add_fn("tl_path_join", path_join_type);

    let path_exists_type = context.bool_type().fn_type(&[void_ptr.into()], false);
    add_fn("tl_path_exists", path_exists_type);
    add_fn("tl_path_is_dir", path_exists_type);
    add_fn("tl_path_is_file", path_exists_type);

    // tl_mem_get_buffer(slot_id: i64, min_size: i64) -> *mut c_void
    let get_buffer_type = void_ptr.fn_type(&[i64_type.into(), i64_type.into()], false);
    add_fn("tl_mem_get_buffer", get_buffer_type);
    
    // tl_log_alloc(ptr, size, file, line)
    let log_alloc_type = void_type.fn_type(&[void_ptr.into(), i64_type.into(), i8_ptr.into(), i32_type.into()], false);
    add_fn("tl_log_alloc", log_alloc_type);

    // tl_log_free(ptr, file, line)
    let log_free_type = void_type.fn_type(&[void_ptr.into(), i8_ptr.into(), i32_type.into()], false);
    add_fn("tl_log_free", log_free_type);
    
    // tl_trace_mem(file, line, col, tag)
    let trace_mem_type = void_type.fn_type(
        &[
             i8_ptr.into(),
             i32_type.into(),
             i32_type.into(),
             i8_ptr.into(),
        ],
        false
    );
    add_fn("tl_trace_mem", trace_mem_type);

    if let Some(f) = module.get_function("tl_log_alloc") {
        execution_engine.add_global_mapping(&f, runtime::tl_log_alloc as usize);
    }
    if let Some(f) = module.get_function("tl_log_free") {
        execution_engine.add_global_mapping(&f, runtime::tl_log_free as usize);
    }
    if let Some(f) = module.get_function("tl_trace_mem") {
        execution_engine.add_global_mapping(&f, runtime::tl_trace_mem as usize);
    }
    
    if let Some(f) = module.get_function("tl_mem_get_buffer") {
        execution_engine.add_global_mapping(&f, runtime::tl_mem_get_buffer as usize);
    }

    // Function Frames
    let fn_enter_type = void_type.fn_type(&[i64_type.into()], false);
    add_fn("tl_mem_function_enter", fn_enter_type);
    let fn_exit_type = void_type.fn_type(&[], false);
    add_fn("tl_mem_function_exit", fn_exit_type);

    if let Some(f) = module.get_function("tl_mem_function_enter") {
        execution_engine.add_global_mapping(&f, runtime::tl_mem_function_enter as usize);
    }
    if let Some(f) = module.get_function("tl_mem_function_exit") {
        execution_engine.add_global_mapping(&f, runtime::tl_mem_function_exit as usize);
    }
    
    let path_to_str_type = i8_ptr.fn_type(&[void_ptr.into()], false);
    add_fn("tl_path_to_string", path_to_str_type);

    // DEBUG: Explicitly map tl_print_string
    if let Some(f) = module.get_function("tl_print_string") {
        execution_engine.add_global_mapping(&f, runtime::tl_print_string as usize);
    }
    
    // Explicitly map tl_mem_function_exit
    let exit_fn_type = void_type.fn_type(&[], false);
    add_fn("tl_mem_function_exit", exit_fn_type);
    if let Some(f) = module.get_function("tl_mem_function_exit") {
        execution_engine.add_global_mapping(&f, runtime::memory_manager::tl_mem_function_exit as usize);
    }

    // Explicitly map tl_arena_init
    let arena_init_type = void_type.fn_type(&[i64_type.into()], false);
    add_fn("tl_arena_init", arena_init_type);
    // Duplicate mapping removed (handled at 1835)

    let path_free_type = void_type.fn_type(&[void_ptr.into()], false);
    add_fn("tl_path_free", path_free_type);

    // Http
    let http_dl_type = context
        .bool_type()
        .fn_type(&[i8_ptr.into(), i8_ptr.into()], false);
    add_fn("tl_http_download", http_dl_type);

    let http_get_type = i8_ptr.fn_type(&[i8_ptr.into()], false);
    add_fn("tl_http_get", http_get_type);

    // Env
    let env_get_type = i8_ptr.fn_type(&[i8_ptr.into()], false);
    add_fn("tl_env_get", env_get_type);

    let env_set_type = void_type.fn_type(&[i8_ptr.into(), i8_ptr.into()], false);
    add_fn("tl_env_set", env_set_type);

    // Args (command line arguments)
    let args_count_type = i64_type.fn_type(&[], false);
    add_fn("tl_args_count", args_count_type);

    let args_get_type = i8_ptr.fn_type(&[i64_type.into()], false);
    add_fn("tl_args_get", args_get_type);

    // tl_string_to_i64(s: *const i8) -> i64
    let str_to_int_type = i64_type.fn_type(&[i8_ptr.into()], false);
    add_fn("tl_string_to_i64", str_to_int_type);

    // (tl_string_char_at は上で宣言済み - 重複削除)


    let string_len_type = i64_type.fn_type(&[i8_ptr.into()], false);
    add_fn("tl_string_len", string_len_type);

    // tl_read_line(prompt: *const i8) -> *mut i8
    let read_line_type = i8_ptr.fn_type(&[i8_ptr.into()], false);
    add_fn("tl_read_line", read_line_type);

    // System
    let sys_time_type = f32_type.fn_type(&[], false); // Return f32 timestamp
    add_fn("tl_system_time", sys_time_type);

    let sys_sleep_type = void_type.fn_type(&[f32_type.into()], false);
    add_fn("tl_system_sleep", sys_sleep_type);

    let mem_mb_type = i64_type.fn_type(&[], false);
    add_fn("tl_get_memory_mb", mem_mb_type);
    add_fn("tl_get_metal_pool_bytes", mem_mb_type);
    add_fn("tl_get_metal_pool_mb", mem_mb_type);
    add_fn("tl_get_metal_pool_count", mem_mb_type);
    add_fn("tl_get_pool_count", mem_mb_type);
    add_fn("tl_get_refcount_count", mem_mb_type);
    add_fn("tl_get_scope_depth", mem_mb_type);
    let trace_type = void_type.fn_type(
        &[
            i8_ptr.into(),
            context.i32_type().into(),
            context.i32_type().into(),
            i8_ptr.into(),
        ],
        false,
    );
    add_fn("tl_trace_mem", trace_type);
    let metal_sync_type = void_type.fn_type(&[], false);
    add_fn("tl_metal_sync", metal_sync_type);

    // Memory Scope
    let enter_scope_type = void_type.fn_type(&[], false);
    add_fn("tl_mem_enter_scope", enter_scope_type);

    let exit_scope_type = void_type.fn_type(&[], false);
    add_fn("tl_mem_exit_scope", exit_scope_type);

    let reg_struct_type = void_type.fn_type(&[void_ptr.into()], false);
    add_fn("tl_mem_register_struct", reg_struct_type);
    add_fn("tl_mem_register_tensor", reg_struct_type);
    add_fn("tl_mem_unregister", reg_struct_type);

    let prepare_ret_type = void_ptr.fn_type(&[void_ptr.into()], false);
    add_fn("tl_tensor_prepare_return", prepare_ret_type);

    // Pool / Arena
    let pool_acq = void_ptr.fn_type(&[i64_type.into()], false);
    add_fn("tl_pool_acquire", pool_acq);

    let pool_rel = void_type.fn_type(&[void_ptr.into(), i64_type.into()], false);
    add_fn("tl_pool_release", pool_rel);

    let arena_init = void_type.fn_type(&[i64_type.into()], false);
    add_fn("tl_arena_init", arena_init);

    let arena_alloc = void_ptr.fn_type(&[i64_type.into()], false);
    add_fn("tl_arena_alloc", arena_alloc);

    let arena_free = void_type.fn_type(&[], false);
    add_fn("tl_arena_free", arena_free);

    let arena_active = context.bool_type().fn_type(&[], false);
    add_fn("tl_arena_is_active", arena_active);

    let arena_reset = void_type.fn_type(&[], false);
    add_fn("tl_arena_reset", arena_reset);

    let arena_offset = i64_type.fn_type(&[], false);
    add_fn("tl_arena_get_offset", arena_offset);

    let arena_capacity = i64_type.fn_type(&[], false);
    add_fn("tl_arena_get_capacity", arena_capacity);

    // --- LLM Builtins ---
    // tl_tokenizer_new(path: *const c_char) -> i64 (handle)
    let tok_new_type = i64_type.fn_type(&[i8_ptr.into()], false);
    add_fn("tl_tokenizer_new", tok_new_type);

    // tl_tokenizer_encode(tok: i64, prompt: *const c_char) -> *mut OpaqueTensor
    let tok_enc_type = void_ptr.fn_type(&[i64_type.into(), i8_ptr.into()], false);
    add_fn("tl_tokenizer_encode", tok_enc_type);

    // tl_tokenizer_decode(tok: i64, ids: *mut OpaqueTensor) -> *const c_char
    let tok_dec_type = i8_ptr.fn_type(&[i64_type.into(), void_ptr.into()], false);
    add_fn("tl_tokenizer_decode", tok_dec_type);

    // tl_gguf_load(path: *const c_char) -> *mut OpaqueTensorMap
    let gguf_load_type = void_ptr.fn_type(&[i8_ptr.into()], false);
    add_fn("tl_gguf_load", gguf_load_type);

    // tl_tensor_map_get(map: *mut Map, name: *const c_char) -> *mut Tensor
    let map_get_type = void_ptr.fn_type(&[void_ptr.into(), i8_ptr.into()], false);
    add_fn("tl_tensor_map_get", map_get_type);

    // tl_tensor_cat(tensors: *mut Vec, dim: i64) -> *mut Tensor
    let cat_type = void_ptr.fn_type(&[void_ptr.into(), i64_type.into()], false);
    add_fn("tl_tensor_cat", cat_type);

    // tl_tensor_silu(t: *mut) -> *mut
    add_fn("tl_tensor_silu", unary_type);

    let scale_type = void_ptr.fn_type(&[void_ptr.into(), f32_type.into()], false);
    add_fn("tl_tensor_scale", scale_type);
    // tl_tensor_cat2(a: *mut, b: *mut, dim: i64) -> *mut
    let cat2_type = void_ptr.fn_type(&[void_ptr.into(), void_ptr.into(), i64_type.into()], false);
    add_fn("tl_tensor_cat2", cat2_type);

    // tl_tensor_cat_4d(a: *mut, b: *mut, dim: i64) -> *mut (alias for type safety)
    add_fn("tl_tensor_cat_4d", cat2_type);

    // tl_tensor_rms_norm(x: *mut, w: *mut, eps: f32) -> *mut
    let rms_type = void_ptr.fn_type(&[void_ptr.into(), void_ptr.into(), f32_type.into()], false);
    add_fn("tl_tensor_rms_norm", rms_type);

    // tl_tensor_apply_rope(x: *mut, cos: *mut, sin: *mut) -> *mut
    let rope_type = void_ptr.fn_type(&[void_ptr.into(), void_ptr.into(), void_ptr.into()], false);
    add_fn("tl_tensor_apply_rope", rope_type);

    // Aliases for different ranks
    add_fn("tl_tensor_transpose_2d", transpose_type);
    add_fn("tl_tensor_matmul_4d", bin_type);
    add_fn("tl_tensor_add_4d", bin_type);
    add_fn("tl_tensor_silu_4d", unary_type);
    let tensor_reshape_2d_type =
        void_ptr.fn_type(&[void_ptr.into(), void_ptr.into(), i64_type.into()], false);
    add_fn("tl_tensor_reshape_2d", tensor_reshape_2d_type);
    add_fn("tl_tensor_reshape_3d_to_2d", tensor_reshape_2d_type); // alias

    // Map get alias (first arg is ptr handle from tl_gguf_load)
    let map_get_1d_type = void_ptr.fn_type(&[void_ptr.into(), i8_ptr.into()], false);
    add_fn("tl_tensor_map_get_1d", map_get_1d_type);

    // Narrow (slice): tl_tensor_narrow(t, dim, start, length) -> t
    let narrow_type = void_ptr.fn_type(
        &[
            void_ptr.into(),
            i64_type.into(),
            i64_type.into(),
            i64_type.into(),
        ],
        false,
    );
    add_fn("tl_tensor_narrow", narrow_type);

    // Repeat interleave: tl_tensor_repeat_interleave(t, repeats, dim) -> t
    let repeat_interleave_type =
        void_ptr.fn_type(&[void_ptr.into(), i64_type.into(), i64_type.into()], false);
    add_fn("tl_tensor_repeat_interleave", repeat_interleave_type);

    // RoPE Factories: tl_tensor_rope_new_cos(dim: i64, len: i64, theta: f32) -> *mut
    let rope_new_type =
        void_ptr.fn_type(&[i64_type.into(), i64_type.into(), f32_type.into()], false);
    add_fn("tl_tensor_rope_new_cos", rope_new_type);
    add_fn("tl_tensor_rope_new_sin", rope_new_type);

    // Causal Mask: tl_tensor_new_causal_mask(dim: i64) -> tensor
    let causal_mask_type = void_ptr.fn_type(&[i64_type.into()], false);
    add_fn("tl_tensor_new_causal_mask", causal_mask_type);

    // tl_tensor_cat_i64(a, b, dim) -> t
    let cat_i64_type =
        void_ptr.fn_type(&[void_ptr.into(), void_ptr.into(), i64_type.into()], false);
    add_fn("tl_tensor_cat_i64", cat_i64_type);

    // tl_tensor_device_id(t) -> i64
    let device_id_type = i64_type.fn_type(&[void_ptr.into()], false);
    add_fn("tl_tensor_device_id", device_id_type);

    // --- Global Mappings ---
    // Mapping symbols is critical for JIT.
    // We do it here to keep CodeGenerator::new clean.

    if let Some(f) = module.get_function("tl_print_string") {
        execution_engine.add_global_mapping(&f, runtime::tl_print_string as *const () as usize);
    }
    if let Some(f) = module.get_function("tl_display_string") {
        execution_engine.add_global_mapping(&f, runtime::tl_display_string as *const () as usize);
    }
    if let Some(f) = module.get_function("tl_prompt") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_prompt as *const () as usize);
    }
    if let Some(f) = module.get_function("tl_print_f32") {
        execution_engine.add_global_mapping(&f, runtime::tl_print_f32 as *const () as usize);
    }
    if let Some(f) = module.get_function("tl_display_f32") {
        execution_engine.add_global_mapping(&f, runtime::tl_display_f32 as *const () as usize);
    }
    if let Some(f) = module.get_function("tl_print_f64") {
        execution_engine.add_global_mapping(&f, runtime::tl_print_f64 as *const () as usize);
    }
    if let Some(f) = module.get_function("tl_display_f64") {
        execution_engine.add_global_mapping(&f, runtime::tl_display_f64 as *const () as usize);
    }
    if let Some(f) = module.get_function("tl_print_bool") {
        execution_engine.add_global_mapping(&f, runtime::tl_print_bool as *const () as usize);
    }
    if let Some(f) = module.get_function("tl_display_bool") {
        execution_engine.add_global_mapping(&f, runtime::tl_display_bool as *const () as usize);
    }
    let f32_unary_mappings: [(&str, usize); 31] = [
        ("tl_f32_abs", runtime::tl_f32_abs as *const () as usize),
        ("tl_f32_acos", runtime::tl_f32_acos as *const () as usize),
        ("tl_f32_acosh", runtime::tl_f32_acosh as *const () as usize),
        ("tl_f32_asin", runtime::tl_f32_asin as *const () as usize),
        ("tl_f32_asinh", runtime::tl_f32_asinh as *const () as usize),
        ("tl_f32_atan", runtime::tl_f32_atan as *const () as usize),
        ("tl_f32_atanh", runtime::tl_f32_atanh as *const () as usize),
        ("tl_f32_cbrt", runtime::tl_f32_cbrt as *const () as usize),
        ("tl_f32_ceil", runtime::tl_f32_ceil as *const () as usize),
        ("tl_f32_cos", runtime::tl_f32_cos as *const () as usize),
        ("tl_f32_cosh", runtime::tl_f32_cosh as *const () as usize),
        ("tl_f32_exp", runtime::tl_f32_exp as *const () as usize),
        ("tl_f32_exp2", runtime::tl_f32_exp2 as *const () as usize),
        (
            "tl_f32_exp_m1",
            runtime::tl_f32_exp_m1 as *const () as usize,
        ),
        ("tl_f32_floor", runtime::tl_f32_floor as *const () as usize),
        ("tl_f32_fract", runtime::tl_f32_fract as *const () as usize),
        ("tl_f32_ln", runtime::tl_f32_ln as *const () as usize),
        ("tl_f32_ln_1p", runtime::tl_f32_ln_1p as *const () as usize),
        ("tl_f32_log10", runtime::tl_f32_log10 as *const () as usize),
        ("tl_f32_log2", runtime::tl_f32_log2 as *const () as usize),
        ("tl_f32_recip", runtime::tl_f32_recip as *const () as usize),
        ("tl_f32_round", runtime::tl_f32_round as *const () as usize),
        (
            "tl_f32_signum",
            runtime::tl_f32_signum as *const () as usize,
        ),
        ("tl_f32_sin", runtime::tl_f32_sin as *const () as usize),
        ("tl_f32_sinh", runtime::tl_f32_sinh as *const () as usize),
        ("tl_f32_sqrt", runtime::tl_f32_sqrt as *const () as usize),
        ("tl_f32_tan", runtime::tl_f32_tan as *const () as usize),
        ("tl_f32_tanh", runtime::tl_f32_tanh as *const () as usize),
        (
            "tl_f32_to_degrees",
            runtime::tl_f32_to_degrees as *const () as usize,
        ),
        (
            "tl_f32_to_radians",
            runtime::tl_f32_to_radians as *const () as usize,
        ),
        ("tl_f32_trunc", runtime::tl_f32_trunc as *const () as usize),
    ];
    for (name, addr) in f32_unary_mappings {
        if let Some(f) = module.get_function(name) {
            execution_engine.add_global_mapping(&f, addr);
        }
    }
    let f32_binary_mappings: [(&str, usize); 5] = [
        ("tl_f32_atan2", runtime::tl_f32_atan2 as *const () as usize),
        (
            "tl_f32_copysign",
            runtime::tl_f32_copysign as *const () as usize,
        ),
        ("tl_f32_hypot", runtime::tl_f32_hypot as *const () as usize),
        ("tl_f32_log", runtime::tl_f32_log as *const () as usize),
        ("tl_f32_powf", runtime::tl_f32_powf as *const () as usize),
    ];
    for (name, addr) in f32_binary_mappings {
        if let Some(f) = module.get_function(name) {
            execution_engine.add_global_mapping(&f, addr);
        }
    }
    if let Some(f) = module.get_function("tl_f32_powi") {
        execution_engine.add_global_mapping(&f, runtime::tl_f32_powi as *const () as usize);
    }
    let f64_unary_mappings: [(&str, usize); 31] = [
        ("tl_f64_abs", runtime::tl_f64_abs as *const () as usize),
        ("tl_f64_acos", runtime::tl_f64_acos as *const () as usize),
        ("tl_f64_acosh", runtime::tl_f64_acosh as *const () as usize),
        ("tl_f64_asin", runtime::tl_f64_asin as *const () as usize),
        ("tl_f64_asinh", runtime::tl_f64_asinh as *const () as usize),
        ("tl_f64_atan", runtime::tl_f64_atan as *const () as usize),
        ("tl_f64_atanh", runtime::tl_f64_atanh as *const () as usize),
        ("tl_f64_cbrt", runtime::tl_f64_cbrt as *const () as usize),
        ("tl_f64_ceil", runtime::tl_f64_ceil as *const () as usize),
        ("tl_f64_cos", runtime::tl_f64_cos as *const () as usize),
        ("tl_f64_cosh", runtime::tl_f64_cosh as *const () as usize),
        ("tl_f64_exp", runtime::tl_f64_exp as *const () as usize),
        ("tl_f64_exp2", runtime::tl_f64_exp2 as *const () as usize),
        (
            "tl_f64_exp_m1",
            runtime::tl_f64_exp_m1 as *const () as usize,
        ),
        ("tl_f64_floor", runtime::tl_f64_floor as *const () as usize),
        ("tl_f64_fract", runtime::tl_f64_fract as *const () as usize),
        ("tl_f64_ln", runtime::tl_f64_ln as *const () as usize),
        ("tl_f64_ln_1p", runtime::tl_f64_ln_1p as *const () as usize),
        ("tl_f64_log10", runtime::tl_f64_log10 as *const () as usize),
        ("tl_f64_log2", runtime::tl_f64_log2 as *const () as usize),
        ("tl_f64_recip", runtime::tl_f64_recip as *const () as usize),
        ("tl_f64_round", runtime::tl_f64_round as *const () as usize),
        (
            "tl_f64_signum",
            runtime::tl_f64_signum as *const () as usize,
        ),
        ("tl_f64_sin", runtime::tl_f64_sin as *const () as usize),
        ("tl_f64_sinh", runtime::tl_f64_sinh as *const () as usize),
        ("tl_f64_sqrt", runtime::tl_f64_sqrt as *const () as usize),
        ("tl_f64_tan", runtime::tl_f64_tan as *const () as usize),
        ("tl_f64_tanh", runtime::tl_f64_tanh as *const () as usize),
        (
            "tl_f64_to_degrees",
            runtime::tl_f64_to_degrees as *const () as usize,
        ),
        (
            "tl_f64_to_radians",
            runtime::tl_f64_to_radians as *const () as usize,
        ),
        ("tl_f64_trunc", runtime::tl_f64_trunc as *const () as usize),
    ];
    for (name, addr) in f64_unary_mappings {
        if let Some(f) = module.get_function(name) {
            execution_engine.add_global_mapping(&f, addr);
        }
    }
    let f64_binary_mappings: [(&str, usize); 5] = [
        ("tl_f64_atan2", runtime::tl_f64_atan2 as *const () as usize),
        (
            "tl_f64_copysign",
            runtime::tl_f64_copysign as *const () as usize,
        ),
        ("tl_f64_hypot", runtime::tl_f64_hypot as *const () as usize),
        ("tl_f64_log", runtime::tl_f64_log as *const () as usize),
        ("tl_f64_powf", runtime::tl_f64_powf as *const () as usize),
    ];
    for (name, addr) in f64_binary_mappings {
        if let Some(f) = module.get_function(name) {
            execution_engine.add_global_mapping(&f, addr);
        }
    }
    if let Some(f) = module.get_function("tl_f64_powi") {
        execution_engine.add_global_mapping(&f, runtime::tl_f64_powi as *const () as usize);
    }
    let i64_mappings: [(&str, usize); 5] = [
        ("tl_i64_abs", runtime::tl_i64_abs as *const () as usize),
        (
            "tl_i64_signum",
            runtime::tl_i64_signum as *const () as usize,
        ),
        ("tl_i64_pow", runtime::tl_i64_pow as *const () as usize),
        (
            "tl_i64_div_euclid",
            runtime::tl_i64_div_euclid as *const () as usize,
        ),
        (
            "tl_i64_rem_euclid",
            runtime::tl_i64_rem_euclid as *const () as usize,
        ),
    ];
    for (name, addr) in i64_mappings {
        if let Some(f) = module.get_function(name) {
            execution_engine.add_global_mapping(&f, addr);
        }
    }
    if let Some(f) = module.get_function("tl_i64_is_positive") {
        execution_engine.add_global_mapping(&f, runtime::tl_i64_is_positive as *const () as usize);
    }
    if let Some(f) = module.get_function("tl_i64_is_negative") {
        execution_engine.add_global_mapping(&f, runtime::tl_i64_is_negative as *const () as usize);
    }
    let i32_mappings: [(&str, usize); 5] = [
        ("tl_i32_abs", runtime::tl_i32_abs as *const () as usize),
        (
            "tl_i32_signum",
            runtime::tl_i32_signum as *const () as usize,
        ),
        ("tl_i32_pow", runtime::tl_i32_pow as *const () as usize),
        (
            "tl_i32_div_euclid",
            runtime::tl_i32_div_euclid as *const () as usize,
        ),
        (
            "tl_i32_rem_euclid",
            runtime::tl_i32_rem_euclid as *const () as usize,
        ),
    ];
    for (name, addr) in i32_mappings {
        if let Some(f) = module.get_function(name) {
            execution_engine.add_global_mapping(&f, addr);
        }
    }
    if let Some(f) = module.get_function("tl_i32_is_positive") {
        execution_engine.add_global_mapping(&f, runtime::tl_i32_is_positive as *const () as usize);
    }
    if let Some(f) = module.get_function("tl_i32_is_negative") {
        execution_engine.add_global_mapping(&f, runtime::tl_i32_is_negative as *const () as usize);
    }
    if let Some(f) = module.get_function("tl_print_i64") {
        execution_engine.add_global_mapping(&f, runtime::tl_print_i64 as *const () as usize);
    }
    if let Some(f) = module.get_function("tl_print_i32") {
        execution_engine.add_global_mapping(&f, runtime::tl_print_i32 as *const () as usize);
    }
    if let Some(f) = module.get_function("tl_display_i64") {
        execution_engine.add_global_mapping(&f, runtime::tl_display_i64 as *const () as usize);
    }
    if let Some(f) = module.get_function("tl_display_i32") {
        execution_engine.add_global_mapping(&f, runtime::tl_display_i32 as *const () as usize);
    }
    if let Some(f) = module.get_function("tl_print_char") {
        execution_engine.add_global_mapping(&f, runtime::tl_print_char as *const () as usize);
    }
    if let Some(f) = module.get_function("tl_display_char") {
        execution_engine.add_global_mapping(&f, runtime::tl_display_char as *const () as usize);
    }
    if let Some(f) = module.get_function("tl_print_ptr") {
        execution_engine.add_global_mapping(&f, runtime::tl_print_ptr as *const () as usize);
    }
    // ========== テンソル作成/メモリ/情報: CPU/GPU 切替 ==========
    map_tensor_fn!("tl_tensor_new", runtime::tl_tensor_new, cpu_ffi::tl_cpu_tensor_new);
    map_tensor_fn!("tl_tensor_new_i64", runtime::tl_tensor_new_i64, cpu_ffi::tl_cpu_tensor_new_i64);
    if let Some(f) = module.get_function("tl_tensor_from_i64_array") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_from_i64_array as usize);
    }
    map_tensor_fn!("tl_tensor_matmul", runtime::tl_tensor_matmul, cpu_ffi::tl_cpu_tensor_matmul);
    map_tensor_fn!("tl_tensor_contiguous", runtime::tl_tensor_contiguous, cpu_ffi::tl_cpu_tensor_contiguous);
    map_tensor_fn!("tl_tensor_print", runtime::tl_tensor_print, cpu_ffi::tl_cpu_tensor_print);
    if let Some(f) = module.get_function("tl_tensor_display") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_display as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_print_1") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_print_1 as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_print_2") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_print_2 as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_print_3") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_print_3 as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_device_id") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_device_id as usize);
    }
    map_tensor_fn!("tl_tensor_free", runtime::tl_tensor_free, cpu_ffi::tl_cpu_tensor_free);
    map_tensor_fn!("tl_tensor_clone", runtime::tl_tensor_clone, cpu_ffi::tl_cpu_tensor_clone);
    map_tensor_fn!("tl_tensor_acquire", runtime::memory_ffi::tl_tensor_acquire, cpu_ffi::tl_cpu_tensor_acquire);
    map_tensor_fn!("tl_tensor_release", runtime::tl_tensor_release, cpu_ffi::tl_cpu_tensor_release);
    if let Some(f) = module.get_function("tl_tensor_prepare_return") {
        execution_engine.add_global_mapping(
            &f,
            runtime::tl_tensor_prepare_return as usize,
        );
    }
    map_tensor_fn!("tl_tensor_len", runtime::tl_tensor_len, cpu_ffi::tl_cpu_tensor_len);

    map_tensor_fn!("tl_tensor_dim", runtime::tl_tensor_dim, cpu_ffi::tl_cpu_tensor_dim);
    if let Some(f) = module.get_function("tl_tensor_get_f32_md") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_get_f32_md as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_get_i64_md") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_get_i64_md as usize);
    }
    map_tensor_fn!("tl_tensor_neg", runtime::tl_tensor_neg, cpu_ffi::tl_cpu_tensor_neg);
    map_tensor_fn!("tl_tensor_transpose", runtime::tl_tensor_transpose, cpu_ffi::tl_cpu_tensor_transpose);
    map_tensor_fn!("tl_tensor_reshape_new", runtime::tl_tensor_reshape_new, cpu_ffi::tl_cpu_tensor_reshape_new);

    if let Some(f) = module.get_function("tl_tensor_get") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_get as usize);
    }
    map_tensor_fn!("tl_tensor_slice", runtime::tl_tensor_slice, cpu_ffi::tl_cpu_tensor_slice);
    if let Some(f) = module.get_function("tl_register_tensor") {
        execution_engine.add_global_mapping(&f, runtime::registry::tl_register_tensor as usize);
    }
    // Additional mappings from previous list...
    map_tensor_fn!("tl_tensor_randn_debug", runtime::tl_tensor_randn_debug, cpu_ffi::tl_cpu_tensor_randn_debug);
    map_tensor_fn!("tl_tensor_zeros", runtime::tl_tensor_zeros, cpu_ffi::tl_cpu_tensor_zeros);
    map_tensor_fn!("tl_tensor_backward", runtime::tl_tensor_backward, cpu_ffi::tl_cpu_tensor_backward);
    map_tensor_fn!("tl_tensor_grad", runtime::tl_tensor_grad, cpu_ffi::tl_cpu_tensor_grad);
    map_tensor_fn!("tl_tensor_detach", runtime::tl_tensor_detach, cpu_ffi::tl_cpu_tensor_detach);
    map_tensor_fn!("tl_tensor_enable_grad", runtime::tl_tensor_enable_grad, cpu_ffi::tl_cpu_tensor_enable_grad);
    map_tensor_fn!("tl_tensor_softmax", runtime::tl_tensor_softmax, cpu_ffi::tl_cpu_tensor_softmax);
    if let Some(f) = module.get_function("tl_tensor_cross_entropy") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_cross_entropy as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_conv2d") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_conv2d as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_clamp") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_clamp as usize);
    }
    // ========== テンソル演算: CPU/GPU 切替 (map_tensor_fn! マクロ使用) ==========
    map_tensor_fn!("tl_tensor_ones", runtime::tl_tensor_ones, cpu_ffi::tl_cpu_tensor_ones);
    map_tensor_fn!("tl_tensor_sub_assign", runtime::tl_tensor_sub_assign, cpu_ffi::tl_cpu_tensor_sub_assign);
    map_tensor_fn!("tl_tensor_sum", runtime::tl_tensor_sum, cpu_ffi::tl_cpu_tensor_sum);
    map_tensor_fn!("tl_tensor_add", runtime::tl_tensor_add, cpu_ffi::tl_cpu_tensor_add);
    map_tensor_fn!("tl_tensor_sub", runtime::tl_tensor_sub, cpu_ffi::tl_cpu_tensor_sub);
    map_tensor_fn!("tl_tensor_mul", runtime::tl_tensor_mul, cpu_ffi::tl_cpu_tensor_mul);
    map_tensor_fn!("tl_tensor_div", runtime::tl_tensor_div, cpu_ffi::tl_cpu_tensor_div);
    map_tensor_fn!("tl_tensor_rem", runtime::tl_tensor_rem, cpu_ffi::tl_cpu_tensor_rem);
    // Comparisons
    map_tensor_fn!("tl_tensor_eq", runtime::tl_tensor_eq, cpu_ffi::tl_cpu_tensor_eq);
    map_tensor_fn!("tl_tensor_neq", runtime::tl_tensor_neq, cpu_ffi::tl_cpu_tensor_neq);
    map_tensor_fn!("tl_tensor_gt", runtime::tl_tensor_gt, cpu_ffi::tl_cpu_tensor_gt);
    map_tensor_fn!("tl_tensor_lt", runtime::tl_tensor_lt, cpu_ffi::tl_cpu_tensor_lt);
    map_tensor_fn!("tl_tensor_ge", runtime::tl_tensor_ge, cpu_ffi::tl_cpu_tensor_ge);
    map_tensor_fn!("tl_tensor_le", runtime::tl_tensor_le, cpu_ffi::tl_cpu_tensor_le);

    map_tensor_fn!("tl_tensor_pow", runtime::tl_tensor_pow, cpu_ffi::tl_cpu_tensor_pow_scalar);
    map_tensor_fn!("tl_tensor_pow_scalar", runtime::tl_tensor_pow_scalar, cpu_ffi::tl_cpu_tensor_pow_scalar);
    map_tensor_fn!("tl_tensor_add_assign", runtime::tl_tensor_add_assign, cpu_ffi::tl_cpu_tensor_add_assign);
    if let Some(f) = module.get_function("tl_tensor_set_f32_md") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_set_f32_md as usize);
    }
    map_tensor_fn!("tl_tensor_mul_assign", runtime::tl_tensor_mul_assign, cpu_ffi::tl_cpu_tensor_mul_assign);
    map_tensor_fn!("tl_tensor_div_assign", runtime::tl_tensor_div_assign, cpu_ffi::tl_cpu_tensor_div_assign);
    // Scalar assign variants
    map_tensor_fn!("tl_tensor_mul_assign_scalar_f32", runtime::tl_tensor_mul_assign_scalar_f32, cpu_ffi::tl_cpu_tensor_mul_assign_scalar_f32);
    map_tensor_fn!("tl_tensor_div_assign_scalar_f32", runtime::tl_tensor_div_assign_scalar_f32, cpu_ffi::tl_cpu_tensor_div_assign_scalar_f32);
    map_tensor_fn!("tl_tensor_mod_assign_scalar_f32", runtime::tl_tensor_mod_assign_scalar_f32, cpu_ffi::tl_cpu_tensor_mod_assign_scalar_f32);

    if let Some(f) = module.get_function("tl_kb_add_entity") {
        execution_engine.add_global_mapping(&f, runtime::knowledge_base::tl_kb_add_entity as usize);
    }
    if let Some(f) = module.get_function("tl_kb_add_fact") {
        execution_engine.add_global_mapping(&f, runtime::knowledge_base::tl_kb_add_fact as usize);
    }
    if let Some(f) = module.get_function("tl_kb_infer") {
        execution_engine.add_global_mapping(&f, runtime::knowledge_base::tl_kb_infer as usize);
    }
    if let Some(f) = module.get_function("tl_kb_rule_start") {
        execution_engine.add_global_mapping(&f, runtime::knowledge_base::tl_kb_rule_start as usize);
    }
    if let Some(f) = module.get_function("tl_kb_rule_add_head_arg_var") {
        execution_engine.add_global_mapping(
            &f,
            runtime::knowledge_base::tl_kb_rule_add_head_arg_var as usize,
        );
    }
    if let Some(f) = module.get_function("tl_kb_add_fact_serialized") {
        execution_engine.add_global_mapping(
            &f,
            runtime::knowledge_base::tl_kb_add_fact_serialized as usize,
        );
    }
    if let Some(f) = module.get_function("tl_kb_fact_args_clear") {
        execution_engine.add_global_mapping(
            &f,
            runtime::knowledge_base::tl_kb_fact_args_clear as usize,
        );
    }
    if let Some(f) = module.get_function("tl_kb_fact_args_add_int") {
        execution_engine.add_global_mapping(
            &f,
            runtime::knowledge_base::tl_kb_fact_args_add_int as usize,
        );
    }
    if let Some(f) = module.get_function("tl_kb_fact_args_add_float") {
        execution_engine.add_global_mapping(
            &f,
            runtime::knowledge_base::tl_kb_fact_args_add_float as usize,
        );
    }
    if let Some(f) = module.get_function("tl_kb_fact_args_add_bool") {
        execution_engine.add_global_mapping(
            &f,
            runtime::knowledge_base::tl_kb_fact_args_add_bool as usize,
        );
    }
    if let Some(f) = module.get_function("tl_kb_fact_args_add_entity") {
        execution_engine.add_global_mapping(
            &f,
            runtime::knowledge_base::tl_kb_fact_args_add_entity as usize,
        );
    }
    if let Some(f) = module.get_function("tl_kb_fact_args_add_string") {
        execution_engine.add_global_mapping(
            &f,
            runtime::knowledge_base::tl_kb_fact_args_add_string as usize,
        );
    }
    if let Some(f) = module.get_function("tl_kb_rule_add_head_arg_const_int") {
        execution_engine.add_global_mapping(
            &f,
            runtime::knowledge_base::tl_kb_rule_add_head_arg_const_int as usize,
        );
    }
    if let Some(f) = module.get_function("tl_kb_rule_add_head_arg_const_float") {
        execution_engine.add_global_mapping(
            &f,
            runtime::knowledge_base::tl_kb_rule_add_head_arg_const_float as usize,
        );
    }
    if let Some(f) = module.get_function("tl_kb_rule_add_head_arg_const_entity") {
        execution_engine.add_global_mapping(
            &f,
            runtime::knowledge_base::tl_kb_rule_add_head_arg_const_entity as usize,
        );
    }
    if let Some(f) = module.get_function("tl_kb_rule_add_body_atom") {
        execution_engine.add_global_mapping(
            &f,
            runtime::knowledge_base::tl_kb_rule_add_body_atom as usize,
        );
    }
    if let Some(f) = module.get_function("tl_kb_rule_add_body_atom_neg") {
        execution_engine.add_global_mapping(
            &f,
            runtime::knowledge_base::tl_kb_rule_add_body_atom_neg as usize,
        );
    }
    if let Some(f) = module.get_function("tl_kb_rule_add_body_arg_var") {
        execution_engine.add_global_mapping(
            &f,
            runtime::knowledge_base::tl_kb_rule_add_body_arg_var as usize,
        );
    }
    if let Some(f) = module.get_function("tl_kb_rule_add_body_arg_const_int") {
        execution_engine.add_global_mapping(
            &f,
            runtime::knowledge_base::tl_kb_rule_add_body_arg_const_int as usize,
        );
    }
    if let Some(f) = module.get_function("tl_kb_rule_add_body_arg_const_float") {
        execution_engine.add_global_mapping(
            &f,
            runtime::knowledge_base::tl_kb_rule_add_body_arg_const_float as usize,
        );
    }
    if let Some(f) = module.get_function("tl_kb_rule_add_body_arg_const_entity") {
        execution_engine.add_global_mapping(
            &f,
            runtime::knowledge_base::tl_kb_rule_add_body_arg_const_entity as usize,
        );
    }
    if let Some(f) = module.get_function("tl_kb_rule_finish") {
        execution_engine
            .add_global_mapping(&f, runtime::knowledge_base::tl_kb_rule_finish as usize);
    }
    map_tensor_fn!("tl_tensor_add_assign_scalar_f32", runtime::tl_tensor_add_assign_scalar_f32, cpu_ffi::tl_cpu_tensor_add_assign_scalar_f32);
    map_tensor_fn!("tl_tensor_sub_assign_scalar_f32", runtime::tl_tensor_sub_assign_scalar_f32, cpu_ffi::tl_cpu_tensor_sub_assign_scalar_f32);
    map_tensor_fn!("tl_tensor_mod_assign", runtime::tl_tensor_mod_assign, cpu_ffi::tl_cpu_tensor_mod_assign);
    map_tensor_fn!("tl_tensor_mod_assign_scalar_f32", runtime::tl_tensor_mod_assign_scalar_f32, cpu_ffi::tl_cpu_tensor_mod_assign_scalar_f32);
    map_tensor_fn!("tl_tensor_exp", runtime::tl_tensor_exp, cpu_ffi::tl_cpu_tensor_exp);
    map_tensor_fn!("tl_tensor_log", runtime::tl_tensor_log, cpu_ffi::tl_cpu_tensor_log);
    // tl_tensor_sqrt はCPU版実装済み
    map_tensor_fn!("tl_tensor_sqrt", runtime::tl_tensor_sqrt, cpu_ffi::tl_cpu_tensor_sqrt);
    map_tensor_fn!("tl_tensor_sin", runtime::tl_tensor_sin, cpu_ffi::tl_cpu_tensor_sin);
    map_tensor_fn!("tl_tensor_cos", runtime::tl_tensor_cos, cpu_ffi::tl_cpu_tensor_cos);
    map_tensor_fn!("tl_tensor_relu", runtime::tl_tensor_relu, cpu_ffi::tl_cpu_tensor_relu);
    // tl_tensor_gelu, tl_tensor_tril, tl_tensor_sum_dim, tl_tensor_embedding: CPU版実装済み
    map_tensor_fn!("tl_tensor_gelu", runtime::tl_tensor_gelu, cpu_ffi::tl_cpu_tensor_gelu);
    map_tensor_fn!("tl_tensor_tril", runtime::tl_tensor_tril, cpu_ffi::tl_cpu_tensor_tril);
    map_tensor_fn!("tl_tensor_sum_dim", runtime::tl_tensor_sum_dim, cpu_ffi::tl_cpu_tensor_sum_dim);
    map_tensor_fn!("tl_tensor_embedding", runtime::tl_tensor_embedding, cpu_ffi::tl_cpu_tensor_embedding);
    if let Some(f) = module.get_function("tl_tensor_save") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_save as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_load") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_load as usize);
    }
    if let Some(f) = module.get_function("tl_save_all_params") {
        execution_engine.add_global_mapping(&f, runtime::tl_save_all_params as usize);
    }
    if let Some(f) = module.get_function("tl_add_parameter") {
        execution_engine.add_global_mapping(&f, runtime::tl_add_parameter as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_map_new") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_map_new as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_map_insert") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_map_insert as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_map_save") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_map_save as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_map_load") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_map_load as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_map_get") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_map_get as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_map_free") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_map_free as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_set_f32_md") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_set_f32_md as usize);
    }
    if let Some(f) = module.get_function("tl_mem_unregister") {
        execution_engine
            .add_global_mapping(&f, runtime::memory_manager::tl_mem_unregister as usize);
    }
    if let Some(f) = module.get_function("tl_load_all_params") {
        execution_engine.add_global_mapping(&f, runtime::tl_load_all_params as usize);
    }
    if let Some(f) = module.get_function("tl_register_parameter") {
        execution_engine.add_global_mapping(&f, runtime::tl_register_parameter as usize);
    }
    if let Some(f) = module.get_function("tl_string_new") {
        execution_engine.add_global_mapping(&f, runtime::tl_string_new as usize);
    }

    // VarBuilder-based parameter management
    if let Some(f) = module.get_function("tl_varbuilder_get") {
        execution_engine.add_global_mapping(&f, runtime::tl_varbuilder_get as usize);
    }
    if let Some(f) = module.get_function("tl_varbuilder_get_from_tensor") {
        execution_engine.add_global_mapping(&f, runtime::tl_varbuilder_get_from_tensor as usize);
    }
    if let Some(f) = module.get_function("tl_varbuilder_grad") {
        execution_engine.add_global_mapping(&f, runtime::tl_varbuilder_grad as usize);
    }
    if let Some(f) = module.get_function("tl_update_all_params") {
        execution_engine.add_global_mapping(&f, runtime::tl_update_all_params as usize);
    }

    if let Some(f) = module.get_function("tl_get_memory_mb") {
        execution_engine.add_global_mapping(&f, runtime::tl_get_memory_mb as usize);
    }
    if let Some(f) = module.get_function("tl_get_metal_pool_bytes") {
        execution_engine.add_global_mapping(&f, runtime::tl_get_metal_pool_bytes as usize);
    }
    if let Some(f) = module.get_function("tl_get_metal_pool_mb") {
        execution_engine.add_global_mapping(&f, runtime::tl_get_metal_pool_mb as usize);
    }
    if let Some(f) = module.get_function("tl_get_metal_pool_count") {
        execution_engine.add_global_mapping(&f, runtime::tl_get_metal_pool_count as usize);
    }
    if let Some(f) = module.get_function("tl_metal_sync") {
        execution_engine.add_global_mapping(&f, runtime::tl_metal_sync as usize);
    }
    if let Some(f) = module.get_function("tl_trace_mem") {
        execution_engine.add_global_mapping(&f, runtime::tl_trace_mem as usize);
    }
    if let Some(f) = module.get_function("tl_get_metal_pool_bytes") {
        execution_engine.add_global_mapping(&f, runtime::tl_get_metal_pool_bytes as usize);
    }
    if let Some(f) = module.get_function("tl_get_metal_pool_mb") {
        execution_engine.add_global_mapping(&f, runtime::tl_get_metal_pool_mb as usize);
    }
    if let Some(f) = module.get_function("tl_get_metal_pool_count") {
        execution_engine.add_global_mapping(&f, runtime::tl_get_metal_pool_count as usize);
    }
    if let Some(f) = module.get_function("tl_get_pool_count") {
        execution_engine.add_global_mapping(&f, runtime::tl_get_pool_count as usize);
    }
    if let Some(f) = module.get_function("tl_get_refcount_count") {
        execution_engine.add_global_mapping(&f, runtime::tl_get_refcount_count as usize);
    }
    if let Some(f) = module.get_function("tl_get_scope_depth") {
        execution_engine.add_global_mapping(&f, runtime::tl_get_scope_depth as usize);
    }

    // Args (command line arguments)
    if let Some(f) = module.get_function("tl_args_count") {
        execution_engine.add_global_mapping(&f, runtime::args::tl_args_count as usize);
    }
    if let Some(f) = module.get_function("tl_args_get") {
        execution_engine.add_global_mapping(&f, runtime::args::tl_args_get as usize);
    }
    if let Some(f) = module.get_function("tl_string_to_i64") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_string_to_i64 as usize);
    }

    // String utilities
    if let Some(f) = module.get_function("tl_string_char_at") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_string_char_at as usize);
    }
    if let Some(f) = module.get_function("tl_string_len") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_string_len as usize);
    }

    // Memory manager mappings
    if let Some(f) = module.get_function("tl_mem_enter_scope") {
        execution_engine
            .add_global_mapping(&f, runtime::memory_manager::tl_mem_enter_scope as usize);
    }
    if let Some(f) = module.get_function("tl_mem_exit_scope") {
        execution_engine
            .add_global_mapping(&f, runtime::memory_manager::tl_mem_exit_scope as usize);
    }
    if let Some(f) = module.get_function("tl_mem_register_struct") {
        execution_engine
            .add_global_mapping(&f, runtime::memory_manager::tl_mem_register_struct as usize);
    }
    if let Some(f) = module.get_function("tl_mem_register_struct_named") {
        execution_engine.add_global_mapping(
            &f,
            runtime::memory_manager::tl_mem_register_struct_named as usize,
        );
    }
    if let Some(f) = module.get_function("tl_mem_register_tensor") {
        execution_engine
            .add_global_mapping(&f, runtime::memory_manager::tl_mem_register_tensor as usize);
    }
    if let Some(f) = module.get_function("tl_mem_unregister") {
        execution_engine
            .add_global_mapping(&f, runtime::memory_manager::tl_mem_unregister as usize);
    }
    if let Some(f) = module.get_function("tl_ptr_dec_ref") {
        execution_engine.add_global_mapping(
            &f,
            runtime::memory_manager::tl_ptr_dec_ref as usize,
        );
    }
    if let Some(f) = module.get_function("tl_ptr_inc_ref") {
        execution_engine.add_global_mapping(
            &f,
            runtime::memory_manager::tl_ptr_inc_ref as usize,
        );
    }
    if let Some(f) = module.get_function("tl_ptr_acquire") {
        execution_engine.add_global_mapping(
            &f,
            runtime::memory_manager::tl_ptr_acquire as usize,
        );
    }
    if let Some(f) = module.get_function("tl_ptr_release") {
        execution_engine.add_global_mapping(
            &f,
            runtime::memory_manager::tl_ptr_release as usize,
        );
    }
    if let Some(f) = module.get_function("tl_mem_free") {
        execution_engine.add_global_mapping(&f, runtime::tl_mem_free as usize);
    }

    // Arena Allocator mappings
    if let Some(f) = module.get_function("tl_arena_init") {
        execution_engine.add_global_mapping(&f, runtime::arena::tl_arena_init as usize);
    }
    if let Some(f) = module.get_function("tl_arena_alloc") {
        execution_engine.add_global_mapping(&f, runtime::arena::tl_arena_alloc as usize);
    }
    if let Some(f) = module.get_function("tl_arena_free") {
        execution_engine.add_global_mapping(&f, runtime::arena::tl_arena_free as usize);
    }
    if let Some(f) = module.get_function("tl_arena_is_active") {
        execution_engine.add_global_mapping(&f, runtime::arena::tl_arena_is_active as usize);
    }
    if let Some(f) = module.get_function("tl_arena_reset") {
        execution_engine.add_global_mapping(&f, runtime::arena::tl_arena_reset as usize);
    }
    if let Some(f) = module.get_function("tl_arena_get_offset") {
        execution_engine.add_global_mapping(&f, runtime::arena::tl_arena_get_offset as usize);
    }
    if let Some(f) = module.get_function("tl_arena_get_capacity") {
        execution_engine.add_global_mapping(&f, runtime::arena::tl_arena_get_capacity as usize);
    }
    if let Some(f) = module.get_function("tl_arena_malloc") {
        execution_engine.add_global_mapping(&f, runtime::arena::tl_arena_malloc as usize);
    }

    if let Some(f) = module.get_function("tl_update_all_params") {
        execution_engine.add_global_mapping(&f, runtime::tl_update_all_params as usize);
    }
    if let Some(f) = module.get_function("tl_varbuilder_grad") {
        execution_engine.add_global_mapping(&f, runtime::tl_varbuilder_grad as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_reshape_dims") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_reshape_dims as usize);
    }

    // Map Support
    if let Some(f) = module.get_function("tl_tensor_map_new") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_map_new as usize);
    }
    if let Some(f) = module.get_function("tl_alloc_tmp") {
        execution_engine.add_global_mapping(&f, runtime::tl_alloc_tmp as usize);
    }
    if let Some(f) = module.get_function("tl_free_tmp") {
        execution_engine.add_global_mapping(&f, runtime::tl_free_tmp as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_map_insert") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_map_insert as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_map_save") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_map_save as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_map_load") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_map_load as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_map_get") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_map_get as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_map_free") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_map_free as usize);
    }

    if let Some(f) = module.get_function("tl_tensor_argmax") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_argmax as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_to_i64") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_to_i64 as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_item_i64") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_item_i64 as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_item") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_item as usize);
    }

    // CLI Args

    // Added missing ones (Define type locally as declared later)
    let _tensor_type = Type::Tensor(Box::new(Type::F32), 1);
    let _tensor_type_local = Type::Tensor(Box::new(Type::I64), 1);
    // Removed duplicate tl_tensor_to_i64

    // Added for Tensor Refactor
    add_fn("tl_tensor_tan", unary_type);
    if let Some(f) = module.get_function("tl_tensor_tan") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_tan as usize);
    }
    add_fn("tl_tensor_abs", unary_type);
    if let Some(f) = module.get_function("tl_tensor_abs") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_abs as usize);
    }
    add_fn("tl_tensor_sigmoid", unary_type);
    if let Some(f) = module.get_function("tl_tensor_sigmoid") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_sigmoid as usize);
    }
    add_fn("tl_tensor_tanh", unary_type);
    if let Some(f) = module.get_function("tl_tensor_tanh") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_tanh as usize);
    }
    add_fn("tl_tensor_max", unary_type);
    if let Some(f) = module.get_function("tl_tensor_max") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_max as usize);
    }
    add_fn("tl_tensor_max_dim", sum_dim_type);
    if let Some(f) = module.get_function("tl_tensor_max_dim") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_max_dim as usize);
    }
    add_fn("tl_tensor_min", unary_type);
    if let Some(f) = module.get_function("tl_tensor_min") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_min as usize);
    }
    add_fn("tl_tensor_min_dim", sum_dim_type);
    if let Some(f) = module.get_function("tl_tensor_min_dim") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_min_dim as usize);
    }
    add_fn("tl_tensor_mean", unary_type);
    if let Some(f) = module.get_function("tl_tensor_mean") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_mean as usize);
    }
    add_fn("tl_tensor_mean_dim", sum_dim_type);
    if let Some(f) = module.get_function("tl_tensor_mean_dim") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_mean_dim as usize);
    }
    add_fn("tl_tensor_argmin", sum_dim_type);
    if let Some(f) = module.get_function("tl_tensor_argmin") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_argmin as usize);
    }


    // --- LLM Mappings ---
    if let Some(f) = module.get_function("tl_tokenizer_new") {
        execution_engine.add_global_mapping(&f, runtime::llm::tl_tokenizer_new as usize);
    }
    if let Some(f) = module.get_function("tl_tokenizer_encode") {
        execution_engine.add_global_mapping(&f, runtime::llm::tl_tokenizer_encode as usize);
    }
    if let Some(f) = module.get_function("tl_tokenizer_decode") {
        execution_engine.add_global_mapping(&f, runtime::llm::tl_tokenizer_decode as usize);
    }
    if let Some(f) = module.get_function("tl_gguf_load") {
        execution_engine.add_global_mapping(&f, runtime::llm::tl_gguf_load as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_map_get") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_map_get as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_cat") {
        execution_engine.add_global_mapping(&f, runtime::llm::tl_tensor_cat as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_silu") {
        execution_engine.add_global_mapping(&f, runtime::llm::tl_tensor_silu as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_apply_rope") {
        execution_engine.add_global_mapping(&f, runtime::llm::tl_tensor_apply_rope as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_rms_norm") {
        execution_engine.add_global_mapping(&f, runtime::llm::tl_tensor_rms_norm as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_cat2") {
        execution_engine.add_global_mapping(&f, runtime::llm::tl_tensor_cat2 as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_cat_4d") {
        execution_engine.add_global_mapping(&f, runtime::llm::tl_tensor_cat_4d as usize);
    }

    // Alias Mappings
    if let Some(f) = module.get_function("tl_tensor_transpose_2d") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_transpose as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_matmul_4d") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_matmul as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_add_4d") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_add as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_silu_4d") {
        execution_engine.add_global_mapping(&f, runtime::llm::tl_tensor_silu as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_scale") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_scale as usize);
    }

    if let Some(f) = module.get_function("tl_tensor_rope_new_cos") {
        execution_engine.add_global_mapping(&f, runtime::llm::tl_tensor_rope_new_cos as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_rope_new_sin") {
        execution_engine.add_global_mapping(&f, runtime::llm::tl_tensor_rope_new_sin as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_new_causal_mask") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_new_causal_mask as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_cat_i64") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_cat_i64 as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_narrow") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_narrow as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_repeat_interleave") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_repeat_interleave as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_get_shape") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_get_shape as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_reshape_2d") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_reshape_dims as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_reshape_3d_to_2d") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_reshape_dims as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_map_get_1d") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_map_get as usize);
    }


    if let Some(f) = module.get_function("tl_tensor_from_vec_u8") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_from_vec_u8 as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_from_u8_labels") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_from_u8_labels as usize);
    }


    // Binary file I/O mappings
    if let Some(f) = module.get_function("tl_file_read_binary") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_file_read_binary as usize);
    }
    if let Some(f) = module.get_function("tl_file_write_binary") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_file_write_binary as usize);
    }

    // Image loading mappings
    if let Some(f) = module.get_function("tl_image_load_grayscale") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_image_load_grayscale as usize);
    }
    if let Some(f) = module.get_function("tl_image_width") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_image_width as usize);
    }
    if let Some(f) = module.get_function("tl_image_height") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_image_height as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_to_f32") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_to_f32 as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_to_i64") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_to_i64 as usize);
    }

    // End of function

    let _tensor_type = Type::Tensor(Box::new(Type::F32), 1); // Common return type for many tensor ops

    let f32_unary_methods = [
        "abs",
        "acos",
        "acosh",
        "asin",
        "asinh",
        "atan",
        "atanh",
        "cbrt",
        "ceil",
        "cos",
        "cosh",
        "exp",
        "exp2",
        "exp_m1",
        "floor",
        "fract",
        "ln",
        "ln_1p",
        "log10",
        "log2",
        "recip",
        "round",
        "signum",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
        "to_degrees",
        "to_radians",
        "trunc",
    ];
    for _name in f32_unary_methods {
    }
    let f32_binary_methods = ["atan2", "copysign", "hypot", "log", "powf"];
    for _name in f32_binary_methods {
    }
    let f64_unary_methods = [
        "abs",
        "acos",
        "acosh",
        "asin",
        "asinh",
        "atan",
        "atanh",
        "cbrt",
        "ceil",
        "cos",
        "cosh",
        "exp",
        "exp2",
        "exp_m1",
        "floor",
        "fract",
        "ln",
        "ln_1p",
        "log10",
        "log2",
        "recip",
        "round",
        "signum",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
        "to_degrees",
        "to_radians",
        "trunc",
    ];
    for _name in f64_unary_methods {
    }
    let f64_binary_methods = ["atan2", "copysign", "hypot", "log", "powf"];
    for _name in f64_binary_methods {
    }
    let i64_unary_methods = ["abs", "signum"];
    for _name in i64_unary_methods {
    }
    let i64_binary_methods = ["div_euclid", "rem_euclid", "pow"];
    for _name in i64_binary_methods {
    }
    let i32_unary_methods = ["abs", "signum"];
    for _name in i32_unary_methods {
    }
    let i32_binary_methods = ["div_euclid", "rem_euclid", "pow"];
    for _name in i32_binary_methods {
    }
    
    // ... complete as needed based on original CodeGen
                                                                    // tl_tensor_reshape_dims(tensor: *mut OpaqueTensor, dims: *const i64, num_dims: i64) -> *mut OpaqueTensor
    let tensor_reshape_dims_type = void_ptr.fn_type(
        &[
            void_ptr.into(),                                           // tensor
            context.ptr_type(inkwell::AddressSpace::default()).into(), // dims ptr
            context.i64_type().into(),                                 // num_dims
        ],
        false,
    );
    module.add_function("tl_tensor_reshape_dims", tensor_reshape_dims_type, None);

    // tl_tensor_reshape(tensor: *mut OpaqueTensor, new_shape: *mut OpaqueTensor) -> *mut OpaqueTensor
    let tensor_reshape_type = void_ptr.fn_type(&[void_ptr.into(), void_ptr.into()], false);
    module.add_function("tl_tensor_reshape", tensor_reshape_type, None);

    // tl_tensor_randn(rank: usize, shape: *const usize, req_grad: bool) -> *mut OpaqueTensor

    // VarBuilder-based parameter management (following Candle's official pattern)
    // tl_varbuilder_get(name: *const c_char, rank: i64, shape: *const usize) -> *mut OpaqueTensor
    let varbuilder_get_type =
        void_ptr.fn_type(&[i8_ptr.into(), i64_type.into(), usize_ptr.into()], false);
    module.add_function("tl_varbuilder_get", varbuilder_get_type, None);

    // tl_varbuilder_get_from_tensor(name: *const c_char, shape_tensor: *mut OpaqueTensor)
    let varbuilder_get_tensor_type = void_ptr.fn_type(&[i8_ptr.into(), void_ptr.into()], false);
    module.add_function(
        "tl_varbuilder_get_from_tensor",
        varbuilder_get_tensor_type,
        None,
    );

    // tl_update_all_params(learning_rate: f32) -> void
    let update_params_type = void_type.fn_type(&[context.f32_type().into()], false);
    module.add_function("tl_update_all_params", update_params_type, None);

    // tl_varbuilder_grad(name: *const c_char) -> *mut OpaqueTensor
    let varbuilder_grad_type = void_ptr.fn_type(&[i8_ptr.into()], false);
    module.add_function("tl_varbuilder_grad", varbuilder_grad_type, None);

    // tl_tensor_backward(t: *mut OpaqueTensor) -> void
    let backward_type = void_type.fn_type(&[void_ptr.into()], false);
    module.add_function("tl_tensor_backward", backward_type, None);

    // tl_tensor_grad(t: *mut OpaqueTensor) -> *mut OpaqueTensor
    let grad_type = void_ptr.fn_type(&[void_ptr.into()], false);
    module.add_function("tl_tensor_grad", grad_type, None);

    // tl_tensor_detach(t: *mut, req_grad: bool) -> *mut
    let detach_type = void_ptr.fn_type(&[void_ptr.into(), context.bool_type().into()], false);
    module.add_function("tl_tensor_detach", detach_type, None);

    // tl_tensor_softmax(t: *mut, dim: i64) -> *mut
    let softmax_type = void_ptr.fn_type(&[void_ptr.into(), i64_type.into()], false);
    module.add_function("tl_tensor_softmax", softmax_type, None);

    // tl_tensor_cross_entropy(logits: *mut, targets: *mut) -> *mut
    let ce_type = void_ptr.fn_type(&[void_ptr.into(), void_ptr.into()], false);
    module.add_function("tl_tensor_cross_entropy", ce_type, None);

    // tl_tensor_save(path: *const i8, t: *mut OpaqueTensor) -> void
    let save_type = void_type.fn_type(&[i8_ptr.into(), void_ptr.into()], false);
    module.add_function("tl_tensor_save", save_type, None);

    // tl_tensor_load(path: *const i8) -> *mut OpaqueTensor
    let load_type = void_ptr.fn_type(&[i8_ptr.into()], false);
    module.add_function("tl_tensor_load", load_type, None);

    // Cast
    let cast_type = void_ptr.fn_type(&[void_ptr.into()], false);
    add_fn("tl_tensor_to_f32", cast_type);
    add_fn("tl_tensor_to_i64", cast_type);

    // tl_save_all_params(path: *const i8) -> void
    let save_all_type = void_type.fn_type(&[i8_ptr.into()], false);
    module.add_function("tl_save_all_params", save_all_type, None);

    // tl_add_parameter(name: *str, t: *mut Tensor) -> void
    let add_param_type = void_type.fn_type(&[void_ptr.into(), void_ptr.into()], false);
    module.add_function("tl_add_parameter", add_param_type, None);

    // tl_tensor_argmax(t: *mut, dim: i64, keep_dim: bool) -> *mut
    let argmax_type = void_ptr.fn_type(
        &[void_ptr.into(), i64_type.into(), context.bool_type().into()],
        false,
    );
    module.add_function("tl_tensor_argmax", argmax_type, None);

    // tl_tensor_item(t: *mut) -> f32
    let item_type = f32_type.fn_type(&[void_ptr.into()], false);
    module.add_function("tl_tensor_item", item_type, None);

    // tl_tensor_item_i64(t: *mut) -> i64
    let item_i64_type = i64_type.fn_type(&[void_ptr.into()], false);
    module.add_function("tl_tensor_item_i64", item_i64_type, None);

    // tl_tensor_to_i64(t: *mut) -> *mut
    let to_i64_type = void_ptr.fn_type(&[void_ptr.into()], false);
    module.add_function("tl_tensor_to_i64", to_i64_type, None);

    // tl_tensor_shape(t: *mut) -> *mut Vec
    let shape_type = void_ptr.fn_type(&[void_ptr.into()], false);
    module.add_function("tl_tensor_shape", shape_type, None);

    // tl_load_all_params(path: *const i8) -> void
    let load_all_type = void_type.fn_type(&[i8_ptr.into()], false);
    module.add_function("tl_load_all_params", load_all_type, None);

    // tl_tensor_sub_assign(ref_t: *mut, val: *mut) -> void
    let sub_assign_type = void_type.fn_type(&[void_ptr.into(), void_ptr.into()], false);
    module.add_function("tl_tensor_sub_assign", sub_assign_type, None);

    // tl_add_parameter(name: *const i8, t: *mut OpaqueTensor) -> void
    let add_param_type = void_type.fn_type(&[i8_ptr.into(), void_ptr.into()], false);
    module.add_function("tl_add_parameter", add_param_type, None);

    // tl_register_parameter(t: *mut OpaqueTensor) -> *mut OpaqueTensor
    let reg_param_type = void_ptr.fn_type(&[void_ptr.into()], false);
    module.add_function("tl_register_parameter", reg_param_type, None);

    // tl_tensor_conv2d(input: *mut, weight: *mut, padding: i64, stride: i64) -> *mut
    let conv2d_type = void_ptr.fn_type(
        &[
            void_ptr.into(),
            void_ptr.into(),
            i64_type.into(),
            i64_type.into(),
        ],
        false,
    );
    module.add_function("tl_tensor_conv2d", conv2d_type, None);

    // tl_tensor_clamp(t: *mut, min: f32, max: f32) -> *mut
    let clamp_type = void_ptr.fn_type(&[void_ptr.into(), f32_type.into(), f32_type.into()], false);
    module.add_function("tl_tensor_clamp", clamp_type, None);

    // tl_tensor_ones(rank: i64, shape: *const usize, req_grad: bool) -> *mut OpaqueTensor
    let ones_type = void_ptr.fn_type(
        &[
            i64_type.into(),
            usize_ptr.into(),
            context.bool_type().into(),
        ],
        false,
    );
    module.add_function("tl_tensor_ones", ones_type, None);

    // Register new return types

    // VarBuilder-based parameter management

    // --- Standard Library Phase 1 ---

    let i8_ptr = context.ptr_type(AddressSpace::default());

    // Strings
    let str_concat_type = i8_ptr.fn_type(&[i8_ptr.into(), i8_ptr.into()], false);
    module.add_function("tl_string_concat", str_concat_type, None);

    // File I/O
    // tl_file_open(path: *const i8, mode: *const i8) -> *mut File
    let file_open_type = void_ptr.fn_type(&[i8_ptr.into(), i8_ptr.into()], false);
    module.add_function("tl_file_open", file_open_type, None);

    // tl_file_read_string(file: *mut File) -> *mut i8
    let file_read_type = i8_ptr.fn_type(&[void_ptr.into()], false);
    module.add_function("tl_file_read_string", file_read_type, None);

    // tl_file_write_string(file: *mut File, content: *const i8) -> void
    let file_write_type = void_type.fn_type(&[void_ptr.into(), i8_ptr.into()], false);
    module.add_function("tl_file_write_string", file_write_type, None);

    // tl_file_close(file: *mut File) -> void
    let file_close_type = void_type.fn_type(&[void_ptr.into()], false);
    module.add_function("tl_file_close", file_close_type, None);

    // Path
    // tl_path_new(path: *const i8) -> *mut Path
    let path_new_type = void_ptr.fn_type(&[i8_ptr.into()], false);
    module.add_function("tl_path_new", path_new_type, None);

    // tl_path_join(base: *mut Path, part: *const i8) -> *mut Path
    let path_join_type = void_ptr.fn_type(&[void_ptr.into(), i8_ptr.into()], false);
    module.add_function("tl_path_join", path_join_type, None);

    // tl_path_exists(path: *mut Path) -> bool
    let path_exists_type = context.bool_type().fn_type(&[void_ptr.into()], false);
    module.add_function("tl_path_exists", path_exists_type, None);

    // tl_path_is_dir(path: *mut Path) -> bool
    let path_is_dir_type = context.bool_type().fn_type(&[void_ptr.into()], false);
    module.add_function("tl_path_is_dir", path_is_dir_type, None);

    // tl_path_is_file(path: *mut Path) -> bool
    let path_is_file_type = context.bool_type().fn_type(&[void_ptr.into()], false);
    module.add_function("tl_path_is_file", path_is_file_type, None);

    // tl_path_to_string(path: *mut Path) -> *mut i8
    let path_to_str_type = i8_ptr.fn_type(&[void_ptr.into()], false);
    module.add_function("tl_path_to_string", path_to_str_type, None);

    // tl_path_free(path: *mut Path) -> void
    let path_free_type = void_type.fn_type(&[void_ptr.into()], false);
    module.add_function("tl_path_free", path_free_type, None);

    // Http
    // tl_http_download(url: *const i8, dest: *const i8) -> bool
    let http_dl_type = context
        .bool_type()
        .fn_type(&[i8_ptr.into(), i8_ptr.into()], false);
    module.add_function("tl_http_download", http_dl_type, None);

    // tl_http_get(url: *const i8) -> *mut i8
    let http_get_type = i8_ptr.fn_type(&[i8_ptr.into()], false);
    module.add_function("tl_http_get", http_get_type, None);

    // Env
    // tl_env_get(key: *const i8) -> *mut i8
    let env_get_type = i8_ptr.fn_type(&[i8_ptr.into()], false);
    module.add_function("tl_env_get", env_get_type, None);

    // tl_env_set(key: *const i8, value: *const i8) -> void
    let env_set_type = void_type.fn_type(&[i8_ptr.into(), i8_ptr.into()], false);
    module.add_function("tl_env_set", env_set_type, None);

    // System
    // tl_system_time() -> f32
    let system_time_type = context.f32_type().fn_type(&[], false);
    module.add_function("tl_system_time", system_time_type, None);

    // tl_system_sleep(seconds: f32) -> void
    let system_sleep_type = void_type.fn_type(&[context.f32_type().into()], false);
    module.add_function("tl_system_sleep", system_sleep_type, None);

    // Mappings
    if let Some(f) = module.get_function("tl_string_concat") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_string_concat as usize);
    }
    if let Some(f) = module.get_function("tl_string_from_int") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_string_from_int as usize);
    }
    if let Some(f) = module.get_function("tl_file_open") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_file_open as usize);
    }
    if let Some(f) = module.get_function("tl_file_read_string") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_file_read_string as usize);
    }
    if let Some(f) = module.get_function("tl_file_write_string") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_file_write_string as usize);
    }
    if let Some(f) = module.get_function("tl_file_close") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_file_close as usize);
    }
    if let Some(f) = module.get_function("tl_path_new") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_path_new as usize);
    }
    if let Some(f) = module.get_function("tl_path_join") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_path_join as usize);
    }
    if let Some(f) = module.get_function("tl_path_exists") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_path_exists as usize);
    }
    if let Some(f) = module.get_function("tl_path_is_dir") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_path_is_dir as usize);
    }
    if let Some(f) = module.get_function("tl_path_is_file") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_path_is_file as usize);
    }
    if let Some(f) = module.get_function("tl_path_to_string") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_path_to_string as usize);
    }
    if let Some(f) = module.get_function("tl_path_free") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_path_free as usize);
    }
    if let Some(f) = module.get_function("tl_http_download") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_http_download as usize);
    }
    if let Some(f) = module.get_function("tl_http_get") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_http_get as usize);
    }
    if let Some(f) = module.get_function("tl_env_get") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_env_get as usize);
    }
    if let Some(f) = module.get_function("tl_env_set") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_env_set as usize);
    }
    if let Some(f) = module.get_function("tl_system_time") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_system_time as usize);
    }
    if let Some(f) = module.get_function("tl_system_sleep") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_system_sleep as usize);
    }
    if let Some(f) = module.get_function("tl_read_line") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_read_line as usize);
    }

    // Memory Manager mappings
    if let Some(f) = module.get_function("tl_mem_enter_scope") {
        execution_engine
            .add_global_mapping(&f, runtime::memory_manager::tl_mem_enter_scope as usize);
    }
    if let Some(f) = module.get_function("tl_mem_exit_scope") {
        execution_engine
            .add_global_mapping(&f, runtime::memory_manager::tl_mem_exit_scope as usize);
    }
    if let Some(f) = module.get_function("tl_mem_register_struct") {
        execution_engine
            .add_global_mapping(&f, runtime::memory_manager::tl_mem_register_struct as usize);
    }
    if let Some(f) = module.get_function("tl_mem_register_tensor") {
        execution_engine
            .add_global_mapping(&f, runtime::memory_manager::tl_mem_register_tensor as usize);
    }
    if let Some(f) = module.get_function("tl_mem_unregister") {
        execution_engine
            .add_global_mapping(&f, runtime::memory_manager::tl_mem_unregister as usize);
    }

    if let Some(f) = module.get_function("tl_query") {
        execution_engine.add_global_mapping(&f, runtime::logic::tl_query as usize);
    }


    // tl_get_memory_mb() -> i64
    let get_memory_type = i64_type.fn_type(&[], false);
    module.add_function("tl_get_memory_mb", get_memory_type, None);

    // tl_get_metal_pool_bytes() -> i64
    let get_metal_pool_bytes_type = i64_type.fn_type(&[], false);
    module.add_function("tl_get_metal_pool_bytes", get_metal_pool_bytes_type, None);

    // tl_get_metal_pool_mb() -> i64
    let get_metal_pool_mb_type = i64_type.fn_type(&[], false);
    module.add_function("tl_get_metal_pool_mb", get_metal_pool_mb_type, None);

    // tl_get_metal_pool_count() -> i64
    let get_metal_pool_count_type = i64_type.fn_type(&[], false);
    module.add_function("tl_get_metal_pool_count", get_metal_pool_count_type, None);
    // tl_metal_sync() -> void
    let metal_sync_type = context.void_type().fn_type(&[], false);
    module.add_function("tl_metal_sync", metal_sync_type, None);
    // tl_trace_mem(file: i8*, line: i32, col: i32, tag: i8*) -> void
    let trace_type = context.void_type().fn_type(
        &[
            i8_ptr.into(),
            context.i32_type().into(),
            context.i32_type().into(),
            i8_ptr.into(),
        ],
        false,
    );
    module.add_function("tl_trace_mem", trace_type, None);
    module.add_function("tl_get_pool_count", get_memory_type, None);
    module.add_function("tl_get_refcount_count", get_memory_type, None);
    module.add_function("tl_get_scope_depth", get_memory_type, None);

    // Memory manager functions
    let void_type = context.void_type();
    let ptr_type = context.ptr_type(inkwell::AddressSpace::default());

    // tl_mem_enter_scope() -> void
    let mem_enter_type = void_type.fn_type(&[], false);
    module.add_function("tl_mem_enter_scope", mem_enter_type, None);

    // tl_mem_exit_scope() -> void
    let mem_exit_type = void_type.fn_type(&[], false);
    module.add_function("tl_mem_exit_scope", mem_exit_type, None);

    // tl_mem_register_struct(ptr) -> void
    let mem_register_struct_type = void_type.fn_type(&[ptr_type.into()], false);
    module.add_function("tl_mem_register_struct", mem_register_struct_type, None);

    // tl_mem_register_tensor(ptr) -> void
    let mem_register_tensor_type = void_type.fn_type(&[ptr_type.into()], false);
    module.add_function("tl_mem_register_tensor", mem_register_tensor_type, None);

    // tl_mem_unregister(ptr) -> void
    let mem_unregister_type = void_type.fn_type(&[ptr_type.into()], false);
    module.add_function("tl_mem_unregister", mem_unregister_type, None);

    // tl_mem_register_struct_named(ptr, name) -> void
    let reg_struct_named_type = void_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);
    module.add_function("tl_mem_register_struct_named", reg_struct_named_type, None);

    // tl_ptr_acquire(ptr) -> void
    let ptr_acquire_type = void_type.fn_type(&[ptr_type.into()], false);
    module.add_function("tl_ptr_acquire", ptr_acquire_type, None);

    // tl_pool_acquire(usize) -> ptr
    let pool_acquire_type = ptr_type.fn_type(&[i64_type.into()], false);
    module.add_function("tl_pool_acquire", pool_acquire_type, None);

    // tl_pool_release(ptr, usize) -> void
    let pool_release_type = void_type.fn_type(&[ptr_type.into(), i64_type.into()], false);
    module.add_function("tl_pool_release", pool_release_type, None);

    // --- KV Cache ---

    // tl_kv_cache_new(layers: i64) -> i64
    let kv_new_type = i64_type.fn_type(&[i64_type.into()], false);
    add_fn("tl_kv_cache_new", kv_new_type);

    if let Some(f) = module.get_function("tl_kv_cache_new") {
        execution_engine.add_global_mapping(&f, runtime::llm::tl_kv_cache_new as usize);
    }

    // tl_kv_cache_free(ptr: i64) -> void
    let kv_free_type = void_type.fn_type(&[i64_type.into()], false);
    add_fn("tl_kv_cache_free", kv_free_type);

    if let Some(f) = module.get_function("tl_kv_cache_free") {
        execution_engine.add_global_mapping(&f, runtime::llm::tl_kv_cache_free as usize);
    }

    // tl_kv_cache_get_k(ptr: i64, layer: i64) -> Tensor
    let kv_get_type = ptr_type.fn_type(&[i64_type.into(), i64_type.into()], false);
    add_fn("tl_kv_cache_get_k", kv_get_type);

    if let Some(f) = module.get_function("tl_kv_cache_get_k") {
        execution_engine.add_global_mapping(&f, runtime::llm::tl_kv_cache_get_k as usize);
    }

    // tl_kv_cache_get_v(ptr: i64, layer: i64) -> Tensor
    add_fn("tl_kv_cache_get_v", kv_get_type);

    if let Some(f) = module.get_function("tl_kv_cache_get_v") {
        execution_engine.add_global_mapping(&f, runtime::llm::tl_kv_cache_get_v as usize);
    }

    // tl_kv_cache_update(ptr, layer, k, v) -> void
    let kv_update_type = void_type.fn_type(
        &[
            i64_type.into(),
            i64_type.into(),
            ptr_type.into(),
            ptr_type.into(),
        ],
        false,
    );
    add_fn("tl_kv_cache_update", kv_update_type);

    if let Some(f) = module.get_function("tl_kv_cache_update") {
        execution_engine.add_global_mapping(&f, runtime::llm::tl_kv_cache_update as usize);
    }

    // --- KV Cache ---

    // tl_kv_cache_new(layers: i64) -> i64
    let kv_new_type = i64_type.fn_type(&[i64_type.into()], false);
    add_fn("tl_kv_cache_new", kv_new_type);

    if let Some(f) = module.get_function("tl_kv_cache_new") {
        execution_engine.add_global_mapping(&f, runtime::llm::tl_kv_cache_new as usize);
    }

    // tl_kv_cache_free(ptr: i64) -> void
    let kv_free_type = void_type.fn_type(&[i64_type.into()], false);
    add_fn("tl_kv_cache_free", kv_free_type);

    if let Some(f) = module.get_function("tl_kv_cache_free") {
        execution_engine.add_global_mapping(&f, runtime::llm::tl_kv_cache_free as usize);
    }

    // tl_kv_cache_get_k(ptr: i64, layer: i64) -> Tensor
    let kv_get_type = ptr_type.fn_type(&[i64_type.into(), i64_type.into()], false);
    add_fn("tl_kv_cache_get_k", kv_get_type);

    if let Some(f) = module.get_function("tl_kv_cache_get_k") {
        execution_engine.add_global_mapping(&f, runtime::llm::tl_kv_cache_get_k as usize);
    }

    // tl_kv_cache_get_v(ptr: i64, layer: i64) -> Tensor
    add_fn("tl_kv_cache_get_v", kv_get_type);

    if let Some(f) = module.get_function("tl_kv_cache_get_v") {
        execution_engine.add_global_mapping(&f, runtime::llm::tl_kv_cache_get_v as usize);
    }

    // tl_kv_cache_update(ptr, layer, k, v) -> void
    let kv_update_type = void_type.fn_type(
        &[
            i64_type.into(),
            i64_type.into(),
            ptr_type.into(),
            ptr_type.into(),
        ],
        false,
    );
    add_fn("tl_kv_cache_update", kv_update_type);

    if let Some(f) = module.get_function("tl_kv_cache_update") {
        execution_engine.add_global_mapping(&f, runtime::llm::tl_kv_cache_update as usize);
    }



    // tl_alloc_tmp(size) -> void*
    let alloc_tmp_type = ptr_type.fn_type(&[i64_type.into()], false);
    module.add_function("tl_alloc_tmp", alloc_tmp_type, None);

    // tl_free_tmp(ptr) -> void
    let free_tmp_type = void_type.fn_type(&[ptr_type.into()], false);
    module.add_function("tl_free_tmp", free_tmp_type, None);

    // tl_query(name: *i8, mask: i64, args: *Tensor, tags: *u8) -> *Tensor
    let query_type = void_ptr.fn_type(
        &[i8_ptr.into(), i64_type.into(), void_ptr.into(), i8_ptr.into()],
        false,
    );
    module.add_function("tl_query", query_type, None);




    // --- Binary file I/O ---
    let i8_ptr = context.ptr_type(inkwell::AddressSpace::default());

    // tl_file_read_binary(path) -> *mut Vec<u8>
    let file_read_binary_type = ptr_type.fn_type(&[i8_ptr.into()], false);
    module.add_function("tl_file_read_binary", file_read_binary_type, None);

    // tl_file_write_binary(path, data) -> bool
    let file_write_binary_type = context
        .bool_type()
        .fn_type(&[i8_ptr.into(), ptr_type.into()], false);
    module.add_function("tl_file_write_binary", file_write_binary_type, None);

    // --- Image loading functions ---

    // tl_image_load_grayscale(path) -> *mut Vec<u8>
    let image_load_type = ptr_type.fn_type(&[i8_ptr.into()], false);
    module.add_function("tl_image_load_grayscale", image_load_type, None);

    // tl_image_width(path) -> i64
    let image_dim_type = i64_type.fn_type(&[i8_ptr.into()], false);
    module.add_function("tl_image_width", image_dim_type, None);

    // tl_image_height(path) -> i64
    module.add_function("tl_image_height", image_dim_type, None);
    // --- Missing Mappings for Llama 3 ---

    // tl_tensor_map_get_quantized -> ptr (OpaqueQTensor*)
    let qt_get_type = void_ptr.fn_type(&[void_ptr.into(), i8_ptr.into()], false);
    add_fn("tl_tensor_map_get_quantized", qt_get_type);
    if let Some(f) = module.get_function("tl_tensor_map_get_quantized") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_map_get_quantized as usize);
    }

    // tl_qtensor_matmul -> ptr (handle) as 2nd arg
    let qmatmul_type = void_ptr.fn_type(&[void_ptr.into(), void_ptr.into()], false);
    add_fn("tl_qtensor_matmul", qmatmul_type);
    if let Some(f) = module.get_function("tl_qtensor_matmul") {
        execution_engine.add_global_mapping(&f, runtime::tl_qtensor_matmul as usize);
    }

    // tl_qtensor_free
    let qfree_type = void_type.fn_type(&[i64_type.into()], false);
    add_fn("tl_qtensor_free", qfree_type);
    if let Some(f) = module.get_function("tl_qtensor_free") {
        execution_engine.add_global_mapping(&f, runtime::tl_qtensor_free as usize);
    }

    // tl_string_contains
    let str_contains_type = context
        .bool_type()
        .fn_type(&[i8_ptr.into(), i8_ptr.into()], false);
    add_fn("tl_string_contains", str_contains_type);
    if let Some(f) = module.get_function("tl_string_contains") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_string_contains as usize);
    }

    // tl_string_from_int
    let str_from_int_type = i8_ptr.fn_type(&[i64_type.into()], false);
    add_fn("tl_string_from_int", str_from_int_type);
    if let Some(f) = module.get_function("tl_string_from_int") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_string_from_int as usize);
    }

    // tl_tensor_argmax(t, dim, keepdim) -> CTensorResult
    let argmax_type = c_tensor_result_type.fn_type(
        &[void_ptr.into(), i64_type.into(), context.bool_type().into()],
        false,
    );
    add_fn("tl_tensor_argmax", argmax_type);
    if let Some(f) = module.get_function("tl_tensor_argmax") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_argmax as usize);
    }

    // tl_tensor_item_i64
    let item_i64_type = i64_type.fn_type(&[void_ptr.into()], false);
    add_fn("tl_tensor_item_i64", item_i64_type);
    if let Some(f) = module.get_function("tl_tensor_item_i64") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_item_i64 as usize);
    }

    // tl_tensor_cat_i64
    if let Some(f) = module.get_function("tl_tensor_cat_i64") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_cat_i64 as usize);
    }
    // Fix: register return type

    // tl_tensor_reshape -> tl_tensor_reshape_new alias for generic fallback
    // Define types locally since this is late in function
    let void_ptr_local = context.ptr_type(AddressSpace::default());
    let reshape_type =
        void_ptr_local.fn_type(&[void_ptr_local.into(), void_ptr_local.into()], false);

    if module.get_function("tl_tensor_reshape").is_none() {
        module.add_function("tl_tensor_reshape", reshape_type, None);
    }
    // Also need return type
    let _tensor_type_local = Type::Tensor(Box::new(Type::F32), 1);

    if let Some(f) = module.get_function("tl_tensor_reshape") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_reshape_new as usize);
    }

    // tl_tensor_sample(t, temp, topp) -> tensor (llm.rs)
    let sample_type = void_ptr.fn_type(&[void_ptr.into(), f32_type.into(), f32_type.into()], false);
    add_fn("tl_tensor_sample", sample_type);
    if let Some(f) = module.get_function("tl_tensor_sample") {
        execution_engine.add_global_mapping(&f, runtime::llm::tl_tensor_sample as usize);
    }
}
