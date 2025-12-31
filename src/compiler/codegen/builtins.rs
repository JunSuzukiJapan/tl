use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::module::Module;
use inkwell::AddressSpace;
use std::collections::HashMap;

use crate::compiler::ast::Type;
use crate::runtime;

pub fn declare_runtime_functions<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    execution_engine: &ExecutionEngine<'ctx>,
    fn_return_types: &mut HashMap<String, Type>,
) {
    let i64_type = context.i64_type(); // usize
    let f32_type = context.f32_type();
    let f32_ptr = context.ptr_type(AddressSpace::default());
    let usize_ptr = context.ptr_type(AddressSpace::default());
    let i64_ptr = context.ptr_type(AddressSpace::default());
    let void_ptr = context.ptr_type(AddressSpace::default()); // OpaqueTensor*
    let void_type = context.void_type();

    let print_i64_type = void_type.fn_type(&[i64_type.into()], false);
    module.add_function("tl_print_i64", print_i64_type, None);

    let print_f32_type = void_type.fn_type(&[f32_type.into()], false);
    module.add_function("tl_print_f32", print_f32_type, None);

    let print_str_type = void_type.fn_type(&[void_ptr.into()], false);
    module.add_function("tl_print_string", print_str_type, None);

    // malloc(size: i64) -> *u8
    let malloc_type = void_ptr.fn_type(&[i64_type.into()], false);
    module.add_function("malloc", malloc_type, None);

    // calloc(num: i64, size: i64) -> *u8
    let calloc_type = void_ptr.fn_type(&[i64_type.into(), i64_type.into()], false);
    module.add_function("calloc", calloc_type, None);

    // tl_tensor_dim(t: *mut OpaqueTensor, dim_idx: usize) -> i64
    let dim_type = i64_type.fn_type(&[void_ptr.into(), i64_type.into()], false);
    module.add_function("tl_tensor_dim", dim_type, None);

    // tl_tensor_get_f32_md(t: *mut OpaqueTensor, indices: *const i64, rank: usize) -> f32
    let get_md_type = f32_type.fn_type(&[void_ptr.into(), i64_ptr.into(), i64_type.into()], false);
    module.add_function("tl_tensor_get_f32_md", get_md_type, None);

    // tl_tensor_new(data: *const f32, rank: usize, shape: *const usize) -> *mut OpaqueTensor
    let new_type = void_ptr.fn_type(&[f32_ptr.into(), i64_type.into(), usize_ptr.into()], false);
    module.add_function("tl_tensor_new", new_type, None);

    let binop_type = void_ptr.fn_type(&[void_ptr.into(), void_ptr.into()], false);
    module.add_function("tl_tensor_sub", binop_type, None);

    // tl_tensor_free(t: *mut) -> void
    let free_type = void_type.fn_type(&[void_ptr.into()], false);
    module.add_function("tl_tensor_free", free_type, None);

    // tl_tensor_clone(t: *mut) -> *mut
    let clone_type = void_ptr.fn_type(&[void_ptr.into()], false);
    module.add_function("tl_tensor_clone", clone_type, None);

    // tl_tensor_add(a: *mut, b: *mut) -> *mut
    let bin_type = void_ptr.fn_type(&[void_ptr.into(), void_ptr.into()], false);
    module.add_function("tl_tensor_add", bin_type, None);
    module.add_function("tl_tensor_mul", bin_type, None);

    // tl_tensor_print(t: *mut) -> void
    let print_type = void_type.fn_type(&[void_ptr.into()], false);
    module.add_function("tl_tensor_print", print_type, None);

    // tl_tensor_get(t: *mut, index: usize) -> f32 (Simplification: 1D get)
    let get_type = f32_type.fn_type(&[void_ptr.into(), i64_type.into()], false);
    module.add_function("tl_tensor_get", get_type, None);

    // tl_tensor_slice(t: *mut, start: usize, len: usize) -> *mut
    let slice_type = void_ptr.fn_type(&[void_ptr.into(), i64_type.into(), i64_type.into()], false);
    module.add_function("tl_tensor_slice", slice_type, None);

    // tl_tensor_len(t: *mut) -> i64
    let len_type = i64_type.fn_type(&[void_ptr.into()], false);
    module.add_function("tl_tensor_len", len_type, None);

    // tl_tensor_neg(t: *mut) -> *mut
    let neg_type = void_ptr.fn_type(&[void_ptr.into()], false);
    module.add_function("tl_tensor_neg", neg_type, None);

    // tl_tensor_transpose(t: *mut, d0: usize, d1: usize) -> *mut
    let transpose_type =
        void_ptr.fn_type(&[void_ptr.into(), i64_type.into(), i64_type.into()], false);
    module.add_function("tl_tensor_transpose", transpose_type, None);

    // tl_tensor_reshape(t: *mut, shape: *mut) -> *mut
    let reshape_type = void_ptr.fn_type(&[void_ptr.into(), void_ptr.into()], false);
    module.add_function("tl_tensor_reshape", reshape_type, None);

    // tl_tensor_sum(t: *mut) -> *mut
    let sum_type = void_ptr.fn_type(&[void_ptr.into()], false);
    module.add_function("tl_tensor_sum", sum_type, None);

    // tl_tensor_div(a: *mut, b: *mut) -> *mut
    module.add_function("tl_tensor_div", bin_type, None);

    // tl_tensor_matmul(a: *mut, b: *mut) -> *mut
    module.add_function("tl_tensor_matmul", bin_type, None);

    // Unary ops: exp, log, sqrt
    let unary_type = void_ptr.fn_type(&[void_ptr.into()], false);
    module.add_function("tl_tensor_exp", unary_type, None);
    module.add_function("tl_tensor_log", unary_type, None);
    module.add_function("tl_tensor_sqrt", unary_type, None);

    // Binary ops: pow
    module.add_function("tl_tensor_pow", bin_type, None);

    // Assign ops: add_assign, sub_assign, mul_assign, div_assign (return void)
    let assign_type = void_type.fn_type(&[void_ptr.into(), void_ptr.into()], false);
    module.add_function("tl_tensor_add_assign", assign_type, None);
    module.add_function("tl_tensor_sub_assign", assign_type, None);
    module.add_function("tl_tensor_mul_assign", assign_type, None);
    module.add_function("tl_tensor_div_assign", assign_type, None);

    let i8_ptr = context.ptr_type(AddressSpace::default());
    let register_type = void_type.fn_type(&[i8_ptr.into(), void_ptr.into()], false);
    module.add_function("tl_register_tensor", register_type, None);

    // strcmp(s1: *const i8, s2: *const i8) -> i32
    let strcmp_type = context
        .i32_type()
        .fn_type(&[i8_ptr.into(), i8_ptr.into()], false);
    module.add_function("strcmp", strcmp_type, None);

    // --- Global Mappings ---
    // Mapping symbols is critical for JIT.
    // We do it here to keep CodeGenerator::new clean.

    if let Some(f) = module.get_function("tl_print_string") {
        execution_engine.add_global_mapping(&f, runtime::tl_print_string as *const () as usize);
    }
    if let Some(f) = module.get_function("tl_print_f32") {
        execution_engine.add_global_mapping(&f, runtime::tl_print_f32 as *const () as usize);
    }
    if let Some(f) = module.get_function("tl_print_i64") {
        execution_engine.add_global_mapping(&f, runtime::tl_print_i64 as *const () as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_new") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_new as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_matmul") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_matmul as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_print") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_print as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_free") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_free as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_clone") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_clone as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_len") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_len as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_dim") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_dim as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_get_f32_md") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_get_f32_md as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_neg") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_neg as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_transpose") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_transpose as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_reshape") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_reshape as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_get") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_get as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_slice") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_slice as usize);
    }
    if let Some(f) = module.get_function("tl_register_tensor") {
        execution_engine.add_global_mapping(&f, runtime::registry::tl_register_tensor as usize);
    }
    // Additional mappings from previous list...
    if let Some(f) = module.get_function("tl_tensor_randn") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_randn as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_backward") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_backward as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_grad") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_grad as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_detach") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_detach as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_softmax") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_softmax as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_cross_entropy") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_cross_entropy as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_sub_assign") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_sub_assign as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_sum") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_sum as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_add") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_add as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_sub") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_sub as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_mul") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_mul as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_div") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_div as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_pow") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_pow as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_add_assign") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_add_assign as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_mul_assign") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_mul_assign as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_div_assign") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_div_assign as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_exp") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_exp as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_log") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_log as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_sqrt") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_sqrt as usize);
    }

    // Populate return types for lookups
    fn_return_types.insert(
        "tl_tensor_new".to_string(),
        Type::Tensor(Box::new(Type::F32), 1),
    );
    fn_return_types.insert(
        "tl_tensor_add".to_string(),
        Type::Tensor(Box::new(Type::F32), 1),
    );
    fn_return_types.insert(
        "tl_tensor_mul".to_string(),
        Type::Tensor(Box::new(Type::F32), 1),
    );
    fn_return_types.insert(
        "tl_tensor_neg".to_string(),
        Type::Tensor(Box::new(Type::F32), 1),
    );
    fn_return_types.insert(
        "tl_tensor_slice".to_string(),
        Type::Tensor(Box::new(Type::F32), 1),
    );
    fn_return_types.insert("tl_tensor_print".to_string(), Type::Void);
    fn_return_types.insert("tl_print_i64".to_string(), Type::Void);
    fn_return_types.insert("tl_print_f32".to_string(), Type::Void);
    fn_return_types.insert("tl_tensor_len".to_string(), Type::I64);
    fn_return_types.insert("tl_tensor_get".to_string(), Type::F32);
    // Add missing types that were likely in the original file but I need to make sure are present
    fn_return_types.insert(
        "tl_tensor_transpose".to_string(),
        Type::Tensor(Box::new(Type::F32), 1),
    );
    fn_return_types.insert(
        "tl_tensor_reshape".to_string(),
        Type::Tensor(Box::new(Type::F32), 1),
    );
    fn_return_types.insert(
        "tl_tensor_matmul".to_string(),
        Type::Tensor(Box::new(Type::F32), 1),
    );
    fn_return_types.insert("tl_tensor_sum".to_string(), Type::F32); // Or tensor 0D? Usually returns scalar in simple implementation
                                                                    // ... complete as needed based on original CodeGen
                                                                    // tl_tensor_randn(rank: usize, shape: *const usize, req_grad: bool) -> *mut OpaqueTensor
    let randn_type = void_ptr.fn_type(
        &[
            i64_type.into(),
            usize_ptr.into(),
            context.bool_type().into(),
        ],
        false,
    );
    module.add_function("tl_tensor_randn", randn_type, None);

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

    // tl_tensor_sub_assign(ref_t: *mut, val: *mut) -> void
    let sub_assign_type = void_type.fn_type(&[void_ptr.into(), void_ptr.into()], false);
    module.add_function("tl_tensor_sub_assign", sub_assign_type, None);

    // Register new return types
    fn_return_types.insert(
        "tl_tensor_randn".to_string(),
        Type::Tensor(Box::new(Type::F32), 1),
    );
    fn_return_types.insert(
        "tl_tensor_grad".to_string(),
        Type::Tensor(Box::new(Type::F32), 1),
    );
    fn_return_types.insert(
        "tl_tensor_detach".to_string(),
        Type::Tensor(Box::new(Type::F32), 1),
    );
    fn_return_types.insert(
        "tl_tensor_softmax".to_string(),
        Type::Tensor(Box::new(Type::F32), 1),
    );
    fn_return_types.insert(
        "tl_tensor_cross_entropy".to_string(),
        Type::Tensor(Box::new(Type::F32), 1),
    );
    fn_return_types.insert(
        "tl_tensor_sum".to_string(),
        Type::Tensor(Box::new(Type::F32), 1),
    );
    fn_return_types.insert("tl_tensor_backward".to_string(), Type::Void);
    fn_return_types.insert("tl_tensor_sub_assign".to_string(), Type::Void);

    // --- Standard Library Phase 1 ---

    let i8_ptr = context.ptr_type(AddressSpace::default());

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

    // Mappings
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
    if let Some(f) = module.get_function("tl_http_download") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_http_download as usize);
    }
    if let Some(f) = module.get_function("tl_http_get") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_http_get as usize);
    }
    if let Some(f) = module.get_function("tl_env_get") {
        execution_engine.add_global_mapping(&f, runtime::stdlib::tl_env_get as usize);
    }

    // Return types
    fn_return_types.insert(
        "tl_file_open".to_string(),
        Type::UserDefined("File".to_string()),
    );
    fn_return_types.insert(
        "tl_file_read_string".to_string(),
        Type::UserDefined("String".to_string()),
    );
    fn_return_types.insert("tl_file_write_string".to_string(), Type::Void);
    fn_return_types.insert("tl_file_close".to_string(), Type::Void);
    fn_return_types.insert("tl_http_download".to_string(), Type::Bool);
    fn_return_types.insert(
        "tl_http_get".to_string(),
        Type::UserDefined("String".to_string()),
    );
    fn_return_types.insert(
        "tl_env_get".to_string(),
        Type::UserDefined("String".to_string()),
    );
}
