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

    // tl_tensor_pow(t: *mut Tensor, exponent: f32) -> *mut Tensor
    let pow_type = void_ptr.fn_type(&[void_ptr.into(), f32_type.into()], false);
    module.add_function("tl_tensor_pow", pow_type, None);

    // tl_tensor_sqrt(t: *mut Tensor) -> *mut Tensor
    let sqrt_type = void_ptr.fn_type(&[void_ptr.into()], false);
    module.add_function("tl_tensor_sqrt", sqrt_type, None);

    // Transformer Ops
    // tl_tensor_sin(t: *mut Tensor) -> *mut Tensor
    module.add_function("tl_tensor_sin", sqrt_type, None); // Same signature as sqrt

    // tl_tensor_cos(t: *mut Tensor) -> *mut Tensor
    module.add_function("tl_tensor_cos", sqrt_type, None);

    // tl_tensor_relu(t: *mut Tensor) -> *mut Tensor
    module.add_function("tl_tensor_relu", sqrt_type, None);

    // tl_tensor_gelu(t: *mut Tensor) -> *mut Tensor
    module.add_function("tl_tensor_gelu", sqrt_type, None);

    // tl_tensor_tril(t: *mut Tensor, diagonal: i32) -> *mut Tensor
    let i32_type = context.i32_type();
    let tril_type = void_ptr.fn_type(&[void_ptr.into(), i32_type.into()], false);
    module.add_function("tl_tensor_tril", tril_type, None);

    // tl_tensor_sum_dim(t: *mut Tensor, dim: usize, keep: bool) -> *mut Tensor
    // usize -> i64 on 64-bit
    let sum_dim_type = void_ptr.fn_type(
        &[void_ptr.into(), i64_type.into(), context.bool_type().into()],
        false,
    );
    module.add_function("tl_tensor_sum_dim", sum_dim_type, None);

    // tl_tensor_embedding(indices: *mut Tensor, weights: *mut Tensor) -> *mut Tensor
    let embedding_type = void_ptr.fn_type(&[void_ptr.into(), void_ptr.into()], false);
    module.add_function("tl_tensor_embedding", embedding_type, None);

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
    // module.add_function("tl_tensor_sqrt", unary_type, None); // Already declared above with specific type

    // Binary ops: pow
    // module.add_function("tl_tensor_pow", bin_type, None); // Already declared above with specific type

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
    if let Some(f) = module.get_function("tl_tensor_sin") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_sin as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_cos") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_cos as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_relu") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_relu as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_gelu") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_gelu as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_tril") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_tril as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_sum_dim") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_sum_dim as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_embedding") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_embedding as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_save") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_save as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_load") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_load as usize);
    }
    if let Some(f) = module.get_function("tl_save_all_params") {
        execution_engine.add_global_mapping(&f, runtime::tl_save_all_params as usize);
    }
    if let Some(f) = module.get_function("tl_load_all_params") {
        execution_engine.add_global_mapping(&f, runtime::tl_load_all_params as usize);
    }
    if let Some(f) = module.get_function("tl_add_parameter") {
        execution_engine.add_global_mapping(&f, runtime::tl_add_parameter as usize);
    }
    if let Some(f) = module.get_function("tl_register_parameter") {
        execution_engine.add_global_mapping(&f, runtime::tl_register_parameter as usize);
    }

    // VarBuilder-based parameter management
    if let Some(f) = module.get_function("tl_varbuilder_get") {
        execution_engine.add_global_mapping(&f, runtime::tl_varbuilder_get as usize);
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

    // Populate return types for lookups
    let tensor_type = Type::Tensor(Box::new(Type::F32), 1); // Common return type for many tensor ops

    fn_return_types.insert("tl_tensor_new".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_add".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_mul".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_neg".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_slice".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_print".to_string(), Type::Void);
    fn_return_types.insert("tl_print_i64".to_string(), Type::Void);
    fn_return_types.insert("tl_print_f32".to_string(), Type::Void);
    fn_return_types.insert("tl_tensor_len".to_string(), Type::I64);
    fn_return_types.insert("tl_tensor_get".to_string(), Type::F32);
    // Add missing types that were likely in the original file but I need to make sure are present
    fn_return_types.insert("tl_tensor_transpose".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_reshape".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_reshape_dims".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_sum_dim".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_matmul".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_sum".to_string(), Type::F32); // Or tensor 0D? Usually returns scalar in simple implementation
                                                                    // ... complete as needed based on original CodeGen
                                                                    // tl_tensor_reshape_dims(tensor: *mut OpaqueTensor, dims: *const i64, num_dims: i64) -> *mut OpaqueTensor
    let tensor_reshape_dims_type = void_ptr.fn_type(
        &[
            void_ptr.into(), // tensor
            context
                .i64_type()
                .ptr_type(inkwell::AddressSpace::default())
                .into(), // dims ptr
            context.i64_type().into(), // num_dims
        ],
        false,
    );
    module.add_function("tl_tensor_reshape_dims", tensor_reshape_dims_type, None);

    // tl_tensor_reshape(tensor: *mut OpaqueTensor, new_shape: *mut OpaqueTensor) -> *mut OpaqueTensor
    let tensor_reshape_type = void_ptr.fn_type(&[void_ptr.into(), void_ptr.into()], false);
    module.add_function("tl_tensor_reshape", tensor_reshape_type, None);

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

    // VarBuilder-based parameter management (following Candle's official pattern)
    // tl_varbuilder_get(name: *const c_char, rank: usize, shape: *const usize) -> *mut OpaqueTensor
    let i8_ptr = context.ptr_type(AddressSpace::default());
    let varbuilder_get_type =
        void_ptr.fn_type(&[i8_ptr.into(), i64_type.into(), usize_ptr.into()], false);
    module.add_function("tl_varbuilder_get", varbuilder_get_type, None);

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

    // tl_save_all_params(path: *const i8) -> void
    let save_all_type = void_type.fn_type(&[i8_ptr.into()], false);
    module.add_function("tl_save_all_params", save_all_type, None);

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

    // Register new return types
    fn_return_types.insert("tl_tensor_randn".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_grad".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_detach".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_softmax".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_cross_entropy".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_sum".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_backward".to_string(), Type::Void);
    fn_return_types.insert("tl_tensor_sub_assign".to_string(), Type::Void);
    fn_return_types.insert("tl_tensor_pow".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_sqrt".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_sin".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_cos".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_relu".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_gelu".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_tril".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_sum_dim".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_embedding".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_save".to_string(), Type::Void);
    fn_return_types.insert("tl_tensor_load".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_save_all_params".to_string(), Type::Void);
    fn_return_types.insert("tl_load_all_params".to_string(), Type::Void);
    fn_return_types.insert("tl_add_parameter".to_string(), Type::Void);
    fn_return_types.insert("tl_register_parameter".to_string(), tensor_type.clone());

    // VarBuilder-based parameter management
    fn_return_types.insert("tl_varbuilder_get".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_update_all_params".to_string(), Type::Void);
    fn_return_types.insert("tl_varbuilder_grad".to_string(), tensor_type.clone());

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

    // Return types
    fn_return_types.insert(
        "tl_string_concat".to_string(),
        Type::UserDefined("String".to_string()),
    );
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
    fn_return_types.insert(
        "tl_path_new".to_string(),
        Type::UserDefined("Path".to_string()),
    );
    fn_return_types.insert(
        "tl_path_join".to_string(),
        Type::UserDefined("Path".to_string()),
    );
    fn_return_types.insert("tl_path_exists".to_string(), Type::Bool);
    fn_return_types.insert("tl_path_is_dir".to_string(), Type::Bool);
    fn_return_types.insert("tl_path_is_file".to_string(), Type::Bool);
    fn_return_types.insert(
        "tl_path_to_string".to_string(),
        Type::UserDefined("String".to_string()),
    );
    fn_return_types.insert("tl_path_free".to_string(), Type::Void);
    fn_return_types.insert("tl_http_download".to_string(), Type::Bool);
    fn_return_types.insert(
        "tl_http_get".to_string(),
        Type::UserDefined("String".to_string()),
    );
    fn_return_types.insert(
        "tl_env_get".to_string(),
        Type::UserDefined("String".to_string()),
    );
    fn_return_types.insert("tl_env_set".to_string(), Type::Void);
    fn_return_types.insert("tl_system_time".to_string(), Type::F32); // Using F32 as default float for now
    fn_return_types.insert("tl_system_sleep".to_string(), Type::Void);
}
