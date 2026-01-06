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

    // Helper to add function only if not exists
    let add_fn = |name: &str, ty: inkwell::types::FunctionType<'ctx>| {
        if module.get_function(name).is_none() {
            module.add_function(name, ty, None);
        }
    };

    let print_i64_type = void_type.fn_type(&[i64_type.into()], false);
    add_fn("tl_print_i64", print_i64_type);

    let print_f32_type = void_type.fn_type(&[f32_type.into()], false);
    add_fn("tl_print_f32", print_f32_type);

    let print_str_type = void_type.fn_type(&[void_ptr.into()], false);
    add_fn("tl_print_string", print_str_type);

    // tl_print_ptr for debugging tensor pointers
    let print_ptr_type = void_type.fn_type(&[void_ptr.into()], false);
    add_fn("tl_print_ptr", print_ptr_type);

    // malloc(size: i64) -> *u8
    let malloc_type = void_ptr.fn_type(&[i64_type.into()], false);
    add_fn("malloc", malloc_type);

    // calloc(num: i64, size: i64) -> *u8
    let calloc_type = void_ptr.fn_type(&[i64_type.into(), i64_type.into()], false);
    add_fn("calloc", calloc_type);

    // free(ptr: *u8) -> void
    let free_type = void_type.fn_type(&[void_ptr.into()], false);
    add_fn("free", free_type);

    // tl_tensor_dim(t: *mut OpaqueTensor, dim_idx: usize) -> i64
    let dim_type = i64_type.fn_type(&[void_ptr.into(), i64_type.into()], false);
    add_fn("tl_tensor_dim", dim_type);

    // tl_tensor_get_f32_md(t: *mut OpaqueTensor, indices: *const i64, rank: usize) -> f32
    let get_md_type = f32_type.fn_type(&[void_ptr.into(), i64_ptr.into(), i64_type.into()], false);
    add_fn("tl_tensor_get_f32_md", get_md_type);

    // tl_tensor_new(data: *const f32, rank: usize, shape: *const usize) -> *mut OpaqueTensor
    let new_type = void_ptr.fn_type(&[f32_ptr.into(), i64_type.into(), usize_ptr.into()], false);
    add_fn("tl_tensor_new", new_type);

    // tl_tensor_from_i64_array(data: *const i64, len: usize) -> *mut OpaqueTensor
    let from_i64_type = void_ptr.fn_type(&[i64_ptr.into(), i64_type.into()], false);
    add_fn("tl_tensor_from_i64_array", from_i64_type);

    let binop_type = void_ptr.fn_type(&[void_ptr.into(), void_ptr.into()], false);
    add_fn("tl_tensor_sub", binop_type);

    // tl_tensor_free(t: *mut) -> void
    let free_type = void_type.fn_type(&[void_ptr.into()], false);
    add_fn("tl_tensor_free", free_type);

    // tl_tensor_clone(t: *mut) -> *mut
    let clone_type = void_ptr.fn_type(&[void_ptr.into()], false);
    add_fn("tl_tensor_clone", clone_type);

    // tl_tensor_add(a: *mut, b: *mut) -> *mut
    let bin_type = void_ptr.fn_type(&[void_ptr.into(), void_ptr.into()], false);
    add_fn("tl_tensor_add", bin_type);
    add_fn("tl_tensor_mul", bin_type);

    // tl_tensor_print(t: *mut) -> void
    let print_type = void_type.fn_type(&[void_ptr.into()], false);
    add_fn("tl_tensor_print", print_type);
    add_fn("tl_tensor_print_1", print_type.clone());
    add_fn("tl_tensor_print_2", print_type.clone());
    add_fn("tl_tensor_print_3", print_type.clone());

    // tl_tensor_get(t: *mut, index: usize) -> f32 (Simplification: 1D get)
    let get_type = f32_type.fn_type(&[void_ptr.into(), i64_type.into()], false);
    add_fn("tl_tensor_get", get_type);

    // tl_tensor_slice(t: *mut, start: usize, len: usize) -> *mut
    let slice_type = void_ptr.fn_type(&[void_ptr.into(), i64_type.into(), i64_type.into()], false);
    add_fn("tl_tensor_slice", slice_type);

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

    // tl_tensor_pow(t: *mut Tensor, exponent: *mut Tensor) -> *mut Tensor
    let pow_type = void_ptr.fn_type(&[void_ptr.into(), void_ptr.into()], false);
    add_fn("tl_tensor_pow", pow_type);

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
    
    // Manual mapping for tl_clear_grads to avoid dlsym issues
    if let Some(f) = module.get_function("tl_clear_grads") {
        execution_engine.add_global_mapping(&f, runtime::tl_clear_grads as usize);
    }

    // tl_checkpoint(ctx: *mut, func: *mut, input: *mut) -> *mut
    let checkpoint_type = void_ptr.fn_type(&[void_ptr.into(), void_ptr.into(), void_ptr.into()], false);
    add_fn("tl_checkpoint", checkpoint_type);
    
    if let Some(f) = module.get_function("tl_checkpoint") {
        execution_engine.add_global_mapping(&f, runtime::checkpoint::tl_checkpoint as usize);
    }

    // tl_tensor_sum_dim(t: *mut Tensor, dim: usize, keep: bool) -> *mut Tensor
    // usize -> i64 on 64-bit
    let sum_dim_type = void_ptr.fn_type(
        &[void_ptr.into(), i64_type.into(), context.bool_type().into()],
        false,
    );
    add_fn("tl_tensor_sum_dim", sum_dim_type);

    // tl_tensor_embedding(indices: *mut Tensor, weights: *mut Tensor) -> *mut Tensor
    let embedding_type = void_ptr.fn_type(&[void_ptr.into(), void_ptr.into()], false);
    add_fn("tl_tensor_embedding", embedding_type);

    // tl_tensor_sum(t: *mut) -> *mut
    add_fn("tl_tensor_sum", unary_type);

    // tl_tensor_div(a: *mut, b: *mut) -> *mut
    add_fn("tl_tensor_div", bin_type);

    // tl_tensor_matmul(a: *mut, b: *mut) -> *mut
    add_fn("tl_tensor_matmul", bin_type);

    // Unary ops: exp, log
    add_fn("tl_tensor_exp", unary_type);
    add_fn("tl_tensor_log", unary_type);

    // Assign ops: add_assign, sub_assign, mul_assign, div_assign (return void)
    let assign_type = void_type.fn_type(&[void_ptr.into(), void_ptr.into()], false);
    add_fn("tl_tensor_add_assign", assign_type);
    add_fn("tl_tensor_sub_assign", assign_type);
    add_fn("tl_tensor_mul_assign", assign_type);
    add_fn("tl_tensor_div_assign", assign_type);

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

    // tl_tensor_map_get(map, name) -> *mut Tensor
    let map_get_type = void_ptr.fn_type(&[void_ptr.into(), i8_ptr.into()], false);
    add_fn("tl_tensor_map_get", map_get_type);

    // tl_tensor_map_free(map)
    let map_free_type = void_type.fn_type(&[void_ptr.into()], false);
    add_fn("tl_tensor_map_free", map_free_type);

    // Reshape
    let reshape_dims_type =
        void_ptr.fn_type(&[void_ptr.into(), i64_ptr.into(), i64_type.into()], false);
    add_fn("tl_tensor_reshape_dims", reshape_dims_type);

    let reshape_tensor_type = void_ptr.fn_type(&[void_ptr.into(), void_ptr.into()], false);
    add_fn("tl_tensor_reshape", reshape_tensor_type);

    // Randn
    // tl_tensor_randn(rank: usize, shape: *const usize, req_grad: bool) -> *mut OpaqueTensor
    let randn_type = void_ptr.fn_type(
        &[
            i64_type.into(),            // Rank
            void_ptr.into(),            // Shape Ptr
            context.bool_type().into(), // Req Grad
        ],
        false,
    );
    add_fn("tl_tensor_randn_debug", randn_type);

    // VarBuilder
    // tl_varbuilder_get(name: *const c_char, rank: usize, shape: *const usize)
    let varbuilder_get_type =
        void_ptr.fn_type(&[i8_ptr.into(), i64_type.into(), usize_ptr.into()], false);
    add_fn("tl_varbuilder_get", varbuilder_get_type);

    // tl_varbuilder_get_from_tensor(name: *const c_char, shape_tensor: *mut OpaqueTensor)
    let varbuilder_get_tensor_type = void_ptr.fn_type(&[i8_ptr.into(), void_ptr.into()], false);
    add_fn("tl_varbuilder_get_from_tensor", varbuilder_get_tensor_type);

    // update_all_params(lr: f32)
    let update_type = void_type.fn_type(&[f32_type.into()], false);
    add_fn("tl_update_all_params", update_type);

    // grad(name: *const c_char) -> Tensor
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

    let path_to_str_type = i8_ptr.fn_type(&[void_ptr.into()], false);
    add_fn("tl_path_to_string", path_to_str_type);

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

    // System
    let sys_time_type = f32_type.fn_type(&[], false); // Return f32 timestamp
    add_fn("tl_system_time", sys_time_type);

    let sys_sleep_type = void_type.fn_type(&[f32_type.into()], false);
    add_fn("tl_system_sleep", sys_sleep_type);

    let mem_mb_type = i64_type.fn_type(&[], false);
    add_fn("tl_get_memory_mb", mem_mb_type);

    // Memory Scope
    let enter_scope_type = void_type.fn_type(&[], false);
    add_fn("tl_mem_enter_scope", enter_scope_type);

    let exit_scope_type = void_type.fn_type(&[], false);
    add_fn("tl_mem_exit_scope", exit_scope_type);

    let reg_struct_type = void_type.fn_type(&[void_ptr.into()], false);
    add_fn("tl_mem_register_struct", reg_struct_type);
    add_fn("tl_mem_register_tensor", reg_struct_type);
    add_fn("tl_mem_unregister", reg_struct_type);

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
    if let Some(f) = module.get_function("tl_print_ptr") {
        execution_engine.add_global_mapping(&f, runtime::tl_print_ptr as *const () as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_new") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_new as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_from_i64_array") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_from_i64_array as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_matmul") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_matmul as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_contiguous") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_contiguous as usize);
    }
    if let Some(f) = module.get_function("tl_tensor_print") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_print as usize);
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
    if let Some(f) = module.get_function("tl_tensor_randn_debug") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_randn_debug as usize);
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
    if let Some(f) = module.get_function("tl_load_all_params") {
        execution_engine.add_global_mapping(&f, runtime::tl_load_all_params as usize);
    }
    if let Some(f) = module.get_function("tl_register_parameter") {
        execution_engine.add_global_mapping(&f, runtime::tl_register_parameter as usize);
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
    if let Some(f) = module.get_function("tl_mem_register_tensor") {
        execution_engine
            .add_global_mapping(&f, runtime::memory_manager::tl_mem_register_tensor as usize);
    }
    if let Some(f) = module.get_function("tl_mem_unregister") {
        execution_engine
            .add_global_mapping(&f, runtime::memory_manager::tl_mem_unregister as usize);
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
    if let Some(f) = module.get_function("tl_tensor_item_i64") {
        execution_engine.add_global_mapping(&f, runtime::tl_tensor_item_i64 as usize);
    } // End of function
    let tensor_type = Type::Tensor(Box::new(Type::F32), 1); // Common return type for many tensor ops

    fn_return_types.insert("tl_tensor_new".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_add".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_mul".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_neg".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_slice".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_print".to_string(), Type::Void);
    fn_return_types.insert("tl_tensor_print_1".to_string(), Type::Void);
    fn_return_types.insert("tl_tensor_print_2".to_string(), Type::Void);
    fn_return_types.insert("tl_tensor_print_3".to_string(), Type::Void);
    fn_return_types.insert("tl_print_i64".to_string(), Type::Void);
    fn_return_types.insert("tl_print_f32".to_string(), Type::Void);
    fn_return_types.insert("tl_tensor_len".to_string(), Type::I64);
    fn_return_types.insert("tl_tensor_get".to_string(), Type::F32);
    fn_return_types.insert("tl_tensor_dim".to_string(), Type::I64);
    // Add missing types that were likely in the original file but I need to make sure are present
    fn_return_types.insert("tl_tensor_transpose".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_reshape".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_reshape_dims".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_sum_dim".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_matmul".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_contiguous".to_string(), tensor_type.clone());
    fn_return_types.insert("tl_tensor_sum".to_string(), Type::F32); // Or tensor 0D? Usually returns scalar in simple implementation
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

    // tl_save_all_params(path: *const i8) -> void
    let save_all_type = void_type.fn_type(&[i8_ptr.into()], false);
    module.add_function("tl_save_all_params", save_all_type, None);

    // tl_add_parameter(name: *str, t: *mut Tensor) -> void
    let add_param_type = void_type.fn_type(&[void_ptr.into(), void_ptr.into()], false);
    module.add_function("tl_add_parameter", add_param_type, None);

    // tl_tensor_argmax(t: *mut, dim: i64, keep_dim: bool) -> *mut
    let argmax_type = void_ptr.fn_type(&[void_ptr.into(), i64_type.into(), context.bool_type().into()], false);
    module.add_function("tl_tensor_argmax", argmax_type, None);

    // tl_tensor_item_i64(t: *mut) -> i64
    let item_i64_type = i64_type.fn_type(&[void_ptr.into()], false);
    module.add_function("tl_tensor_item_i64", item_i64_type, None);


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
    fn_return_types.insert("tl_arena_get_offset".to_string(), Type::I64);
    fn_return_types.insert("tl_arena_get_capacity".to_string(), Type::I64);
    fn_return_types.insert("tl_arena_is_active".to_string(), Type::Bool);
    fn_return_types.insert("tl_arena_alloc".to_string(), Type::I64);
    fn_return_types.insert("tl_arena_reset".to_string(), Type::Void);
    fn_return_types.insert("tl_get_memory_mb".to_string(), Type::I64);

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

    // tl_get_memory_mb() -> i64
    let get_memory_type = i64_type.fn_type(&[], false);
    module.add_function("tl_get_memory_mb", get_memory_type, None);
    fn_return_types.insert("tl_get_memory_mb".to_string(), Type::I64);

    // Memory manager functions
    let void_type = context.void_type();
    let ptr_type = context.ptr_type(inkwell::AddressSpace::default());

    // tl_mem_enter_scope() -> void
    let mem_enter_type = void_type.fn_type(&[], false);
    module.add_function("tl_mem_enter_scope", mem_enter_type, None);
    fn_return_types.insert("tl_mem_enter_scope".to_string(), Type::Void);

    // tl_mem_exit_scope() -> void
    let mem_exit_type = void_type.fn_type(&[], false);
    module.add_function("tl_mem_exit_scope", mem_exit_type, None);
    fn_return_types.insert("tl_mem_exit_scope".to_string(), Type::Void);

    // tl_mem_register_struct(ptr) -> void
    let mem_register_struct_type = void_type.fn_type(&[ptr_type.into()], false);
    module.add_function("tl_mem_register_struct", mem_register_struct_type, None);
    fn_return_types.insert("tl_mem_register_struct".to_string(), Type::Void);

    // tl_mem_register_tensor(ptr) -> void
    let mem_register_tensor_type = void_type.fn_type(&[ptr_type.into()], false);
    module.add_function("tl_mem_register_tensor", mem_register_tensor_type, None);
    fn_return_types.insert("tl_mem_register_tensor".to_string(), Type::Void);

    // tl_mem_unregister(ptr) -> void
    let mem_unregister_type = void_type.fn_type(&[ptr_type.into()], false);
    module.add_function("tl_mem_unregister", mem_unregister_type, None);
    fn_return_types.insert("tl_mem_unregister".to_string(), Type::Void);

    // tl_pool_acquire(usize) -> ptr
    let pool_acquire_type = ptr_type.fn_type(&[i64_type.into()], false);
    module.add_function("tl_pool_acquire", pool_acquire_type, None);
    fn_return_types.insert(
        "tl_pool_acquire".to_string(),
        Type::Tensor(Box::new(Type::F32), 1),
    ); // Simplified rank

    // tl_pool_release(ptr, usize) -> void
    let pool_release_type = void_type.fn_type(&[ptr_type.into(), i64_type.into()], false);
    module.add_function("tl_pool_release", pool_release_type, None);
    fn_return_types.insert("tl_pool_release".to_string(), Type::Void);

    // Arena Allocator Functions
    // tl_arena_init(capacity: i64) -> void
    let arena_init_type = void_type.fn_type(&[i64_type.into()], false);
    module.add_function("tl_arena_init", arena_init_type, None);
    fn_return_types.insert("tl_arena_init".to_string(), Type::Void);

    // tl_arena_alloc(size) -> i64 (address)
    let arena_alloc_type = i64_type.fn_type(&[i64_type.into()], false);
    module.add_function("tl_arena_alloc", arena_alloc_type, None);
    // fn_return_types.insert("tl_arena_alloc".to_string(), Type::I64); // Already inserted at line 786

    // tl_arena_malloc(size) -> void*
    let arena_malloc_type = ptr_type.fn_type(&[i64_type.into()], false);
    module.add_function("tl_arena_malloc", arena_malloc_type, None);
    fn_return_types.insert(
        "tl_arena_malloc".to_string(),
        Type::Tensor(Box::new(Type::F32), 1),
    );

    // tl_arena_is_active() -> bool
    let arena_is_active_type = context.bool_type().fn_type(&[], false);
    module.add_function("tl_arena_is_active", arena_is_active_type, None);
    fn_return_types.insert("tl_arena_is_active".to_string(), Type::Bool);

    // tl_arena_free() -> void
    let arena_free_type = void_type.fn_type(&[], false);
    module.add_function("tl_arena_free", arena_free_type, None);
    fn_return_types.insert("tl_arena_free".to_string(), Type::Void);

    // tl_alloc_tmp(size) -> void*
    let alloc_tmp_type = ptr_type.fn_type(&[i64_type.into()], false);
    module.add_function("tl_alloc_tmp", alloc_tmp_type, None);
    fn_return_types.insert(
        "tl_alloc_tmp".to_string(),
        Type::Tensor(Box::new(Type::F32), 1),
    );

    // tl_free_tmp(ptr) -> void
    let free_tmp_type = void_type.fn_type(&[ptr_type.into()], false);
    module.add_function("tl_free_tmp", free_tmp_type, None);
    fn_return_types.insert("tl_free_tmp".to_string(), Type::Void);

    fn_return_types.insert("tl_env_set".to_string(), Type::Void);
    fn_return_types.insert("tl_system_time".to_string(), Type::F32); // Using F32 as default float for now
    fn_return_types.insert("tl_system_sleep".to_string(), Type::Void);
}
