// src/compiler/codegen.rs
use crate::compiler::ast::*;
use crate::runtime::{
    tl_print_f32, tl_print_i64, tl_tensor_add, tl_tensor_backward, tl_tensor_clone, tl_tensor_dim,
    tl_tensor_free, tl_tensor_get, tl_tensor_get_f32_md, tl_tensor_grad, tl_tensor_len,
    tl_tensor_mul, tl_tensor_neg, tl_tensor_new, tl_tensor_print, tl_tensor_randn, tl_tensor_slice,
    tl_tensor_sub_assign, tl_tensor_sum,
}; // Import runtime functions
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::module::Module as InkwellModule;
use inkwell::types::{BasicMetadataTypeEnum, StructType};
// use inkwell::values::Either; // Not used directlyimizationLevel;
use inkwell::values::{BasicValueEnum, FunctionValue, ValueKind};
use inkwell::OptimizationLevel;
use std::collections::HashMap;

pub struct CodeGenerator<'ctx> {
    context: &'ctx Context,
    module: InkwellModule<'ctx>,
    builder: Builder<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
    variables: Vec<HashMap<String, (BasicValueEnum<'ctx>, Type, bool)>>,
    fn_return_types: HashMap<String, Type>,
    struct_types: HashMap<String, StructType<'ctx>>,
    struct_defs: HashMap<String, StructDef>,
}

impl<'ctx> CodeGenerator<'ctx> {
    pub fn new(context: &'ctx Context, module_name: &str) -> Self {
        let module = context.create_module(module_name);
        let builder = context.create_builder();
        let execution_engine = module
            .create_jit_execution_engine(OptimizationLevel::Aggressive)
            .unwrap();

        let mut codegen = CodeGenerator {
            context,
            module,
            builder,
            execution_engine,
            variables: vec![HashMap::new()],
            fn_return_types: HashMap::new(),
            struct_types: HashMap::new(),
            struct_defs: HashMap::new(),
        };

        codegen.declare_runtime_functions();
        codegen
    }

    fn declare_runtime_functions(&mut self) {
        let i64_type = self.context.i64_type(); // usize
        let f32_type = self.context.f32_type();
        let f32_ptr = self.context.ptr_type(inkwell::AddressSpace::default());
        let usize_ptr = self.context.ptr_type(inkwell::AddressSpace::default());
        let i64_ptr = self.context.ptr_type(inkwell::AddressSpace::default());
        let void_ptr = self.context.ptr_type(inkwell::AddressSpace::default()); // OpaqueTensor*
        let void_type = self.context.void_type();

        let print_i64_type = void_type.fn_type(&[i64_type.into()], false);
        self.module
            .add_function("tl_print_i64", print_i64_type, None);

        let print_f32_type = void_type.fn_type(&[self.context.f32_type().into()], false);
        self.module
            .add_function("tl_print_f32", print_f32_type, None);

        // malloc(size: i64) -> *u8
        let malloc_type = void_ptr.fn_type(&[i64_type.into()], false);
        self.module.add_function("malloc", malloc_type, None);

        // calloc(num: i64, size: i64) -> *u8
        let calloc_type = void_ptr.fn_type(&[i64_type.into(), i64_type.into()], false);
        self.module.add_function("calloc", calloc_type, None);

        // tl_tensor_dim(t: *mut OpaqueTensor, dim_idx: usize) -> i64
        let dim_type = i64_type.fn_type(&[void_ptr.into(), i64_type.into()], false);
        self.module.add_function("tl_tensor_dim", dim_type, None);

        // tl_tensor_get_f32_md(t: *mut OpaqueTensor, indices: *const i64, rank: usize) -> f32
        let get_md_type =
            f32_type.fn_type(&[void_ptr.into(), i64_ptr.into(), i64_type.into()], false);
        self.module
            .add_function("tl_tensor_get_f32_md", get_md_type, None);

        // tl_tensor_new(data: *const f32, rank: usize, shape: *const usize) -> *mut OpaqueTensor
        let _new_type =
            void_ptr.fn_type(&[f32_ptr.into(), i64_type.into(), usize_ptr.into()], false);
        // tl_tensor_new(data: *const f32, rank: usize, shape: *const usize) -> *mut OpaqueTensor
        let new_type =
            void_ptr.fn_type(&[f32_ptr.into(), i64_type.into(), usize_ptr.into()], false);
        self.module.add_function("tl_tensor_new", new_type, None);

        let binop_type = void_ptr.fn_type(&[void_ptr.into(), void_ptr.into()], false);
        self.module.add_function("tl_tensor_sub", binop_type, None);

        // tl_tensor_free(t: *mut) -> void
        let free_type = void_type.fn_type(&[void_ptr.into()], false);
        self.module.add_function("tl_tensor_free", free_type, None);

        // tl_tensor_clone(t: *mut) -> *mut
        let clone_type = void_ptr.fn_type(&[void_ptr.into()], false);
        self.module
            .add_function("tl_tensor_clone", clone_type, None);

        // tl_tensor_add(a: *mut, b: *mut) -> *mut
        let bin_type = void_ptr.fn_type(&[void_ptr.into(), void_ptr.into()], false);
        self.module.add_function("tl_tensor_add", bin_type, None);
        self.module.add_function("tl_tensor_mul", bin_type, None);

        // tl_tensor_print(t: *mut) -> void
        let print_type = void_type.fn_type(&[void_ptr.into()], false);
        self.module
            .add_function("tl_tensor_print", print_type, None);

        // tl_tensor_get(t: *mut, index: usize) -> f32 (Simplification: 1D get)
        let get_type = self
            .context
            .f32_type()
            .fn_type(&[void_ptr.into(), i64_type.into()], false);
        self.module.add_function("tl_tensor_get", get_type, None);

        // tl_tensor_slice(t: *mut, start: usize, len: usize) -> *mut
        let slice_type =
            void_ptr.fn_type(&[void_ptr.into(), i64_type.into(), i64_type.into()], false);
        self.module
            .add_function("tl_tensor_slice", slice_type, None);

        // tl_tensor_len(t: *mut) -> i64
        let len_type = i64_type.fn_type(&[void_ptr.into()], false);
        self.module.add_function("tl_tensor_len", len_type, None);

        // tl_tensor_neg(t: *mut) -> *mut
        let neg_type = void_ptr.fn_type(&[void_ptr.into()], false);
        self.module.add_function("tl_tensor_neg", neg_type, None);

        // tl_tensor_transpose(t: *mut, d0: usize, d1: usize) -> *mut
        let transpose_type =
            void_ptr.fn_type(&[void_ptr.into(), i64_type.into(), i64_type.into()], false);
        self.module
            .add_function("tl_tensor_transpose", transpose_type, None);

        // tl_tensor_reshape(t: *mut, shape: *mut) -> *mut
        let reshape_type = void_ptr.fn_type(&[void_ptr.into(), void_ptr.into()], false);
        self.module
            .add_function("tl_tensor_reshape", reshape_type, None);

        // tl_tensor_sum(t: *mut) -> *mut
        let sum_type = void_ptr.fn_type(&[void_ptr.into()], false);
        self.module.add_function("tl_tensor_sum", sum_type, None);

        // tl_tensor_div(a: *mut, b: *mut) -> *mut
        let bin_type = void_ptr.fn_type(&[void_ptr.into(), void_ptr.into()], false);
        self.module.add_function("tl_tensor_div", bin_type, None);
        self.module.add_function("tl_tensor_sub", bin_type, None);

        // tl_tensor_matmul(a: *mut, b: *mut) -> *mut
        self.module.add_function("tl_tensor_matmul", bin_type, None);

        // Unary ops: exp, log, sqrt
        let unary_type = void_ptr.fn_type(&[void_ptr.into()], false);
        self.module.add_function("tl_tensor_exp", unary_type, None);
        self.module.add_function("tl_tensor_log", unary_type, None);
        self.module.add_function("tl_tensor_sqrt", unary_type, None);

        // Map symbols
        if let Some(f) = self.module.get_function("tl_tensor_new") {
            self.execution_engine
                .add_global_mapping(&f, tl_tensor_new as usize);
        }
        if let Some(f) = self.module.get_function("tl_tensor_randn") {
            self.execution_engine
                .add_global_mapping(&f, tl_tensor_randn as usize);
        }
        if let Some(f) = self.module.get_function("tl_tensor_backward") {
            self.execution_engine
                .add_global_mapping(&f, tl_tensor_backward as usize);
        }
        if let Some(f) = self.module.get_function("tl_tensor_grad") {
            self.execution_engine
                .add_global_mapping(&f, tl_tensor_grad as usize);
        }
        if let Some(f) = self.module.get_function("tl_tensor_detach") {
            self.execution_engine
                .add_global_mapping(&f, crate::runtime::tl_tensor_detach as usize);
        }
        if let Some(f) = self.module.get_function("tl_tensor_sub_assign") {
            self.execution_engine
                .add_global_mapping(&f, tl_tensor_sub_assign as usize);
        }
        if let Some(f) = self.module.get_function("tl_tensor_sum") {
            self.execution_engine
                .add_global_mapping(&f, tl_tensor_sum as usize);
        }
        if let Some(f) = self.module.get_function("tl_tensor_add") {
            self.execution_engine
                .add_global_mapping(&f, tl_tensor_add as usize);
        }
        if let Some(f) = self.module.get_function("tl_tensor_sub") {
            self.execution_engine
                .add_global_mapping(&f, crate::runtime::tl_tensor_sub as usize);
        }
        if let Some(f) = self.module.get_function("tl_tensor_mul") {
            self.execution_engine
                .add_global_mapping(&f, tl_tensor_mul as usize);
        }
        if let Some(f) = self.module.get_function("tl_tensor_div") {
            self.execution_engine
                .add_global_mapping(&f, crate::runtime::tl_tensor_div as usize);
        }
        if let Some(f) = self.module.get_function("tl_tensor_matmul") {
            self.execution_engine
                .add_global_mapping(&f, crate::runtime::tl_tensor_matmul as usize);
        }
        if let Some(f) = self.module.get_function("tl_tensor_exp") {
            self.execution_engine
                .add_global_mapping(&f, crate::runtime::tl_tensor_exp as usize);
        }
        if let Some(f) = self.module.get_function("tl_tensor_log") {
            self.execution_engine
                .add_global_mapping(&f, crate::runtime::tl_tensor_log as usize);
        }
        if let Some(f) = self.module.get_function("tl_tensor_sqrt") {
            self.execution_engine
                .add_global_mapping(&f, crate::runtime::tl_tensor_sqrt as usize);
        }
        if let Some(f) = self.module.get_function("tl_tensor_free") {
            self.execution_engine
                .add_global_mapping(&f, tl_tensor_free as usize);
        }
        if let Some(f) = self.module.get_function("tl_tensor_clone") {
            self.execution_engine
                .add_global_mapping(&f, tl_tensor_clone as usize);
        }
        if let Some(f) = self.module.get_function("tl_tensor_print") {
            self.execution_engine
                .add_global_mapping(&f, tl_tensor_print as usize);
        }
        if let Some(f) = self.module.get_function("tl_print_i64") {
            self.execution_engine
                .add_global_mapping(&f, tl_print_i64 as usize);
        }
        if let Some(f) = self.module.get_function("tl_print_f32") {
            self.execution_engine
                .add_global_mapping(&f, tl_print_f32 as usize);
        }
        if let Some(f) = self.module.get_function("tl_tensor_len") {
            self.execution_engine
                .add_global_mapping(&f, tl_tensor_len as usize);
        }
        if let Some(f) = self.module.get_function("tl_tensor_dim") {
            self.execution_engine
                .add_global_mapping(&f, tl_tensor_dim as usize);
        }
        if let Some(f) = self.module.get_function("tl_tensor_get_f32_md") {
            self.execution_engine
                .add_global_mapping(&f, tl_tensor_get_f32_md as usize);
        }
        if let Some(f) = self.module.get_function("tl_tensor_neg") {
            self.execution_engine
                .add_global_mapping(&f, crate::runtime::tl_tensor_neg as usize);
        }
        if let Some(f) = self.module.get_function("tl_tensor_transpose") {
            self.execution_engine
                .add_global_mapping(&f, crate::runtime::tl_tensor_transpose as usize);
        }
        if let Some(f) = self.module.get_function("tl_tensor_reshape") {
            self.execution_engine
                .add_global_mapping(&f, crate::runtime::tl_tensor_reshape as usize);
        }

        // Add internal functions for printing debugging
        // ... (existing)
        if let Some(f) = self.module.get_function("tl_tensor_get") {
            self.execution_engine
                .add_global_mapping(&f, tl_tensor_get as usize);
        }
        if let Some(f) = self.module.get_function("tl_tensor_neg") {
            self.execution_engine
                .add_global_mapping(&f, tl_tensor_neg as usize);
        }
        if let Some(f) = self.module.get_function("tl_tensor_slice") {
            self.execution_engine
                .add_global_mapping(&f, tl_tensor_slice as usize);
        }
        if let Some(f) = self.module.get_function("tl_register_tensor") {
            self.execution_engine
                .add_global_mapping(&f, crate::runtime::registry::tl_register_tensor as usize);
        }

        // Register types for runtime functions (critical for FnCall)
        self.fn_return_types.insert(
            "tl_tensor_new".to_string(),
            Type::Tensor(Box::new(Type::F32), 1),
        );
        self.fn_return_types.insert(
            "tl_tensor_add".to_string(),
            Type::Tensor(Box::new(Type::F32), 1),
        );
        self.fn_return_types.insert(
            "tl_tensor_mul".to_string(),
            Type::Tensor(Box::new(Type::F32), 1),
        );
        self.fn_return_types.insert(
            "tl_tensor_neg".to_string(),
            Type::Tensor(Box::new(Type::F32), 1),
        );
        self.fn_return_types.insert(
            "tl_tensor_slice".to_string(),
            Type::Tensor(Box::new(Type::F32), 1),
        );
        self.fn_return_types
            .insert("tl_tensor_print".to_string(), Type::Void);
        self.fn_return_types
            .insert("tl_print_i64".to_string(), Type::Void);
        self.fn_return_types
            .insert("tl_print_f32".to_string(), Type::Void);
        self.fn_return_types
            .insert("tl_tensor_len".to_string(), Type::I64);
        self.fn_return_types
            .insert("tl_tensor_get".to_string(), Type::F32);

        let i8_ptr = self.context.ptr_type(inkwell::AddressSpace::default());
        let register_type = void_type.fn_type(&[i8_ptr.into(), void_ptr.into()], false);
        self.module
            .add_function("tl_register_tensor", register_type, None);

        // tl_tensor_randn(rank: usize, shape: *const usize, req_grad: bool) -> *mut OpaqueTensor
        let randn_type = void_ptr.fn_type(
            &[
                i64_type.into(),
                usize_ptr.into(),
                self.context.bool_type().into(),
            ],
            false,
        );
        self.module
            .add_function("tl_tensor_randn", randn_type, None);

        // tl_tensor_backward(t: *mut OpaqueTensor) -> void
        let backward_type = void_type.fn_type(&[void_ptr.into()], false);
        self.module
            .add_function("tl_tensor_backward", backward_type, None);

        // tl_tensor_grad(t: *mut OpaqueTensor) -> *mut OpaqueTensor
        let grad_type = void_ptr.fn_type(&[void_ptr.into()], false);
        self.module.add_function("tl_tensor_grad", grad_type, None);

        // tl_tensor_detach(t: *mut, req_grad: bool) -> *mut
        let detach_type =
            void_ptr.fn_type(&[void_ptr.into(), self.context.bool_type().into()], false);
        self.module
            .add_function("tl_tensor_detach", detach_type, None);

        // tl_tensor_sub_assign(ref_t: *mut, val: *mut) -> void
        let sub_assign_type = void_type.fn_type(&[void_ptr.into(), void_ptr.into()], false);
        self.module
            .add_function("tl_tensor_sub_assign", sub_assign_type, None);

        // Register new return types
        self.fn_return_types.insert(
            "tl_tensor_randn".to_string(),
            Type::Tensor(Box::new(Type::F32), 1),
        );
        self.fn_return_types.insert(
            "tl_tensor_grad".to_string(),
            Type::Tensor(Box::new(Type::F32), 1),
        );
        self.fn_return_types.insert(
            "tl_tensor_detach".to_string(),
            Type::Tensor(Box::new(Type::F32), 1),
        );
        self.fn_return_types.insert(
            "tl_tensor_sum".to_string(),
            Type::Tensor(Box::new(Type::F32), 1),
        );
        self.fn_return_types
            .insert("tl_tensor_backward".to_string(), Type::Void);
        self.fn_return_types
            .insert("tl_tensor_sub_assign".to_string(), Type::Void);
    }

    pub fn jit_execute(&self, function_name: &str) -> Result<u64, String> {
        unsafe {
            let function = self
                .execution_engine
                .get_function::<unsafe extern "C" fn() -> u64>(function_name)
                .map_err(|e| format!("JIT compile error: {}", e))?;
            Ok(function.call())
        }
    }

    // Enter a new scope
    fn enter_scope(&mut self) {
        self.variables.push(std::collections::HashMap::new());
    }

    // Emit cleanup for ALL active scopes (reverse order)
    // Used for Return statements to ensure everything is freed before returning
    fn emit_all_scopes_cleanup(&self) {
        if let Some(free_fn) = self.module.get_function("tl_tensor_free") {
            for scope in self.variables.iter().rev() {
                for (_name, (ptr, ty, should_free)) in scope {
                    if *should_free {
                        if let Type::Tensor(_, _) = ty {
                            if ptr.is_pointer_value() {
                                let ptr_val = ptr.into_pointer_value();
                                let void_ptr_type =
                                    self.context.ptr_type(inkwell::AddressSpace::default());
                                if let Ok(loaded) =
                                    self.builder
                                        .build_load(void_ptr_type, ptr_val, "load_for_free")
                                {
                                    let _ = self.builder.build_call(free_fn, &[loaded.into()], "");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Exit the current scope
    fn exit_scope(&mut self) {
        // Only emit cleanup if the current block is NOT terminated
        let is_terminated = self
            .builder
            .get_insert_block()
            .map(|b| b.get_terminator().is_some())
            .unwrap_or(false);

        if !is_terminated {
            // Emit free calls for all tensors in this scope
            if let Some(scope) = self.variables.last() {
                if let Some(free_fn) = self.module.get_function("tl_tensor_free") {
                    for (_name, (ptr, ty, should_free)) in scope {
                        // Only free Tensors if owned
                        if *should_free {
                            if let Type::Tensor(_, _) = ty {
                                // ptr is the Alloca (Pointer to Pointer)
                                // We need to Load the actual Tensor Pointer
                                if ptr.is_pointer_value() {
                                    let ptr_val = ptr.into_pointer_value();
                                    let void_ptr_type =
                                        self.context.ptr_type(inkwell::AddressSpace::default());
                                    let loaded = self
                                        .builder
                                        .build_load(void_ptr_type, ptr_val, "load_for_free")
                                        .unwrap();
                                    self.builder
                                        .build_call(free_fn, &[loaded.into()], "")
                                        .unwrap();
                                }
                            }
                        }
                    }
                }
            }
        }
        self.variables.pop();
    }

    fn compile_struct_defs(&mut self, structs: &[StructDef]) -> Result<(), String> {
        // Pass 1: Opaque
        for s in structs {
            self.struct_types
                .insert(s.name.clone(), self.context.opaque_struct_type(&s.name));
            self.struct_defs.insert(s.name.clone(), s.clone());
        }

        // Pass 2: Body
        for s in structs {
            let mut field_types = Vec::new();
            for (_field_name, field_type) in &s.fields {
                let llvm_type = match field_type {
                    Type::F32 => self.context.f32_type().into(),
                    Type::I64 => self.context.i64_type().into(),
                    Type::Bool => self.context.bool_type().into(),
                    Type::Tensor(_, _) => self
                        .context
                        .ptr_type(inkwell::AddressSpace::default())
                        .into(), // OpaqueTensor*
                    Type::Struct(name) => {
                        if self.struct_types.contains_key(name) {
                            self.context
                                .ptr_type(inkwell::AddressSpace::default())
                                .into()
                        } else {
                            return Err(format!("Struct {} not found", name));
                        }
                    }
                    Type::Vec(_) => self
                        .context
                        .ptr_type(inkwell::AddressSpace::default())
                        .into(),
                    _ => {
                        return Err(format!(
                            "Unsupported field type in struct {}: {:?}",
                            s.name, field_type
                        ))
                    }
                };
                field_types.push(llvm_type);
            }
            if let Some(st) = self.struct_types.get(&s.name) {
                st.set_body(&field_types, false);
            }
        }
        Ok(())
    }

    fn compile_impl_blocks(&mut self, impls: &[ImplBlock]) -> Result<(), String> {
        for imp in impls {
            // Check if struct exists
            let self_type_ptr = if self.struct_types.contains_key(&imp.target_type) {
                self.context.ptr_type(inkwell::AddressSpace::default())
            } else {
                return Err(format!("Impl block for unknown struct {}", imp.target_type));
            };

            for method in &imp.methods {
                let mangled_name = format!("{}_{}", imp.target_type, method.name);

                let mut param_types: Vec<BasicMetadataTypeEnum> = Vec::new();
                // implicit self
                param_types.push(self_type_ptr.into());

                for (_, arg_ty) in &method.args {
                    let ty: BasicMetadataTypeEnum = match arg_ty {
                        Type::F32 => self.context.f32_type().into(),
                        Type::I64 => self.context.i64_type().into(),
                        Type::Tensor(_, _) => self
                            .context
                            .ptr_type(inkwell::AddressSpace::default())
                            .into(),
                        _ => self
                            .context
                            .ptr_type(inkwell::AddressSpace::default())
                            .into(),
                    };
                    param_types.push(ty);
                }

                let fn_type = match &method.return_type {
                    Type::F32 => self.context.f32_type().fn_type(&param_types, false),
                    Type::I64 => self.context.i64_type().fn_type(&param_types, false),
                    Type::Bool => self.context.bool_type().fn_type(&param_types, false),
                    Type::Void => self.context.void_type().fn_type(&param_types, false),
                    Type::Tensor(_, _) | Type::Struct(_) | Type::UserDefined(_) => self
                        .context
                        .ptr_type(inkwell::AddressSpace::default())
                        .fn_type(&param_types, false),
                    _ => self.context.void_type().fn_type(&param_types, false),
                };

                let function = self.module.add_function(&mangled_name, fn_type, None);
                self.fn_return_types
                    .insert(mangled_name.clone(), method.return_type.clone());

                // Compile Body
                let entry = self.context.append_basic_block(function, "entry");
                self.builder.position_at_end(entry);
                self.enter_scope();

                // Add self
                let self_val = function.get_nth_param(0).unwrap();
                let self_alloca = self.create_entry_block_alloca(
                    function,
                    "self",
                    &Type::Struct(imp.target_type.clone()),
                );
                self.builder
                    .build_store(self_alloca, self_val)
                    .map_err(|e| e.to_string())?;
                self.variables.last_mut().unwrap().insert(
                    "self".to_string(),
                    (
                        self_alloca.into(),
                        Type::Struct(imp.target_type.clone()),
                        false,
                    ),
                );

                for (i, (arg_name, arg_type)) in method.args.iter().enumerate() {
                    let arg_val = function.get_nth_param((i + 1) as u32).unwrap();
                    let arg_alloca = self.create_entry_block_alloca(function, arg_name, arg_type);
                    self.builder
                        .build_store(arg_alloca, arg_val)
                        .map_err(|e| e.to_string())?;
                    self.variables.last_mut().unwrap().insert(
                        arg_name.clone(),
                        (arg_alloca.into(), arg_type.clone(), false),
                    );
                }

                for stmt in &method.body {
                    self.compile_stmt(stmt)?;
                }

                if let Type::Void = method.return_type {
                    let is_terminated = self
                        .builder
                        .get_insert_block()
                        .map(|b| b.get_terminator().is_some())
                        .unwrap_or(false);
                    if !is_terminated {
                        self.builder.build_return(None).unwrap();
                    }
                }

                self.exit_scope();

                if !function.verify(true) {
                    function.print_to_stderr();
                    return Err(format!("Invalid generated method {}", mangled_name));
                }
            }
        }
        Ok(())
    }

    pub fn compile_module(&mut self, ast_module: &Module) -> Result<(), String> {
        // 0. Declare runtime functions
        self.declare_runtime_functions();

        // 1. Declare structs (types) and methods
        self.compile_struct_defs(&ast_module.structs)?;
        self.compile_impl_blocks(&ast_module.impls)?;

        // Prepare functions list, potentially adding synthetic main
        let mut synthetic_main = None;
        let mut functions_refs = Vec::new();
        let mut main_exists = false;

        for func in &ast_module.functions {
            if func.name == "main" {
                main_exists = true;
            }
            functions_refs.push(func);
        }

        if !main_exists && !ast_module.tensor_decls.is_empty() {
            let syn_main = FunctionDef {
                name: "main".to_string(),
                args: vec![],
                return_type: Type::Void,
                body: vec![],
                generics: vec![],
            };
            synthetic_main = Some(syn_main);
            // We can't push reference to local variable easily here due to lifetimes.
            // But we can iterate separately or handle logic below.
        }

        // 2. Declare functions (prototypes)
        for func in &functions_refs {
            self.compile_fn_proto(func)?;
        }
        if let Some(func) = &synthetic_main {
            self.compile_fn_proto(func)?;
        }

        // 3. Compile function bodies
        for func in &functions_refs {
            let extra: &[Stmt] = if func.name == "main" {
                &ast_module.tensor_decls
            } else {
                &[]
            };
            self.compile_fn(func, extra)?;
        }
        if let Some(func) = &synthetic_main {
            self.compile_fn(func, &ast_module.tensor_decls)?;
        }

        self.module.print_to_file("dump.ll").unwrap();
        Ok(())
    }

    fn lookup_variable(&self, name: &str) -> Option<(BasicValueEnum<'ctx>, Type)> {
        for scope in self.variables.iter().rev() {
            if let Some((v, t, _)) = scope.get(name) {
                return Some((v.clone(), t.clone()));
            }
        }
        None
    }

    fn compile_fn_proto(&mut self, func: &FunctionDef) -> Result<FunctionValue<'ctx>, String> {
        self.fn_return_types
            .insert(func.name.clone(), func.return_type.clone());
        let ret_type: Option<inkwell::types::BasicTypeEnum> = match &func.return_type {
            Type::Void => None, // Void is not BasicValue
            Type::I64 => Some(self.context.i64_type().into()),
            Type::F32 => Some(self.context.f32_type().into()),
            Type::Bool => Some(self.context.bool_type().into()),
            Type::Tensor(_, _) | Type::UserDefined(_) | Type::Struct(_) => Some(
                self.context
                    .ptr_type(inkwell::AddressSpace::default())
                    .into(),
            ),
            _ => Some(self.context.i64_type().into()), // default
        };

        let mut args_types = Vec::new();
        for (_, val) in &func.args {
            let arg_ty: inkwell::types::BasicMetadataTypeEnum = match val {
                Type::I64 => self.context.i64_type().into(),
                Type::F32 => self.context.f32_type().into(),
                Type::Bool => self.context.bool_type().into(),
                Type::Tensor(_, _) | Type::UserDefined(_) | Type::Struct(_) => self
                    .context
                    .ptr_type(inkwell::AddressSpace::default())
                    .into(),
                _ => self.context.i64_type().into(),
            };
            args_types.push(arg_ty);
        }

        let fn_type = match ret_type {
            Some(inkwell::types::BasicTypeEnum::IntType(i)) => i.fn_type(&args_types, false),
            Some(inkwell::types::BasicTypeEnum::FloatType(f)) => f.fn_type(&args_types, false),
            Some(inkwell::types::BasicTypeEnum::PointerType(p)) => p.fn_type(&args_types, false),
            _ => self.context.void_type().fn_type(&args_types, false), // Void fallback
        };

        let val = self.module.add_function(&func.name, fn_type, None);

        // Add param names for debug
        for (i, arg) in val.get_param_iter().enumerate() {
            arg.set_name(&func.args[i].0);
        }

        Ok(val)
    }

    fn compile_fn(&mut self, func: &FunctionDef, extra_stmts: &[Stmt]) -> Result<(), String> {
        let function = self
            .module
            .get_function(&func.name)
            .ok_or("Function not found")?;
        // Initialize entry block
        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);

        // Clear variables for new function scope?
        // Actually functions have their own scope stack context usually.
        // But here we might be reusing the generator.
        // Ideally we should reset scopes relative to function, but we kept global context?
        // Let's assume a fresh function scope.
        // self.variables.clear(); // DANGEROUS if we have globals?
        // Better: Push a new scope for function arguments
        self.enter_scope(); // Function scope

        // Register arguments
        for (i, arg) in function.get_param_iter().enumerate() {
            let (arg_name, arg_type) = &func.args[i];
            let alloca = self.create_entry_block_alloca(function, arg_name, arg_type);
            self.builder.build_store(alloca, arg).unwrap();

            // Insert into current scope
            self.variables
                .last_mut()
                .unwrap()
                .insert(arg_name.clone(), (alloca.into(), arg_type.clone(), false));
        }

        // Compile extra statements (e.g. top-level tensor decls)
        for stmt in extra_stmts {
            self.compile_stmt(stmt)?;
        }

        // Compile body
        let body_len = func.body.len();
        for (i, stmt) in func.body.iter().enumerate() {
            if i == body_len - 1 && func.return_type != Type::Void {
                // Check if it's an expression that should be returned
                if let Stmt::Expr(expr) = stmt {
                    let (val, _) = self.compile_expr(expr)?;
                    self.emit_all_scopes_cleanup();
                    self.builder
                        .build_return(Some(&val))
                        .map_err(|e| e.to_string())?;
                    continue;
                }
            }
            self.compile_stmt(stmt)?;
        }

        self.exit_scope(); // End function scope

        // Add implicit return void if needed (not perfect but ok for now)
        if func.return_type == Type::Void {
            if self
                .builder
                .get_insert_block()
                .unwrap()
                .get_terminator()
                .is_none()
            {
                self.builder.build_return(None).map_err(|e| e.to_string())?;
            }
        }

        if !function.verify(true) {
            function.print_to_stderr();
            return Err(format!("Invalid generated function {}", func.name));
        }

        Ok(())
    }

    fn compile_stmt(&mut self, stmt: &Stmt) -> Result<(), String> {
        match stmt {
            Stmt::FieldAssign { obj, field, value } => {
                let (obj_val, obj_ty) = self.compile_expr(obj)?;
                let struct_name = match obj_ty {
                    Type::Struct(name) => name,
                    Type::UserDefined(name) => name,
                    _ => return Err(format!("Field assignment on non-struct type {:?}", obj_ty)),
                };

                let struct_def = self
                    .struct_defs
                    .get(&struct_name)
                    .ok_or(format!("Struct definition for {} not found", struct_name))?;

                let field_idx = struct_def
                    .fields
                    .iter()
                    .position(|(n, _)| n == field)
                    .ok_or(format!(
                        "Field {} not found in struct {}",
                        field, struct_name
                    ))?;

                if !obj_val.is_pointer_value() {
                    return Err("Cannot assign field of non-pointer struct".into());
                }
                let ptr = obj_val.into_pointer_value();
                let st_llvm_ty = self.struct_types.get(&struct_name).unwrap().clone();

                let field_ptr = self
                    .builder
                    .build_struct_gep(st_llvm_ty, ptr, field_idx as u32, &format!("ptr_{}", field))
                    .map_err(|e| e.to_string())?;

                let (val, _) = self.compile_expr(value)?;

                self.builder
                    .build_store(field_ptr, val)
                    .map_err(|e| e.to_string())?;
                Ok(())
            }
            Stmt::TensorDecl {
                name,
                type_annotation: _,
                init,
            } => {
                if let Some(expr) = init {
                    let (val_ir, val_ty) = self.compile_expr(expr)?;
                    let fn_val = self
                        .builder
                        .get_insert_block()
                        .unwrap()
                        .get_parent()
                        .unwrap();
                    let ptr = self.create_entry_block_alloca(fn_val, name, &val_ty);
                    self.builder
                        .build_store(ptr, val_ir)
                        .map_err(|e| e.to_string())?;

                    self.variables
                        .last_mut()
                        .unwrap()
                        .insert(name.clone(), (ptr.into(), val_ty, true));
                }
                Ok(())
            }
            Stmt::Let {
                name,
                indices,
                value,
                ..
            } => {
                if let Some(idxs) = indices {
                    // Tensor Equation
                    return self
                        .compile_tensor_equation(name, idxs, value)
                        .map_err(|e| e.to_string());
                }

                let val = self.compile_expr(value)?;
                // Verify stack allocation needed? For now simple register mapping
                // But for mutable variables we need alloca.
                // Let's use alloca for everything for simplicity.
                let current_function = self
                    .builder
                    .get_insert_block()
                    .unwrap()
                    .get_parent()
                    .unwrap();
                let alloca = self.create_entry_block_alloca(current_function, name, &val.1);
                self.builder
                    .build_store(alloca, val.0)
                    .map_err(|e| e.to_string())?;
                self.variables
                    .last_mut()
                    .unwrap()
                    .insert(name.clone(), (alloca.into(), val.1.clone(), true)); // Store pointer and type

                // Register tensor with runtime if it is a tensor
                if let Type::Tensor(_, _) = val.1 {
                    if let Some(register_fn) = self.module.get_function("tl_register_tensor") {
                        // Create global string for name
                        let name_global = self
                            .builder
                            .build_global_string_ptr(name, "tensor_name")
                            .map_err(|e| e.to_string())?;
                        // val.0 is pointer to tensor (OpaqueTensor*)
                        // register call: tl_register_tensor(name_ptr, tensor_ptr)
                        self.builder
                            .build_call(
                                register_fn,
                                &[name_global.as_pointer_value().into(), val.0.into()],
                                "",
                            )
                            .map_err(|e| e.to_string())?;
                    }
                }
                Ok(())
            }
            Stmt::Return(expr) => {
                // If returning a variable, mark it as moved (should_free = false)
                if let Expr::Variable(name) = expr {
                    for scope in self.variables.iter_mut().rev() {
                        if let Some(entry) = scope.get_mut(name) {
                            entry.2 = false;
                            break;
                        }
                    }
                }
                let val = self.compile_expr(expr)?;

                // Emit cleanup for ALL active scopes (reverse order)
                self.emit_all_scopes_cleanup();

                self.builder
                    .build_return(Some(&val.0))
                    .map_err(|e| e.to_string())?;
                Ok(())
            }
            Stmt::Assign {
                name,
                indices,
                op,
                value,
            } => {
                if let Some(_idxs) = indices {
                    // Determine if this is a supported tensor equation assignment
                    if *op == AssignOp::Assign {
                        // C[i, k] = ...
                        // This is treated as "redefining" or "filling" C.
                        // But compile_tensor_equation creates a NEW tensor currently (Stmt::Let logic).
                        // For Assign, we want to write into EXISTING tensor.
                        // TODO: support in-place update.
                        // For M3, maybe 'let' is enough for equations?
                        // User Example: let C[i, k] = ...
                        // If user does C[i, k] = ..., we need In-Place logic.
                        // Given current implementation of compile_tensor_equation allocates new buffer,
                        // let's error for now or defer?
                        // Or reuse the logic but write to existing pointer?
                        return Err(
                            "In-place indexed assignment not yet fully supported (use 'let')."
                                .into(),
                        );
                    } else {
                        return Err(
                            "Only direct assignment supported for tensor equations currently"
                                .into(),
                        );
                    }
                }

                // Compile value first
                let (val, val_type) = self.compile_expr(value)?;

                // Lookup variable
                let mut found_var_ptr = None;
                let mut found_var_type = None;
                for scope in self.variables.iter().rev() {
                    if let Some((v, t, _)) = scope.get(name) {
                        found_var_ptr = Some(v.clone());
                        found_var_type = Some(t.clone());
                        break;
                    }
                }

                let var_ptr = found_var_ptr.ok_or(format!("Variable {} not found", name))?;
                let var_type = found_var_type.ok_or(format!("Variable {} not found", name))?;

                if let Some(idxs) = indices {
                    if !idxs.is_empty() {
                        return Err("Indexed assignment not yet supported".into());
                    }
                }

                // Handle assignment operator (e.g., +=, -=, =)
                let final_val = match op {
                    AssignOp::Assign => {
                        // Free old value if it is a Tensor
                        if let Type::Tensor(_, _) = var_type {
                            let load_type = self.context.ptr_type(inkwell::AddressSpace::default());
                            let current_val = self
                                .builder
                                .build_load(
                                    load_type,
                                    var_ptr.into_pointer_value(),
                                    "old_val_to_free",
                                )
                                .map_err(|e| e.to_string())?
                                .into_pointer_value();

                            if let Some(free_fn) = self.module.get_function("tl_tensor_free") {
                                self.builder
                                    .build_call(free_fn, &[current_val.into()], "")
                                    .map_err(|e| e.to_string())?;
                            }
                        }
                        val
                    }
                    AssignOp::AddAssign => {
                        // Load current value
                        let load_type: inkwell::types::BasicTypeEnum = match var_type {
                            Type::I64 => self.context.i64_type().into(),
                            Type::F32 => self.context.f32_type().into(),
                            Type::Bool => self.context.bool_type().into(),
                            Type::Tensor(_, _) => self
                                .context
                                .ptr_type(inkwell::AddressSpace::default())
                                .into(),
                            _ => {
                                return Err(format!(
                                    "Unsupported type for assignment operation: {:?}",
                                    var_type
                                ))
                            }
                        };

                        let current_val = self
                            .builder
                            .build_load(
                                load_type,
                                var_ptr.into_pointer_value(),
                                &format!("{}_current", name),
                            )
                            .map_err(|e| e.to_string())?;

                        // For +=, we are computing New = Old + Val.
                        // The `compile_bin_op` creates a NEW tensor result.
                        // We must free the OLD `current_val` after we use it (or rely on `dl_tensor_add` to NOT consume it? Candle ops return new tensors).
                        // Current `tl_tensor_add` returns new tensor.
                        // So `current_val` (pointer to old tensor) is now orphaned unless we free it.
                        // BUT: `compile_bin_op` emits `tl_tensor_add(lhs, rhs)`.
                        // Does `tl_tensor_add` take ownership? No, specific implementation just reads.
                        // So we MUST free `current_val` here before overwriting `var_ptr`.

                        if let Type::Tensor(_, _) = var_type {
                            // For AddAssign on Tensor:
                            // We need to free the OLD tensor because `tl_tensor_add` returns a NEW tensor.
                            // The old tensor pointer `current_val` will be lost when we overwrite `var_ptr`.
                            // So we free it here.

                            if let Some(free_fn) = self.module.get_function("tl_tensor_free") {
                                self.builder
                                    .build_call(free_fn, &[current_val.into()], "")
                                    .map_err(|e| e.to_string())?;
                            }
                        }

                        let (op_res, _) = self.compile_bin_op(
                            current_val,
                            var_type.clone(),
                            val,
                            val_type,
                            BinOp::Add,
                        )?;
                        op_res
                    }
                    AssignOp::SubAssign => {
                        // SubAssign logic (In-Place for Tensor)
                        if let Type::Tensor(_, _) = var_type {
                            // Load current val
                            let load_type = self.context.ptr_type(inkwell::AddressSpace::default());
                            let current_val = self
                                .builder
                                .build_load(
                                    load_type,
                                    var_ptr.into_pointer_value(),
                                    &format!("{}_current", name),
                                )
                                .map_err(|e| e.to_string())?;

                            // Call sub_assign
                            let sub_assign_fn =
                                self.module.get_function("tl_tensor_sub_assign").unwrap();
                            self.builder
                                .build_call(sub_assign_fn, &[current_val.into(), val.into()], "")
                                .map_err(|e| e.to_string())?;

                            // Return early to avoid store (in-place)
                            return Ok(());
                        } else {
                            return Err("SubAssign -= only supported for Tensors currently via in-place optimization".into());
                        }
                    }
                    _ => return Err(format!("Unsupported assignment op: {:?}", op)),
                };

                self.builder
                    .build_store(var_ptr.into_pointer_value(), final_val)
                    .map_err(|e| e.to_string())?;
                Ok(())
            }
            Stmt::If {
                cond,
                then_block,
                else_block,
            } => {
                let parent = self
                    .builder
                    .get_insert_block()
                    .unwrap()
                    .get_parent()
                    .unwrap();

                let (cond_val, _) = self.compile_expr(cond)?;
                let cond_int = self
                    .builder
                    .build_int_cast(
                        cond_val.into_int_value(),
                        self.context.bool_type(),
                        "boolcast",
                    )
                    .unwrap();

                let then_bb = self.context.append_basic_block(parent, "then");
                let else_bb = self.context.append_basic_block(parent, "else");
                let merge_bb = self.context.append_basic_block(parent, "merge");

                self.builder
                    .build_conditional_branch(cond_int, then_bb, else_bb)
                    .unwrap();

                // Then
                self.builder.position_at_end(then_bb);
                self.enter_scope();
                for stmt in then_block {
                    self.compile_stmt(stmt)?;
                }
                // Branch to merge if not returned
                if then_bb.get_terminator().is_none() {
                    self.builder.build_unconditional_branch(merge_bb).unwrap();
                }
                self.exit_scope();

                // Else
                self.builder.position_at_end(else_bb);
                self.enter_scope();
                if let Some(else_stmts) = else_block {
                    for stmt in else_stmts {
                        self.compile_stmt(stmt)?;
                    }
                }
                if else_bb.get_terminator().is_none() {
                    self.builder.build_unconditional_branch(merge_bb).unwrap();
                }
                self.exit_scope();

                // Merge
                self.builder.position_at_end(merge_bb);
                Ok(())
            }
            Stmt::For {
                loop_var: _,
                iterator: _,
                body: _,
            } => {
                return Err("For loop not yet fully implemented".into());
            }
            Stmt::While { cond, body } => {
                let parent = self
                    .builder
                    .get_insert_block()
                    .unwrap()
                    .get_parent()
                    .unwrap();

                let cond_block = self.context.append_basic_block(parent, "while_cond");
                let body_block = self.context.append_basic_block(parent, "while_body");
                let end_block = self.context.append_basic_block(parent, "while_end");

                // Jump to condition from current
                self.builder
                    .build_unconditional_branch(cond_block)
                    .map_err(|e| e.to_string())?;

                // Compile condition
                self.builder.position_at_end(cond_block);
                let (cond_val, _) = self.compile_expr(cond)?;
                let cond_bool = self
                    .builder
                    .build_int_compare(
                        inkwell::IntPredicate::NE,
                        cond_val.into_int_value(),
                        self.context.bool_type().const_zero(),
                        "while_cond_check",
                    )
                    .map_err(|e| e.to_string())?;

                self.builder
                    .build_conditional_branch(cond_bool, body_block, end_block)
                    .map_err(|e| e.to_string())?;

                // Compile body
                self.builder.position_at_end(body_block);
                self.enter_scope();
                for stmt in body {
                    self.compile_stmt(stmt)?;
                }
                self.exit_scope();

                // Loop back to condition
                if self
                    .builder
                    .get_insert_block()
                    .unwrap()
                    .get_terminator()
                    .is_none()
                {
                    self.builder
                        .build_unconditional_branch(cond_block)
                        .map_err(|e| e.to_string())?;
                }

                // Continue at end
                self.builder.position_at_end(end_block);
                Ok(())
            }
            Stmt::Expr(expr) => {
                self.compile_expr(expr)?;
                Ok(())
            }
        }
    }

    // Helper for BinOp
    fn compile_bin_op(
        &self,
        lhs: BasicValueEnum<'ctx>,
        lhs_type: Type,
        rhs: BasicValueEnum<'ctx>,
        rhs_type: Type,
        op: BinOp,
    ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        match (&lhs_type, &rhs_type) {
            (Type::I64, Type::I64) => {
                let l = lhs.into_int_value();
                let r = rhs.into_int_value();
                let res = match op {
                    BinOp::Add => self.builder.build_int_add(l, r, "addtmp"),
                    BinOp::Sub => self.builder.build_int_sub(l, r, "subtmp"),
                    BinOp::Mul => self.builder.build_int_mul(l, r, "multmp"),
                    BinOp::Div => self.builder.build_int_signed_div(l, r, "divtmp"),
                    BinOp::Eq => {
                        self.builder
                            .build_int_compare(inkwell::IntPredicate::EQ, l, r, "eqtmp")
                    }
                    BinOp::Neq => {
                        self.builder
                            .build_int_compare(inkwell::IntPredicate::NE, l, r, "neqtmp")
                    }
                    BinOp::Lt => {
                        self.builder
                            .build_int_compare(inkwell::IntPredicate::SLT, l, r, "lttmp")
                    }
                    BinOp::Gt => {
                        self.builder
                            .build_int_compare(inkwell::IntPredicate::SGT, l, r, "gttmp")
                    }
                    BinOp::Le => {
                        self.builder
                            .build_int_compare(inkwell::IntPredicate::SLE, l, r, "letmp")
                    }
                    BinOp::Ge => {
                        self.builder
                            .build_int_compare(inkwell::IntPredicate::SGE, l, r, "getmp")
                    }
                    BinOp::And => self.builder.build_and(l, r, "andtmp"),
                    BinOp::Or => self.builder.build_or(l, r, "ortmp"),
                }
                .map_err(|e| e.to_string())?;

                if res.get_type().get_bit_width() == 1 {
                    Ok((res.into(), Type::Bool))
                } else {
                    Ok((res.into(), Type::I64))
                }
            }
            (Type::F32, Type::F32) => {
                let l = lhs.into_float_value();
                let r = rhs.into_float_value();
                let res: BasicValueEnum = match op {
                    BinOp::Add => self
                        .builder
                        .build_float_add(l, r, "faddtmp")
                        .map(|v| v.into()),
                    BinOp::Sub => self
                        .builder
                        .build_float_sub(l, r, "fsubtmp")
                        .map(|v| v.into()),
                    BinOp::Mul => self
                        .builder
                        .build_float_mul(l, r, "fmultmp")
                        .map(|v| v.into()),
                    BinOp::Div => self
                        .builder
                        .build_float_div(l, r, "fdivtmp")
                        .map(|v| v.into()),
                    BinOp::Eq => self
                        .builder
                        .build_float_compare(inkwell::FloatPredicate::OEQ, l, r, "feqtmp")
                        .map(|v| v.into()),
                    BinOp::Neq => self
                        .builder
                        .build_float_compare(inkwell::FloatPredicate::ONE, l, r, "fneqtmp")
                        .map(|v| v.into()),
                    BinOp::Lt => self
                        .builder
                        .build_float_compare(inkwell::FloatPredicate::OLT, l, r, "flttmp")
                        .map(|v| v.into()),
                    BinOp::Gt => self
                        .builder
                        .build_float_compare(inkwell::FloatPredicate::OGT, l, r, "fgttmp")
                        .map(|v| v.into()),
                    BinOp::Le => self
                        .builder
                        .build_float_compare(inkwell::FloatPredicate::OLE, l, r, "fletmp")
                        .map(|v| v.into()),
                    BinOp::Ge => self
                        .builder
                        .build_float_compare(inkwell::FloatPredicate::OGE, l, r, "fgetmp")
                        .map(|v| v.into()),
                    _ => return Err("Unsupported float op".into()),
                }
                .map_err(|e| e.to_string())?;

                if res.is_int_value() {
                    Ok((res, Type::Bool))
                } else {
                    Ok((res, Type::F32))
                }
            }
            (Type::Bool, Type::Bool) => {
                let l = lhs.into_int_value();
                let r = rhs.into_int_value();
                let res = match op {
                    BinOp::And => self.builder.build_and(l, r, "andtmp"),
                    BinOp::Or => self.builder.build_or(l, r, "ortmp"),
                    BinOp::Eq => {
                        self.builder
                            .build_int_compare(inkwell::IntPredicate::EQ, l, r, "eqtmp")
                    }
                    BinOp::Neq => {
                        self.builder
                            .build_int_compare(inkwell::IntPredicate::NE, l, r, "neqtmp")
                    }
                    _ => return Err("Unsupported bool op".into()),
                }
                .map_err(|e| e.to_string())?;
                Ok((res.into(), Type::Bool))
            }
            (Type::Tensor(_, _), Type::Tensor(_, _)) => {
                let l = lhs.into_pointer_value();
                let r = rhs.into_pointer_value();

                let fn_name = match op {
                    BinOp::Add => "tl_tensor_add",
                    BinOp::Mul => "tl_tensor_mul",
                    BinOp::Div => "tl_tensor_div",
                    BinOp::Sub => "tl_tensor_sub",
                    _ => return Err("Unsupported tensor op".into()),
                };

                let fn_val = self
                    .module
                    .get_function(fn_name)
                    .ok_or(format!("Runtime function {} not found", fn_name))?;
                let call = self
                    .builder
                    .build_call(fn_val, &[l.into(), r.into()], "binop_res")
                    .map_err(|e| e.to_string())?;

                let res_ptr = match call.try_as_basic_value() {
                    ValueKind::Basic(v) => v.into_pointer_value(),
                    _ => return Err("Invalid return from runtime binop".into()),
                };
                Ok((res_ptr.into(), lhs_type.clone()))
            }
            // Handling mixed types (F32 vs I64) for convenience
            (Type::F32, Type::I64) => {
                let l = lhs.into_float_value();
                let r = rhs.into_int_value();
                let r_f32 = self
                    .builder
                    .build_signed_int_to_float(r, self.context.f32_type(), "cast_r_f32")
                    .map_err(|e| e.to_string())?;

                // Recurse with F32, F32
                self.compile_bin_op(l.into(), Type::F32, r_f32.into(), Type::F32, op)
            }
            (Type::I64, Type::F32) => {
                let l = lhs.into_int_value();
                let r = rhs.into_float_value();
                let l_f32 = self
                    .builder
                    .build_signed_int_to_float(l, self.context.f32_type(), "cast_l_f32")
                    .map_err(|e| e.to_string())?;

                // Recurse with F32, F32
                self.compile_bin_op(l_f32.into(), Type::F32, r.into(), Type::F32, op)
            }
            (Type::Tensor(inner, _), Type::F32) if **inner == Type::F32 => {
                // Broadcasting Tensor op Scalar
                // Create scalar tensor
                let val = rhs.into_float_value();
                let f32_type = self.context.f32_type();
                let i64_type = self.context.i64_type();

                // 1. Data Alloca (1 elem)
                let data_alloca = self
                    .builder
                    .build_alloca(f32_type, "scalar_data")
                    .map_err(|e| e.to_string())?;
                self.builder
                    .build_store(data_alloca, val)
                    .map_err(|e| e.to_string())?;

                // 2. Shape Alloca (0 elem)
                let shape_alloca = self
                    .builder
                    .build_array_alloca(i64_type, i64_type.const_int(0, false), "scalar_shape")
                    .map_err(|e| e.to_string())?;

                // 3. New Tensor
                let new_fn = self.module.get_function("tl_tensor_new").unwrap();
                let rank_val = i64_type.const_int(0, false); // Rank 0
                let call = self
                    .builder
                    .build_call(
                        new_fn,
                        &[data_alloca.into(), rank_val.into(), shape_alloca.into()],
                        "scalar_tensor",
                    )
                    .map_err(|e| e.to_string())?;
                let scalar_tensor = match call.try_as_basic_value() {
                    ValueKind::Basic(v) => v.into_pointer_value(),
                    _ => return Err("Invalid tensor new return".into()),
                };

                // 4. Call Op
                let fn_name = match op {
                    BinOp::Add => "tl_tensor_add",
                    BinOp::Mul => "tl_tensor_mul",
                    _ => return Err("Unsupported tensor op".into()),
                };
                let fn_val = self
                    .module
                    .get_function(fn_name)
                    .ok_or("Runtime fn not found")?;

                let call = self
                    .builder
                    .build_call(
                        fn_val,
                        &[lhs.into_pointer_value().into(), scalar_tensor.into()],
                        "binop_res",
                    )
                    .map_err(|e| e.to_string())?;

                let res_ptr = match call.try_as_basic_value() {
                    ValueKind::Basic(v) => v.into_pointer_value(),
                    _ => return Err("Invalid return from runtime binop".into()),
                };
                Ok((res_ptr.into(), lhs_type.clone()))
            }
            (Type::F32, Type::Tensor(inner, _)) if **inner == Type::F32 => {
                // Scalar op Tensor (Broadcasting)
                // Create scalar tensor
                let val = lhs.into_float_value();
                let f32_type = self.context.f32_type();
                let i64_type = self.context.i64_type();

                let data_alloca = self
                    .builder
                    .build_alloca(f32_type, "scalar_data")
                    .map_err(|e| e.to_string())?;
                self.builder
                    .build_store(data_alloca, val)
                    .map_err(|e| e.to_string())?;

                let shape_alloca = self
                    .builder
                    .build_array_alloca(i64_type, i64_type.const_int(0, false), "scalar_shape")
                    .map_err(|e| e.to_string())?;

                let new_fn = self.module.get_function("tl_tensor_new").unwrap();
                let rank_val = i64_type.const_int(0, false);
                let call = self
                    .builder
                    .build_call(
                        new_fn,
                        &[data_alloca.into(), rank_val.into(), shape_alloca.into()],
                        "scalar_tensor",
                    )
                    .map_err(|e| e.to_string())?;
                let scalar_tensor = match call.try_as_basic_value() {
                    ValueKind::Basic(v) => v.into_pointer_value(),
                    _ => return Err("Invalid tensor new return".into()),
                };

                let fn_name = match op {
                    BinOp::Add => "tl_tensor_add",
                    BinOp::Mul => "tl_tensor_mul",
                    _ => return Err("Unsupported tensor op".into()),
                };
                let fn_val = self
                    .module
                    .get_function(fn_name)
                    .ok_or("Runtime fn not found")?;

                let call = self
                    .builder
                    .build_call(
                        fn_val,
                        &[scalar_tensor.into(), rhs.into_pointer_value().into()],
                        "binop_res",
                    )
                    .map_err(|e| e.to_string())?;

                let res_ptr = match call.try_as_basic_value() {
                    ValueKind::Basic(v) => v.into_pointer_value(),
                    _ => return Err("Invalid return from runtime binop".into()),
                };
                Ok((res_ptr.into(), rhs_type.clone()))
            }
            _ => Err(format!(
                "Type mismatch in BinOp {:?}: {:?} vs {:?}",
                op, lhs_type, rhs_type
            )),
        }
    }

    fn compile_expr(&mut self, expr: &Expr) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        match expr {
            Expr::Block(stmts) => {
                self.enter_scope();
                let mut last_val = None;
                for (i, stmt) in stmts.iter().enumerate() {
                    if i == stmts.len() - 1 {
                        if let Stmt::Expr(e) = stmt {
                            last_val = Some(self.compile_expr(e)?);
                        } else {
                            self.compile_stmt(stmt)?;
                        }
                    } else {
                        self.compile_stmt(stmt)?;
                    }
                }
                self.exit_scope();

                Ok(last_val.unwrap_or((
                    self.context.i64_type().const_int(0, false).into(),
                    Type::Void,
                )))
            }
            Expr::Int(i) => {
                let i64_type = self.context.i64_type();
                Ok((i64_type.const_int(*i as u64, true).into(), Type::I64))
            }
            Expr::Float(f) => {
                let f32_type = self.context.f32_type();
                Ok((f32_type.const_float(*f).into(), Type::F32))
            }
            Expr::Bool(b) => {
                let bool_type = self.context.bool_type();
                Ok((
                    bool_type.const_int(if *b { 1 } else { 0 }, false).into(),
                    Type::Bool,
                ))
            }
            Expr::StringLiteral(_) => Err(
                "String literals not yet supported in codegen (need runtime string type)".into(),
            ),
            Expr::FieldAccess(obj, field) => {
                let (obj_val, obj_ty) = self.compile_expr(obj)?;
                let struct_name = match obj_ty {
                    Type::Struct(name) => name,
                    Type::UserDefined(name) => name,
                    _ => return Err(format!("Field access on non-struct type {:?}", obj_ty)),
                };

                let struct_def = self
                    .struct_defs
                    .get(&struct_name)
                    .ok_or(format!("Struct definition for {} not found", struct_name))?;

                let field_idx = struct_def
                    .fields
                    .iter()
                    .position(|(n, _)| n == field)
                    .ok_or(format!(
                        "Field {} not found in struct {}",
                        field, struct_name
                    ))?;
                let (_, field_ty) = &struct_def.fields[field_idx];

                if !obj_val.is_pointer_value() {
                    return Err("Cannot access field of non-pointer struct".into());
                }
                let ptr = obj_val.into_pointer_value();
                let st_llvm_ty = self.struct_types.get(&struct_name).unwrap();

                let field_ptr = self
                    .builder
                    .build_struct_gep(
                        st_llvm_ty.clone(),
                        ptr,
                        field_idx as u32,
                        &format!("ptr_{}", field),
                    )
                    .map_err(|e| e.to_string())?;

                let llvm_ty: inkwell::types::BasicTypeEnum = match field_ty {
                    Type::I64 => self.context.i64_type().into(),
                    Type::F32 => self.context.f32_type().into(),
                    Type::Bool => self.context.bool_type().into(),
                    Type::Tensor(_, _) | Type::Struct(_) => self
                        .context
                        .ptr_type(inkwell::AddressSpace::default())
                        .into(),
                    _ => self.context.i64_type().into(), // Placeholder
                };

                let loaded = self
                    .builder
                    .build_load(llvm_ty, field_ptr, field)
                    .map_err(|e| e.to_string())?;
                Ok((loaded, field_ty.clone()))
            }

            Expr::Variable(name) => {
                for scope in self.variables.iter().rev() {
                    if let Some((val, ty, _)) = scope.get(name) {
                        if val.is_pointer_value() {
                            let ptr = val.into_pointer_value();
                            let llvm_ty: inkwell::types::BasicTypeEnum = match ty {
                                Type::I64 => self.context.i64_type().into(),
                                Type::F32 => self.context.f32_type().into(),
                                Type::Bool => self.context.bool_type().into(),
                                Type::Tensor(_, _) | Type::Struct(_) | Type::UserDefined(_) => self
                                    .context
                                    .ptr_type(inkwell::AddressSpace::default())
                                    .into(),
                                _ => self.context.i64_type().into(), // Fallback
                            };
                            let loaded = self
                                .builder
                                .build_load(llvm_ty, ptr, name)
                                .map_err(|e| e.to_string())?;
                            return Ok((loaded, ty.clone()));
                        } else {
                            return Ok((val.clone(), ty.clone()));
                        }
                    }
                }
                Err(format!("Variable {} not found in scopes", name))
            }
            Expr::BinOp(lhs, op, rhs) => {
                let left = self.compile_expr(lhs)?;
                let right = self.compile_expr(rhs)?;
                self.compile_bin_op(left.0, left.1, right.0, right.1, op.clone())
            }
            Expr::TensorLiteral(elements) => {
                // Helper to flatten nested tensor literals
                fn flatten_tensor(exprs: &[Expr]) -> Result<(Vec<f64>, Vec<usize>), String> {
                    if exprs.is_empty() {
                        return Ok((vec![], vec![0]));
                    }

                    // Check if elements are nested tensors or scalars
                    let is_nested = matches!(exprs[0], Expr::TensorLiteral(_));

                    if is_nested {
                        let mut flat_data = Vec::new();
                        let mut child_shapes = Vec::new();
                        let mut first_shape = None;

                        for e in exprs {
                            if let Expr::TensorLiteral(children) = e {
                                let (mut data, shape) = flatten_tensor(children)?;

                                if let Some(ref s) = first_shape {
                                    if s != &shape {
                                        return Err("Ragged tensors not supported".into());
                                    }
                                } else {
                                    first_shape = Some(shape.clone());
                                }

                                flat_data.append(&mut data);
                                child_shapes.push(shape);
                            } else {
                                return Err("Mixed types in tensor literal".into());
                            }
                        }

                        let mut shape = vec![exprs.len()];
                        if let Some(s) = first_shape {
                            shape.extend(s);
                        }
                        Ok((flat_data, shape))
                    } else {
                        // Leaf level (Scalars)
                        let mut data = Vec::new(); // Use f64 for simplicity, convert later
                        for e in exprs {
                            match e {
                                Expr::Float(f) => data.push(*f),
                                Expr::Int(i) => data.push(*i as f64),
                                _ => return Err("Tensor elements must be numbers".into()),
                            }
                        }
                        Ok((data, vec![exprs.len()]))
                    }
                }

                let (flat_data, shape) = flatten_tensor(elements)?;
                let rank = shape.len();
                let len = flat_data.len() as u64;

                let f32_type = self.context.f32_type();
                let i64_type = self.context.i64_type();

                // 1. Alloca for data
                let data_alloca = self
                    .builder
                    .build_array_alloca(f32_type, i64_type.const_int(len, false), "tensor_data")
                    .unwrap();

                // 2. Store elements
                for (i, val) in flat_data.iter().enumerate() {
                    let float_val = f32_type.const_float(*val);
                    unsafe {
                        let ptr = self
                            .builder
                            .build_in_bounds_gep(
                                f32_type,
                                data_alloca,
                                &[i64_type.const_int(i as u64, false)],
                                "elem_ptr",
                            )
                            .unwrap();
                        self.builder.build_store(ptr, float_val).unwrap();
                    }
                }

                // 3. Alloca for shape
                let shape_alloca = self
                    .builder
                    .build_array_alloca(
                        i64_type,
                        i64_type.const_int(rank as u64, false),
                        "tensor_shape",
                    )
                    .unwrap();
                for (i, dim) in shape.iter().enumerate() {
                    unsafe {
                        let ptr = self
                            .builder
                            .build_in_bounds_gep(
                                i64_type,
                                shape_alloca,
                                &[i64_type.const_int(i as u64, false)],
                                "shape_val",
                            )
                            .unwrap();
                        self.builder
                            .build_store(ptr, i64_type.const_int(*dim as u64, false))
                            .unwrap();
                    }
                }

                // 4. Call Runtime
                let fn_name = "tl_tensor_new";
                let f = self
                    .module
                    .get_function(fn_name)
                    .ok_or("Runtime fn not found")?;
                let call = self
                    .builder
                    .build_call(
                        f,
                        &[
                            data_alloca.into(),
                            i64_type.const_int(rank as u64, false).into(),
                            shape_alloca.into(),
                        ],
                        "new_tensor",
                    )
                    .map_err(|e| e.to_string())?;

                let res = match call.try_as_basic_value() {
                    ValueKind::Basic(v) => v,
                    _ => return Err("Invalid call return".into()),
                };

                // Type is Tensor<f32, Rank>
                Ok((res, Type::Tensor(Box::new(Type::F32), rank)))
            }
            Expr::MethodCall(obj, method, args) => {
                let (obj_val, obj_ty) = self.compile_expr(obj)?;

                let maybe_struct_name = match &obj_ty {
                    Type::Struct(name) => Some(name.clone()),
                    Type::UserDefined(name) => Some(name.clone()),
                    _ => None,
                };

                if let Some(struct_name) = maybe_struct_name {
                    let mangled_name = format!("{}_{}", struct_name, method);
                    let func_val = self.module.get_function(&mangled_name).ok_or(format!(
                        "Method {} not found in struct {}",
                        mangled_name, struct_name
                    ))?;

                    let mut compiled_args = Vec::with_capacity(args.len() + 1);
                    compiled_args.push(obj_val.into()); // self

                    for arg in args {
                        let (val, _) = self.compile_expr(arg)?;
                        compiled_args.push(val.into());
                    }

                    let call = self
                        .builder
                        .build_call(func_val, &compiled_args, "call_method")
                        .map_err(|e| e.to_string())?;

                    let ret_ty = self
                        .fn_return_types
                        .get(&mangled_name)
                        .unwrap_or(&Type::Void)
                        .clone();
                    if let Type::Void = ret_ty {
                        Ok((
                            self.context.i64_type().const_int(0, false).into(),
                            Type::Void,
                        ))
                    } else {
                        match call.try_as_basic_value() {
                            ValueKind::Basic(v) => Ok((v, ret_ty)),
                            _ => Err("Invalid return value".into()),
                        }
                    }
                } else {
                    match method.as_str() {
                        "backward" => {
                            let fn_val = self.module.get_function("tl_tensor_backward").unwrap();
                            self.builder
                                .build_call(fn_val, &[obj_val.into()], "backward_call")
                                .map_err(|e| e.to_string())?;
                            Ok((
                                self.context.i64_type().const_int(0, false).into(),
                                Type::Void,
                            ))
                        }
                        "clone" => {
                            let fn_val = self.module.get_function("tl_tensor_clone").unwrap();
                            let call = self
                                .builder
                                .build_call(fn_val, &[obj_val.into()], "clone_res")
                                .map_err(|e| e.to_string())?;
                            let res = match call.try_as_basic_value() {
                                ValueKind::Basic(v) => v,
                                _ => return Err("Invalid clone return".into()),
                            };
                            Ok((res, obj_ty))
                        }
                        "detach" => {
                            let fn_val = self
                                .module
                                .get_function("tl_tensor_detach")
                                .ok_or("Runtime fn tl_tensor_detach not found")?;

                            let mut compiled_args = Vec::with_capacity(args.len() + 1);
                            compiled_args.push(obj_val.into());
                            for arg in args {
                                let (val, _) = self.compile_expr(arg)?;
                                compiled_args.push(val.into());
                            }

                            let call = self
                                .builder
                                .build_call(fn_val, &compiled_args, "detach_res")
                                .map_err(|e| e.to_string())?;

                            let res = match call.try_as_basic_value() {
                                ValueKind::Basic(v) => v,
                                _ => return Err("Invalid detach return".into()),
                            };
                            Ok((res, obj_ty))
                        }
                        "grad" => {
                            let fn_val = self.module.get_function("tl_tensor_grad").unwrap();
                            let call = self
                                .builder
                                .build_call(fn_val, &[obj_val.into()], "grad_res")
                                .map_err(|e| e.to_string())?;
                            let res = match call.try_as_basic_value() {
                                ValueKind::Basic(v) => v,
                                _ => return Err("Invalid grad return".into()),
                            };
                            // Assuming grad has same rank as obj, but for now just opaque tensor
                            Ok((res, obj_ty))
                        }
                        "reshape" => {
                            if args.len() != 1 {
                                return Err("reshape method requires 1 argument (shape)".into());
                            }
                            let (s_val, _) = self.compile_expr(&args[0])?;
                            let reshape_fn = self.module.get_function("tl_tensor_reshape").unwrap();
                            let call = self
                                .builder
                                .build_call(
                                    reshape_fn,
                                    &[obj_val.into(), s_val.into()],
                                    "reshape_res",
                                )
                                .map_err(|e| e.to_string())?;
                            let res = match call.try_as_basic_value() {
                                ValueKind::Basic(v) => v,
                                _ => return Err("Invalid reshape return".into()),
                            };
                            Ok((res, obj_ty))
                        }
                        "sum" => {
                            let fn_val = self.module.get_function("tl_tensor_sum").unwrap();
                            let call = self
                                .builder
                                .build_call(fn_val, &[obj_val.into()], "sum_res")
                                .map_err(|e| e.to_string())?;
                            let res = match call.try_as_basic_value() {
                                ValueKind::Basic(v) => v,
                                _ => return Err("Invalid sum return".into()),
                            };
                            // sum returns scalar tensor (rank 0 or 1 depending on impl).
                            // Assuming it returns Tensor<f32, 0> or 1.
                            Ok((res, obj_ty)) // Currently preserving type/rank info is hard, returning same opaque type
                        }
                        _ => Err(format!("Unknown method: {}", method)),
                    }
                }
            }
            Expr::FnCall(name, args) => {
                if let Some(struct_def) = self.struct_defs.get(name).cloned() {
                    let st_llvm_ty = self.struct_types.get(name).unwrap().clone();
                    let size = st_llvm_ty.size_of().unwrap();

                    let malloc_fn = self
                        .module
                        .get_function("malloc")
                        .expect("malloc not found");
                    let call = self
                        .builder
                        .build_call(malloc_fn, &[size.into()], "struct_malloc")
                        .map_err(|e| e.to_string())?;
                    let raw_ptr = match call.try_as_basic_value() {
                        ValueKind::Basic(v) => v.into_pointer_value(),
                        _ => return Err("malloc returned instruction value".into()),
                    };

                    // Cast to Struct* (Opaque)
                    let struct_ptr = self
                        .builder
                        .build_pointer_cast(
                            raw_ptr,
                            self.context.ptr_type(inkwell::AddressSpace::default()),
                            "struct_ptr",
                        )
                        .map_err(|e| e.to_string())?;

                    // Assign fields
                    if args.len() != struct_def.fields.len() {
                        return Err(format!(
                            "Struct constructor {} expects {} args, got {}",
                            name,
                            struct_def.fields.len(),
                            args.len()
                        ));
                    }

                    for (i, arg_expr) in args.iter().enumerate() {
                        let (val, _) = self.compile_expr(arg_expr)?;
                        let field_ptr = self
                            .builder
                            .build_struct_gep(
                                st_llvm_ty.clone(),
                                struct_ptr,
                                i as u32,
                                "init_field",
                            )
                            .map_err(|e| e.to_string())?;
                        self.builder
                            .build_store(field_ptr, val)
                            .map_err(|e| e.to_string())?;
                    }

                    return Ok((struct_ptr.into(), Type::Struct(name.clone())));
                }

                match name.as_str() {
                    "print" => {
                        if args.len() != 1 {
                            return Err("print requires 1 argument".into());
                        }
                        // Check type of arg
                        let arg_expr = &args[0];
                        let (arg_val, arg_type) = self.compile_expr(arg_expr)?;

                        match arg_type {
                            Type::I64 => {
                                let fn_val = self.module.get_function("tl_print_i64").unwrap();
                                self.builder
                                    .build_call(fn_val, &[arg_val.into()], "print_call")
                                    .map_err(|e| e.to_string())?;
                            }
                            Type::F32 => {
                                let fn_val = self.module.get_function("tl_print_f32").unwrap();
                                self.builder
                                    .build_call(fn_val, &[arg_val.into()], "print_call")
                                    .map_err(|e| e.to_string())?;
                            }
                            Type::Tensor(_, _) => {
                                let fn_val = self.module.get_function("tl_tensor_print").unwrap();
                                self.builder
                                    .build_call(fn_val, &[arg_val.into()], "print_call")
                                    .map_err(|e| e.to_string())?;
                            }
                            _ => return Err(format!("Cannot print type {:?}", arg_type)),
                        }
                        return Ok((
                            self.context.i64_type().const_int(0, false).into(),
                            Type::Void,
                        ));
                    }
                    "transpose" => {
                        // transpose(tensor, d0, d1)
                        if args.len() != 3 {
                            return Err("transpose requires 3 arguments: tensor, dim0, dim1".into());
                        }
                        let (t_val, t_ty) = self.compile_expr(&args[0])?;
                        let (d0_val, _d0_ty) = self.compile_expr(&args[1])?;
                        let (d1_val, _d1_ty) = self.compile_expr(&args[2])?;

                        if !matches!(t_ty, Type::Tensor(_, _)) {
                            return Err("First argument to transpose must be a tensor".into());
                        }

                        let transpose_fn = self
                            .module
                            .get_function("tl_tensor_transpose")
                            .ok_or("tl_tensor_transpose not found")?;

                        let call = self
                            .builder
                            .build_call(
                                transpose_fn,
                                &[t_val.into(), d0_val.into(), d1_val.into()],
                                "transpose_res",
                            )
                            .map_err(|e| e.to_string())?;

                        let res = match call.try_as_basic_value() {
                            ValueKind::Basic(v) => v,
                            _ => return Err("Invalid transpose return".into()),
                        };
                        Ok((res, t_ty)) // Returns same type (Tensor)
                    }
                    "reshape" => {
                        // reshape(tensor, shape_tensor)
                        if args.len() != 2 {
                            return Err("reshape requires 2 arguments: tensor, new_shape".into());
                        }
                        let (t_val, t_ty) = self.compile_expr(&args[0])?;
                        let (s_val, _s_ty) = self.compile_expr(&args[1])?;

                        if !matches!(t_ty, Type::Tensor(_, _)) {
                            return Err("First argument to reshape must be a tensor".into());
                        }

                        let reshape_fn = self
                            .module
                            .get_function("tl_tensor_reshape")
                            .ok_or("tl_tensor_reshape not found")?;

                        let call = self
                            .builder
                            .build_call(reshape_fn, &[t_val.into(), s_val.into()], "reshape_res")
                            .map_err(|e| e.to_string())?;

                        let res = match call.try_as_basic_value() {
                            ValueKind::Basic(v) => v,
                            _ => return Err("Invalid reshape return".into()),
                        };
                        Ok((res, t_ty)) // Returns same type (Tensor)
                    }
                    "exp" => {
                        if args.len() != 1 {
                            return Err("exp requires 1 argument".into());
                        }
                        let (arg_val, arg_ty) = self.compile_expr(&args[0])?;
                        if !matches!(arg_ty, Type::Tensor(_, _)) {
                            return Err("exp requires a tensor".into());
                        }
                        let fn_val = self.module.get_function("tl_tensor_exp").unwrap();
                        let call = self
                            .builder
                            .build_call(fn_val, &[arg_val.into()], "exp_res")
                            .map_err(|e| e.to_string())?;
                        let res = match call.try_as_basic_value() {
                            ValueKind::Basic(v) => v,
                            _ => return Err("Invalid exp return".into()),
                        };
                        return Ok((res, arg_ty));
                    }
                    "log" => {
                        if args.len() != 1 {
                            return Err("log requires 1 argument".into());
                        }
                        let (arg_val, arg_ty) = self.compile_expr(&args[0])?;
                        let fn_val = self.module.get_function("tl_tensor_log").unwrap();
                        let call = self
                            .builder
                            .build_call(fn_val, &[arg_val.into()], "log_res")
                            .unwrap();
                        let res = match call.try_as_basic_value() {
                            ValueKind::Basic(v) => v,
                            _ => return Err("Invalid log return".into()),
                        };
                        return Ok((res, arg_ty));
                    }
                    "len" => {
                        if args.len() != 1 {
                            return Err("len requires 1 argument".into());
                        }
                        let (arg_val, arg_ty) = self.compile_expr(&args[0])?;
                        if !matches!(arg_ty, Type::Tensor(_, _)) {
                            return Err("len requires a tensor".into());
                        }
                        let fn_val = self
                            .module
                            .get_function("tl_tensor_len")
                            .ok_or("tl_tensor_len not found")?;
                        let call = self
                            .builder
                            .build_call(fn_val, &[arg_val.into()], "len_res")
                            .map_err(|e| e.to_string())?;
                        let res = match call.try_as_basic_value() {
                            ValueKind::Basic(v) => v,
                            _ => return Err("Invalid len return".into()),
                        };
                        return Ok((res, Type::I64));
                    }

                    "sqrt" => {
                        if args.len() != 1 {
                            return Err("sqrt requires 1 argument".into());
                        }
                        let (arg_val, arg_ty) = self.compile_expr(&args[0])?;
                        if !matches!(arg_ty, Type::Tensor(_, _)) {
                            return Err("sqrt requires a tensor".into());
                        }
                        let fn_val = self.module.get_function("tl_tensor_sqrt").unwrap();
                        let call = self
                            .builder
                            .build_call(fn_val, &[arg_val.into()], "sqrt_res")
                            .map_err(|e| e.to_string())?;
                        let res = match call.try_as_basic_value() {
                            ValueKind::Basic(v) => v,
                            _ => return Err("Invalid sqrt return".into()),
                        };
                        return Ok((res, arg_ty));
                    }
                    "matmul" => {
                        if args.len() != 2 {
                            return Err("matmul requires 2 arguments".into());
                        }
                        let (lhs_val, lhs_ty) = self.compile_expr(&args[0])?;
                        let (rhs_val, rhs_ty) = self.compile_expr(&args[1])?;
                        if !matches!(lhs_ty, Type::Tensor(_, _))
                            || !matches!(rhs_ty, Type::Tensor(_, _))
                        {
                            return Err("matmul requires tensors".into());
                        }
                        let fn_val = self.module.get_function("tl_tensor_matmul").unwrap();
                        let call = self
                            .builder
                            .build_call(fn_val, &[lhs_val.into(), rhs_val.into()], "matmul_res")
                            .map_err(|e| e.to_string())?;
                        let res = match call.try_as_basic_value() {
                            ValueKind::Basic(v) => v,
                            _ => return Err("Invalid matmul return".into()),
                        };
                        // Only supporting basic tensor type propagation for now
                        return Ok((res, lhs_ty));
                    }
                    "grad" => {
                        if args.len() != 1 {
                            return Err("grad requires 1 argument".into());
                        }
                        let (arg_val, arg_ty) = self.compile_expr(&args[0])?;
                        let fn_val = self.module.get_function("tl_tensor_grad").unwrap();
                        let call = self
                            .builder
                            .build_call(fn_val, &[arg_val.into()], "grad_res")
                            .map_err(|e| e.to_string())?;
                        let res = match call.try_as_basic_value() {
                            ValueKind::Basic(v) => v,
                            _ => return Err("Invalid grad return".into()),
                        };
                        return Ok((res, arg_ty));
                    }
                    "backward" => {
                        if args.len() != 1 {
                            return Err("backward requires 1 argument".into());
                        }
                        let (arg_val, _arg_ty) = self.compile_expr(&args[0])?;
                        let fn_val = self.module.get_function("tl_tensor_backward").unwrap();
                        self.builder
                            .build_call(fn_val, &[arg_val.into()], "")
                            .map_err(|e| e.to_string())?;
                        return Ok((
                            self.context.i64_type().const_int(0, false).into(),
                            Type::Void,
                        ));
                    }
                    "sum" => {
                        if args.len() != 1 {
                            return Err("sum requires 1 argument".into());
                        }
                        let (arg_val, _arg_ty) = self.compile_expr(&args[0])?;
                        let fn_val = self.module.get_function("tl_tensor_sum").unwrap();
                        let call = self
                            .builder
                            .build_call(fn_val, &[arg_val.into()], "sum_res")
                            .map_err(|e| e.to_string())?;

                        let res = match call.try_as_basic_value() {
                            ValueKind::Basic(v) => v,
                            _ => return Err("Invalid sum return".into()),
                        };
                        // Return type is Tensor (scalar)
                        return Ok((res, Type::Tensor(Box::new(Type::F32), 1)));
                    }
                    _ => {
                        // Generic function call logic
                        let llvm_func_name = match name.as_str() {
                            "slice" => "tl_tensor_slice",
                            "sum" => "tl_tensor_sum",
                            "randn" => {
                                // randn(shape, requires_grad)
                                // Handle specially to pass rank/shape pointer and bool
                                // args[0] must be tensor (shape) or literal?
                                // Actually, user might pass [10, 10].
                                // But `compile_expr` for TensorLiteral returns a Tensor pointer.
                                // We need shape as generic array?
                                // Existing `tl_tensor_new` logic handled parsing TensorLiteral manually to create C-array.
                                // But here `args[0]` is an Expr.
                                // If it's a TensorLiteral, we can do similar logic.
                                // If it's a variable, it is a Tensor*.
                                // `tl_tensor_randn` needs `rank, shape_ptr`.
                                // Let's support only Literal Shape for now for simplicity, OR
                                // use a version of randn that takes a shape TENSOR.
                                // To match `tl_tensor_new`, we need raw shape.
                                // Let's assume usage: let x = randn([10, 20], true);
                                // The parser gives `TensorLiteral`.
                                if args.is_empty() {
                                    return Err("randn requires shape".into());
                                }

                                let shape_expr = &args[0];
                                let (rank, shape_vals) = if let Expr::TensorLiteral(el) = shape_expr
                                {
                                    let mut vals = Vec::new();
                                    for e in el {
                                        // Compile each dimension expression
                                        let (v, t) = self.compile_expr(e)?;
                                        let int_val = match t {
                                            Type::I64 => v.into_int_value(),
                                            Type::I32 => self
                                                .builder
                                                .build_int_z_extend(
                                                    v.into_int_value(),
                                                    self.context.i64_type(),
                                                    "dim_ext",
                                                )
                                                .map_err(|e| e.to_string())?,
                                            _ => {
                                                return Err(format!(
                                                    "Dimension must be integer, found {:?}",
                                                    t
                                                ))
                                            }
                                        };
                                        vals.push(int_val);
                                    }
                                    (el.len(), vals)
                                } else {
                                    return Err("randn currently requires array literal [dim, ...] for shape".into());
                                };

                                let requires_grad = if args.len() > 1 {
                                    match &args[1] {
                                        Expr::Bool(b) => *b,
                                        _ => false,
                                    }
                                } else {
                                    false
                                };

                                let i64_type = self.context.i64_type();
                                let usize_type = self.context.i64_type(); // usize is 64-bit

                                // Stack allocate shape array
                                let current_block = self.builder.get_insert_block().unwrap();
                                let entry = current_block
                                    .get_parent()
                                    .unwrap()
                                    .get_first_basic_block()
                                    .unwrap();
                                self.builder.position_at_end(entry);

                                let shape_array_type = usize_type.array_type(rank as u32);
                                let shape_alloca = self
                                    .builder
                                    .build_alloca(shape_array_type, "shape_arr")
                                    .map_err(|e| e.to_string())?;

                                self.builder.position_at_end(current_block);

                                // Store compiled shape values
                                for (i, val) in shape_vals.iter().enumerate() {
                                    let ptr = unsafe {
                                        self.builder.build_in_bounds_gep(
                                            shape_array_type,
                                            shape_alloca,
                                            &[
                                                i64_type.const_int(0, false),
                                                i64_type.const_int(i as u64, false),
                                            ],
                                            "shape_ptr_in",
                                        )
                                    }
                                    .map_err(|e| e.to_string())?;
                                    self.builder
                                        .build_store(ptr, *val)
                                        .map_err(|e| e.to_string())?;
                                }

                                let req_grad_val = self
                                    .context
                                    .bool_type()
                                    .const_int(if requires_grad { 1 } else { 0 }, false);

                                let f = self.module.get_function("tl_tensor_randn").unwrap();
                                let call = self
                                    .builder
                                    .build_call(
                                        f,
                                        &[
                                            i64_type.const_int(rank as u64, false).into(),
                                            shape_alloca.into(),
                                            req_grad_val.into(),
                                        ],
                                        "randn_res",
                                    )
                                    .map_err(|e| e.to_string())?;

                                let res = match call.try_as_basic_value() {
                                    ValueKind::Basic(v) => v,
                                    _ => return Err("Invalid call return".into()),
                                };
                                return Ok((res, Type::Tensor(Box::new(Type::F32), rank)));
                            }
                            _ => name,
                        };

                        let func = self
                            .module
                            .get_function(llvm_func_name)
                            .ok_or(format!("Function {} not found", name))?;

                        let mut compiled_args = Vec::new();
                        for arg in args {
                            let (val, _) = self.compile_expr(arg)?;
                            compiled_args.push(val.into());
                        }

                        let call = self
                            .builder
                            .build_call(func, &compiled_args, "call_tmp")
                            .map_err(|e| e.to_string())?;

                        // Lookup return type
                        let lookup_name = match name.as_str() {
                            "slice" => "tl_tensor_slice",
                            _ => name,
                        };

                        let ret_type = self
                            .fn_return_types
                            .get(lookup_name)
                            .cloned()
                            .unwrap_or(Type::Void);

                        match call.try_as_basic_value() {
                            ValueKind::Basic(v) => Ok((v, ret_type)),
                            _ => {
                                // Void return
                                Ok((
                                    self.context.i64_type().const_int(0, false).into(),
                                    Type::Void,
                                ))
                            }
                        }
                    }
                }
            }
            Expr::IndexAccess(target, indices) => {
                let (val, val_type) = self.compile_expr(target)?;
                match val_type {
                    Type::Tensor(_, _) => {
                        // Prepare indices array
                        let rank = indices.len();
                        let i64_type = self.context.i64_type();

                        // Create array on stack
                        let array_type = i64_type.array_type(rank as u32);
                        let array_alloca = self
                            .builder
                            .build_alloca(array_type, "idx_arr")
                            .map_err(|e| e.to_string())?;

                        for (i, idx_str) in indices.iter().enumerate() {
                            let idx_val = if let Ok(n) = idx_str.parse::<u64>() {
                                i64_type.const_int(n, false).into()
                            } else {
                                // Lookup variable
                                let (ptr_val, _) = self
                                    .lookup_variable(idx_str)
                                    .ok_or(format!("Index {} not found", idx_str))?;
                                self.builder
                                    .build_load(i64_type, ptr_val.into_pointer_value(), "idx_load")
                                    .map_err(|e| e.to_string())?
                            };

                            let elem_ptr = unsafe {
                                self.builder
                                    .build_gep(
                                        array_type,
                                        array_alloca,
                                        &[
                                            i64_type.const_int(0, false),
                                            i64_type.const_int(i as u64, false),
                                        ],
                                        "idx_ptr",
                                    )
                                    .map_err(|e| e.to_string())?
                            };
                            self.builder
                                .build_store(elem_ptr, idx_val)
                                .map_err(|e| e.to_string())?;
                        }

                        // Call tl_tensor_get_f32_md
                        let get_fn = self.module.get_function("tl_tensor_get_f32_md").unwrap();
                        let tensor_ptr = val.into_pointer_value();
                        let array_ptr = self
                            .builder
                            .build_pointer_cast(
                                array_alloca,
                                self.context.ptr_type(inkwell::AddressSpace::default()),
                                "arr_ptr",
                            )
                            .map_err(|e| e.to_string())?;
                        let rank_val = i64_type.const_int(rank as u64, false);

                        let call = self
                            .builder
                            .build_call(
                                get_fn,
                                &[tensor_ptr.into(), array_ptr.into(), rank_val.into()],
                                "get_md_call",
                            )
                            .map_err(|e| e.to_string())?;

                        let res = match call.try_as_basic_value() {
                            ValueKind::Basic(v) => v,
                            _ => return Err("Invalid get return".into()),
                        };

                        Ok((res, Type::F32))
                    }
                    _ => Err("Index access only on Tensor".into()),
                }
            }
            Expr::UnOp(op, expr) => {
                let (val, ty) = self.compile_expr(expr)?;
                match op {
                    UnOp::Neg => match ty {
                        Type::I64 => {
                            let i = val.into_int_value();
                            let res = self
                                .builder
                                .build_int_neg(i, "negtmp")
                                .map_err(|e| e.to_string())?;
                            Ok((res.into(), Type::I64))
                        }
                        Type::F32 => {
                            let f = val.into_float_value();
                            let res = self
                                .builder
                                .build_float_neg(f, "negtmp")
                                .map_err(|e| e.to_string())?;
                            Ok((res.into(), Type::F32))
                        }
                        Type::Tensor(inner, rank) => {
                            let neg_fn = self.module.get_function("tl_tensor_neg").unwrap();
                            let call = self
                                .builder
                                .build_call(neg_fn, &[val.into()], "neg")
                                .map_err(|e| e.to_string())?;
                            let res = match call.try_as_basic_value() {
                                ValueKind::Basic(v) => v,
                                _ => return Err("Failed neg".into()),
                            };
                            Ok((res, Type::Tensor(inner, rank)))
                        }
                        _ => Err("Negation only on int/float/tensor".into()),
                    },
                    UnOp::Not => {
                        match ty {
                            Type::Bool => {
                                let b = val.into_int_value(); // i1
                                let res = self
                                    .builder
                                    .build_not(b, "nottmp")
                                    .map_err(|e| e.to_string())?;
                                Ok((res.into(), Type::Bool))
                            }
                            _ => Err("Not only on bool".into()),
                        }
                    }
                }
            }

            Expr::Aggregation {
                op,
                expr,
                var,
                range,
                condition,
            } => {
                // For now, implement a simple version:
                // Assume range is a tensor/array and we iterate over its length
                // sum(arr[i] for i in arr) -> loop over arr indices

                let function = self
                    .builder
                    .get_insert_block()
                    .unwrap()
                    .get_parent()
                    .unwrap();

                // Get range expression to determine loop bounds
                let (range_val, range_ty) = self.compile_expr(range)?;

                // For Tensor types, get the length from the first dimension
                let loop_count = match &range_ty {
                    Type::Tensor(_, _) => {
                        let len_fn = self
                            .module
                            .get_function("tl_tensor_len")
                            .ok_or("tl_tensor_len not found")?;
                        let call = self
                            .builder
                            .build_call(len_fn, &[range_val.into()], "len")
                            .map_err(|e| e.to_string())?;
                        match call.try_as_basic_value() {
                            ValueKind::Basic(v) => v.into_int_value(),
                            _ => return Err("Failed to get tensor length".into()),
                        }
                    }
                    _ => return Err("Aggregation range must be a tensor".into()),
                };

                // Create blocks for the loop
                let preheader_bb = self.builder.get_insert_block().unwrap();
                let loop_bb = self.context.append_basic_block(function, "agg_loop");
                let body_bb = self.context.append_basic_block(function, "agg_body");
                let after_bb = self.context.append_basic_block(function, "agg_after");

                // Initialize accumulator based on op (0 for sum, etc.)
                let f64_type = self.context.f64_type();
                let init_val = match op {
                    AggregateOp::Sum | AggregateOp::Avg => f64_type.const_float(0.0),
                    AggregateOp::Max => f64_type.const_float(f64::NEG_INFINITY),
                    AggregateOp::Min => f64_type.const_float(f64::INFINITY),
                    AggregateOp::Count => f64_type.const_float(0.0),
                };

                // Branch to loop
                self.builder
                    .build_unconditional_branch(loop_bb)
                    .map_err(|e| e.to_string())?;

                // Loop header with phi nodes
                self.builder.position_at_end(loop_bb);
                let i64_type = self.context.i64_type();
                let counter_phi = self
                    .builder
                    .build_phi(i64_type, "i")
                    .map_err(|e| e.to_string())?;
                let acc_phi = self
                    .builder
                    .build_phi(f64_type, "acc")
                    .map_err(|e| e.to_string())?;

                counter_phi.add_incoming(&[(&i64_type.const_int(0, false), preheader_bb)]);
                acc_phi.add_incoming(&[(&init_val, preheader_bb)]);

                let current_i = counter_phi.as_basic_value().into_int_value();
                let current_acc = acc_phi.as_basic_value().into_float_value();

                // Check if i < loop_count
                let cond = self
                    .builder
                    .build_int_compare(inkwell::IntPredicate::SLT, current_i, loop_count, "cond")
                    .map_err(|e| e.to_string())?;
                self.builder
                    .build_conditional_branch(cond, body_bb, after_bb)
                    .map_err(|e| e.to_string())?;

                // Body: compute expression with var = element value
                self.builder.position_at_end(body_bb);
                self.enter_scope();

                // Determine element type from range_ty
                let elem_ty = match &range_ty {
                    Type::Tensor(t, _) => *t.clone(),
                    _ => Type::I64, // Fallback (should be checked earlier)
                };

                // Load element value using tl_tensor_get(tensor, index) -> f32
                let get_fn = self.module.get_function("tl_tensor_get").unwrap();
                let call_res = self
                    .builder
                    .build_call(get_fn, &[range_val.into(), current_i.into()], "get_elem")
                    .map_err(|e| e.to_string())?;

                let val_f32 = match call_res.try_as_basic_value() {
                    ValueKind::Basic(v) => v.into_float_value(),
                    _ => return Err("tl_tensor_get returned void".into()),
                };

                // Cast f32 to var type (if i64)
                let var_val: BasicValueEnum = match elem_ty {
                    Type::I64 => self
                        .builder
                        .build_float_to_signed_int(val_f32, i64_type, "cast_i64")
                        .map_err(|e| e.to_string())?
                        .into(),
                    Type::F32 => val_f32.into(),
                    _ => return Err("Unsupported tensor element type for aggregation".into()),
                };

                // Store the loop variable
                let var_alloca = self.create_entry_block_alloca(function, var, &elem_ty);
                self.builder
                    .build_store(var_alloca, var_val)
                    .map_err(|e| e.to_string())?;
                self.variables
                    .last_mut()
                    .unwrap()
                    .insert(var.clone(), (var_alloca.into(), elem_ty, false));

                // Compile the aggregated expression
                let (expr_val, _expr_ty) = self.compile_expr(expr)?;

                // Check condition if present
                let should_include = if let Some(cond_expr) = condition {
                    let (cond_val, _) = self.compile_expr(cond_expr)?;
                    cond_val.into_int_value()
                } else {
                    self.context.bool_type().const_int(1, false)
                };

                self.exit_scope();

                // Update accumulator based on op
                let expr_f64 = if expr_val.is_float_value() {
                    self.builder
                        .build_float_ext(expr_val.into_float_value(), f64_type, "ext")
                        .map_err(|e| e.to_string())?
                } else if expr_val.is_int_value() {
                    self.builder
                        .build_signed_int_to_float(expr_val.into_int_value(), f64_type, "itof")
                        .map_err(|e| e.to_string())?
                } else {
                    return Err("Aggregation expression must be numeric".into());
                };

                let new_acc = match op {
                    AggregateOp::Sum | AggregateOp::Avg => {
                        let add_val = self
                            .builder
                            .build_float_add(current_acc, expr_f64, "add")
                            .map_err(|e| e.to_string())?;
                        // Select based on condition
                        self.builder
                            .build_select(should_include, add_val, current_acc, "sel")
                            .map_err(|e| e.to_string())?
                            .into_float_value()
                    }
                    AggregateOp::Count => {
                        let one = f64_type.const_float(1.0);
                        let add_val = self
                            .builder
                            .build_float_add(current_acc, one, "inc")
                            .map_err(|e| e.to_string())?;
                        self.builder
                            .build_select(should_include, add_val, current_acc, "sel")
                            .map_err(|e| e.to_string())?
                            .into_float_value()
                    }
                    AggregateOp::Max => {
                        let is_greater = self
                            .builder
                            .build_float_compare(
                                inkwell::FloatPredicate::OGT,
                                expr_f64,
                                current_acc,
                                "gt",
                            )
                            .map_err(|e| e.to_string())?;
                        let max_val = self
                            .builder
                            .build_select(is_greater, expr_f64, current_acc, "max")
                            .map_err(|e| e.to_string())?
                            .into_float_value();
                        self.builder
                            .build_select(should_include, max_val, current_acc, "sel")
                            .map_err(|e| e.to_string())?
                            .into_float_value()
                    }
                    AggregateOp::Min => {
                        let is_less = self
                            .builder
                            .build_float_compare(
                                inkwell::FloatPredicate::OLT,
                                expr_f64,
                                current_acc,
                                "lt",
                            )
                            .map_err(|e| e.to_string())?;
                        let min_val = self
                            .builder
                            .build_select(is_less, expr_f64, current_acc, "min")
                            .map_err(|e| e.to_string())?
                            .into_float_value();
                        self.builder
                            .build_select(should_include, min_val, current_acc, "sel")
                            .map_err(|e| e.to_string())?
                            .into_float_value()
                    }
                };

                // Increment counter
                let next_i = self
                    .builder
                    .build_int_add(current_i, i64_type.const_int(1, false), "next_i")
                    .map_err(|e| e.to_string())?;

                // Branch back to loop header
                let body_end_bb = self.builder.get_insert_block().unwrap();
                self.builder
                    .build_unconditional_branch(loop_bb)
                    .map_err(|e| e.to_string())?;

                // Add incoming edges to phi nodes
                counter_phi.add_incoming(&[(&next_i, body_end_bb)]);
                acc_phi.add_incoming(&[(&new_acc, body_end_bb)]);

                // After loop
                self.builder.position_at_end(after_bb);

                // For avg, divide by count
                let result = if matches!(op, AggregateOp::Avg) {
                    let count_f64 = self
                        .builder
                        .build_signed_int_to_float(loop_count, f64_type, "count")
                        .map_err(|e| e.to_string())?;
                    self.builder
                        .build_float_div(
                            acc_phi.as_basic_value().into_float_value(),
                            count_f64,
                            "avg",
                        )
                        .map_err(|e| e.to_string())?
                } else {
                    acc_phi.as_basic_value().into_float_value()
                };

                // Convert back to f32 for consistency
                let result_f32 = self
                    .builder
                    .build_float_trunc(result, self.context.f32_type(), "trunc")
                    .map_err(|e| e.to_string())?;

                Ok((result_f32.into(), Type::F32))
            }

            _ => Err("Unsupported expression".into()),
        }
    }

    fn create_entry_block_alloca(
        &self,
        function: FunctionValue<'ctx>,
        name: &str,
        ty: &Type,
    ) -> inkwell::values::PointerValue<'ctx> {
        let builder = self.context.create_builder();
        let entry = function.get_first_basic_block().unwrap();
        match entry.get_first_instruction() {
            Some(first_instr) => builder.position_before(&first_instr),
            None => builder.position_at_end(entry),
        }

        let llvm_type: inkwell::types::BasicTypeEnum = match ty {
            Type::I64 => self.context.i64_type().into(),
            Type::F32 => self.context.f32_type().into(),
            // Tensor is a pointer to OpaqueTensor struct.
            // We represent it as a generic pointer (ptr) in LLVM 15+, or i8* in older.
            // Inkwell Context has ptr_type
            Type::Tensor(_, _) | Type::UserDefined(_) | Type::Struct(_) => self
                .context
                .ptr_type(inkwell::AddressSpace::default())
                .into(),
            _ => self.context.i64_type().into(),
        };

        builder.build_alloca(llvm_type, name).unwrap()
    }

    // Debug method to print IR
    pub fn dump_llvm_ir(&self) {
        self.module.print_to_stderr();
    }

    fn extract_index_bounds(
        &mut self,
        expr: &Expr,
        bounds: &mut HashMap<String, inkwell::values::IntValue<'ctx>>,
    ) -> Result<(), String> {
        match expr {
            Expr::IndexAccess(target, indices) => {
                // Target should be Expr::Ident for variable access
                // Instead of compiling, look up the variable directly
                let tensor_ptr = match target.as_ref() {
                    Expr::Variable(name) => {
                        let (val, ty) = self
                            .lookup_variable(name)
                            .ok_or(format!("Variable {} not found", name))?;
                        // Load pointer if needed
                        match ty {
                            Type::Tensor(_, _) => {
                                // val is a pointer to the tensor pointer
                                let loaded = self
                                    .builder
                                    .build_load(
                                        self.context.ptr_type(inkwell::AddressSpace::default()),
                                        val.into_pointer_value(),
                                        name,
                                    )
                                    .map_err(|e| e.to_string())?
                                    .into_pointer_value();
                                loaded
                            }
                            _ => return Err("Expected tensor variable".into()),
                        }
                    }
                    _ => {
                        return Err("Complex index target not supported in bounds extraction".into())
                    }
                };

                let dim_fn = self.module.get_function("tl_tensor_dim").unwrap();
                for (i, idx_name) in indices.iter().enumerate() {
                    if idx_name.parse::<u64>().is_ok() {
                        continue;
                    }
                    if !bounds.contains_key(idx_name) {
                        let dim_idx_val = self.context.i64_type().const_int(i as u64, false);
                        let call_result = self
                            .builder
                            .build_call(
                                dim_fn,
                                &[tensor_ptr.into(), dim_idx_val.into()],
                                "dim_size",
                            )
                            .map_err(|e| e.to_string())?;
                        let dim_size = match call_result.try_as_basic_value() {
                            ValueKind::Basic(v) => v.into_int_value(),
                            _ => return Err("Invalid dim return".into()),
                        };
                        bounds.insert(idx_name.clone(), dim_size);
                    }
                }
            }
            Expr::BinOp(lhs, _, rhs) => {
                self.extract_index_bounds(lhs, bounds)?;
                self.extract_index_bounds(rhs, bounds)?;
            }
            Expr::UnOp(_, inner) => self.extract_index_bounds(inner, bounds)?,
            _ => {}
        }
        Ok(())
    }

    fn compile_tensor_equation(
        &mut self,
        name: &str,
        lhs_indices: &[String],
        value: &Expr,
    ) -> Result<(), String> {
        let i64_type = self.context.i64_type();
        let f32_type = self.context.f32_type();

        let mut index_bounds = HashMap::new();
        self.extract_index_bounds(value, &mut index_bounds)?;

        for idx in lhs_indices {
            if !index_bounds.contains_key(idx) {
                return Err(format!("Bound not found for {}", idx));
            }
        }
        let reduction_indices: Vec<String> = index_bounds
            .keys()
            .filter(|k| !lhs_indices.contains(k))
            .cloned()
            .collect();

        let mut total_size = i64_type.const_int(1, false);
        for idx in lhs_indices {
            let bound = index_bounds.get(idx).unwrap();
            total_size = self
                .builder
                .build_int_mul(total_size, *bound, "sz_acc")
                .map_err(|e| e.to_string())?;
        }

        let calloc_fn = self.module.get_function("calloc").unwrap();
        let call_result = self
            .builder
            .build_call(
                calloc_fn,
                &[total_size.into(), i64_type.const_int(4, false).into()],
                "buf_void",
            )
            .map_err(|e| e.to_string())?;
        let buffer_void = match call_result.try_as_basic_value() {
            ValueKind::Basic(v) => v.into_pointer_value(),
            _ => return Err("Invalid calloc return".into()),
        };
        let buffer_ptr = self
            .builder
            .build_pointer_cast(
                buffer_void,
                self.context.ptr_type(inkwell::AddressSpace::default()),
                "buf_f32",
            )
            .map_err(|e| e.to_string())?;

        let parent_fn = self
            .builder
            .get_insert_block()
            .unwrap()
            .get_parent()
            .unwrap();
        let after_bb = self.context.append_basic_block(parent_fn, "eq_after");

        // Use tuple instead of struct to avoid lifetime issues
        type LoopInfo<'a> = (
            String,
            inkwell::basic_block::BasicBlock<'a>,
            inkwell::basic_block::BasicBlock<'a>,
            inkwell::values::PhiValue<'a>,
        );
        let mut loops: Vec<LoopInfo<'ctx>> = Vec::new();

        let all_indices = [lhs_indices, reduction_indices.as_slice()].concat();
        let mut current_bb = self.builder.get_insert_block().unwrap();

        self.enter_scope();

        for idx_name in &all_indices {
            let limit = *index_bounds.get(idx_name).unwrap();
            let cond_bb = self.context.append_basic_block(parent_fn, "loop_cond");
            let body_bb = self.context.append_basic_block(parent_fn, "loop_body");
            let aft_bb = self.context.append_basic_block(parent_fn, "loop_aft");

            self.builder
                .build_unconditional_branch(cond_bb)
                .map_err(|e| e.to_string())?;
            self.builder.position_at_end(cond_bb);
            let phi = self
                .builder
                .build_phi(i64_type, "i")
                .map_err(|e| e.to_string())?;
            phi.add_incoming(&[(&i64_type.const_int(0, false), current_bb)]);

            let iv = phi.as_basic_value().into_int_value();
            let cmp = self
                .builder
                .build_int_compare(inkwell::IntPredicate::SLT, iv, limit, "cmp")
                .map_err(|e| e.to_string())?;
            self.builder
                .build_conditional_branch(cmp, body_bb, aft_bb)
                .map_err(|e| e.to_string())?;

            self.builder.position_at_end(body_bb);
            let alloca = self.create_entry_block_alloca(parent_fn, idx_name, &Type::I64);
            self.builder
                .build_store(alloca, iv)
                .map_err(|e| e.to_string())?;
            // Insert variable directly into scope
            self.variables
                .last_mut()
                .unwrap()
                .insert(idx_name.clone(), (alloca.into(), Type::I64, false));

            loops.push((idx_name.clone(), cond_bb, aft_bb, phi));
            current_bb = body_bb;
        }

        let (rhs_val, _) = self.compile_expr(value)?;
        let rhs_float = rhs_val.into_float_value();

        let mut offset = i64_type.const_int(0, false);
        let mut stride = i64_type.const_int(1, false);
        for idx_name in lhs_indices.iter().rev() {
            let limit = *index_bounds.get(idx_name).unwrap();
            let (ptr_val, _) = self.lookup_variable(idx_name).unwrap();
            let iv = self
                .builder
                .build_load(i64_type, ptr_val.into_pointer_value(), "iv")
                .map_err(|e| e.to_string())?
                .into_int_value();
            let term = self
                .builder
                .build_int_mul(iv, stride, "term")
                .map_err(|e| e.to_string())?;
            offset = self
                .builder
                .build_int_add(offset, term, "off")
                .map_err(|e| e.to_string())?;
            stride = self
                .builder
                .build_int_mul(stride, limit, "str")
                .map_err(|e| e.to_string())?;
        }

        let elem_ptr = unsafe {
            self.builder
                .build_gep(f32_type, buffer_ptr, &[offset], "ptr")
                .map_err(|e| e.to_string())?
        };
        let cur = self
            .builder
            .build_load(f32_type, elem_ptr, "cur")
            .map_err(|e| e.to_string())?
            .into_float_value();
        let new_val = self
            .builder
            .build_float_add(cur, rhs_float, "new")
            .map_err(|e| e.to_string())?;
        self.builder
            .build_store(elem_ptr, new_val)
            .map_err(|e| e.to_string())?;

        let mut last_bb = self.builder.get_insert_block().unwrap();
        for (loop_name, cond_bb, aft_bb, phi) in loops.iter().rev() {
            self.builder.position_at_end(last_bb);
            let (alloca_val, _) = self.lookup_variable(loop_name).unwrap();
            let iv = self
                .builder
                .build_load(i64_type, alloca_val.into_pointer_value(), "iv")
                .map_err(|e| e.to_string())?
                .into_int_value();
            let next = self
                .builder
                .build_int_add(iv, i64_type.const_int(1, false), "next")
                .map_err(|e| e.to_string())?;
            phi.add_incoming(&[(&next, last_bb)]);
            self.builder
                .build_unconditional_branch(*cond_bb)
                .map_err(|e| e.to_string())?;
            last_bb = *aft_bb;
        }
        self.exit_scope();

        self.builder.position_at_end(last_bb);
        self.builder
            .build_unconditional_branch(after_bb)
            .map_err(|e| e.to_string())?;
        self.builder.position_at_end(after_bb);

        let new_fn = self.module.get_function("tl_tensor_new").unwrap();
        let rank = lhs_indices.len();
        let shape_alloca = self
            .builder
            .build_alloca(i64_type.array_type(rank as u32), "shape")
            .map_err(|e| e.to_string())?;
        for (i, idx) in lhs_indices.iter().enumerate() {
            let limit = *index_bounds.get(idx).unwrap();
            let elem_ptr = unsafe {
                self.builder
                    .build_gep(
                        i64_type.array_type(rank as u32),
                        shape_alloca,
                        &[
                            i64_type.const_int(0, false),
                            i64_type.const_int(i as u64, false),
                        ],
                        "shape_ptr",
                    )
                    .map_err(|e| e.to_string())?
            };
            self.builder
                .build_store(elem_ptr, limit)
                .map_err(|e| e.to_string())?;
        }
        let shape_ptr = self
            .builder
            .build_pointer_cast(
                shape_alloca,
                self.context.ptr_type(inkwell::AddressSpace::default()),
                "sptr",
            )
            .map_err(|e| e.to_string())?;
        let call_result = self
            .builder
            .build_call(
                new_fn,
                &[
                    buffer_ptr.into(),
                    i64_type.const_int(rank as u64, false).into(),
                    shape_ptr.into(),
                ],
                "t",
            )
            .map_err(|e| e.to_string())?;
        let tptr = match call_result.try_as_basic_value() {
            ValueKind::Basic(v) => v,
            _ => return Err("Invalid tl_tensor_new return".into()),
        };

        let v_alloca = self.create_entry_block_alloca(
            parent_fn,
            name,
            &Type::Tensor(Box::new(Type::F32), rank),
        );
        self.builder
            .build_store(v_alloca, tptr)
            .map_err(|e| e.to_string())?;
        self.variables.last_mut().unwrap().insert(
            name.to_string(),
            (
                v_alloca.into(),
                Type::Tensor(Box::new(Type::F32), rank),
                true,
            ),
        );

        Ok(())
    }
}
