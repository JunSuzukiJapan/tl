use crate::compiler::ast::*;
use crate::compiler::shape_analysis::ShapeAnalyzer;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::module::Module as InkwellModule;
use inkwell::types::{BasicMetadataTypeEnum, StructType};
// use inkwell::values::Either; // Not used directly
use inkwell::values::{BasicValueEnum, FunctionValue};
use inkwell::OptimizationLevel;
use std::collections::HashMap;

pub mod builtins;
pub mod expr;
pub mod stmt;
pub mod tensor;

pub struct CodeGenerator<'ctx> {
    pub(crate) context: &'ctx Context,
    pub(crate) module: InkwellModule<'ctx>,
    pub(crate) builder: Builder<'ctx>,
    pub(crate) execution_engine: ExecutionEngine<'ctx>,
    pub(crate) variables: Vec<HashMap<String, (BasicValueEnum<'ctx>, Type, bool)>>,
    pub(crate) fn_return_types: HashMap<String, Type>,
    pub(crate) struct_types: HashMap<String, StructType<'ctx>>,
    pub(crate) struct_defs: HashMap<String, StructDef>,
    pub(crate) fn_entry_scope_depth: usize,
}

impl<'ctx> CodeGenerator<'ctx> {
    pub fn new(context: &'ctx Context, module_name: &str) -> Self {
        let module = context.create_module(module_name);
        let builder = context.create_builder();
        let execution_engine = module
            .create_jit_execution_engine(OptimizationLevel::Aggressive)
            .map_err(|e| e.to_string())
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
            fn_entry_scope_depth: 0,
        };

        // Delegate to runtime module
        builtins::declare_runtime_functions(
            codegen.context,
            &codegen.module,
            &codegen.execution_engine,
            &mut codegen.fn_return_types,
        );

        codegen
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
        if let Some(f) = self.module.get_function("tl_mem_enter_scope") {
            self.builder.build_call(f, &[], "").unwrap();
        }
    }

    // Helper to generate free calls for variables in a specific scope index
    fn emit_cleanup_vars_in_scope(&self, scope_idx: usize) {
        if let Some(scope) = self.variables.get(scope_idx) {
            for (_name, (val_enum, ty, should_free)) in scope {
                if *should_free {
                    if matches!(ty, Type::Struct(_) | Type::UserDefined(_)) {
                        // Load the struct pointer from the stack variable (Alloca)
                        let ptr = val_enum.into_pointer_value();
                        let load_type = self.context.ptr_type(inkwell::AddressSpace::default());
                        // We must check if the pointer stored in alloca is not null
                        if let Ok(struct_val) =
                            self.builder.build_load(load_type, ptr, "struct_to_free")
                        {
                            // Recursive free handles null check
                            let _ = self.emit_recursive_free(struct_val, ty);

                            // CRITICAL: Unregister from MemoryManager to prevent double-free via tl_mem_exit_scope
                            if let Some(unreg_fn) = self.module.get_function("tl_mem_unregister") {
                                let _ = self.builder.build_call(unreg_fn, &[struct_val.into()], "");
                            }
                        }
                    } else if matches!(ty, Type::Tensor(_, _)) {
                        let ptr = val_enum.into_pointer_value();
                        let load_type = self.context.ptr_type(inkwell::AddressSpace::default());
                        if let Ok(tensor_val) =
                            self.builder.build_load(load_type, ptr, "tensor_to_free")
                        {
                            let _ = self.emit_recursive_free(tensor_val, ty);
                        }
                    }
                }
            }
        }
    }

    fn emit_all_scopes_cleanup(&self) {
        if let Some(f) = self.module.get_function("tl_mem_exit_scope") {
            // Only clean up scopes pushed WITHIN the current function
            // Iterate in REVERSE order (from inner to outer)
            let start = self.fn_entry_scope_depth;
            let end = self.variables.len();

            for i in (start..end).rev() {
                // 1. Emit cleanup for variables in this scope
                self.emit_cleanup_vars_in_scope(i);

                // 2. Call runtime exit_scope
                self.builder.build_call(f, &[], "").unwrap();
            }
        }
    }

    // Emit cleanup for the current scope (without popping).
    pub(crate) fn emit_top_scope_cleanup(&self) {
        if self.variables.is_empty() {
            return;
        }

        // Cleanup current scope
        self.emit_cleanup_vars_in_scope(self.variables.len() - 1);

        if let Some(f) = self.module.get_function("tl_mem_exit_scope") {
            self.builder.build_call(f, &[], "").unwrap();
        }
    }

    // Exit the current scope
    fn exit_scope(&mut self) {
        // Only emit cleanup if the current block is NOT terminated.
        // Note: This causes enter/exit imbalance at runtime for return statements.
        // The proper fix is in Stmt::Return to emit cleanup BEFORE the return.
        let is_terminated = self
            .builder
            .get_insert_block()
            .map(|b| b.get_terminator().is_some())
            .unwrap_or(false);

        if !is_terminated {
            self.emit_top_scope_cleanup();
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
                    Type::Struct(name) | Type::UserDefined(name) => {
                        if self.struct_types.contains_key(name) || name == "String" {
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
        // Pass 1: Declare all methods (Prototypes) and register return types
        for imp in impls {
            for method in &imp.methods {
                let mangled_name = format!("tl_{}_{}", imp.target_type, method.name);

                let mut param_types: Vec<BasicMetadataTypeEnum> = Vec::new();

                // Add 'self' param type explicitly if we want strict compatibility,
                // but parser puts 'self' in args so we treat it as arg 0.
                for (_arg_name, arg_ty) in &method.args {
                    let resolved_ty = if let Type::UserDefined(name) = arg_ty {
                        if name == "Self" {
                            Type::UserDefined(imp.target_type.clone())
                        } else {
                            arg_ty.clone()
                        }
                    } else {
                        arg_ty.clone()
                    };

                    let ty: BasicMetadataTypeEnum = match &resolved_ty {
                        Type::F32 => self.context.f32_type().into(),
                        Type::I64 => self.context.i64_type().into(),
                        Type::Bool => self.context.bool_type().into(),
                        Type::Tensor(_, _) => self
                            .context
                            .ptr_type(inkwell::AddressSpace::default())
                            .into(),
                        Type::Struct(_) | Type::UserDefined(_) => self
                            .context
                            .ptr_type(inkwell::AddressSpace::default())
                            .into(),
                        _ => self
                            .context
                            .ptr_type(inkwell::AddressSpace::default()) // Fallback
                            .into(),
                    };
                    param_types.push(ty);
                }

                let fn_type = match &method.return_type {
                    Type::F32 => self.context.f32_type().fn_type(&param_types, false),
                    Type::I64 => self.context.i64_type().fn_type(&param_types, false),
                    Type::Bool => self.context.bool_type().fn_type(&param_types, false),
                    Type::Void => self.context.void_type().fn_type(&param_types, false),
                    Type::Tensor(_, _) => self
                        .context
                        .ptr_type(inkwell::AddressSpace::default())
                        .fn_type(&param_types, false),
                    Type::Struct(_) | Type::UserDefined(_) => self
                        .context
                        .ptr_type(inkwell::AddressSpace::default())
                        .fn_type(&param_types, false),
                    _ => self.context.void_type().fn_type(&param_types, false),
                };

                let _function = self.module.add_function(&mangled_name, fn_type, None);
                self.fn_return_types
                    .insert(mangled_name.clone(), method.return_type.clone());
            }
        }

        // Pass 2: Compile Bodies
        for imp in impls {
            for method in &imp.methods {
                let mangled_name = format!("tl_{}_{}", imp.target_type, method.name);
                let function = self
                    .module
                    .get_function(&mangled_name)
                    .ok_or(format!("Function {} not found", mangled_name))?;

                // Compile Body
                let entry = self.context.append_basic_block(function, "entry");
                self.builder.position_at_end(entry);
                self.fn_entry_scope_depth = self.variables.len();
                self.enter_scope();

                // Get params and store them
                for (i, (arg_name, arg_ty)) in method.args.iter().enumerate() {
                    let resolved_ty = if let Type::UserDefined(name) = arg_ty {
                        if name == "Self" {
                            Type::UserDefined(imp.target_type.clone())
                        } else {
                            arg_ty.clone()
                        }
                    } else {
                        arg_ty.clone()
                    };

                    if let Some(param_val) = function.get_nth_param(i as u32) {
                        param_val.set_name(arg_name);
                        let alloca =
                            self.create_entry_block_alloca(function, arg_name, &resolved_ty);
                        self.builder.build_store(alloca, param_val).unwrap();

                        // Register in scope
                        self.variables
                            .last_mut()
                            .unwrap()
                            .insert(arg_name.clone(), (alloca.into(), resolved_ty, false));
                    }
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
                        // CRITICAL FIX: Emit cleanup BEFORE the implicit return
                        self.emit_all_scopes_cleanup();
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
        builtins::declare_runtime_functions(
            self.context,
            &self.module,
            &self.execution_engine,
            &mut self.fn_return_types,
        );

        // 1. Declare structs (types) and methods
        // 1. Declare structs (types) and methods
        self.compile_struct_defs(&ast_module.structs)?;

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
                is_extern: false,
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

        // 3. Compile Impl Blocks (Declare Method Prototypes + Body)
        // Moved after function proto conversion so methods can call global functions
        self.compile_impl_blocks(&ast_module.impls)?;

        // 4. Compile function bodies
        for func in &functions_refs {
            if func.is_extern {
                continue;
            }
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

        // Apply LLVM optimizations
        self.apply_optimizations();

        self.module.print_to_file("dump.ll").unwrap();
        Ok(())
    }

    pub(crate) fn lookup_variable(&self, name: &str) -> Option<(BasicValueEnum<'ctx>, Type)> {
        for scope in self.variables.iter().rev() {
            if let Some((v, t, _)) = scope.get(name) {
                return Some((*v, t.clone()));
            }
        }
        None
    }

    pub(crate) fn lookup_scope_depth(&self, name: &str) -> Option<usize> {
        for (i, scope) in self.variables.iter().enumerate().rev() {
            if scope.contains_key(name) {
                return Some(i);
            }
        }
        None
    }

    pub(crate) fn is_outer_scope(&self, name: &str) -> bool {
        if let Some(depth) = self.lookup_scope_depth(name) {
            // Current depth is self.variables.len() - 1
            if depth < self.variables.len() - 1 {
                return true;
            }
        }
        false
    }

    fn compile_fn_proto(&mut self, func: &FunctionDef) -> Result<FunctionValue<'ctx>, String> {
        self.fn_return_types
            .insert(func.name.clone(), func.return_type.clone());
        let ret_type: Option<inkwell::types::BasicTypeEnum> = match &func.return_type {
            Type::Void => None, // Void is not BasicValue
            Type::I64 => Some(self.context.i64_type().into()),
            Type::F32 => Some(self.context.f32_type().into()),
            Type::Bool => Some(self.context.bool_type().into()),
            Type::Tensor(_, _) => Some(
                self.context
                    .ptr_type(inkwell::AddressSpace::default())
                    .into(),
            ),
            Type::Struct(_) | Type::UserDefined(_) => Some(
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
                Type::Tensor(_, _) => self
                    .context
                    .ptr_type(inkwell::AddressSpace::default())
                    .into(),
                Type::UserDefined(_) | Type::Struct(_) => self
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

        // Push a new scope for function arguments
        self.fn_entry_scope_depth = self.variables.len();
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

        // Initialize Arena in main if needed
        if func.name == "main" {
            let mut analyzer = ShapeAnalyzer::new();
            let profile = analyzer.analyze_block(&func.body);
            // Heuristic: If we have static tensors or significant allocations, init arena
            // We assume safe upper bound for OpaqueTensor size (around 32-48 bytes usually, use 64 for safety)
            // Plus the actual static tensor data size.
            let mut total_capacity = profile.total_static_size.unwrap_or(0);
            total_capacity += profile.max_allocations * 512; // Increased per-allocation overhead

            if total_capacity > 0 || profile.max_allocations > 0 {
                // Minimum arena size: 64KB
                total_capacity = total_capacity.max(65536);
                // Align to page size (4096) roughly, or just use what we need.
                // call tl_arena_init(capacity)
                let init_fn = self
                    .module
                    .get_function("tl_arena_init")
                    .or_else(|| {
                        // Declare if missing (should be in builtins but just in case)
                        let i64_type = self.context.i64_type();
                        let fn_type = self.context.void_type().fn_type(&[i64_type.into()], false);
                        let f = self.module.add_function("tl_arena_init", fn_type, None);
                        Some(f)
                    })
                    .unwrap();

                self.builder
                    .build_call(
                        init_fn,
                        &[self
                            .context
                            .i64_type()
                            .const_int(total_capacity as u64, false)
                            .into()],
                        "",
                    )
                    .unwrap();
            }
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
                    let (val, ty) = self.compile_expr(expr)?;

                    // IMPORTANT: Unregister return value (same as Stmt::Return)
                    self.emit_recursive_unregister(val, &ty)?;

                    self.emit_all_scopes_cleanup();

                    // CRITICAL FIX: Pop the function scope from variables stack
                    // emit_all_scopes_cleanup only emits LLVM IR calls to tl_mem_exit_scope
                    // but doesn't update the Rust compiler's scope tracking
                    self.variables.pop();

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
        if func.return_type == Type::Void
            && self
                .builder
                .get_insert_block()
                .unwrap()
                .get_terminator()
                .is_none()
        {
            self.builder.build_return(None).map_err(|e| e.to_string())?;
        }

        if !function.verify(true) {
            function.print_to_stderr();
            return Err(format!("Invalid generated function {}", func.name));
        }

        Ok(())
    }

    fn apply_optimizations(&self) {
        // OptimizationLevel::Aggressive is already set in execution_engine initialization.
        // Manual pass management requires exact matching of inkwell/LLVM versioned methods.
    }
}
