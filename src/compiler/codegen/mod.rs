// src/compiler/codegen.rs
use crate::compiler::ast::*;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::module::Module as InkwellModule;
use inkwell::types::{BasicMetadataTypeEnum, StructType};
// use inkwell::values::Either; // Not used directlyimizationLevel;
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
}

impl<'ctx> CodeGenerator<'ctx> {
    pub fn new(context: &'ctx Context, module_name: &str) -> Self {
        let module = context.create_module(module_name);
        let builder = context.create_builder();
        let execution_engine = module
            .create_jit_execution_engine(OptimizationLevel::None)
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
        builtins::declare_runtime_functions(
            self.context,
            &self.module,
            &self.execution_engine,
            &mut self.fn_return_types,
        );

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

    pub(crate) fn lookup_variable(&self, name: &str) -> Option<(BasicValueEnum<'ctx>, Type)> {
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
}
