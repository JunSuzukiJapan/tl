use crate::compiler::ast::*;
use crate::compiler::shape_analysis::ShapeAnalyzer;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::module::Module as InkwellModule;
use inkwell::types::{BasicMetadataTypeEnum, StructType};
use inkwell::values::ValueKind; // Used in relation wrappers
use inkwell::values::{BasicValueEnum, FunctionValue};
use inkwell::OptimizationLevel;
use std::collections::HashMap;

pub mod builtins;
pub mod expr;
pub mod kb;
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
    pub(crate) enum_types: HashMap<String, StructType<'ctx>>,
    pub(crate) enum_defs: HashMap<String, EnumDef>,
    pub(crate) fn_entry_scope_depth: usize,
    pub(crate) builtin_manager: expr::BuiltinManager,
    pub(crate) instance_methods: HashMap<String, expr::InstanceMethodManager>,
    pub(crate) static_methods: HashMap<String, expr::StaticMethodManager>,
    pub(crate) loop_stack: Vec<(
        inkwell::basic_block::BasicBlock<'ctx>,
        inkwell::basic_block::BasicBlock<'ctx>,
    )>,
    pub(crate) relations: std::collections::HashSet<String>,
    pub(crate) current_span: Option<crate::compiler::error::Span>,
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
            enum_types: HashMap::new(),
            enum_defs: HashMap::new(),
            fn_entry_scope_depth: 0,
            builtin_manager: expr::BuiltinManager::new(),
            instance_methods: HashMap::new(),
            static_methods: HashMap::new(),
            loop_stack: Vec::new(),
            relations: std::collections::HashSet::new(),
            current_span: None,
        };

        // Register all methods (instance and static)
        codegen.register_all_methods();

        // Register builtins (Enums, etc.)
        codegen.register_builtins();

        // Delegate to runtime module
        builtins::declare_runtime_functions(
            codegen.context,
            &codegen.module,
            &codegen.execution_engine,
            &mut codegen.fn_return_types,
        );

        codegen
    }
    pub fn dump_ir(&self) {
        self.module.print_to_stderr();
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

    pub fn emit_object_file(&self, path: &std::path::Path) -> Result<(), String> {
        use inkwell::targets::{
            CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
        };

        Target::initialize_native(&InitializationConfig::default()).map_err(|e| e.to_string())?;

        let triple = TargetMachine::get_default_triple();
        let target = Target::from_triple(&triple).map_err(|e| e.to_string())?;

        let target_machine = target
            .create_target_machine(
                &triple,
                "generic",
                "",
                OptimizationLevel::Default,
                RelocMode::Default,
                CodeModel::Default,
            )
            .ok_or("Failed to create target machine")?;

        target_machine
            .write_to_file(&self.module, FileType::Object, path)
            .map_err(|e| e.to_string())
    }

    pub fn emit_assembly_file(&self, path: &std::path::Path) -> Result<(), String> {
        use inkwell::targets::{
            CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
        };

        Target::initialize_native(&InitializationConfig::default()).map_err(|e| e.to_string())?;

        let triple = TargetMachine::get_default_triple();
        let target = Target::from_triple(&triple).map_err(|e| e.to_string())?;

        let target_machine = target
            .create_target_machine(
                &triple,
                "generic",
                "",
                OptimizationLevel::Default,
                RelocMode::Default,
                CodeModel::Default,
            )
            .ok_or("Failed to create target machine")?;

        target_machine
            .write_to_file(&self.module, FileType::Assembly, path)
            .map_err(|e| e.to_string())
    }

    pub(crate) fn check_tensor_result(
        &self,
        call_site_value: inkwell::values::CallSiteValue<'ctx>,
        context_msg: &str,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        let (file, line, col) = if let Some(span) = &self.current_span {
            (span.file.as_deref(), span.line as u32, span.column as u32)
        } else {
            (None, 0, 0)
        };
        self.check_tensor_result_with_loc(call_site_value, context_msg, file, line, col)
    }

    pub(crate) fn check_tensor_result_with_loc(
        &self,
        call_site_value: inkwell::values::CallSiteValue<'ctx>,
        _context_msg: &str,
        file: Option<&str>,
        line: u32,
        col: u32,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        let basic_value = match call_site_value.try_as_basic_value() {
            inkwell::values::ValueKind::Basic(v) => v,
            _ => return Err("Call returned void".into()),
        };

        if !basic_value.is_pointer_value() {
            return Ok(basic_value);
        }

        let ptr_val = basic_value.into_pointer_value();

        let current_bb = self.builder.get_insert_block().unwrap();
        let function = current_bb.get_parent().unwrap();

        // Check is_null
        let is_null = self.builder.build_is_null(ptr_val, "is_null").unwrap();

        let error_bb = self.context.append_basic_block(function, "runtime_error");
        let success_bb = self.context.append_basic_block(function, "runtime_success");

        self.builder
            .build_conditional_branch(is_null, error_bb, success_bb)
            .unwrap();

        // Error Handler
        self.builder.position_at_end(error_bb);

        // Prepare location args
        let file_str = file.unwrap_or("unknown");
        let file_ptr = self
            .builder
            .build_global_string_ptr(file_str, "file_str")
            .map_err(|e| e.to_string())?
            .as_pointer_value();

        // Call tl_amend_error_loc(file, line, col)
        let i32_type = self.context.i32_type();
        let amend_fn = self
            .module
            .get_function("tl_amend_error_loc")
            .expect("tl_amend_error_loc missing");

        // tl_amend_error_loc accepts (i8*, i32, i32)
        // file_ptr is i8* (from build_global_string_ptr)

        self.builder
            .build_call(
                amend_fn,
                &[
                    file_ptr.into(),
                    i32_type.const_int(line as u64, false).into(),
                    i32_type.const_int(col as u64, false).into(),
                ],
                "",
            )
            .unwrap();

        // Return zero/null
        let return_type = function.get_type().get_return_type();
        if let Some(rt) = return_type {
            if rt.is_pointer_type() {
                self.builder
                    .build_return(Some(&rt.into_pointer_type().const_null()))
                    .unwrap();
            } else if rt.is_int_type() {
                self.builder
                    .build_return(Some(&rt.into_int_type().const_zero()))
                    .unwrap();
            } else if rt.is_float_type() {
                self.builder
                    .build_return(Some(&rt.into_float_type().const_zero()))
                    .unwrap();
            } else {
                self.builder.build_return(None).unwrap();
            }
        } else {
            self.builder.build_return(None).unwrap();
        }

        // Success Path
        self.builder.position_at_end(success_bb);

        Ok(basic_value) // Return the original pointer (which is valid here)
    }

    fn register_builtins(&mut self) {
        let device_enum = EnumDef {
            name: "Device".to_string(),
            generics: vec![],
            variants: vec![
                VariantDef {
                    name: "Auto".to_string(),
                    fields: vec![],
                },
                VariantDef {
                    name: "Cpu".to_string(),
                    fields: vec![],
                },
                VariantDef {
                    name: "Metal".to_string(),
                    fields: vec![],
                },
                VariantDef {
                    name: "Cuda".to_string(),
                    fields: vec![],
                },
            ],
        };
        // Compile and register the builtin enum
        // We use unwrap() here because failure to compile a builtin is a compiler bug
        self.compile_enum_defs(&[device_enum]).unwrap();
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
                    if let Type::UserDefined(name) = ty {
                        let ptr = val_enum.into_pointer_value();
                        let load_type = self.context.ptr_type(inkwell::AddressSpace::default());
                        if let Ok(obj_val) = self.builder.build_load(load_type, ptr, "obj_to_free")
                        {
                            match name.as_str() {
                                "String" => {}
                                "File" | "Path" => {}
                                "Env" | "Http" => {}
                                _ => {
                                    let _ = self.emit_recursive_free(obj_val, ty);
                                    if let Some(unreg_fn) =
                                        self.module.get_function("tl_mem_unregister")
                                    {
                                        let _ = self.builder.build_call(
                                            unreg_fn,
                                            &[obj_val.into()],
                                            "",
                                        );
                                    }
                                    if let Some(free_fn) = self.module.get_function("free") {
                                        let void_ptr = self
                                            .builder
                                            .build_pointer_cast(
                                                obj_val.into_pointer_value(),
                                                self.context
                                                    .ptr_type(inkwell::AddressSpace::default()),
                                                "void_ptr",
                                            )
                                            .unwrap();
                                        let _ = self.builder.build_call(
                                            free_fn,
                                            &[void_ptr.into()],
                                            "",
                                        );
                                    }
                                }
                            }
                        }
                    } else if matches!(ty, Type::Struct(_)) {
                        // Load the struct pointer from the stack variable (Alloca)
                        let ptr = val_enum.into_pointer_value();
                        let load_type = self.context.ptr_type(inkwell::AddressSpace::default());
                        // We must check if the pointer stored in alloca is not null
                        if let Ok(struct_val) =
                            self.builder.build_load(load_type, ptr, "struct_to_free")
                        {
                            // Recursive free handles null check
                            let _ = self.emit_recursive_free(struct_val, ty);

                            if let Some(unreg_fn) = self.module.get_function("tl_mem_unregister") {
                                let _ = self.builder.build_call(unreg_fn, &[struct_val.into()], "");
                            }

                            // CRITICAL FIX: Free the struct pointer itself (container)
                            // recursive_free only freed the fields. unregister removed it from Runtime auto-free.
                            // We must explicitly free the malloc'd struct now.
                            // Use 'free' from libc (or tl_free_tmp if appropriate, but standard free is safer for general malloc)
                            if let Some(free_fn) = self.module.get_function("free") {
                                let void_ptr = self
                                    .builder
                                    .build_pointer_cast(
                                        struct_val.into_pointer_value(),
                                        self.context.ptr_type(inkwell::AddressSpace::default()),
                                        "void_ptr",
                                    )
                                    .unwrap();
                                let _ = self.builder.build_call(free_fn, &[void_ptr.into()], "");
                            } else {
                                // Try to declare free if not found (unlikely if stdlib loaded, but safe fallback)
                                let free_type = self.context.void_type().fn_type(
                                    &[self
                                        .context
                                        .ptr_type(inkwell::AddressSpace::default())
                                        .into()],
                                    false,
                                );
                                let free_fn = self.module.add_function("free", free_type, None);
                                let void_ptr = self
                                    .builder
                                    .build_pointer_cast(
                                        struct_val.into_pointer_value(),
                                        self.context.ptr_type(inkwell::AddressSpace::default()),
                                        "void_ptr",
                                    )
                                    .unwrap();
                                let _ = self.builder.build_call(free_fn, &[void_ptr.into()], "");
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
        // The proper fix is in StmtKind::Return to emit cleanup BEFORE the return.
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

    /// Null out a variable (Move Semantics) so it won't be double-freed
    #[allow(dead_code)]
    pub(crate) fn null_out_variable(&self, name: &str) -> Result<(), String> {
        for scope in self.variables.iter().rev() {
            if let Some((val, ty, _should_free)) = scope.get(name) {
                // Only for types that would be freed recursively
                if matches!(
                    ty,
                    Type::Tensor(_, _) | Type::Struct(_) | Type::UserDefined(_)
                ) {
                    let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
                    let null_ptr = ptr_type.const_null();

                    self.builder
                        .build_store(val.into_pointer_value(), null_ptr)
                        .map_err(|e| e.to_string())?;
                }
                return Ok(());
            }
        }
        Ok(())
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
                    Type::I64 | Type::Entity => self.context.i64_type().into(),
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

    fn compile_enum_defs(&mut self, enums: &[EnumDef]) -> Result<(), String> {
        // Pass 1: Opaque
        for e in enums {
            self.enum_types
                .insert(e.name.clone(), self.context.opaque_struct_type(&e.name));
            self.enum_defs.insert(e.name.clone(), e.clone());
        }

        // Pass 2: Body (Tag + Union)
        // We need data layout to calculate variant sizes
        let target_data = self.execution_engine.get_target_data();

        for e in enums {
            let mut max_payload_size = 0;

            for v in &e.variants {
                let mut field_types: Vec<inkwell::types::BasicTypeEnum> = Vec::new();
                for (_, ty) in &v.fields {
                    let field_llvm_ty = match ty {
                        Type::F32 => self.context.f32_type().into(),
                        Type::I64 | Type::Entity => self.context.i64_type().into(),
                        Type::Bool => self.context.bool_type().into(),
                        Type::Tensor(_, _) => self
                            .context
                            .ptr_type(inkwell::AddressSpace::default())
                            .into(),
                        Type::Struct(_) | Type::Enum(_) | Type::UserDefined(_) => {
                            // Objects are pointers
                            self.context
                                .ptr_type(inkwell::AddressSpace::default())
                                .into()
                        }
                        Type::Vec(_) => self
                            .context
                            .ptr_type(inkwell::AddressSpace::default())
                            .into(),
                        _ => {
                            return Err(format!(
                                "Unsupported type in enum variant {}: {:?}",
                                v.name, ty
                            ))
                        }
                    };
                    field_types.push(field_llvm_ty);
                }

                // Create anonymous struct type to measure size
                let variant_struct_ty = self.context.struct_type(&field_types, false);
                let size = target_data.get_store_size(&variant_struct_ty);
                if size > max_payload_size {
                    max_payload_size = size;
                }
            }

            // Enum body: { i32 tag, [i64 x N] payload }
            // Alignment Issue: If we use [i8], alignment is 1. If we store i64, we need alignment 8.
            // By using [i64], we enforce alignment 8 for the payload area.
            let tag_type = self.context.i32_type();

            // Calculate number of i64s needed
            let payload_size = std::cmp::max(max_payload_size, 1); // Bytes needed
            let element_count = (payload_size + 7) / 8; // CEIL(bytes / 8)

            let payload_type = self.context.i64_type().array_type(element_count as u32);

            if let Some(st) = self.enum_types.get(&e.name) {
                st.set_body(&[tag_type.into(), payload_type.into()], false);
            }
        }
        Ok(())
    }

    fn compile_impl_blocks(&mut self, impls: &[ImplBlock]) -> Result<(), String> {
        // Pass 1: Declare all methods (Prototypes) and register return types
        for imp in impls {
            for method in &imp.methods {
                let simple_target = if imp.target_type.contains("::") {
                    imp.target_type.split("::").last().unwrap()
                } else {
                    &imp.target_type
                };
                let mangled_name = format!("tl_{}_{}", simple_target, method.name);

                // SRET is disabled; keep signatures simple.
                let uses_sret = false;

                let mut param_types: Vec<BasicMetadataTypeEnum> = Vec::new();

                // If sret, add hidden pointer argument at the beginning
                if uses_sret {
                    param_types.push(
                        self.context
                            .ptr_type(inkwell::AddressSpace::default())
                            .into(),
                    );
                }

                // Add regular arguments
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
                        Type::I64 | Type::Entity => self.context.i64_type().into(),
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
                            .ptr_type(inkwell::AddressSpace::default())
                            .into(),
                    };
                    param_types.push(ty);
                }

                // Build function type
                let fn_type = if uses_sret {
                    self.context.void_type().fn_type(&param_types, false)
                } else {
                    match &method.return_type {
                        Type::F32 => self.context.f32_type().fn_type(&param_types, false),
                        Type::I64 | Type::Entity => {
                            self.context.i64_type().fn_type(&param_types, false)
                        }
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
                    }
                };

                let _function = self.module.add_function(&mangled_name, fn_type, None);
                self.fn_return_types
                    .insert(mangled_name.clone(), method.return_type.clone());
            }
        }

        // Pass 2: Compile Bodies
        for imp in impls {
            for method in &imp.methods {
                let simple_target = if imp.target_type.contains("::") {
                    imp.target_type.split("::").last().unwrap()
                } else {
                    &imp.target_type
                };
                let mangled_name = format!("tl_{}_{}", simple_target, method.name);
                let function = self
                    .module
                    .get_function(&mangled_name)
                    .ok_or(format!("Function {} not found", mangled_name))?;

                // Compile Body
                let entry = self.context.append_basic_block(function, "entry");
                self.builder.position_at_end(entry);
                self.fn_entry_scope_depth = self.variables.len();
                self.enter_scope();

                // Check if this method uses sret
                let uses_sret = false; /* SRET DISABLED */
                let param_offset = if uses_sret { 1 } else { 0 };

                // Get params and store them (skip sret param if present)
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

                    if let Some(param_val) = function.get_nth_param((i + param_offset) as u32) {
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

    pub fn compile_module(&mut self, ast_module: &Module, module_name: &str) -> Result<(), String> {
        // 0. Declare runtime functions
        builtins::declare_runtime_functions(
            self.context,
            &self.module,
            &self.execution_engine,
            &mut self.fn_return_types,
        );

        // Generate Logic KB initialization function
        self.compile_kb_init_function(ast_module, module_name)?;

        // Compile submodules recursively
        for (sub_name, submodule) in &ast_module.submodules {
            self.compile_module(submodule, sub_name)?;
        }

        // 1. Declare structs (types) and methods
        self.compile_struct_defs(&ast_module.structs)?;
        self.compile_enum_defs(&ast_module.enums)?;
        self.compile_struct_defs(&ast_module.structs)?;
        self.compile_enum_defs(&ast_module.enums)?;
        self.compile_relation_wrappers(&ast_module.relations)?;

        for r in &ast_module.relations {
            self.relations.insert(r.name.clone());
        }

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
        if let Err(e) = self.module.verify() {
            // self.module.print_to_stderr();
            return Err(format!("Module verification failed: {}", e.to_string()));
        }

        self.apply_optimizations();
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

        // Check if this function returns a struct (requires sret)
        let uses_sret = false; /* SRET DISABLED */

        let mut args_types = Vec::new();

        // If sret, add hidden pointer argument at the beginning
        if uses_sret {
            args_types.push(
                self.context
                    .ptr_type(inkwell::AddressSpace::default())
                    .into(),
            );
        }

        // Add regular arguments
        for (_, val) in &func.args {
            let arg_ty: inkwell::types::BasicMetadataTypeEnum = match val {
                Type::I64 | Type::Entity => self.context.i64_type().into(),
                Type::F32 => self.context.f32_type().into(),
                Type::Bool => self.context.bool_type().into(),
                Type::Tensor(_, _) => self
                    .context
                    .ptr_type(inkwell::AddressSpace::default())
                    .into(),
                Type::UserDefined(_) | Type::Struct(_) | Type::Enum(_) => self
                    .context
                    .ptr_type(inkwell::AddressSpace::default())
                    .into(),
                _ => self.context.i64_type().into(),
            };
            args_types.push(arg_ty);
        }

        // Build function type
        let fn_type = if uses_sret {
            // Sret functions return void
            self.context.void_type().fn_type(&args_types, false)
        } else {
            let ret_type: Option<inkwell::types::BasicTypeEnum> = match &func.return_type {
                Type::Void => None,
                Type::I64 => Some(self.context.i64_type().into()),
                Type::F32 => Some(self.context.f32_type().into()),
                Type::Bool => Some(self.context.bool_type().into()),
                Type::Tensor(_, _) => Some(
                    self.context
                        .ptr_type(inkwell::AddressSpace::default())
                        .into(),
                ),
                Type::Struct(_) | Type::UserDefined(_) | Type::Enum(_) => Some(
                    self.context
                        .ptr_type(inkwell::AddressSpace::default())
                        .into(),
                ),
                _ => Some(self.context.i64_type().into()),
            };
            match ret_type {
                Some(inkwell::types::BasicTypeEnum::IntType(i)) => i.fn_type(&args_types, false),
                Some(inkwell::types::BasicTypeEnum::FloatType(f)) => f.fn_type(&args_types, false),
                Some(inkwell::types::BasicTypeEnum::PointerType(p)) => {
                    p.fn_type(&args_types, false)
                }
                _ => self.context.void_type().fn_type(&args_types, false),
            }
        };

        let val = self.module.add_function(&func.name, fn_type, None);

        // Add param names for debug
        let param_offset = if uses_sret { 1 } else { 0 };
        if uses_sret {
            if let Some(sret_param) = val.get_nth_param(0) {
                sret_param.set_name("sret");
            }
        }
        for (i, arg) in val.get_param_iter().skip(param_offset).enumerate() {
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

        // Check if this function uses sret
        let uses_sret = false; /* SRET DISABLED */
        let param_offset = if uses_sret { 1 } else { 0 };

        // Register arguments (skip sret param if present)
        for (i, arg) in function.get_param_iter().skip(param_offset).enumerate() {
            let (arg_name, arg_type) = &func.args[i];
            let alloca = self.create_entry_block_alloca(function, arg_name, arg_type);

            // Borrowing Semantics: Just store the pointer/value.
            // Do NOT Acquire/DeepClone. The caller owns the data.
            match arg {
                inkwell::values::BasicValueEnum::PointerValue(p) => {
                    self.builder.build_store(alloca, p).unwrap()
                }
                inkwell::values::BasicValueEnum::FloatValue(f) => {
                    self.builder.build_store(alloca, f).unwrap()
                }
                inkwell::values::BasicValueEnum::IntValue(v) => {
                    self.builder.build_store(alloca, v).unwrap()
                }
                _ => panic!("Unsupported arg type"),
            };

            // Insert into current scope with should_free=FALSE
            // Arguments are BORROWED. Function must NOT free them on exit.
            self.variables
                .last_mut()
                .unwrap()
                .insert(arg_name.clone(), (alloca.into(), arg_type.clone(), false));
        }

        // Initialize Arena in main if needed
        if func.name == "main" {
            // Logic Engine Init
            if let Some(init_kb) = self.module.get_function("_tl_init_kb") {
                self.builder.build_call(init_kb, &[], "").unwrap();
            }

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
                if let StmtKind::Expr(expr) = &stmt.inner {
                    let (val, ty) = self.compile_expr(expr)?;

                    // IMPORTANT: Unregister return value (same as StmtKind::Return)
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
            // Try to get more specific error from LLVM
            eprintln!("=== LLVM VERIFICATION FAILED FOR: {} ===", func.name);
            return Err(format!("Invalid generated function {}", func.name));
        }

        Ok(())
    }

    fn apply_optimizations(&self) {
        // OptimizationLevel::Aggressive is already set in execution_engine initialization.
        // Manual pass management requires exact matching of inkwell/LLVM versioned methods.
    }

    fn compile_relation_wrappers(&mut self, relations: &[RelationDecl]) -> Result<(), String> {
        if relations.is_empty() {
            return Ok(());
        }

        // Register return types FIRST so recursive/future calls find it
        for relation in relations {
            self.fn_return_types
                .insert(relation.name.clone(), Type::Tensor(Box::new(Type::F32), 1));
        }

        let i64_type = self.context.i64_type();
        let tensor_ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());

        // Runtime function tl_query
        let tl_query_fn = self
            .module
            .get_function("tl_query")
            .ok_or("tl_query must be declared")?;
        // Runtime function tl_tensor_from_i64_array
        let tl_tensor_from_arr_fn = self
            .module
            .get_function("tl_tensor_from_i64_array")
            .ok_or("tl_tensor_from_i64_array must be declared")?;
        // Runtime function tl_tensor_free
        let tl_tensor_free_fn = self
            .module
            .get_function("tl_tensor_free")
            .ok_or("tl_tensor_free must be declared")?;

        for rel in relations {
            let func_name = &rel.name;
            // Args: mask (i64), arg1 (i64), arg2 (i64)...
            let mut arg_types = vec![i64_type.into()]; // mask
            for _ in 0..rel.args.len() {
                arg_types.push(i64_type.into());
            }

            let fn_type = tensor_ptr_type.fn_type(&arg_types, false);
            let function = self.module.add_function(func_name, fn_type, None);

            let basic_block = self.context.append_basic_block(function, "entry");
            self.builder.position_at_end(basic_block);

            // Build name string
            let name_global = self
                .builder
                .build_global_string_ptr(func_name, "rel_name")
                .unwrap();
            let name_ptr = name_global.as_pointer_value();

            // Get mask
            let mask_arg = function.get_nth_param(0).unwrap().into_int_value();

            // Pack other args into array
            let num_args = rel.args.len();
            if num_args > 0 {
                let arr_type = i64_type.array_type(num_args as u32);
                let arr_alloca = self.builder.build_alloca(arr_type, "args_arr").unwrap();

                for i in 0..num_args {
                    let arg_val = function
                        .get_nth_param((i + 1) as u32)
                        .unwrap()
                        .into_int_value();
                    // Store in array
                    // GEP to element i
                    let ptr = unsafe {
                        self.builder
                            .build_gep(
                                arr_type,
                                arr_alloca,
                                &[
                                    i64_type.const_int(0, false),
                                    i64_type.const_int(i as u64, false),
                                ],
                                "",
                            )
                            .unwrap()
                    };
                    self.builder.build_store(ptr, arg_val).unwrap();
                }

                // Create tensor from array
                // tl_tensor_from_i64_array expects pointer to i64 (decayed array) and size
                let decayed_ptr = unsafe {
                    self.builder
                        .build_gep(
                            arr_type,
                            arr_alloca,
                            &[i64_type.const_int(0, false), i64_type.const_int(0, false)],
                            "decayed",
                        )
                        .unwrap()
                };

                let call = self
                    .builder
                    .build_call(
                        tl_tensor_from_arr_fn,
                        &[
                            decayed_ptr.into(),
                            i64_type.const_int(num_args as u64, false).into(),
                        ],
                        "args_tensor",
                    )
                    .unwrap();
                let args_tensor = self.check_tensor_result(call, "args_tensor_error")?;

                // Call tl_query
                let result_tensor = match self
                    .builder
                    .build_call(
                        tl_query_fn,
                        &[name_ptr.into(), mask_arg.into(), args_tensor.into()],
                        "res",
                    )
                    .unwrap()
                    .try_as_basic_value()
                {
                    ValueKind::Basic(v) => v,
                    _ => return Err("Expected value from tl_query".to_string()),
                };

                // Free args_tensor (it was created for this call)
                self.builder
                    .build_call(tl_tensor_free_fn, &[args_tensor.into()], "")
                    .unwrap();

                self.builder.build_return(Some(&result_tensor)).unwrap();
            } else {
                // No args case
                // Create empty tensor
                // Or pass null? tl_query check for null args
                let null_ptr = tensor_ptr_type.const_null();
                let result_tensor = match self
                    .builder
                    .build_call(
                        tl_query_fn,
                        &[name_ptr.into(), mask_arg.into(), null_ptr.into()],
                        "res",
                    )
                    .unwrap()
                    .try_as_basic_value()
                {
                    ValueKind::Basic(v) => v,
                    _ => return Err("Expected value from tl_query".to_string()),
                };
                self.builder.build_return(Some(&result_tensor)).unwrap();
            }

            if !function.verify(true) {
                return Err(format!("Invalid generated relation wrapper {}", func_name));
            }
        }
        Ok(())
    }
}
