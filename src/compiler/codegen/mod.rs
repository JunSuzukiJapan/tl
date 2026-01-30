use crate::compiler::ast::*;

use crate::compiler::shape_analysis::ShapeAnalyzer;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::module::Module as InkwellModule;
use inkwell::AddressSpace;
use inkwell::types::{BasicMetadataTypeEnum, StructType};
use inkwell::values::ValueKind; // Used in relation wrappers
use inkwell::values::{BasicValueEnum, FunctionValue};
use inkwell::OptimizationLevel;
use std::collections::HashMap;

pub const CLEANUP_NONE: u8 = 0;
pub const CLEANUP_FULL: u8 = 1;
pub const CLEANUP_FINALIZE: u8 = 2;
pub const CLEANUP_STACK: u8 = 3;

pub mod builtins;
pub mod expr;
pub mod kb;
pub mod mono;
pub mod stmt;
pub mod tensor;
pub mod type_manager;
pub mod builtin_types;

pub struct CodeGenerator<'ctx> {
    pub(crate) context: &'ctx Context,
    pub(crate) module: InkwellModule<'ctx>,
    pub(crate) builder: Builder<'ctx>,
    pub(crate) execution_engine: ExecutionEngine<'ctx>,
    pub(crate) variables: Vec<HashMap<String, (BasicValueEnum<'ctx>, Type, u8)>>,
    pub(crate) struct_types: HashMap<String, StructType<'ctx>>,
    pub(crate) struct_defs: HashMap<String, StructDef>,
    pub(crate) enum_types: HashMap<String, StructType<'ctx>>,
    pub(crate) enum_defs: HashMap<String, EnumDef>,
    pub(crate) generic_fn_defs: HashMap<String, FunctionDef>,
    pub(crate) generic_impls: HashMap<String, Vec<ImplBlock>>,
    pub(crate) fn_entry_scope_depth: usize,
    pub(crate) builtin_manager: expr::BuiltinManager,
    pub(crate) instance_methods: HashMap<String, expr::InstanceMethodManager>,
    pub(crate) static_methods: HashMap<String, expr::StaticMethodManager>,
    pub(crate) destructors: HashMap<String, String>, // TypeName -> FreeFnName
    pub(crate) method_return_types: HashMap<String, Type>, // MangledName -> ReturnType
    pub(crate) loop_stack: Vec<(
        inkwell::basic_block::BasicBlock<'ctx>,
        inkwell::basic_block::BasicBlock<'ctx>,
        usize,
    )>,
    pub(crate) relations: std::collections::HashSet<String>,
    pub(crate) current_span: Option<crate::compiler::error::Span>,
    pub(crate) function_analysis: Option<crate::compiler::liveness::FunctionAnalysis>,
    pub(crate) current_sret_dest: Option<inkwell::values::PointerValue<'ctx>>,
    pub(crate) temporaries: Vec<Vec<(BasicValueEnum<'ctx>, Type, u8)>>,
    pub(crate) variable_liveness: Vec<HashMap<String, usize>>, // Parallel to variables: Last Use Time
    pub(crate) current_time: usize,
    pub(crate) type_manager: type_manager::TypeManager,
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
            struct_types: HashMap::new(),
            struct_defs: HashMap::new(),
            enum_types: HashMap::new(),
            enum_defs: HashMap::new(),
            generic_fn_defs: HashMap::new(),
            generic_impls: HashMap::new(),
            fn_entry_scope_depth: 0,
            builtin_manager: expr::BuiltinManager::new(),
            instance_methods: HashMap::new(),
            static_methods: HashMap::new(),
            destructors: HashMap::new(),
            loop_stack: Vec::new(),
            method_return_types: HashMap::new(),
            relations: std::collections::HashSet::new(),
            current_span: None,
            function_analysis: None,
            current_sret_dest: None,
            temporaries: vec![Vec::new()],
            variable_liveness: vec![HashMap::new()],
            current_time: 0,
            type_manager: type_manager::TypeManager::new(),
        };

        // Register all methods (instance and static)
        codegen.register_all_methods();

        // Register builtins (Enums, etc.)
        codegen.register_builtins();
        // Load builtins via TypeManager (Option)
        let option_data = builtin_types::option::load_option_data();
        codegen.type_manager.register_builtin(option_data.clone());
        
        // Propagate ASTs to CodeGen maps (Temporary Bridge)
        if let Some(def) = option_data.enum_def {
            codegen.enum_defs.insert(def.name.clone(), def.clone());
        }
        codegen.generic_impls.entry("Option".to_string()).or_default().extend(option_data.impl_blocks);

        // Load builtins via TypeManager (Result)
        let result_data = builtin_types::result::load_result_data();
        codegen.type_manager.register_builtin(result_data.clone());
        
        // Propagate ASTs to CodeGen maps (Temporary Bridge)
        if let Some(def) = result_data.enum_def {
            codegen.enum_defs.insert(def.name.clone(), def.clone());
        }
        codegen.generic_impls.entry("Result".to_string()).or_default().extend(result_data.impl_blocks);

        let builtin_enums = vec![
            codegen.enum_defs.get("Option").unwrap().clone(),
            codegen.enum_defs.get("Result").unwrap().clone(),
        ];
        for def in &builtin_enums {
            codegen.enum_defs.insert(def.name.clone(), def.clone());
        }
        codegen.compile_enum_defs(&builtin_enums).unwrap();
        
        // Load builtins via TypeManager (Vec)
        let vec_data = builtin_types::vec::load_vec_data();
        codegen.type_manager.register_builtin(vec_data.clone());
        
        if let Some(def) = vec_data.struct_def {
            codegen.struct_defs.insert(def.name.clone(), def);
        }
        codegen.generic_impls.entry("Vec".to_string()).or_default().extend(vec_data.impl_blocks);

        // Load builtins via TypeManager (HashMap)
        let hashmap_data = builtin_types::hashmap::load_hashmap_data();
        codegen.type_manager.register_builtin(hashmap_data.clone());
        
        if let Some(def) = hashmap_data.struct_def {
            codegen.struct_defs.insert(def.name.clone(), def);
        }
        codegen.generic_impls.entry("HashMap".to_string()).or_default().extend(hashmap_data.impl_blocks);

        // Remove legacy builtin_impls calls (Fully Migrated)
        // builtin_impls::register_builtin_structs(&mut codegen.struct_defs);
        // builtin_impls::register_builtin_impls(&mut codegen.generic_impls);
        
        // Compile the struct defs we just added
        let vec_defs = codegen.struct_defs.values().cloned().collect::<Vec<_>>();
        codegen.compile_struct_defs(&vec_defs).unwrap(); // This registers LLVM types.

        // Delegate to runtime module
        builtins::declare_runtime_functions(
            codegen.context,
            &codegen.module,
            &codegen.execution_engine,
        );

        codegen.register_builtin_return_types();

        codegen
    }

    fn register_builtin_return_types(&mut self) {
        // Path
        self.method_return_types.insert("tl_path_new".to_string(), Type::UserDefined("Path".to_string(), vec![]));
        self.method_return_types.insert("tl_path_to_string".to_string(), Type::UserDefined("String".to_string(), vec![]));
        self.method_return_types.insert("tl_path_join".to_string(), Type::UserDefined("Path".to_string(), vec![]));
        
        // String
        self.method_return_types.insert("tl_string_new".to_string(), Type::UserDefined("String".to_string(), vec![]));

        // File
        self.method_return_types.insert("tl_file_open".to_string(), Type::UserDefined("File".to_string(), vec![]));
        self.method_return_types.insert("tl_file_read_string".to_string(), Type::UserDefined("String".to_string(), vec![]));
        self.method_return_types.insert("tl_file_write_string".to_string(), Type::Void);
        self.method_return_types.insert("tl_file_close".to_string(), Type::Void);

        // Env
        self.method_return_types.insert("tl_env_get".to_string(), Type::UserDefined("String".to_string(), vec![]));
        self.method_return_types.insert("tl_env_set".to_string(), Type::Void);
        
        // Map / HashMap (treated as structs but return pointers, so we must register them to avoid Tensor default)
        self.method_return_types.insert("tl_tensor_map_new".to_string(), Type::Struct("Map".to_string(), vec![]));
        self.method_return_types.insert("tl_hashmap_new".to_string(), Type::Struct("HashMap".to_string(), vec![]));
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

    pub fn emit_llvm_file(&self, path: &std::path::Path) -> Result<(), String> {
        self.module
            .print_to_file(path)
            .map_err(|e| e.to_string())
    }

    pub(crate) fn push_temp_scope(&mut self) {
        self.temporaries.push(Vec::new());
    }

    #[allow(dead_code)]
    pub(crate) fn pop_temp_scope(&mut self) -> Result<(), String> {
        let temps = self.temporaries.pop().expect("Temporary stack underflow");
        for (val, ty, cleanup) in temps {
            if cleanup != CLEANUP_NONE {
                self.emit_recursive_free(val, &ty, cleanup)?;
            }
        }
        Ok(())
    }

    pub(crate) fn add_temp(&mut self, val: BasicValueEnum<'ctx>, ty: Type) {
        self.add_temp_with_mode(val, ty, CLEANUP_FULL);
    }

    pub(crate) fn add_temp_with_mode(&mut self, val: BasicValueEnum<'ctx>, ty: Type, mode: u8) {
        // Only track types that need freeing
        match &ty {
            Type::Tensor(_, _) | Type::TensorShaped(_, _) | Type::Struct(_, _) | Type::UserDefined(_, _) | Type::Vec(_) | Type::Tuple(_) | Type::Enum(_, _) => {
                 self.temporaries.last_mut().expect("No temporary context").push((val, ty, mode));
            }
            _ => {}
        }
    }

    pub(crate) fn consume_temp(&mut self, val: BasicValueEnum<'ctx>) {
         let ptr = if val.is_pointer_value() {
             val.into_pointer_value()
         } else {
             return; 
         };

         // Search from innermost scope to outermost
         for scope in self.temporaries.iter_mut().rev() {
             if let Some(pos) = scope.iter().position(|(v, _, _)| v.is_pointer_value() && v.into_pointer_value() == ptr) {
                 scope.remove(pos);
                 return;
             }
         }
    }

    /// Like consume_temp but returns true if a temporary was found and removed.
    pub(crate) fn try_consume_temp(&mut self, val: BasicValueEnum<'ctx>) -> bool {
         if !val.is_pointer_value() { return false; }
         let ptr = val.into_pointer_value();

         // Search from innermost scope to outermost
         for scope in self.temporaries.iter_mut().rev() {
             if let Some(pos) = scope.iter().position(|(v, _, _)| v.is_pointer_value() && v.into_pointer_value() == ptr) {
                 scope.remove(pos);
                 return true;
             }
         }
         false
    }

    pub(crate) fn mark_temp_no_cleanup(&mut self, val: BasicValueEnum<'ctx>) {
         let ptr = if val.is_pointer_value() {
             val.into_pointer_value()
         } else {
             return; 
         };

         // Search from innermost scope to outermost
         for scope in self.temporaries.iter_mut().rev() {
             for (v, _, cleanup) in scope.iter_mut() {
                 if v.is_pointer_value() && v.into_pointer_value() == ptr {
                     *cleanup = CLEANUP_NONE;
                     // We don't return here because there might be duplicates? 
                     // Usually duplicates refer to same value. Safer to mark all.
                 }
             }
         }
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
        context_msg: &str,
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

        // LOG ALLOC (Trace all tensor results from runtime)
        let f_name = "tl_log_alloc";
        if let Some(f) = self.module.get_function(f_name) {
             let size_val = self.context.i64_type().const_int(0, false); // Unknown size from return val
             
             // Use file if available, otherwise context_msg (function name)
             let file_str = if let Some(f) = file {
                 f
             } else {
                 context_msg
             };
             
             // Note: build_global_string_ptr might create duplicates, but LLVM merges constants usually.
             if let Ok(file_ptr_val) = self.builder.build_global_string_ptr(file_str, "log_file") {
                 let file_ptr = file_ptr_val.as_pointer_value();
                 let i32_type = self.context.i32_type();
                 let line_val = i32_type.const_int(line as u64, false);
                 
                 let cast_ptr = self.builder.build_pointer_cast(ptr_val, self.context.ptr_type(inkwell::AddressSpace::default()), "cast_log").unwrap();
    
                 self.builder.build_call(f, &[cast_ptr.into(), size_val.into(), file_ptr.into(), line_val.into()], "").ok();
             }
        }

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
        // Register destructors
        self.destructors.insert("HashMap".to_string(), "tl_hashmap_free".to_string());





        // Register Vec Type - Migrated to BuiltinTypeData (see new() logic)
        // Manual registration removed to support AST-based extern dispatch.



        // Register LLM Types (Tokenizer, KVCache)
        builtin_types::llm::register_llm_types(&mut self.type_manager);

        // Register System Type (Static Methods only)
        let mut system_type = type_manager::CodeGenType::new("System");
        system_type.register_static_method("memory_mb", expr::StaticMethod::Evaluated(expr::compile_system_memory_mb));
        system_type.register_static_method("metal_pool_bytes", expr::StaticMethod::Evaluated(expr::compile_system_metal_pool_bytes));
        system_type.register_static_method("metal_pool_mb", expr::StaticMethod::Evaluated(expr::compile_system_metal_pool_mb));
        system_type.register_static_method("metal_pool_count", expr::StaticMethod::Evaluated(expr::compile_system_metal_pool_count));
        system_type.register_static_method("metal_sync", expr::StaticMethod::Evaluated(expr::compile_system_metal_sync));
        system_type.register_static_method("pool_count", expr::StaticMethod::Evaluated(expr::compile_system_pool_count));
        system_type.register_static_method("refcount_count", expr::StaticMethod::Evaluated(expr::compile_system_refcount_count));
        system_type.register_static_method("scope_depth", expr::StaticMethod::Evaluated(expr::compile_system_scope_depth));
        self.type_manager.register_type(system_type);

        // Register IO Types (File, Path, Env, Http)
        builtin_types::io::register_io_types(&mut self.type_manager);
        builtin_types::system::register_system_types(&mut self.type_manager);
        builtin_types::tensor::register_tensor_types(&mut self.type_manager);

        builtin_types::llm::register_llm_types(&mut self.type_manager);
        // builtin_types::result::register_result_types(&mut self.type_manager);
        builtin_types::string::register_string_types(&mut self.type_manager);



        let device_enum = EnumDef {
            name: "Device".to_string(),
            generics: vec![],
            variants: vec![
                VariantDef {
                    name: "Auto".to_string(),
                    kind: VariantKind::Unit,
                },
                VariantDef {
                    name: "Cpu".to_string(),
                    kind: VariantKind::Unit,
                },
                VariantDef {
                    name: "Metal".to_string(),
                    kind: VariantKind::Unit,
                },
                VariantDef {
                    name: "Cuda".to_string(),
                    kind: VariantKind::Unit,
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
        self.variable_liveness.push(std::collections::HashMap::new());
        self.push_temp_scope(); // Track temporaries for this scope
        if let Some(f) = self.module.get_function("tl_mem_enter_scope") {
            self.builder.build_call(f, &[], "").unwrap();
        }
    }

    // Helper to generate free calls for variables in a specific scope index
    fn emit_cleanup_vars_in_scope(&self, scope_idx: usize) {
        if let Some(scope) = self.variables.get(scope_idx) {
            for (_name, (val_enum, ty, cleanup_mode)) in scope {
                if *cleanup_mode != CLEANUP_NONE {
                    if let Type::UserDefined(name, _) = ty {
                        let ptr = val_enum.into_pointer_value();
                        let load_type = self.context.ptr_type(inkwell::AddressSpace::default());
                        if let Ok(obj_val) = self.builder.build_load(load_type, ptr, "obj_to_free")
                        {
                            match name.as_str() {
                                "String" => {}
                                "File" | "Path" => {}
                                "Env" | "Http" => {}
                                "Map" | "HashMap" | "Tokenizer" | "KVCache" | 
                                "Block" | "RMSNorm" | "Attention" | "MLP" => {}
                                _ => {
                                    // Pass cleanup_mode to recursive free
                                    let _ = self.emit_recursive_free(obj_val, ty, *cleanup_mode);
                                }
                            }
                        }
                    } else if let Type::Struct(name, _) = ty {
                        // Structs in TL now follow "Reference Semantics" for their members.
                        // We do NOT recursively free members when the Struct itself goes out of scope.
                        // This prevents Double Free issues when Structs are copied (shallow copy).
                        // The Struct wrapper itself (if any) is allocated on stack/alloca so it's fine.
                        // Members (Tensors, Maps) are ref-counted or managed elsewhere (or leaked safely).
                        
                        // However, Struct("Tensor") MUST be freed because it holds a handle.
                        if name == "Tensor" {
                            let ptr = val_enum.into_pointer_value();
                            let load_type = self.context.ptr_type(inkwell::AddressSpace::default());
                            if let Ok(struct_val) =
                                self.builder.build_load(load_type, ptr, "tensor_to_free")
                            {
                                let _ = self.emit_recursive_free(struct_val, ty, *cleanup_mode);
                            }
                        }
                    } else if matches!(ty, Type::Tensor(_, _) | Type::TensorShaped(_, _) | Type::Tuple(_) | Type::Vec(_)) {
                         // Tuple and Vec also need loading from Alloca
                        let ptr = val_enum.into_pointer_value();
                        let load_type = self.context.ptr_type(inkwell::AddressSpace::default());
                        if let Ok(val) =
                            self.builder.build_load(load_type, ptr, "val_to_free")
                        {
                            let _ = self.emit_recursive_free(val, ty, *cleanup_mode);
                        }
                    }
                }
            }
        }

        // Cleanup temporaries in this scope
        if let Some(temps) = self.temporaries.get(scope_idx) {
            for (val, ty, cleanup) in temps {
                // Temporaries are always owned (CLEANUP_FULL) unless marked otherwise.
                // They are values (pointers), not allocas.
                if *cleanup != CLEANUP_NONE {
                    let _ = self.emit_recursive_free(*val, ty, *cleanup);
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

    // Emit cleanup for all scopes down to (but not including) target_depth.
    pub(crate) fn emit_cleanup_to_depth(&self, target_depth: usize) {
        if let Some(f) = self.module.get_function("tl_mem_exit_scope") {
            let mut idx = self.variables.len();
            while idx > target_depth {
                idx -= 1;
                self.emit_cleanup_vars_in_scope(idx);
                self.builder.build_call(f, &[], "").unwrap();
            }
        }
    }

    pub(crate) fn emit_trace_mem(&self, tag: &str) -> Result<(), String> {
        let f = match self.module.get_function("tl_trace_mem") {
            Some(f) => f,
            None => return Ok(()),
        };
        let (file, line, col) = if let Some(span) = &self.current_span {
            (
                span.file.as_deref().unwrap_or("unknown"),
                span.line as u32,
                span.column as u32,
            )
        } else {
            ("unknown", 0, 0)
        };
        let file_ptr = self
            .builder
            .build_global_string_ptr(file, "trace_file")
            .map_err(|e| e.to_string())?
            .as_pointer_value();
        let tag_ptr = self
            .builder
            .build_global_string_ptr(tag, "trace_tag")
            .map_err(|e| e.to_string())?
            .as_pointer_value();
        let i32_type = self.context.i32_type();
        self.builder
            .build_call(
                f,
                &[
                    file_ptr.into(),
                    i32_type.const_int(line as u64, false).into(),
                    i32_type.const_int(col as u64, false).into(),
                    tag_ptr.into(),
                ],
                "",
            )
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    #[allow(dead_code)]
    pub(crate) fn emit_log_alloc(&self, ptr: inkwell::values::BasicValueEnum<'ctx>, size: inkwell::values::IntValue<'ctx>) -> Result<(), String> {
        // tl_log_alloc(ptr, size, file, line)
        let f_name = "tl_log_alloc";
        let f = if let Some(f) = self.module.get_function(f_name) {
             f
        } else {
             let void_ty = self.context.void_type();
             let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
             let i64_ty = self.context.i64_type();
             let i32_ty = self.context.i32_type();
             // ptr, size, file, line
             let ft = void_ty.fn_type(&[ptr_ty.into(), i64_ty.into(), ptr_ty.into(), i32_ty.into()], false);
             self.module.add_function(f_name, ft, None)
        };

        let (file, line, _) = if let Some(span) = &self.current_span {
            (
                span.file.as_deref().unwrap_or("unknown"),
                span.line as u32,
                span.column as u32,
            )
        } else {
            ("unknown", 0, 0)
        };
        
        let file_ptr = self
            .builder
            .build_global_string_ptr(file, "log_file")
            .map_err(|e| e.to_string())?
            .as_pointer_value();
        
        let i32_type = self.context.i32_type();
        let ptr_val = if ptr.is_pointer_value() {
            ptr.into_pointer_value()
        } else {
             // Cast to void*? Or error? Just assume correct usage.
             // If not pointer, maybe cast int to ptr? No, memory alloc always returns ptr.
             return Err("emit_log_alloc expects pointer".into());
        };
        
        let cast_ptr = self.builder.build_pointer_cast(ptr_val, self.context.ptr_type(inkwell::AddressSpace::default()), "cast_log").unwrap();

        self.builder.build_call(
            f,
            &[
                cast_ptr.into(),
                size.into(),
                file_ptr.into(),
                i32_type.const_int(line as u64, false).into(),
            ],
            ""
        ).map_err(|e| e.to_string())?;
        
        Ok(())
    }

    pub(crate) fn emit_log_free(&self, ptr: inkwell::values::BasicValueEnum<'ctx>) -> Result<(), String> {
        // tl_log_free(ptr, file, line)
        let f_name = "tl_log_free";
        let f = if let Some(f) = self.module.get_function(f_name) {
             f
        } else {
             let void_ty = self.context.void_type();
             let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
             let i32_ty = self.context.i32_type();
             // ptr, file, line
             let ft = void_ty.fn_type(&[ptr_ty.into(), ptr_ty.into(), i32_ty.into()], false);
             self.module.add_function(f_name, ft, None)
        };

        let (file, line, _) = if let Some(span) = &self.current_span {
            (
                span.file.as_deref().unwrap_or("unknown"),
                span.line as u32,
                span.column as u32,
            )
        } else {
            ("unknown", 0, 0)
        };
        
        let file_ptr = self
            .builder
            .build_global_string_ptr(file, "log_file")
            .map_err(|e| e.to_string())?
            .as_pointer_value();
        
        let i32_type = self.context.i32_type();
        let ptr_val = if ptr.is_pointer_value() {
            ptr.into_pointer_value()
        } else {
             return Err("emit_log_free expects pointer".into());
        };
        let cast_ptr = self.builder.build_pointer_cast(ptr_val, self.context.ptr_type(inkwell::AddressSpace::default()), "cast_log").unwrap();

        self.builder.build_call(
            f,
            &[
                cast_ptr.into(),
                file_ptr.into(),
                i32_type.const_int(line as u64, false).into(),
            ],
            ""
        ).map_err(|e| e.to_string())?;
        
        Ok(())
    }

    // Exit the current scope
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
        self.variable_liveness.pop();
        // Just pop the temporaries stack (cleanup code was emitted by emit_top_scope_cleanup if needed)
        self.temporaries.pop();
    }

    /// Null out a variable (Move Semantics) so it won't be double-freed
    #[allow(dead_code)]
    pub(crate) fn null_out_variable(&self, name: &str) -> Result<(), String> {
        for scope in self.variables.iter().rev() {
            if let Some((val, ty, _should_free)) = scope.get(name) {
                // Only for types that would be freed recursively
                if matches!(
                    ty,
                    Type::Tensor(_, _) | Type::Struct(_, _) | Type::UserDefined(_, _)
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
            // 1. Skip empty specializations generated by TypeChecker/Semantics
            // These usually have names like Vec_i64 but no fields. We rely on generic fallback (Vec) or monomorphization to generate them correctly.
            if (s.name.starts_with("Vec") || s.name.starts_with("HashMap") || s.name.starts_with("Map")) && s.fields.is_empty() {
                 continue;
            }

            // 2. Skip overwriting existing valid (non-empty) definitions with empty ones
            if let Some(existing) = self.struct_defs.get(&s.name) {
                if !existing.fields.is_empty() && s.fields.is_empty() {
                    continue;
                }
            }
            
            // 3. Explicit protection removed - Logic 2 covers the corruption case, and we need to allow initialization to proceed.
            // If we block here, CodeGenerator::new fails to register LLVM types because struct_defs is already populated.


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
                    Type::F64 => self.context.f64_type().into(),
                    Type::I64 | Type::Entity => self.context.i64_type().into(),
                    Type::Bool => self.context.bool_type().into(),
                    Type::Tensor(_, _) => self
                        .context
                        .ptr_type(inkwell::AddressSpace::default())
                        .into(), // OpaqueTensor*
                    Type::Struct(name, _) | Type::UserDefined(name, _) => {
                        // Extract simple name from module path (e.g., "mnist_common::Linear" -> "Linear")
                        let simple_name = if name.contains("::") {
                            name.split("::").last().unwrap()
                        } else {
                            name.as_str()
                        };
                        if self.struct_types.contains_key(simple_name) || name == "String" {
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
                let fields_iter: Box<dyn Iterator<Item = &Type>> = match &v.kind {
                     VariantKind::Unit => Box::new(std::iter::empty()),
                     VariantKind::Tuple(types) => Box::new(types.iter()),
                     VariantKind::Struct(fields) => Box::new(fields.iter().map(|(_, t)| t)),
                };

                for ty in fields_iter {
                    let field_llvm_ty = match ty {
                        Type::F32 => self.context.f32_type().into(),
                        Type::F64 => self.context.f64_type().into(),
                        Type::I64 | Type::Entity => self.context.i64_type().into(),
                        Type::Bool => self.context.bool_type().into(),
                        Type::Tensor(_, _) => self
                            .context
                            .ptr_type(inkwell::AddressSpace::default())
                            .into(),
                        Type::Struct(_, _) | Type::Enum(_, _) | Type::UserDefined(_, _) => {
                            // Objects are pointers
                            self.context
                                .ptr_type(inkwell::AddressSpace::default())
                                .into()
                        }
                        Type::Vec(_) => self
                            .context
                            .ptr_type(inkwell::AddressSpace::default())
                            .into(),
                        Type::Tuple(_) => self
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
            // Check if generic impl
            if !imp.generics.is_empty() {
                let target_name = imp.target_type.get_base_name();
                eprintln!("DEBUG: compile_impl_blocks sees impl for {}", target_name);
                self.generic_impls.entry(target_name).or_default().push(imp.clone());
                continue;
            }

            for method in &imp.methods {
                let target_name = imp.target_type.get_base_name();
                let simple_target = if target_name.contains("::") {
                    target_name.split("::").last().unwrap()
                } else {
                    &target_name
                };
                let mangled_name = if method.is_extern {
                    format!("tl_{}_{}", simple_target.to_lowercase(), method.name)
                } else {
                    format!("tl_{}_{}", simple_target, method.name)
                };

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
                    let resolved_ty = if let Type::UserDefined(name, _) = arg_ty {
                        if name == "Self" {
                            imp.target_type.clone()
                        } else {
                            arg_ty.clone()
                        }
                    } else {
                        arg_ty.clone()
                    };

                    let ty: BasicMetadataTypeEnum = match &resolved_ty {
                        Type::F32 => self.context.f32_type().into(),
                        Type::I64 | Type::Entity | Type::Usize => self.context.i64_type().into(),
                        Type::Bool => self.context.bool_type().into(),
                        Type::Tensor(_, _) => self
                            .context
                            .ptr_type(inkwell::AddressSpace::default())
                            .into(),
                        Type::Struct(_, _) | Type::UserDefined(_, _) => self
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
                        Type::I64 | Type::Entity | Type::Usize => {
                            self.context.i64_type().fn_type(&param_types, false)
                        }
                        Type::Bool => self.context.bool_type().fn_type(&param_types, false),
                        Type::Void => self.context.void_type().fn_type(&param_types, false),
                        Type::Tensor(_, _) => self
                            .context
                            .ptr_type(inkwell::AddressSpace::default())
                            .fn_type(&param_types, false),
                        Type::Struct(_, _) | Type::UserDefined(_, _) | Type::Tuple(_) | Type::Enum(_, _) | Type::Vec(_) => self
                            .context
                            .ptr_type(inkwell::AddressSpace::default())
                            .fn_type(&param_types, false),
                        _ => self.context.void_type().fn_type(&param_types, false),
                    }
                };

                let _function = self.module.add_function(&mangled_name, fn_type, None);
                
                // Register return type for this method
                let return_type = if let Type::UserDefined(n, _) = &method.return_type {
                    if n == "Self" {
                        imp.target_type.clone()
                    } else {
                        method.return_type.clone()
                    }
                } else {
                    method.return_type.clone()
                };
                self.method_return_types.insert(mangled_name.clone(), return_type);
            }
        }

        // Pass 2: Compile Bodies
        // Pass 2: Compile Bodies
        for imp in impls {
            if !imp.generics.is_empty() {
                continue;
            }
            for method in &imp.methods {
                let target_name = imp.target_type.get_base_name();
                let simple_target = if target_name.contains("::") {
                    target_name.split("::").last().unwrap()
                } else {
                    &target_name
                };
                let mangled_name = if method.is_extern {
                     format!("tl_{}_{}", simple_target.to_lowercase(), method.name)
                } else {
                    format!("tl_{}_{}", simple_target, method.name)
                };

                if method.is_extern {
                    continue;
                }
                let function = self
                    .module
                    .get_function(&mangled_name)
                    .ok_or(format!("Function {} not found", mangled_name))?;

                // Compile Body
                let entry = self.context.append_basic_block(function, "entry");
                self.builder.position_at_end(entry);

                // FIX: Must setup function frame for methods too!
                // Liveness Analysis
                self.function_analysis = Some(crate::compiler::liveness::LivenessAnalyzer::analyze(method));
                let num_slots = self.function_analysis.as_ref().map(|a| a.num_slots).unwrap_or(0);
                
                if let Some(enter_fn) = self.module.get_function("tl_mem_function_enter") {
                     self.builder.build_call(
                         enter_fn,
                         &[self.context.i64_type().const_int(num_slots as u64, false).into()],
                         ""
                     ).unwrap();
                } else {
                    // Declare it if missing
                    let i64_type = self.context.i64_type();
                    let fn_type = self.context.void_type().fn_type(&[i64_type.into()], false);
                    let enter_fn = self.module.add_function("tl_mem_function_enter", fn_type, None);
                    self.builder.build_call(
                         enter_fn,
                         &[self.context.i64_type().const_int(num_slots as u64, false).into()],
                         ""
                     ).unwrap();
                }

                self.fn_entry_scope_depth = self.variables.len();
                self.enter_scope();

                // Check if this method uses sret
                let uses_sret = false; /* SRET DISABLED */
                let param_offset = if uses_sret { 1 } else { 0 };

                // Get params and store them (skip sret param if present)
                for (i, (arg_name, arg_ty)) in method.args.iter().enumerate() {
                    let resolved_ty = if let Type::UserDefined(name, _) = arg_ty {
                        if name == "Self" {
                            imp.target_type.clone()
                        } else {
                            arg_ty.clone()
                        }
                    } else {
                        arg_ty.clone()
                    };

                    if let Some(param_val) = function.get_nth_param((i + param_offset) as u32) {
                        param_val.set_name(arg_name);
                        let alloca =
                            self.create_entry_block_alloca(function, arg_name, &resolved_ty)?;
                        self.builder.build_store(alloca, param_val).unwrap();

                        // Register in scope
                        self.variables
                            .last_mut()
                            .unwrap()
                            .insert(arg_name.clone(), (alloca.into(), resolved_ty, CLEANUP_NONE));
                    }
                }

                for (i, stmt) in method.body.iter().enumerate() {
                    if i == method.body.len() - 1 && method.return_type != Type::Void {
                        // Check if it's an expression that should be returned
                        if let StmtKind::Expr(expr) = &stmt.inner {
                            let (val, ty) = self.compile_expr(expr)?;
                            self.emit_recursive_unregister(val, &ty)?;
                            self.emit_all_scopes_cleanup();
                            self.variables.pop();
                            self.builder
                                .build_return(Some(&val))
                                .map_err(|e| e.to_string())?;
                            continue;
                        }
                    }
                    self.compile_stmt(stmt)?;
                }

                self.exit_scope(); // End method scope

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

    pub fn compile_module(&mut self, ast_module: &Module, module_name: &str) -> Result<(), String> {


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
            // If function is generic, skip compilation and store in registry
            if !func.generics.is_empty() {
                self.generic_fn_defs.insert(func.name.clone(), func.clone());
                continue;
            }

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
            eprintln!("DEBUG: Compiling synthetic main proto");
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
            self.module.print_to_stderr();
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

    pub(crate) fn compile_fn_proto(&mut self, func: &FunctionDef) -> Result<FunctionValue<'ctx>, String> {

        // Check if this function returns a struct (requires sret)
        // Check if this function returns a struct (requires sret)
        // Check if this function        // matches!(func.return_type, Type::Struct(_, _) | Type::UserDefined(_, _))
        // Let's assume Structs needs SRET, but Tensors do NOT.
        // String is a pointer, so exclusion is needed.
        let uses_sret = match &func.return_type {
            Type::Struct(name, _) if name != "Vec" && name != "Map" && name != "HashMap" => true,
            Type::UserDefined(name, _) if name != "String" && name != "Vec" && name != "Map" && name != "HashMap" && name != "Path" && name != "File" => true, 
            _ => false,
        };

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
                Type::I64 | Type::Entity | Type::Usize => self.context.i64_type().into(),
                Type::F32 => self.context.f32_type().into(),
                Type::Bool => self.context.bool_type().into(),
                Type::Tensor(_, _) => self
                    .context
                    .ptr_type(inkwell::AddressSpace::default())
                    .into(),
                Type::UserDefined(_, _) | Type::Struct(_, _) | Type::Enum(_, _) => self
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
                Type::I64 | Type::Entity | Type::Usize => Some(self.context.i64_type().into()),
                Type::F32 => Some(self.context.f32_type().into()),
                Type::Bool => Some(self.context.bool_type().into()),
                Type::Tensor(_, _) => Some(
                    self.context
                        .ptr_type(inkwell::AddressSpace::default())
                        .into(),
                ),
                Type::Struct(_, _) | Type::UserDefined(_, _) | Type::Enum(_, _) => Some(
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

        // Register return type for this function
        self.method_return_types.insert(func.name.clone(), func.return_type.clone());

        Ok(val)
    }

    pub(crate) fn compile_fn(&mut self, func: &FunctionDef, extra_stmts: &[Stmt]) -> Result<(), String> {
        let function = self
            .module
            .get_function(&func.name)
            .ok_or("Function not found")?;

        // Run Liveness Analysis
        self.function_analysis = Some(crate::compiler::liveness::LivenessAnalyzer::analyze(func));

        // Initialize entry block
        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);

        // SETUP FUNCTION FRAME (SLOTS)
        let num_slots = self.function_analysis.as_ref().map(|a| a.num_slots).unwrap_or(0);
        
        if let Some(enter_fn) = self.module.get_function("tl_mem_function_enter") {
             self.builder.build_call(
                 enter_fn,
                 &[self.context.i64_type().const_int(num_slots as u64, false).into()],
                 ""
             ).unwrap();
        } else {
            // Declare it if missing
            let i64_type = self.context.i64_type();
            let fn_type = self.context.void_type().fn_type(&[i64_type.into()], false);
            let enter_fn = self.module.add_function("tl_mem_function_enter", fn_type, None);
            self.builder.build_call(
                 enter_fn,
                 &[self.context.i64_type().const_int(num_slots as u64, false).into()],
                 ""
             ).unwrap();
        }

        // Push a new scope for function arguments
        self.fn_entry_scope_depth = self.variables.len();
        self.enter_scope(); // Function scope


        // Check if this function uses sret
        let uses_sret = match &func.return_type {
             Type::Struct(name, _) if name != "Vec" && name != "Map" && name != "HashMap" => true,
             Type::UserDefined(name, _) if name != "String" && name != "Vec" && name != "Map" && name != "HashMap" => true,
             _ => false,
        };
        let param_offset = if uses_sret { 1 } else { 0 };

        if uses_sret {
             if let Some(param) = function.get_nth_param(0) {
                 self.current_sret_dest = Some(param.into_pointer_value());
             }
        } else {
             self.current_sret_dest = None;
        }

        // Register arguments (skip sret param if present)
        for (i, arg) in function.get_param_iter().skip(param_offset).enumerate() {
            let (arg_name, arg_type) = &func.args[i];
            let alloca = self.create_entry_block_alloca(function, arg_name, arg_type)?;

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
                _ => return Err("Unsupported arg type".to_string()),
            };

            // Insert into current scope with should_free=FALSE
            // Arguments are BORROWED. Function must NOT free them on exit.
            self.variables
                .last_mut()
                .unwrap()
                .insert(arg_name.clone(), (alloca.into(), arg_type.clone(), CLEANUP_NONE));
        }

        // Initialize Arena in main if needed
        if func.name == "main" {
            // Logic Engine Init - MUST be before anything else
            if let Some(init_kb) = self.module.get_function("_tl_init_kb") {
                self.builder.build_call(init_kb, &[], "").unwrap();
            }
            // Execute infer to ensure queries work inside main
            if let Some(infer_fn) = self.module.get_function("tl_kb_infer") {
                self.builder.build_call(infer_fn, &[], "").unwrap();
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

                    if uses_sret {
                         // SRET Logic
                         if let Some(dest) = self.current_sret_dest {
                             let src_ptr = val.into_pointer_value();
                             self.emit_struct_copy(dest, src_ptr, &ty)?;
                         }
                         self.emit_all_scopes_cleanup();
                         self.variables.pop(); // Compiler scope cleanup
                         
                         // Calls to function exit must happen before return
                         if let Some(exit_fn) = self.module.get_function("tl_mem_function_exit") {
                              self.builder.build_call(exit_fn, &[], "").unwrap();
                         }

                         self.builder.build_return(None).map_err(|e| e.to_string())?;
                    } else {
                        // IMPORTANT: Unregister return value (same as StmtKind::Return)
                        // If not SRET, we assume returning by value (or pointer ownership transfer)
                        // Note: For Tensors, "Returning by Value" means returning the pointer.
                        // And we must unregister it from scope so it doesn't get freed.
                        self.emit_recursive_unregister(val, &ty)?;

                        self.emit_all_scopes_cleanup();

                        // CRITICAL FIX: Pop the function scope from variables stack
                        self.variables.pop();
                        
                        // Call function exit helper BEFORE return
                        if let Some(exit_fn) = self.module.get_function("tl_mem_function_exit") {
                             self.builder.build_call(exit_fn, &[], "").unwrap();
                        }

                        self.builder
                            .build_return(Some(&val))
                            .map_err(|e| e.to_string())?;
                    }
                    return Ok(()); // RETURN EARLY - Function is done
                }
            }
            self.compile_stmt(stmt)?;
        }

        self.exit_scope(); // End function scope

        // CLEANUP FUNCTION FRAME
        // Only if block is NOT terminated (e.g. no return statement at flow end)
        if self.builder.get_insert_block().unwrap().get_terminator().is_none() {
             if let Some(exit_fn) = self.module.get_function("tl_mem_function_exit") {
                  self.builder.build_call(exit_fn, &[], "").unwrap();
             }

             // Add implicit return void if needed (not perfect but ok for now)
             if func.return_type == Type::Void {
                 self.builder.build_return(None).map_err(|e| e.to_string())?;
             }
        }


        if !function.verify(true) {
            log::error!("=== LLVM VERIFICATION FAILED FOR: {} ===", func.name);
            // function.verify(true) should print the error to stderr
            function.print_to_stderr();
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
        for _relation in relations {
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
        // Runtime function tl_ptr_inc_ref
        let _tl_ptr_inc_ref_fn = self
            .module
            .get_function("tl_ptr_inc_ref")
            .or_else(|| {
                let void_ty = self.context.void_type();
                let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
                let ft = void_ty.fn_type(&[ptr_ty.into()], false);
                Some(self.module.add_function("tl_ptr_inc_ref", ft, None))
            })
            .ok_or("tl_ptr_inc_ref decl failed")?;

        for rel in relations {
            let func_name = &rel.name;
            // Args: mask (i64), arg1 (i64), arg2 (i64)...
            let mut arg_types = vec![i64_type.into()]; // mask
            for _ in 0..rel.args.len() {
                arg_types.push(i64_type.into());
            }
            // Tags pointer (as ALLOCATED INT for safe passing)
            arg_types.push(i64_type.into());

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

                // Get tags passed from call-site
                let tags_arg = function.get_nth_param((num_args + 1) as u32).unwrap();

                // Call tl_query
                let result_tensor = match self
                    .builder
                    .build_call(
                        tl_query_fn,
                        &[
                            name_ptr.into(),
                            mask_arg.into(),
                            args_tensor.into(),
                            self.builder.build_int_to_ptr(tags_arg.into_int_value(), self.context.ptr_type(AddressSpace::default()), "tags_ptr").unwrap().into(),
                        ],
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
                let tags_arg = function.get_nth_param(1).unwrap(); // tags is at index 1 for no-arg relation
                let result_tensor = match self
                    .builder
                    .build_call(
                        tl_query_fn,
                        &[
                            name_ptr.into(),
                            mask_arg.into(),
                            null_ptr.into(),
                            self.builder.build_int_to_ptr(tags_arg.into_int_value(), self.context.ptr_type(AddressSpace::default()), "tags_ptr").unwrap().into(),
                        ],
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
