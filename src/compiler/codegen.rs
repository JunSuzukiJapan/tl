// src/compiler/codegen.rs
use crate::compiler::ast::*;
use crate::runtime::{
    tl_print_f32, tl_print_i64, tl_tensor_add, tl_tensor_dim, tl_tensor_get, tl_tensor_get_f32_md,
    tl_tensor_len, tl_tensor_mul, tl_tensor_neg, tl_tensor_new, tl_tensor_print, tl_tensor_slice,
}; // Import runtime functions
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::module::Module as InkwellModule;
// use inkwell::values::Either; // Not used directlyimizationLevel;
use inkwell::values::{BasicValueEnum, FunctionValue, ValueKind};
use inkwell::OptimizationLevel;
use std::collections::HashMap;

pub struct CodeGenerator<'ctx> {
    context: &'ctx Context,
    module: InkwellModule<'ctx>,
    builder: Builder<'ctx>,
    variables: Vec<HashMap<String, (BasicValueEnum<'ctx>, Type)>>, // Stack of scopes
    execution_engine: ExecutionEngine<'ctx>,
    fn_return_types: HashMap<String, Type>,
}

impl<'ctx> CodeGenerator<'ctx> {
    pub fn new(context: &'ctx Context, module_name: &str) -> Self {
        let module = context.create_module(module_name);
        let builder = context.create_builder();
        let execution_engine = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        CodeGenerator {
            context,
            module,
            builder,
            variables: vec![HashMap::new()], // Start with global scope
            execution_engine,
            fn_return_types: HashMap::new(),
        }
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
        let new_type =
            void_ptr.fn_type(&[f32_ptr.into(), i64_type.into(), usize_ptr.into()], false);
        self.module.add_function("tl_tensor_new", new_type, None);

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

        // Map symbols
        if let Some(f) = self.module.get_function("tl_tensor_new") {
            self.execution_engine
                .add_global_mapping(&f, tl_tensor_new as usize);
        }
        if let Some(f) = self.module.get_function("tl_tensor_add") {
            self.execution_engine
                .add_global_mapping(&f, tl_tensor_add as usize);
        }
        if let Some(f) = self.module.get_function("tl_tensor_mul") {
            self.execution_engine
                .add_global_mapping(&f, tl_tensor_mul as usize);
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

    pub fn compile_module(&mut self, ast_module: &Module) -> Result<(), String> {
        // 0. Declare runtime functions
        self.declare_runtime_functions();

        // 1. Declare structs (types) - Placeholder

        // 2. Declare functions (prototypes)
        for func in &ast_module.functions {
            self.compile_fn_proto(func)?;
        }

        // 3. Compile function bodies
        for func in &ast_module.functions {
            self.compile_fn(func)?;
        }

        Ok(())
    }

    fn enter_scope(&mut self) {
        self.variables.push(HashMap::new());
    }

    fn exit_scope(&mut self) {
        self.variables.pop();
    }

    fn lookup_variable(&self, name: &str) -> Option<(BasicValueEnum<'ctx>, Type)> {
        for scope in self.variables.iter().rev() {
            if let Some((v, t)) = scope.get(name) {
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
            Type::Tensor(_, _) => Some(
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

    fn compile_fn(&mut self, func: &FunctionDef) -> Result<(), String> {
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
                .insert(arg_name.clone(), (alloca.into(), arg_type.clone()));
        }

        // Compile body
        for stmt in &func.body {
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

        Ok(())
    }

    fn compile_stmt(&mut self, stmt: &Stmt) -> Result<(), String> {
        match stmt {
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
                    .insert(name.clone(), (alloca.into(), val.1)); // Store pointer and type
            }
            Stmt::Return(expr) => {
                let val = self.compile_expr(expr)?;
                self.builder
                    .build_return(Some(&val.0))
                    .map_err(|e| e.to_string())?;
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
                    if let Some((v, t)) = scope.get(name) {
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
                    AssignOp::Assign => val,
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

                        let (op_res, _) = self.compile_bin_op(
                            current_val,
                            var_type.clone(),
                            val,
                            val_type,
                            BinOp::Add,
                        )?;
                        op_res
                    }
                    _ => return Err(format!("Unsupported assignment op: {:?}", op)),
                };

                self.builder
                    .build_store(var_ptr.into_pointer_value(), final_val)
                    .map_err(|e| e.to_string())?;
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
            }
            Stmt::For {
                loop_var,
                iterator,
                body,
            } => {
                // 1. Evaluate iterator expression (expecting Tensor)
                let (tensor_val, _tensor_ty) = self.compile_expr(iterator)?;

                // 2. Get Tensor length (runtime call)
                let len_fn = self.module.get_function("tl_tensor_len").unwrap();
                let len_call = self
                    .builder
                    .build_call(len_fn, &[tensor_val.into()], "len")
                    .map_err(|e| e.to_string())?;
                let len_val = match len_call.try_as_basic_value() {
                    ValueKind::Basic(v) => v.into_int_value(),
                    _ => return Err("Failed to get len".into()),
                };

                // 3. Setup Loop Blocks
                let func = self
                    .builder
                    .get_insert_block()
                    .unwrap()
                    .get_parent()
                    .unwrap();
                let cond_block = self.context.append_basic_block(func, "loop_cond");
                let body_block = self.context.append_basic_block(func, "loop_body");
                let end_block = self.context.append_basic_block(func, "loop_end");

                // Capture the current block (preheader) to add to Phi incoming later
                let preheader_block = self.builder.get_insert_block().unwrap();

                self.builder
                    .build_unconditional_branch(cond_block)
                    .map_err(|e| e.to_string())?;

                // --- Condition Block ---
                self.builder.position_at_end(cond_block);

                let i64_type = self.context.i64_type();
                let phi_i = self
                    .builder
                    .build_phi(i64_type, "i")
                    .map_err(|e| e.to_string())?;
                let i_val = phi_i.as_basic_value().into_int_value();

                let cmp = self
                    .builder
                    .build_int_compare(inkwell::IntPredicate::SLT, i_val, len_val, "loop_cond_cmp")
                    .map_err(|e| e.to_string())?;
                self.builder
                    .build_conditional_branch(cmp, body_block, end_block)
                    .map_err(|e| e.to_string())?;

                // --- Body Block ---
                self.builder.position_at_end(body_block);
                self.enter_scope(); // Scope for loop body and loop_var

                // Get element at i
                let get_fn = self.module.get_function("tl_tensor_get").unwrap();
                let val_call = self
                    .builder
                    .build_call(get_fn, &[tensor_val.into(), i_val.into()], "elem")
                    .map_err(|e| e.to_string())?;
                let elem_val = match val_call.try_as_basic_value() {
                    ValueKind::Basic(v) => v,
                    _ => return Err("Failed to get elem".into()),
                };

                // Declare loop variable (alloca)
                // Assuming it is F32 (scalar from tensor)
                let alloca = self.create_entry_block_alloca(func, loop_var, &Type::F32);
                self.builder
                    .build_store(alloca, elem_val)
                    .map_err(|e| e.to_string())?;
                // Insert into current scope
                self.variables
                    .last_mut()
                    .unwrap()
                    .insert(loop_var.clone(), (alloca.into(), Type::F32));

                for stmt in body {
                    self.compile_stmt(stmt)?;
                }

                // Increment i
                let one = i64_type.const_int(1, false);
                let next_i = self
                    .builder
                    .build_int_add(i_val, one, "next_i")
                    .map_err(|e| e.to_string())?;

                // Capture the body latch block
                let body_latch_block = self.builder.get_insert_block().unwrap();

                self.builder
                    .build_unconditional_branch(cond_block)
                    .map_err(|e| e.to_string())?;

                // Add Phi incoming edges
                phi_i.add_incoming(&[
                    (&i64_type.const_int(0, false), preheader_block),
                    (&next_i, body_latch_block),
                ]);

                self.exit_scope(); // End loop scope

                // --- End Block ---
                self.builder.position_at_end(end_block);
            }
            Stmt::Expr(expr) => {
                self.compile_expr(expr)?;
            }
        }
        Ok(())
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
                    _ => return Err("Unsupported tensor op".into()),
                };

                let runtime_fn = self
                    .module
                    .get_function(fn_name)
                    .ok_or(format!("Runtime function {} not found", fn_name))?;
                let call = self
                    .builder
                    .build_call(runtime_fn, &[l.into(), r.into()], "tensor_op")
                    .map_err(|e| e.to_string())?;
                let res = match call.try_as_basic_value() {
                    ValueKind::Basic(v) => v,
                    _ => return Err("Invalid call return: Void".into()),
                };
                Ok((res, lhs_type.clone()))
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
            Expr::Variable(name) => {
                for scope in self.variables.iter().rev() {
                    if let Some((val, ty)) = scope.get(name) {
                        if val.is_pointer_value() {
                            let ptr = val.into_pointer_value();
                            let llvm_ty: inkwell::types::BasicTypeEnum = match ty {
                                Type::I64 => self.context.i64_type().into(),
                                Type::F32 => self.context.f32_type().into(),
                                Type::Bool => self.context.bool_type().into(),
                                Type::Tensor(_, _) => self
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
            Expr::FnCall(name, args) => {
                if name == "print" && args.len() == 1 {
                    // Check type of arg
                    let arg_expr = &args[0];
                    // We need the type of argument.
                    // But compiled_args has values.
                    // Compile_expr returned (value, type).
                    // We need to re-compile to get type?
                    // No, "args" is Vec<expr>. We can check semantics type or compile arg first.
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

                // Generic function call logic
                let llvm_func_name = match name.as_str() {
                    "slice" => "tl_tensor_slice",
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
            Type::Tensor(_, _) => self
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
                .insert(idx_name.clone(), (alloca.into(), Type::I64));

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
            (v_alloca.into(), Type::Tensor(Box::new(Type::F32), rank)),
        );

        Ok(())
    }
}
