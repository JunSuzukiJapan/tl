use super::CodeGenerator;
use crate::compiler::ast::*;
use inkwell::values::*;
use inkwell::types::BasicType;

impl<'ctx> CodeGenerator<'ctx> {
    // Helper to infer free indices from implicit tensor equation (RHS)
    // Returns sorted list of unique variable names used as indices but not bound in scope
    fn infer_free_indices(&self, expr: &Expr) -> Vec<String> {
        let mut indices = std::collections::HashSet::new();
        self.collect_indices(expr, &mut indices);

        // Filter out variables that are defined in current scope (e.g. loops)
        // If a variable is NOT in scope, it is a free index (implicit dimension)
        let mut free_indices: Vec<String> = indices
            .into_iter()
            .filter(|idx| {
                // If variable exists in scope, it's a bound value/loop var, NOT a free dimension
                !self.variable_exists(idx)
            })
            .collect();

        free_indices.sort();
        free_indices
    }

    fn collect_indices(&self, expr: &Expr, indices: &mut std::collections::HashSet<String>) {
        match expr {
            Expr::IndexAccess(_, idxs) => {
                for idx in idxs {
                    if let Expr::Variable(name) = idx {
                        indices.insert(name.clone());
                    }
                    // Recursive check? Indices usually simple vars.
                }
            }
            Expr::BinOp(lhs, _, rhs) => {
                self.collect_indices(lhs, indices);
                self.collect_indices(rhs, indices);
            }
            Expr::UnOp(_, val) => {
                self.collect_indices(val, indices);
            }
            Expr::FnCall(_, args)
            | Expr::MethodCall(_, _, args)
            | Expr::StaticMethodCall(_, _, args) => {
                for arg in args {
                    self.collect_indices(arg, indices);
                }
            }
            Expr::TensorLiteral(elems) => {
                for elem in elems {
                    self.collect_indices(elem, indices);
                }
            }
            Expr::IfExpr(cond, _, _) => {
                self.collect_indices(cond, indices);
            }
            _ => {}
        }
    }

    fn variable_exists(&self, name: &str) -> bool {
        for scope in self.variables.iter().rev() {
            if scope.contains_key(name) {
                return true;
            }
        }
        false
    }

    pub(crate) fn emit_recursive_unregister(
        &self,
        val: BasicValueEnum<'ctx>,
        ty: &Type,
    ) -> Result<(), String> {
        let unreg_fn = self
            .module
            .get_function("tl_mem_unregister")
            .ok_or("tl_mem_unregister not found")?;

        match ty {
            Type::Tensor(_, _) => {
                self.builder
                    .build_call(unreg_fn, &[val.into()], "")
                    .map_err(|e| e.to_string())?;
            }
            Type::Struct(name) | Type::UserDefined(name) => {
                if let Some(struct_def) = self.struct_defs.get(name) {
                    let ptr = val.into_pointer_value();
                    let st_llvm_ty = self.struct_types.get(name).unwrap().clone();

                    for (i, (_, field_type)) in struct_def.fields.iter().enumerate() {
                        if matches!(
                            field_type,
                            Type::Tensor(_, _) | Type::Struct(_) | Type::UserDefined(_)
                        ) {
                            // GEP
                            let field_ptr = self
                                .builder
                                .build_struct_gep(
                                    st_llvm_ty,
                                    ptr,
                                    i as u32,
                                    &format!("unreg_field_{}", i),
                                )
                                .map_err(|e| e.to_string())?;

                            // Load
                            let load_type = self.context.ptr_type(inkwell::AddressSpace::default());
                            let field_val = self
                                .builder
                                .build_load(load_type, field_ptr, "field_val")
                                .map_err(|e| e.to_string())?;

                            // Recurse
                            self.emit_recursive_unregister(field_val, field_type)?;
                        }
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    pub(crate) fn compile_stmt(&mut self, stmt: &Stmt) -> Result<(), String> {
        match stmt {
            Stmt::FieldAssign { obj, field, value } => {
                let (obj_val, obj_ty) = self.compile_expr(obj)?;
                let struct_name = match obj_ty {
                    Type::Struct(name) => name,
                    Type::UserDefined(name) => name,
                    _ => return Err(format!("Field assignment on non-struct type {:?}", obj_ty)),
                };

                let (field_idx, field_type) = {
                    let struct_def = self
                        .struct_defs
                        .get(&struct_name)
                        .ok_or(format!("Struct definition for {} not found", struct_name))?;

                    let idx = struct_def
                        .fields
                        .iter()
                        .position(|(n, _)| n == field)
                        .ok_or(format!(
                            "Field {} not found in struct {}",
                            field, struct_name
                        ))?;
                    (idx, struct_def.fields[idx].1.clone())
                };

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

                // NOTE: Removed explicit tensor clone to preserve gradient graph.
                // Candle's Arc-based tensors should handle aliasing safely.
                // Memory management (freeing old values) is preserved below.

                // Free old value if Tensor
                if let Type::Tensor(_, _) = field_type {
                    let load_type = self.context.ptr_type(inkwell::AddressSpace::default());
                    let _current_val = self
                        .builder
                        .build_load(load_type, field_ptr, "old_field_val")
                        .map_err(|e| e.to_string())?
                        .into_pointer_value();

                    // Free logic removed to prevent double-free with MemoryManager.
                    // Old value remains in scope list and will be freed at scope exit.
                }

                self.builder
                    .build_store(field_ptr, val)
                    .map_err(|e| e.to_string())?;
                Ok(())
            }
            Stmt::TensorDecl {
                name,
                type_annotation,
                init,
            } => {
                if let Some(expr) = init {
                    let val_ir = self.ensure_tensor_v2(expr, 0)?;
                    let val_ty = if matches!(type_annotation, Type::Tensor(_, _)) {
                        type_annotation.clone()
                    } else {
                        // Fallback to compiled type if not explicitly a Tensor
                        let (_, ty) = self.compile_expr(expr)?;
                        ty
                    };

                    // NOTE: Removed clone to preserve gradients

                    if self.variables.last().unwrap().contains_key(name) {
                        // Start of double-free fix logic
                        let (_var_val, _, should_free) = &self.variables.last().unwrap()[name];

                        if *should_free {
                            // Free logic removed to prevent double-free with MemoryManager.
                            // Old value remains in scope list and will be freed at scope exit.
                        }

                        let ptr = self.variables.last().unwrap()[name].0.into_pointer_value();
                        self.builder
                            .build_store(ptr, val_ir)
                            .map_err(|e| e.to_string())?;

                        // Update variable map to mark as owned (should_free = true)
                        self.variables
                            .last_mut()
                            .unwrap()
                            .insert(name.clone(), (ptr.into(), val_ty, true));
                    } else {
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
                }
                Ok(())
            }
            Stmt::Let {
                name,
                type_annotation,
                value,
            } => {
                // 1. Analyze value for Free Indices (Implicit Tensor Equation)
                let free_indices = self.infer_free_indices(value);

                if !free_indices.is_empty() {
                    // Found free indices -> It's a tensor equation! e.g. let C = A[i, j] * B[j, k];
                    // free_indices will be ["i", "k"] (sorted)
                    // Delegate to compile_tensor_equation
                    return self
                        .compile_tensor_equation(name, &free_indices, value)
                        .map_err(|e| e.to_string());
                }

                let (mut val_ir, mut val_ty) = self.compile_expr(value)?;

                // Convert ScalarArray to Tensor if explicitly requested as Tensor
                if let Some(target_ty) = type_annotation {
                    if matches!(target_ty, Type::Tensor(_, _))
                        && matches!(val_ty, Type::ScalarArray(_, _))
                    {
                        val_ir = self.ensure_tensor_v2(value, 0)?;
                        val_ty = target_ty.clone();
                    }
                }

                // Clone if alias (initializing from variable or field)
                let val_ir = if matches!(value, Expr::Variable(_) | Expr::FieldAccess(_, _)) {
                    if let Type::Tensor(_, _) = val_ty {
                        let clone_fn = self
                            .module
                            .get_function("tl_tensor_clone")
                            .expect("tl_tensor_clone not found");
                        let call = self
                            .builder
                            .build_call(clone_fn, &[val_ir.into()], "cloned")
                            .map_err(|e| e.to_string())?;
                        match call.try_as_basic_value() {
                            inkwell::values::ValueKind::Basic(v) => v,
                            _ => return Err("Clone returned void".into()),
                        }
                    } else {
                        val_ir
                    }
                } else {
                    val_ir
                };

                let current_function = self
                    .builder
                    .get_insert_block()
                    .unwrap()
                    .get_parent()
                    .unwrap();

                // Check for shadowing in CURRENT scope
                if let Some(scope) = self.variables.last_mut() {
                    if let Some((_old_ptr, _old_ty, should_free)) = scope.get(name) {
                        // If we are shadowing, and the old value effectively goes away (we overwrite the map entry),
                        // we MUST free it if it's a tensor and we own it.
                        // NOTE: In Rust, shadowing doesn't drop the old var immediately, it lives until end of scope.
                        // BUT in our compiler, we only track variables by name in the map.
                        // If we overwrite the map entry, we lose access to the old variable.
                        // So for our language semantics, we treat shadowing as "replacing".
                        // Use-case: `let x = ...; let x = ...;`
                        if *should_free {
                            // Free logic removed key. MemoryManager handles it.
                        }
                    }
                }

                let alloca = self.create_entry_block_alloca(current_function, name, &val_ty);
                self.builder
                    .build_store(alloca, val_ir)
                    .map_err(|e| e.to_string())?;

                self.variables
                    .last_mut()
                    .unwrap()
                    .insert(name.clone(), (alloca.into(), val_ty.clone(), true)); // Store pointer and type

                // Register tensor with runtime if it is a tensor
                // PANIC INVESTIGATION: tl_register_tensor causes slice alignment panic for some reason.
                // Disabling it for now to verify if tensor creation itself is valid.
                /*
                if let Type::Tensor(_, _) = val_ty {
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
                                &[name_global.as_pointer_value().into(), val_ir.into()],
                                "",
                            )
                            .map_err(|e| e.to_string())?;
                    }
                }
                */
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
                let (val, _) = self.compile_expr(expr)?;

                // Emit cleanup for ALL active scopes (reverse order)
                self.emit_all_scopes_cleanup();

                self.builder
                    .build_return(Some(&val))
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
                        // In-place tensor update would require rewriting into existing buffer.
                        // Current design uses 'let C[i,k] = ...' for tensor equations which creates new tensors.
                        // This is an intentional limitation - users should use 'let' for tensor equations.
                        return Err(
                            "In-place indexed assignment not supported. Use 'let C[i,k] = ...' for tensor equations."
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
                        // Free old value if it is a Struct
                        if matches!(var_type, Type::Struct(_) | Type::UserDefined(_)) {
                            let load_type = self.context.ptr_type(inkwell::AddressSpace::default());
                            let current_val = self
                                .builder
                                .build_load(
                                    load_type,
                                    var_ptr.into_pointer_value(),
                                    "old_struct_to_free",
                                )
                                .map_err(|e| e.to_string())?
                                .into_pointer_value();

                            // Only free if not null
                            let null_ptr = load_type.const_null();
                            let is_not_null = self
                                .builder
                                .build_int_compare(
                                    inkwell::IntPredicate::NE,
                                    current_val,
                                    null_ptr,
                                    "is_not_null",
                                )
                                .map_err(|e| e.to_string())?;

                            let free_block = self.context.append_basic_block(
                                self.builder
                                    .get_insert_block()
                                    .unwrap()
                                    .get_parent()
                                    .unwrap(),
                                "free_struct",
                            );
                            let continue_block = self.context.append_basic_block(
                                self.builder
                                    .get_insert_block()
                                    .unwrap()
                                    .get_parent()
                                    .unwrap(),
                                "continue_after_free",
                            );

                            self.builder
                                .build_conditional_branch(is_not_null, free_block, continue_block)
                                .map_err(|e| e.to_string())?;

                            // Free block
                            self.builder.position_at_end(free_block);

                            // For now, we don't have deep free.
                            // If the old struct was "global" (unregistered), it leaks.
                            // If it was local (registered), scope handles it (but we are overwriting it).
                            // If we overwrite a pointer, the old pointer value is lost.
                            // If the old pointer was managed by MemoryManager, it will be freed at end of scope.
                            // BUT: if we are in a loop, and we overwrite 'model' (which is in outer scope),
                            // the 'model' variable is updated. The OLD model pointer is gone.
                            // Does MemoryManager know?
                            // MemoryManager tracks allocations.
                            // If we allocated Model_1 (Global), assigned to 'model'.
                            // Loop 1: Create Model_2 (Local->Global). Assign to 'model'.
                            // 'model' now points to Model_2. Model_1 is lost.
                            // Model_1 was Global (unregistered). It is LEAKED.
                            // We MUST free Model_1 here.

                            // Since we don't have deep free, we at least UNREGISTER it?
                            // No, unregistering prevents freeing. We want TO FREE.
                            // We need `tl_mem_free_struct`.
                            // Let's implement deep free logic inline? No, reuse unregister traversal?
                            // Actually, if we just call `tl_mem_register_struct(old_val)`, we put it BACK in scope?
                            // Then it gets freed at end of scope!
                            // YES! If old value was Global (leaked), we "adopt" it into current scope.
                            // Then current scope exit frees it.
                            // BUT: deep adoption needed.

                            // Strategy: Unregister new value (Leak/Global).
                            // Old value: Just let it leak for now (2GB max).
                            // Fix properly later with GC or refcounting.

                            self.builder
                                .build_unconditional_branch(continue_block)
                                .map_err(|e| e.to_string())?;

                            // Continue block
                            self.builder.position_at_end(continue_block);
                        }

                        // Free old value if it is a Tensor
                        if let Type::Tensor(_, _) = var_type {
                            let load_type = self.context.ptr_type(inkwell::AddressSpace::default());
                            let _current_val = self
                                .builder
                                .build_load(load_type, var_ptr.into_pointer_value(), "old_val")
                                .map_err(|e| e.to_string())?
                                .into_pointer_value();

                            // Only free if not null
                            let null_ptr = load_type.const_null();
                            let is_not_null = self
                                .builder
                                .build_int_compare(
                                    inkwell::IntPredicate::NE,
                                    _current_val,
                                    null_ptr,
                                    "is_not_null",
                                )
                                .map_err(|e| e.to_string())?;

                            let free_block = self.context.append_basic_block(
                                self.builder
                                    .get_insert_block()
                                    .unwrap()
                                    .get_parent()
                                    .unwrap(),
                                "free_block",
                            );
                            let continue_block = self.context.append_basic_block(
                                self.builder
                                    .get_insert_block()
                                    .unwrap()
                                    .get_parent()
                                    .unwrap(),
                                "continue_block",
                            );

                            self.builder
                                .build_conditional_branch(is_not_null, free_block, continue_block)
                                .map_err(|e| e.to_string())?;

                            self.builder.position_at_end(free_block);
                            // Free logic removed. MemoryManager handles it.
                            self.builder
                                .build_unconditional_branch(continue_block)
                                .map_err(|e| e.to_string())?;

                            self.builder.position_at_end(continue_block);
                        }

                        let new_val_basic = val;

                        // CHECK SCOPE PROMOTION
                        if self.is_outer_scope(name) {
                            // If assigning to outer scope, we must "promote" (unregister) the new value
                            // so it isn't freed when the current inner scope exits.
                            // This effectively leaks it (until program end), but prevents Use-After-Free.

                            let unreg_fn = self.module.get_function("tl_mem_unregister");

                            if let Some(f) = unreg_fn {
                                if let Type::Tensor(_, _) = var_type {
                                    self.builder
                                        .build_call(f, &[new_val_basic.into()], "")
                                        .map_err(|e| e.to_string())?;
                                } else if matches!(var_type, Type::Struct(_) | Type::UserDefined(_))
                                {
                                    // Recursive unregister for struct fields
                                    self.emit_recursive_unregister(new_val_basic, &var_type)?;

                                    // Also unregister the struct pointer itself
                                    self.builder
                                        .build_call(f, &[new_val_basic.into()], "")
                                        .map_err(|e| e.to_string())?;
                                }
                            }
                        }

                        new_val_basic
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

                        // Free logic removed.

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
                // Branch to merge if current block has no terminator
                // Use get_insert_block() because nested statements may have changed current block
                let current_block = self.builder.get_insert_block().unwrap();
                if current_block.get_terminator().is_none() {
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
                // Check current block (not else_bb) since nested if may have changed it
                let current_block = self.builder.get_insert_block().unwrap();
                if current_block.get_terminator().is_none() {
                    self.builder.build_unconditional_branch(merge_bb).unwrap();
                }
                self.exit_scope();

                // Merge
                self.builder.position_at_end(merge_bb);
                Ok(())
            }
            Stmt::For {
                loop_var,
                iterator,
                body,
            } => {
                let parent = self
                    .builder
                    .get_insert_block()
                    .unwrap()
                    .get_parent()
                    .unwrap();

                let i64_type = self.context.i64_type();

                // Check if iterator is a range (BinOp with ".." conceptually - we detect 0..n pattern)
                // Or if it's a tensor/variable
                let (start_val, end_val, is_tensor_iter) = match iterator {
                    Expr::FnCall(name, args) if name == "range" => {
                        // range(start, end)
                        if args.len() != 2 {
                            return Err("range() requires 2 arguments".into());
                        }
                        let (s, _) = self.compile_expr(&args[0])?;
                        let (e, _) = self.compile_expr(&args[1])?;
                        (s.into_int_value(), e.into_int_value(), false)
                    }
                    Expr::Variable(_) | Expr::FieldAccess(_, _) => {
                        // Assume it's a tensor or array iteration
                        let (tensor_val, tensor_ty) = self.compile_expr(iterator)?;
                        let len = match &tensor_ty {
                            Type::Tensor(_, _) => {
                                // Get tensor length
                                let len_fn = self
                                    .module
                                    .get_function("tl_tensor_len")
                                    .ok_or("tl_tensor_len not found")?;
                                let len_call = self
                                    .builder
                                    .build_call(len_fn, &[tensor_val.into()], "tensor_len")
                                    .map_err(|e| e.to_string())?;
                                match len_call.try_as_basic_value() {
                                    inkwell::values::ValueKind::Basic(v) => v.into_int_value(),
                                    _ => return Err("Invalid tensor_len return".into()),
                                }
                            }
                            Type::ScalarArray(_, len) => i64_type.const_int(*len as u64, false),
                            _ => return Err("For loop iterator must be a tensor, array, or range".into()),
                        };

                        // Store tensor/array pointer for use in body
                        let tensor_ptr = tensor_val.into_pointer_value();
                        let tensor_alloca = self
                            .builder
                            .build_alloca(
                                self.context.ptr_type(inkwell::AddressSpace::default()),
                                "for_tensor",
                            )
                            .map_err(|e| e.to_string())?;
                        self.builder
                            .build_store(tensor_alloca, tensor_ptr)
                            .map_err(|e| e.to_string())?;

                        // Register tensor alloca for later use
                        self.variables.last_mut().unwrap().insert(
                            "__for_tensor__".to_string(),
                            (tensor_alloca.into(), tensor_ty.clone(), false),
                        );

                        (i64_type.const_int(0, false), len, true)
                    }
                    _ => {
                        // Try to compile as expression and check type
                        let (iter_val, iter_ty) = self.compile_expr(iterator)?;
                        let len = match &iter_ty {
                            Type::Tensor(_, _) => {
                                let len_fn = self
                                    .module
                                    .get_function("tl_tensor_len")
                                    .ok_or("tl_tensor_len not found")?;
                                let len_call = self
                                    .builder
                                    .build_call(len_fn, &[iter_val.into()], "tensor_len")
                                    .map_err(|e| e.to_string())?;
                                match len_call.try_as_basic_value() {
                                    inkwell::values::ValueKind::Basic(v) => v.into_int_value(),
                                    _ => return Err("Invalid tensor_len return".into()),
                                }
                            }
                            Type::ScalarArray(_, len) => i64_type.const_int(*len as u64, false),
                            _ => return Err("For loop iterator must be a tensor, array, or range".into()),
                        };

                        let tensor_ptr = iter_val.into_pointer_value();
                        let tensor_alloca = self
                            .builder
                            .build_alloca(
                                self.context.ptr_type(inkwell::AddressSpace::default()),
                                "for_tensor",
                            )
                            .map_err(|e| e.to_string())?;
                        self.builder
                            .build_store(tensor_alloca, tensor_ptr)
                            .map_err(|e| e.to_string())?;

                        self.variables.last_mut().unwrap().insert(
                            "__for_tensor__".to_string(),
                            (tensor_alloca.into(), iter_ty.clone(), false),
                        );

                        (i64_type.const_int(0, false), len, true)
                    }
                };

                // Capture preheader block (where we are jumping from)
                let preheader_block = self.builder.get_insert_block().unwrap();

                // Create basic blocks
                let loop_header = self.context.append_basic_block(parent, "for_header");
                let loop_body = self.context.append_basic_block(parent, "for_body");
                let loop_end = self.context.append_basic_block(parent, "for_end");

                // Branch to loop header
                self.builder
                    .build_unconditional_branch(loop_header)
                    .map_err(|e| e.to_string())?;

                // Loop header: PHI for index
                self.builder.position_at_end(loop_header);
                // let current_block = self.builder.get_insert_block().unwrap(); // No longer needed
                let phi = self
                    .builder
                    .build_phi(i64_type, "for_idx")
                    .map_err(|e| e.to_string())?;

                // Add incoming from entry
                // Use preheader_block captured above

                // Check condition: idx < end
                let cond = self
                    .builder
                    .build_int_compare(
                        inkwell::IntPredicate::SLT,
                        phi.as_basic_value().into_int_value(),
                        end_val,
                        "for_cond",
                    )
                    .map_err(|e| e.to_string())?;

                self.builder
                    .build_conditional_branch(cond, loop_body, loop_end)
                    .map_err(|e| e.to_string())?;

                // Get tensor alloca BEFORE entering new scope (it's in current scope)
                let saved_tensor_alloca = if is_tensor_iter {
                    // Search through all scopes to find __for_tensor__
                    let mut found = None;
                    for scope in self.variables.iter().rev() {
                        if let Some((val, _, _)) = scope.get("__for_tensor__") {
                            found = Some(val.into_pointer_value());
                            break;
                        }
                    }
                    found
                } else {
                    None
                };

                // Loop body
                self.builder.position_at_end(loop_body);

                self.enter_scope();

                // Bind loop variable
                let loop_var_val = if is_tensor_iter {
                    // Search through scopes to find the type of __for_tensor__
                    let mut iter_ty = None;
                    for scope in self.variables.iter().rev() {
                        if let Some((_, ty, _)) = scope.get("__for_tensor__") {
                            iter_ty = Some(ty.clone());
                            break;
                        }
                    }
                    let iter_ty = iter_ty.ok_or("Iterator type not found")?;

                    // Get element from tensor/array - use saved alloca since we're in a new scope
                    let tensor_alloca =
                        saved_tensor_alloca.ok_or("Tensor alloca not found for for-loop")?;
                    let load_type = self.context.ptr_type(inkwell::AddressSpace::default());
                    let tensor_ptr = self
                        .builder
                        .build_load(load_type, tensor_alloca, "tensor_ptr")
                        .map_err(|e| e.to_string())?
                        .into_pointer_value();

                    match iter_ty {
                        Type::Tensor(inner_ty, _) => {
                            let get_fn = self
                                .module
                                .get_function("tl_tensor_get")
                                .ok_or("tl_tensor_get not found")?;
                            let get_call = self
                                .builder
                                .build_call(
                                    get_fn,
                                    &[tensor_ptr.into(), phi.as_basic_value().into()],
                                    "elem_val",
                                )
                                .map_err(|e| e.to_string())?;
                            
                            match get_call.try_as_basic_value() {
                                inkwell::values::ValueKind::Basic(v) => {
                                    let f_val = v.into_float_value();
                                    match inner_ty.as_ref() {
                                        Type::I64 => {
                                            let i_val = self.builder.build_float_to_signed_int(
                                                f_val,
                                                self.context.i64_type(),
                                                "f2i"
                                            ).map_err(|e| e.to_string())?;
                                            (i_val.into(), Type::I64)
                                        }
                                        Type::I32 => {
                                            let i_val = self.builder.build_float_to_signed_int(
                                                f_val,
                                                self.context.i32_type(),
                                                "f2i"
                                            ).map_err(|e| e.to_string())?;
                                            (i_val.into(), Type::I32)
                                        }
                                        _ => (v, Type::F32), // Default/Keep as F32
                                    }
                                }
                                _ => return Err("Invalid tensor_get return".into()),
                            }
                        }
                        Type::ScalarArray(elem_ty, len) => {
                            let llvm_elem_type: inkwell::types::BasicTypeEnum = match elem_ty.as_ref() {
                                Type::I64 => self.context.i64_type().into(),
                                Type::F32 => self.context.f32_type().into(),
                                _ => self.context.f32_type().into(),
                            };
                            let array_type = llvm_elem_type.array_type(len as u32);
                            let elem_ptr = unsafe {
                                self.builder
                                    .build_in_bounds_gep(
                                        array_type,
                                        tensor_ptr,
                                        &[
                                            i64_type.const_int(0, false),
                                            phi.as_basic_value().into_int_value(),
                                        ],
                                        "elem_ptr",
                                    )
                                    .map_err(|e| e.to_string())?
                            };
                            let loaded = self
                                .builder
                                .build_load(llvm_elem_type, elem_ptr, "elem_val")
                                .map_err(|e| e.to_string())?;
                            (loaded, *elem_ty.clone())
                        }
                        _ => unreachable!(),
                    }
                } else {
                    // Range iteration: loop var is the index
                    (phi.as_basic_value(), Type::I64)
                };

                // Create alloca for loop var and store
                let var_alloca = self.create_entry_block_alloca(parent, loop_var, &loop_var_val.1);
                self.builder
                    .build_store(var_alloca, loop_var_val.0)
                    .map_err(|e| e.to_string())?;
                self.variables
                    .last_mut()
                    .unwrap()
                    .insert(loop_var.clone(), (var_alloca.into(), loop_var_val.1, false));

                // Compile body
                for stmt in body {
                    self.compile_stmt(stmt)?;
                }

                self.exit_scope();

                // Increment index
                let body_end_block = self.builder.get_insert_block().unwrap();
                let next_idx = self
                    .builder
                    .build_int_add(
                        phi.as_basic_value().into_int_value(),
                        i64_type.const_int(1, false),
                        "next_idx",
                    )
                    .map_err(|e| e.to_string())?;

                // Branch back to header
                // Note: We need to cleanup the loop scope variables before branching back!
                // exit_scope() above already popped the scope from the compiler's tracking,
                // BUT it only emitted cleanup if the block wasn't terminated.
                // Here, we haven't terminated yet (we are about to branch).
                // Wait. exit_scope() was called at line 727.
                // If line 727 emitted cleanup, then we are fine?
                // Line 727: self.exit_scope().
                // exit_scope checks is_terminated.
                // At line 727, are we terminated?
                // self.compile_stmt(stmt) might have terminated (e.g. Return/Break).
                // If body ended naturally, we are NOT terminated. So exit_scope() emitted cleanup.
                // So cleanup logic IS there.

                // Why stack overflow?

                // Maybe the cleanup is emitted AFTER the branch?
                // No, exit_scope is called BEFORE `next_idx` and BEFORE `build_unconditional_branch`.

                if body_end_block.get_terminator().is_none() {
                    self.builder
                        .build_unconditional_branch(loop_header)
                        .map_err(|e| e.to_string())?;
                }

                // Add PHI incoming edges
                phi.add_incoming(&[(&start_val, preheader_block), (&next_idx, body_end_block)]);

                // Continue at loop end
                self.builder.position_at_end(loop_end);

                // Clean up temporary tensor reference
                if is_tensor_iter {
                    for scope in self.variables.iter_mut().rev() {
                        scope.remove("__for_tensor__");
                    }
                }

                Ok(())
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

                // Exit memory scope before back-edge
                if let Some(scope_exit) = self.module.get_function("tl_mem_exit_scope") {
                    self.builder
                        .build_call(scope_exit, &[], "")
                        .map_err(|e| e.to_string())?;
                }

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
    pub(crate) fn compile_bin_op(
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
            (Type::UserDefined(s1), Type::UserDefined(s2)) if s1 == "String" && s2 == "String" => {
                let strcmp_fn = self
                    .module
                    .get_function("strcmp")
                    .ok_or("strcmp not found")?;
                let cmp = self
                    .builder
                    .build_call(strcmp_fn, &[lhs.into(), rhs.into()], "strcmp_res")
                    .map_err(|e| e.to_string())?;

                let cmp_val = match cmp.try_as_basic_value() {
                    ValueKind::Basic(v) => v.into_int_value(),
                    _ => return Err("Invalid strcmp return".into()),
                };

                let zero = self.context.i32_type().const_zero();
                let res = match op {
                    BinOp::Eq => self.builder.build_int_compare(
                        inkwell::IntPredicate::EQ,
                        cmp_val,
                        zero,
                        "streq",
                    ),
                    BinOp::Neq => self.builder.build_int_compare(
                        inkwell::IntPredicate::NE,
                        cmp_val,
                        zero,
                        "strneq",
                    ),
                    BinOp::Add => {
                        let concat_fn = self
                            .module
                            .get_function("tl_string_concat")
                            .ok_or("tl_string_concat not found")?;
                        let res = self
                            .builder
                            .build_call(concat_fn, &[lhs.into(), rhs.into()], "strconcat")
                            .map_err(|e| e.to_string())?;
                        let res_val = match res.try_as_basic_value() {
                            ValueKind::Basic(v) => v,
                            _ => return Err("Invalid string concat return".into()),
                        };
                        return Ok((res_val, Type::UserDefined("String".to_string())));
                    }
                    _ => return Err("Only ==, !=, and + supported for Strings".into()),
                }
                .map_err(|e| e.to_string())?;

                Ok((res.into(), Type::Bool))
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
            // ScalarArray operations: convert to Tensor and use tensor ops
            (Type::ScalarArray(_, len1), Type::ScalarArray(_, len2)) if len1 == len2 => {
                // Convert both ScalarArrays to tensors and perform tensor operation
                let _f32_type = self.context.f32_type();
                let i64_type = self.context.i64_type();

                // Helper to create tensor from ScalarArray pointer
                let create_tensor =
                    |builder: &inkwell::builder::Builder<'ctx>,
                     module: &inkwell::module::Module<'ctx>,
                     ptr: inkwell::values::PointerValue<'ctx>,
                     len: usize|
                     -> Result<inkwell::values::PointerValue<'ctx>, String> {
                        let new_fn = module
                            .get_function("tl_tensor_new")
                            .ok_or("tl_tensor_new not found")?;

                        // Shape: [len]
                        let shape_alloca = builder
                            .build_alloca(i64_type, "shape")
                            .map_err(|e| e.to_string())?;
                        builder
                            .build_store(shape_alloca, i64_type.const_int(len as u64, false))
                            .map_err(|e| e.to_string())?;

                        let call = builder
                            .build_call(
                                new_fn,
                                &[
                                    ptr.into(),
                                    i64_type.const_int(1, false).into(),
                                    shape_alloca.into(),
                                ],
                                "tensor_from_scalar_arr",
                            )
                            .map_err(|e| e.to_string())?;

                        match call.try_as_basic_value() {
                            ValueKind::Basic(v) => Ok(v.into_pointer_value()),
                            _ => Err("Invalid tensor new return".into()),
                        }
                    };

                let l_tensor =
                    create_tensor(&self.builder, &self.module, lhs.into_pointer_value(), *len1)?;
                let r_tensor =
                    create_tensor(&self.builder, &self.module, rhs.into_pointer_value(), *len2)?;

                // Now call tensor binary op
                let fn_name = match op {
                    BinOp::Add => "tl_tensor_add",
                    BinOp::Mul => "tl_tensor_mul",
                    BinOp::Div => "tl_tensor_div",
                    BinOp::Sub => "tl_tensor_sub",
                    _ => return Err("Unsupported ScalarArray op".into()),
                };

                let fn_val = self
                    .module
                    .get_function(fn_name)
                    .ok_or(format!("Runtime function {} not found", fn_name))?;
                let call = self
                    .builder
                    .build_call(fn_val, &[l_tensor.into(), r_tensor.into()], "binop_res")
                    .map_err(|e| e.to_string())?;

                let res_ptr = match call.try_as_basic_value() {
                    ValueKind::Basic(v) => v.into_pointer_value(),
                    _ => return Err("Invalid return from runtime binop".into()),
                };

                // Return as Tensor (since we converted)
                Ok((res_ptr.into(), Type::Tensor(Box::new(Type::F32), 1)))
            }
            _ => Err(format!(
                "Type mismatch in BinOp {:?}: {:?} vs {:?}",
                op, lhs_type, rhs_type
            )),
        }
    }
}
