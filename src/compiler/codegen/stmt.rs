use crate::compiler::ast::*;
use super::CodeGenerator;
use inkwell::values::*;

impl<'ctx> CodeGenerator<'ctx> {
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

                // Free old value if Tensor
                if let Type::Tensor(_, _) = field_type {
                    let load_type = self.context.ptr_type(inkwell::AddressSpace::default());
                    let current_val = self
                        .builder
                        .build_load(load_type, field_ptr, "old_field_val")
                        .map_err(|e| e.to_string())?
                        .into_pointer_value();

                    if let Some(free_fn) = self.module.get_function("tl_tensor_free") {
                        self.builder
                            .build_call(free_fn, &[current_val.into()], "")
                            .map_err(|e| e.to_string())?;
                    }
                }

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
                    if self.variables.last().unwrap().contains_key(name) {
                        // Start of double-free fix logic
                        let (var_val, _, should_free) = &self.variables.last().unwrap()[name];

                        if *should_free {
                            if let Type::Tensor(_, _) = val_ty {
                                if let Some(free_fn) = self.module.get_function("tl_tensor_free") {
                                    // Load ONLY if it's a pointer (alloca)
                                    let ptr_val = var_val.into_pointer_value();
                                    let load_type =
                                        self.context.ptr_type(inkwell::AddressSpace::default());
                                    let current_val = self
                                        .builder
                                        .build_load(load_type, ptr_val, "old_val")
                                        .map_err(|e| e.to_string())?;
                                    self.builder
                                        .build_call(free_fn, &[current_val.into()], "")
                                        .map_err(|e| e.to_string())?;
                                }
                            }
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
                        // Assume it's a tensor iteration
                        let (tensor_val, tensor_ty) = self.compile_expr(iterator)?;
                        if !matches!(tensor_ty, Type::Tensor(_, _)) {
                            return Err("For loop iterator must be a tensor or range".into());
                        }
                        // Get tensor length
                        let len_fn = self
                            .module
                            .get_function("tl_tensor_len")
                            .ok_or("tl_tensor_len not found")?;
                        let len_call = self
                            .builder
                            .build_call(len_fn, &[tensor_val.into()], "tensor_len")
                            .map_err(|e| e.to_string())?;
                        let len = match len_call.try_as_basic_value() {
                            inkwell::values::ValueKind::Basic(v) => v.into_int_value(),
                            _ => return Err("Invalid tensor_len return".into()),
                        };

                        // Store tensor pointer for use in body
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
                        if let Type::Tensor(_, _) = iter_ty {
                            let len_fn = self
                                .module
                                .get_function("tl_tensor_len")
                                .ok_or("tl_tensor_len not found")?;
                            let len_call = self
                                .builder
                                .build_call(len_fn, &[iter_val.into()], "tensor_len")
                                .map_err(|e| e.to_string())?;
                            let len = match len_call.try_as_basic_value() {
                                inkwell::values::ValueKind::Basic(v) => v.into_int_value(),
                                _ => return Err("Invalid tensor_len return".into()),
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
                        } else {
                            return Err("For loop iterator must be a tensor or range".into());
                        }
                    }
                };

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
                let current_block = self.builder.get_insert_block().unwrap();
                let phi = self
                    .builder
                    .build_phi(i64_type, "for_idx")
                    .map_err(|e| e.to_string())?;

                // Add incoming from entry
                let entry_block = current_block
                    .get_previous_basic_block()
                    .unwrap_or(current_block);
                // We'll add proper incoming edges after building body

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
                    // Get element from tensor - use saved alloca since we're in a new scope
                    let tensor_alloca =
                        saved_tensor_alloca.ok_or("Tensor alloca not found for for-loop")?;
                    let load_type = self.context.ptr_type(inkwell::AddressSpace::default());
                    let tensor_ptr = self
                        .builder
                        .build_load(load_type, tensor_alloca, "tensor_ptr")
                        .map_err(|e| e.to_string())?
                        .into_pointer_value();

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
                        inkwell::values::ValueKind::Basic(v) => (v, Type::F32),
                        _ => return Err("Invalid tensor_get return".into()),
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
                if body_end_block.get_terminator().is_none() {
                    self.builder
                        .build_unconditional_branch(loop_header)
                        .map_err(|e| e.to_string())?;
                }

                // Add PHI incoming edges
                phi.add_incoming(&[(&start_val, entry_block), (&next_idx, body_end_block)]);

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

}
