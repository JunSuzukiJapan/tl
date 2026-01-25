use super::CodeGenerator;
use crate::compiler::ast::*;
use inkwell::types::BasicType;
use inkwell::values::*;

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
        match &expr.inner {
            ExprKind::IndexAccess(_, idxs) => {
                for idx in idxs {
                    if let ExprKind::Variable(name) = &idx.inner {
                        indices.insert(name.clone());
                    }
                    // Recursive check? Indices usually simple vars.
                }
            }
            ExprKind::BinOp(lhs, _, rhs) => {
                self.collect_indices(lhs, indices);
                self.collect_indices(rhs, indices);
            }
            ExprKind::UnOp(_, val) => {
                self.collect_indices(val, indices);
            }
            ExprKind::FnCall(_, args)
            | ExprKind::MethodCall(_, _, args)
            | ExprKind::StaticMethodCall(_, _, args) => {
                for arg in args {
                    self.collect_indices(arg, indices);
                }
            }
            ExprKind::TensorLiteral(elems) => {
                for elem in elems {
                    self.collect_indices(elem, indices);
                }
            }
            ExprKind::IfExpr(cond, _, _) => {
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
            Type::UserDefined(name, _) if name == "String" => {} // Skip String
            Type::Tensor(_, _) | Type::Struct(_, _) | Type::UserDefined(_, _) => {
                self.builder
                    .build_call(unreg_fn, &[val.into()], "")
                    .map_err(|e| e.to_string())?;
            }
            _ => {}
        }

        match ty {
            Type::Struct(name, _) | Type::UserDefined(name, _) => {
                if name == "String" {
                    return Ok(());
                }
                if let Some(struct_def) = self.struct_defs.get(name) {
                    let ptr = val.into_pointer_value();
                    let st_llvm_ty = *self.struct_types.get(name).unwrap();

                    for (i, (_, field_type)) in struct_def.fields.iter().enumerate() {
                        if matches!(
                            field_type,
                            Type::Tensor(_, _)
                                | Type::TensorShaped(_, _)
                                | Type::Struct(_, _)
                                | Type::UserDefined(_, _)
                                | Type::Enum(_, _)
                                | Type::Vec(_)
                                | Type::Tuple(_)
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

    /// Copy struct contents from src to dst pointer (used for sret)
    pub(crate) fn emit_struct_copy(
        &self,
        dst: inkwell::values::PointerValue<'ctx>,
        src: inkwell::values::PointerValue<'ctx>,
        ty: &Type,
    ) -> Result<(), String> {
        match ty {
            Type::Struct(name, _) | Type::UserDefined(name, _) => {
                let struct_def = self
                    .struct_defs
                    .get(name)
                    .ok_or(format!("Struct {} not found", name))?
                    .clone();
                let st_llvm_ty = *self
                    .struct_types
                    .get(name)
                    .ok_or(format!("LLVM struct type {} not found", name))?;

                for (i, (field_name, field_ty)) in struct_def.fields.iter().enumerate() {
                    let src_field_ptr = self
                        .builder
                        .build_struct_gep(st_llvm_ty, src, i as u32, &format!("src_{}", field_name))
                        .map_err(|e| e.to_string())?;
                    let dst_field_ptr = self
                        .builder
                        .build_struct_gep(st_llvm_ty, dst, i as u32, &format!("dst_{}", field_name))
                        .map_err(|e| e.to_string())?;

                    // Load field value from src
                    let llvm_field_ty: inkwell::types::BasicTypeEnum = match field_ty {
                        Type::F32 => self.context.f32_type().into(),
                        Type::I64 => self.context.i64_type().into(),
                        Type::I32 => self.context.i32_type().into(),
                        Type::Bool => self.context.bool_type().into(),
                        Type::Tensor(_, _) | Type::Struct(_, _) | Type::UserDefined(_, _) => self
                            .context
                            .ptr_type(inkwell::AddressSpace::default())
                            .into(),
                        _ => self.context.i64_type().into(),
                    };

                    let field_val = self
                        .builder
                        .build_load(llvm_field_ty, src_field_ptr, "field_val")
                        .map_err(|e| e.to_string())?;

                    // Deep Copy Logic:
                    // If field is Tensor/Struct/UserDefined, use emit_deep_clone (Recursively acquire/copy)
                    // Currently emit_deep_clone mallocs new structs, but here we want to store into dst_field_ptr.
                    // But emit_deep_clone returns a Value (Pointer to new struct or Tensor Ptr).
                    // So we store that Value into dst_field_ptr.
                    // This means dst (SRET buffer) will hold Pointers to the Deep Copied fields.
                    // This matches tl semantics (Structs contain pointers).
                    let store_val = if matches!(
                        field_ty,
                        Type::Tensor(_, _)
                            | Type::TensorShaped(_, _)
                            | Type::Struct(_, _)
                            | Type::UserDefined(_, _)
                            | Type::Enum(_, _)
                            | Type::Vec(_)
                            | Type::Tuple(_)
                    ) {
                        self.emit_deep_clone(field_val, field_ty)?
                    } else {
                        field_val
                    };

                    // Store to dst
                    self.builder
                        .build_store(dst_field_ptr, store_val)
                        .map_err(|e| e.to_string())?;
                }
                Ok(())
            }
            _ => Err(format!(
                "emit_struct_copy called on non-struct type: {:?}",
                ty
            )),
        }
    }

    pub(crate) fn emit_recursive_free(
        &self,
        val: BasicValueEnum<'ctx>,
        ty: &Type,
    ) -> Result<(), String> {
        match ty {
            Type::Enum(name, _) => {
                let enum_def = self
                    .enum_defs
                    .get(name)
                    .ok_or(format!("Enum def {} not found", name))?
                    .clone();
                let enum_ty = *self
                    .enum_types
                    .get(name)
                    .ok_or(format!("Enum type {} not found", name))?;

                let ptr = val.into_pointer_value();

                // Runtime Null Check
                let current_block = self.builder.get_insert_block().unwrap();
                let func = current_block.get_parent().unwrap();
                let free_block = self.context.append_basic_block(func, "free_enum");
                let merge_block = self.context.append_basic_block(func, "after_free_enum");

                let is_null = self
                    .builder
                    .build_is_null(ptr, "is_null")
                    .map_err(|e| e.to_string())?;
                self.builder
                    .build_conditional_branch(is_null, merge_block, free_block)
                    .map_err(|e| e.to_string())?;

                self.builder.position_at_end(free_block);

                // Load Tag (Element 0)
                let tag_ptr = self
                    .builder
                    .build_struct_gep(enum_ty, ptr, 0, "tag_ptr")
                    .map_err(|e| e.to_string())?;
                let tag_val = self
                    .builder
                    .build_load(self.context.i32_type(), tag_ptr, "tag")
                    .map_err(|e| e.to_string())?
                    .into_int_value();

                // Prepare Switch
                let after_switch = self.context.append_basic_block(func, "after_enum_switch");
                let mut cases = vec![];

                for (i, variant) in enum_def.variants.iter().enumerate() {
                    let case_block = self
                        .context
                        .append_basic_block(func, &format!("free_variant_{}", variant.name));
                    cases.push((
                        self.context.i32_type().const_int(i as u64, false),
                        case_block,
                    ));
                }

                // Build Switch
                let cases_refs: Vec<(inkwell::values::IntValue, inkwell::basic_block::BasicBlock)> =
                    cases.iter().map(|(i, b)| (*i, *b)).collect();
                self.builder
                    .build_switch(tag_val, after_switch, &cases_refs)
                    .map_err(|e| e.to_string())?;

                // Populate Cases
                for (i, variant) in enum_def.variants.iter().enumerate() {
                    let case_block = cases[i].1;
                    self.builder.position_at_end(case_block);

                    if !variant.fields.is_empty() {
                        // Cast Payload (Element 1 is [i8 x N])
                        let payload_ptr_raw = self
                            .builder
                            .build_struct_gep(enum_ty, ptr, 1, "payload_ptr_raw")
                            .map_err(|e| e.to_string())?;

                        // Reconstruct Variant Struct Type for GEP
                        let mut field_types: Vec<inkwell::types::BasicTypeEnum> = vec![];
                        for (_, ty) in &variant.fields {
                            let llvm_ty = match ty {
                                Type::F32 => self.context.f32_type().into(),
                                Type::I64 => self.context.i64_type().into(),
                                Type::Bool => self.context.bool_type().into(),
                                Type::Tensor(_, _) => self
                                    .context
                                    .ptr_type(inkwell::AddressSpace::default())
                                    .into(),
                                Type::Struct(_, _) | Type::Enum(_, _) | Type::UserDefined(_, _) => self
                                    .context
                                    .ptr_type(inkwell::AddressSpace::default())
                                    .into(),
                                Type::Vec(_) => self
                                    .context
                                    .ptr_type(inkwell::AddressSpace::default())
                                    .into(),
                                _ => self.context.i64_type().into(),
                            };
                            field_types.push(llvm_ty);
                        }
                        let variant_struct_ty = self.context.struct_type(&field_types, false);

                        // Cast payload ptr to variant struct ptr
                        let payload_ptr = self
                            .builder
                            .build_pointer_cast(
                                payload_ptr_raw,
                                self.context.ptr_type(inkwell::AddressSpace::default()), // Opaque ptr
                                "payload_cast",
                            )
                            .unwrap();

                        for (idx, (_, f_ty)) in variant.fields.iter().enumerate() {
                            if matches!(
                                f_ty,
                                Type::Tensor(_, _)
                                    | Type::TensorShaped(_, _)
                                    | Type::Struct(_, _)
                                    | Type::UserDefined(_, _)
                                    | Type::Enum(_, _)
                                    | Type::Vec(_)
                                    | Type::Tuple(_)
                            ) {
                                let f_ptr = self
                                    .builder
                                    .build_struct_gep(
                                        variant_struct_ty,
                                        payload_ptr,
                                        idx as u32,
                                        "field_ptr",
                                    )
                                    .map_err(|e| e.to_string())?;

                                let f_val = self
                                    .builder
                                    .build_load(
                                        self.context.ptr_type(inkwell::AddressSpace::default()),
                                        f_ptr,
                                        "field_val",
                                    )
                                    .map_err(|e| e.to_string())?;

                                self.emit_recursive_free(f_val, f_ty)?;
                            }
                        }
                    }
                    // After recursive calls, branch from current position to after_switch
                    if self.builder.get_insert_block().unwrap().get_terminator().is_none() {
                        self.builder
                            .build_unconditional_branch(after_switch)
                            .unwrap();
                    }
                }

                self.builder.position_at_end(after_switch);
                if self.builder.get_insert_block().unwrap().get_terminator().is_none() {
                    self.builder
                        .build_unconditional_branch(merge_block)
                        .unwrap();
                }

                self.builder.position_at_end(merge_block);
            }
            Type::Tensor(_, _) | Type::TensorShaped(_, _) => {
                if !val.is_pointer_value() {
                    return Err(format!("Tensor value is not pointer: {:?}", val));
                }
                let ptr = val.into_pointer_value();

                // Runtime Null Check
                let current_block = self.builder.get_insert_block().unwrap();
                let func = current_block.get_parent().unwrap();
                let free_block = self.context.append_basic_block(func, "free_tensor");
                let merge_block = self.context.append_basic_block(func, "after_free");

                let is_null = self
                    .builder
                    .build_is_null(ptr, "is_null")
                    .map_err(|e| e.to_string())?;
                self.builder
                    .build_conditional_branch(is_null, merge_block, free_block)
                    .map_err(|e| e.to_string())?;

                self.builder.position_at_end(free_block);

                let free_fn = self
                    .module
                    .get_function("tl_tensor_release")
                    .ok_or("tl_tensor_release not found")?;
                self.builder
                    .build_call(free_fn, &[val.into()], "")
                    .map_err(|e| e.to_string())?;

                self.builder
                    .build_unconditional_branch(merge_block)
                    .map_err(|e| e.to_string())?;
                self.builder.position_at_end(merge_block);
            }
            Type::Struct(name, _) | Type::UserDefined(name, _) => {
                // Skip String
                if name == "String" {
                    return Ok(());
                }

                let struct_def = self
                    .struct_defs
                    .get(name)
                    .ok_or(format!("Struct def {} not found", name))?
                    .clone();
                let struct_ty = *self
                    .struct_types
                    .get(name)
                    .ok_or(format!("Struct type {} not found", name))?;
                let ptr = val.into_pointer_value();

                // Runtime Null Check
                let current_block = self.builder.get_insert_block().unwrap();
                let func = current_block.get_parent().unwrap();
                let free_block = self.context.append_basic_block(func, "free_struct");
                let merge_block = self.context.append_basic_block(func, "after_free");

                let is_null = self
                    .builder
                    .build_is_null(ptr, "is_null")
                    .map_err(|e| e.to_string())?;
                self.builder
                    .build_conditional_branch(is_null, merge_block, free_block)
                    .map_err(|e| e.to_string())?;

                self.builder.position_at_end(free_block);

                // We need to ensure proper control flow when recursively freeing fields.
                // Each recursive call to emit_recursive_free may create new blocks and change
                // the builder's insert position. We must branch to merge_block from wherever
                // the builder ends up after all field cleanup.
                
                for (i, (_, f_ty)) in struct_def.fields.iter().enumerate() {
                    match f_ty {
                        Type::Tensor(_, _)
                        | Type::TensorShaped(_, _)
                        | Type::Struct(_, _)
                        | Type::UserDefined(_, _)
                        | Type::Enum(_, _)
                        | Type::Vec(_)
                        | Type::Tuple(_) => {
                            let f_ptr = self
                                .builder
                                .build_struct_gep(struct_ty, ptr, i as u32, "field_gep")
                                .map_err(|e| e.to_string())?;
                            let f_val = self
                                .builder
                                .build_load(
                                    self.context.ptr_type(inkwell::AddressSpace::default()),
                                    f_ptr,
                                    "field_load",
                                )
                                .map_err(|e| e.to_string())?;
                            // Recursively free - this may change builder position
                            self.emit_recursive_free(f_val, f_ty)?;
                        }
                        _ => {}
                    }
                }

                // After all recursive calls, builder is at some merge block.
                // We need to branch from HERE (current position) to our merge_block.
                // Check if current block already has a terminator
                if self.builder.get_insert_block().unwrap().get_terminator().is_none() {
                    self.builder
                        .build_unconditional_branch(merge_block)
                        .map_err(|e| e.to_string())?;
                }
                self.builder.position_at_end(merge_block);
            }
            Type::Vec(inner_ty) => {
                // Only support Vec<Tensor> or Vec<Struct> (pointer-sized elements) for now
                if matches!(
                    inner_ty.as_ref(),
                    Type::Tensor(_, _) | Type::Struct(_, _) | Type::UserDefined(_, _)
                ) {
                    let len_fn = self
                        .module
                        .get_function("tl_vec_void_len")
                        .ok_or("tl_vec_void_len not found")?;
                    let get_fn = self
                        .module
                        .get_function("tl_vec_void_get")
                        .ok_or("tl_vec_void_get not found")?;
                    let free_fn = self
                        .module
                        .get_function("tl_vec_void_free")
                        .ok_or("tl_vec_void_free not found")?;

                    let void_ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
                    let ptr = val.into_pointer_value();
                    let cast_ptr = self
                        .builder
                        .build_pointer_cast(ptr, void_ptr_type, "vec_ptr")
                        .unwrap();

                    // Get Length
                    let len_call = self
                        .builder
                        .build_call(len_fn, &[cast_ptr.into()], "len")
                        .map_err(|e| e.to_string())?;
                    let len_val = match len_call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => {
                            if v.is_int_value() {
                                v.into_int_value()
                            } else {
                                return Err(format!("len returned non-int: {:?}", v));
                            }
                        }
                        _ => return Err("len call returned non-basic value".to_string()),
                    };

                    // Loop Setup
                    let current_bb = self.builder.get_insert_block().unwrap();
                    let func = current_bb.get_parent().unwrap();
                    let loop_bb = self.context.append_basic_block(func, "vec_free_loop");
                    let body_bb = self.context.append_basic_block(func, "vec_free_body");
                    let end_bb = self.context.append_basic_block(func, "vec_free_end");

                    self.builder
                        .build_unconditional_branch(loop_bb)
                        .map_err(|e| e.to_string())?;

                    // Loop Header (Condition)
                    self.builder.position_at_end(loop_bb);
                    let i64_type = self.context.i64_type();
                    let idx_phi = self
                        .builder
                        .build_phi(i64_type, "i")
                        .map_err(|e| e.to_string())?;
                    idx_phi.add_incoming(&[(&i64_type.const_int(0, false), current_bb)]);

                    let idx_val = idx_phi.as_basic_value().into_int_value();
                    let cmp = self
                        .builder
                        .build_int_compare(inkwell::IntPredicate::ULT, idx_val, len_val, "cmp")
                        .map_err(|e| e.to_string())?;
                    self.builder
                        .build_conditional_branch(cmp, body_bb, end_bb)
                        .map_err(|e| e.to_string())?;

                    // Loop Body
                    self.builder.position_at_end(body_bb);
                    let elem_ptr_call = self
                        .builder
                        .build_call(get_fn, &[cast_ptr.into(), idx_val.into()], "elem_ptr")
                        .map_err(|e| e.to_string())?;
                    let elem_ptr_void = match elem_ptr_call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err("elem_ptr call returned non-basic value".to_string()),
                    };

                    let cast_elem = self
                        .builder
                        .build_pointer_cast(
                            elem_ptr_void.into_pointer_value(),
                            void_ptr_type,
                            "cast_elem",
                        )
                        .unwrap();
                    self.emit_recursive_free(cast_elem.into(), inner_ty)?;

                    // Increment and Jump
                    let next_idx = self
                        .builder
                        .build_int_add(idx_val, i64_type.const_int(1, false), "next_i")
                        .map_err(|e| e.to_string())?;
                    idx_phi.add_incoming(&[(&next_idx, body_bb)]);
                    self.builder
                        .build_unconditional_branch(loop_bb)
                        .map_err(|e| e.to_string())?;

                    // End
                    self.builder.position_at_end(end_bb);
                    self.builder
                        .build_call(free_fn, &[cast_ptr.into()], "")
                        .map_err(|e| e.to_string())?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    pub(crate) fn compile_stmt(&mut self, stmt: &Stmt) -> Result<(), String> {
        let prev_span = self.current_span.clone();
        self.current_span = Some(stmt.span.clone());
        let result = self.compile_stmt_inner(stmt);
        if result.is_ok() {
            let terminated = self
                .builder
                .get_insert_block()
                .and_then(|b| b.get_terminator())
                .is_some();
            if !terminated {
                let tag = stmt_trace_tag(stmt);
                let _ = self.emit_trace_mem(tag);
            }
        }
        self.current_span = prev_span;
        result
    }

    pub(crate) fn compile_stmt_inner(&mut self, stmt: &Stmt) -> Result<(), String> {
        match &stmt.inner {
            StmtKind::Use { .. } => Ok(()),
            StmtKind::FieldAssign {
                obj,
                field,
                op,
                value,
            } => {
                let (obj_val, obj_ty) = self.compile_expr(obj)?;
                let struct_name = match obj_ty {
                    Type::Struct(name, _) => name,
                    Type::UserDefined(name, _) => name,
                    _ => return Err(format!("Field assignment on non-struct type {:?}", obj_ty)),
                };

                let simple_struct_name = if struct_name.contains("::") {
                    struct_name.split("::").last().unwrap()
                } else {
                    &struct_name
                };

                let (field_idx, field_type) = {
                    let struct_def = self
                        .struct_defs
                        .get(simple_struct_name)
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
                let st_llvm_ty = *self.struct_types.get(simple_struct_name).unwrap();

                let field_ptr = self
                    .builder
                    .build_struct_gep(st_llvm_ty, ptr, field_idx as u32, &format!("ptr_{}", field))
                    .map_err(|e| e.to_string())?;

                let (val, val_ty) = self.compile_expr(value)?;

                let final_val = match op {
                    AssignOp::Assign => val,
                    _ => {
                        let load_type: inkwell::types::BasicTypeEnum = match &field_type {
                            Type::I64 => self.context.i64_type().into(),
                            Type::F32 => self.context.f32_type().into(),
                            Type::Tensor(_, _) => self.context.ptr_type(inkwell::AddressSpace::default()).into(),
                            _ => return Err(format!("Unsupported type for FieldAssign op: {:?}", field_type)),
                        };

                        let current_val = self
                            .builder
                            .build_load(load_type, field_ptr, "field_current")
                            .map_err(|e| e.to_string())?;

                        // Tensor optimization path (In-Place)
                        if let Type::Tensor(_, _) = field_type {
                            let in_place_fn_name = match op {
                                AssignOp::SubAssign => Some("tl_tensor_sub_assign"),
                                AssignOp::MulAssign => Some("tl_tensor_mul_assign"),
                                AssignOp::DivAssign => Some("tl_tensor_div_assign"),
                                AssignOp::ModAssign => Some("tl_tensor_mod_assign"),
                                _ => None,
                            };

                            if let Some(base_fn_name) = in_place_fn_name {
                                let (fn_name, is_scalar) = if matches!(val_ty, Type::Tensor(_, _)) {
                                    (base_fn_name.to_string(), false)
                                } else {
                                    (format!("{}_scalar_f32", base_fn_name), true)
                                };

                                let target_fn = self.module.get_function(&fn_name).ok_or(format!("Function {} not found", fn_name))?;

                                let val_arg: inkwell::values::BasicValueEnum = if is_scalar {
                                    // Cast to F32
                                    let scalar_f32: inkwell::values::FloatValue = match val_ty {
                                        Type::F32 => val.into_float_value(),
                                        Type::F64 => self.builder.build_float_cast(val.into_float_value(), self.context.f32_type(), "f32_cast").unwrap(),
                                        Type::I64 | Type::I32 => self.builder.build_signed_int_to_float(val.into_int_value(), self.context.f32_type(), "f32_cast").unwrap(),
                                        _ => return Err(format!("Cannot convert {:?} to f32 for scalar op", val_ty)),
                                    };
                                    scalar_f32.into()
                                } else {
                                    val
                                };

                                self.builder.build_call(target_fn, &[current_val.into(), val_arg.into()], "").map_err(|e| e.to_string())?;
                                return Ok(());
                            }
                        }

                        // Normal path (AddAssign or Tensor generic path)
                        let bin_op = match op {
                            AssignOp::AddAssign => BinOp::Add,
                            AssignOp::SubAssign => BinOp::Sub,
                            AssignOp::MulAssign => BinOp::Mul,
                            AssignOp::DivAssign => BinOp::Div,
                            AssignOp::ModAssign => BinOp::Mod,
                            _ => unreachable!(),
                        };

                        let (res, _) = self.compile_bin_op(current_val, field_type.clone(), val, val_ty, bin_op)?;
                        res
                    }
                };

                // Free old value if it's a structural type (Tensor/Struct)
                if matches!(
                    field_type,
                    Type::Tensor(_, _) | Type::Struct(_, _) | Type::UserDefined(_, _)
                ) {
                    let load_type = self.context.ptr_type(inkwell::AddressSpace::default());
                    let current_val = self
                        .builder
                        .build_load(load_type, field_ptr, "old_field_val")
                        .map_err(|e| e.to_string())?
                        .into_pointer_value();

                    let val_ptr = final_val.into_pointer_value();
                    let are_diff = self
                        .builder
                        .build_int_compare(
                            inkwell::IntPredicate::NE,
                            current_val,
                            val_ptr,
                            "field_free_diff",
                        )
                        .map_err(|e| e.to_string())?;

                    let free_block = self.context.append_basic_block(
                        self.builder.get_insert_block().unwrap().get_parent().unwrap(),
                        "field_free",
                    );
                    let skip_block = self.context.append_basic_block(
                        self.builder.get_insert_block().unwrap().get_parent().unwrap(),
                        "field_skip_free",
                    );

                    self.builder.build_conditional_branch(are_diff, free_block, skip_block).unwrap();
                    self.builder.position_at_end(free_block);
                    self.emit_recursive_free(current_val.into(), &field_type)?;
                    self.builder.build_unconditional_branch(skip_block).unwrap();
                    self.builder.position_at_end(skip_block);
                }

                self.builder.build_store(field_ptr, final_val).map_err(|e| e.to_string())?;

                // Ownership transfer
                if let Some(f) = self.module.get_function("tl_mem_unregister") {
                    let should_unregister = match &field_type {
                        Type::Tensor(_, _) | Type::Struct(_, _) | Type::UserDefined(_, _) => true,
                        _ => false,
                    };
                    if should_unregister {
                        self.builder.build_call(f, &[final_val.into()], "").map_err(|e| e.to_string())?;
                    }
                }

                Ok(())
            }
            StmtKind::TensorDecl {
                name,
                type_annotation,
                init,
            } => {
                if let Some(expr) = init {
                    let (val_ir, _inferred_ty) = self.ensure_tensor_v2(expr, 0)?;
                    let val_ty = if matches!(type_annotation, Type::Tensor(_, _)) {
                        type_annotation.clone()
                    } else if matches!(type_annotation, Type::ScalarArray(_, _)) {
                        type_annotation.clone()
                    } else {
                        // tensor name: f32 means Tensor<f32, 0>
                        Type::Tensor(Box::new(type_annotation.clone()), 0)
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
                        let ptr = self.create_entry_block_alloca(fn_val, name, &val_ty)?;
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
            StmtKind::Let {
                name,
                type_annotation,
                value,
                mutable: _,
            } => {
                // 1. Analyze value for Free Indices (Implicit Tensor Equation)
                let free_indices = self.infer_free_indices(value);

                if !free_indices.is_empty() {
                    // Found free indices -> It's a tensor equation! e.g. let C = A[i, j] * B[j, k];
                    // free_indices will be ["i", "k"] (sorted)
                    // Delegate to compile_tensor_equation
                    // Helper to convert indices string to generator format
                    // Delegate to compile_tensor_equation
                    // Helper to convert indices string to generator format
                    // Note: Implicit equations don't have explicit ranges here, effectively 0..Limit inferred later.
                    // But our AST expects clauses now.
                    // compile_tensor_equation expects `&[ComprehensionClause]`.
                    // But wait, `compile_tensor_equation` generates loops FROM clauses.
                    // Implicit equations: `let C = A[i]`. LHS=C (no explicit indices). RHS has free `i`.
                    // We treat this as `[ i | A[i] ]`.
                    // So we need to synthesize `indices=["i"]` and `clauses=[Generator("i", IMPLICIT)]`?
                    // No, existing logic for explicit comprehension has explicit ranges.
                    // Implicit reduction/equation relies on bounds inference.

                    // We should invoke `compile_tensor_equation` with:
                    // indices = free_indices
                    // clauses = [] (No explicit generators)
                    // body = value

                    // But `compile_tensor_equation` iterates `clauses` to generate loops!
                    // If clauses is empty, it generates NO LOOPS.
                    // This breaks implicit equations.

                    // Fix: `compile_tensor_equation` accepts `indices` (LHS/Output) AND `clauses`.
                    // If `clauses` is empty, it should try to infer loops from `indices` + `free vars`?
                    // OR: We must synthesize clauses effectively.
                    // We don't know bounds easily here without analysis.
                    // Actually, `compile_tensor_equation` does `self.enter_scope()` and bounds lookup.
                    // But it expects clauses to drive the loops.

                    // REVERT/ADJUSTMENT to `compile_tensor_equation`:
                    // It should take `indices: &[String]` (Output dims) and `clauses: &[Clause]`.
                    // It should ALSO take `implicit_indices: &[String]`?
                    // Or we just synthesize clauses.
                    // But we can't synthesize semantic `Expr` for bounds easily.

                    // Let's modify `compile_tensor_equation` to accept an optional "Force Loops for these indices" argument?
                    // Or better: Use `ExprKind::TensorComprehension` AST node effectively.
                    // Implicit equation `C = A[i]` is semantically `C = [i | A[i]]` where `i` range is inferred.
                    // Our new syntax supports `[i | A[i]]` (Implicit body? No, implicit generator?).
                    // `i` is in LHS. RHS has no generator for `i`.
                    // Logic must handle "LHS index NOT in generators".
                    // Existing logic (old): matched `(idx, range_opt)`. If `range_opt` None, inferred.
                    // New logic: `Clause::Generator` HAS `range: Expr`. Mandatory.

                    // So we need a way to represent "Generator with Implicit Range" in `Clause`.
                    // We can add `ExprKind::ImplicitRange`? Or `Option<Expr>` in `Generator` variant?

                    // Let's update `ComprehensionClause` definition to allow optional range?
                    // `Generator { name: String, range: Option<Expr> }`.
                    // This restores compatibility with implicit generators.

                    // Implicit equation `C = A[i]` is semantically `C = [i | { A[i] }]` (Empty clauses).
                    // `compile_tensor_equation` will detect that `i` is in `indices` but not in `clauses`,
                    // and will infer the range from `body` (value).

                    let clauses: Vec<ComprehensionClause> = Vec::new();

                    return self
                        .compile_tensor_equation(name, &free_indices, &clauses, Some(value))
                        .map_err(|e| e.to_string());
                }

                let (mut val_ir, mut val_ty) = self.compile_expr(value)?;

                // Ownership: Shared. The temporary (value) remains in scope and will be released at scope exit.
                // The variable (name) acquires a NEW reference via deep_clone below.
                // We do NOT unregister the temporary. Ref 1 (Temp) + Ref 1 (Var) = 2.
                // Temp Scope Exit -> -1. Var Scope Exit -> -1. Total 0. Safe.

                // Removed: Move Semantics logic.
                // We default to CLONE for variables (see below), so we should NOT disable cleanup for the source.

                // Convert ScalarArray to Tensor if explicitly requested as Tensor
                if let Some(target_ty) = type_annotation {
                    if matches!(target_ty, Type::Tensor(_, _))
                        && matches!(val_ty, Type::ScalarArray(_, _))
                    {
                        let (v, _t) = self.ensure_tensor_v2(value, 0)?;
                    } else if let Type::UserDefined(ref n, _) = target_ty {
                         if n.starts_with("Vec") {
                             // Fix for Vec::new() -> Vec<Void> being assigned to Vec<T> (or Vec_T)
                             if matches!(val_ty, Type::Vec(_)) || matches!(val_ty, Type::UserDefined(ref vn, _) if vn == "Vec") {
                                 val_ty = target_ty.clone();
                             }
                         }
                    }
                }

                // Variable Assignment: Deep Clone (Struct Copy + Tensor Acquire)
                // Optimization: R-value Move Semantics
                // If the value is a temporary (FnCall, BinOp, etc), we take ownership (Move).
                // If the value is an L-value (Variable, FieldAccess), we must Copy (Acquire/Clone).

                let is_rvalue = matches!(
                    &value.inner,
                    ExprKind::FnCall(_, _)
                        | ExprKind::MethodCall(_, _, _)
                        | ExprKind::StaticMethodCall(_, _, _)
                        | ExprKind::BinOp(_, _, _)
                        | ExprKind::UnOp(_, _)
                        | ExprKind::TensorLiteral(_)
                        | ExprKind::IfExpr(_, _, _) // Treating IfExpr as R-value (Assumes IfExpr logic ensures failure-safety)
                        | ExprKind::Block(_)
                );

                let should_deep_clone = match &val_ty {
                    Type::Tensor(_, _) | Type::TensorShaped(_, _) => !is_rvalue, // Clone only if L-value
                    Type::Struct(_, _) | Type::UserDefined(_, _) | Type::Enum(_, _) | Type::Vec(_) | Type::Tuple(_) => {
                        // Structs/UserDefined/Enum/Vec/Tuple: Pointer copy vs Deep Clone
                        // If R-value, we own the pointer. Move.
                        !is_rvalue
                    }
                    _ => false,
                };

                if should_deep_clone {
                    val_ir = self.emit_deep_clone(val_ir, &val_ty)?;
                }

                let current_function = self
                    .builder
                    .get_insert_block()
                    .unwrap()
                    .get_parent()
                    .unwrap();

                // Check for shadowing in CURRENT scope
                let shadow_info = if let Some(scope) = self.variables.last() {
                    if let Some((old_ptr, old_ty, should_free)) = scope.get(name) {
                        if *should_free {
                            Some((*old_ptr, old_ty.clone()))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };

                if let Some((old_ptr_val, old_ty)) = shadow_info {
                    // Load the actual pointer value from the alloca
                    let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
                    let old_value = self
                        .builder
                        .build_load(ptr_type, old_ptr_val.into_pointer_value(), "old_shadowed")
                        .map_err(|e| e.to_string())?;

                    self.emit_recursive_free(old_value, &old_ty)?;
                }

                let alloca = self.create_entry_block_alloca(current_function, name, &val_ty)?;
                self.builder
                    .build_store(alloca, val_ir)
                    .map_err(|e| e.to_string())?;

                self.variables
                    .last_mut()
                    .unwrap()
                    .insert(name.clone(), (alloca.into(), val_ty.clone(), true)); // Store pointer and type
                                                                                  /*
                                                                                  match &val_ty {
                                                                                      Type::Tensor(_, _) => {
                                                                                          if let Some(register_fn) =
                                                                                              self.module.get_function("tl_mem_register_tensor")
                                                                                          {
                                                                                              // Load the pointer from alloca (val_ir is the value to store, so it's the pointer)
                                                                                              // val_ir is the T* (OpaqueTensor*)
                                                                                              self.builder
                                                                                                  .build_call(register_fn, &[val_ir.into()], "")
                                                                                                  .map_err(|e| e.to_string())?;
                                                                                          }
                                                                                      }
                                                                                      Type::Struct(_, _) | Type::UserDefined(_, _) => {
                                                                                          /*
                                                                                          if let Some(register_fn) = self.module.get_function("tl_mem_register_struct") {
                                                                                              self.builder
                                                                                                  .build_call(register_fn, &[val_ir.into()], "")
                                                                                                  .map_err(|e| e.to_string())?;
                                                                                          }
                                                                                          */
                                                                                      }
                                                                                      _ => {}
                                                                                  }
                                                                                  */
                Ok(())
            }
            StmtKind::Return(expr_opt) => {
                if let Some(expr) = expr_opt {
                    // If returning a variable, mark it as moved (should_free = false)
                    if let ExprKind::Variable(name) = &expr.inner {
                        for scope in self.variables.iter_mut().rev() {
                            if let Some(entry) = scope.get_mut(name) {
                                entry.2 = false;
                                break;
                            }
                        }
                    }
                    let (val, ty) = self.compile_expr(expr)?;

                    // Check if this is a struct return (uses sret)
                    let uses_sret = false; /* SRET DISABLED */

                    // IMPORTANT: Do NOT unregister. Instead Acquire/Copy to preserve for caller.
                    // If we unregister, it releases (decrements refcount).
                    // If we exit scope, it releases (decrements refcount).
                    // Result: Double decrement -> Free.
                    // Fix:
                    // 1. For SRET: emit_struct_copy (above) now does Deep Copy + Acquire.
                    // 2. For Tensor Return: We must Acquire.
                    // 3. For Struct Return: We must Unregister to prevent exit_scope from freeing it.
                    if !uses_sret {
                        match &ty {
                            Type::Tensor(_, _) => {
                                if let Some(acquire_fn) =
                                    self.module.get_function("tl_tensor_acquire")
                                {
                                    let ptr = val.into_pointer_value();
                                    let void_ptr_type =
                                        self.context.ptr_type(inkwell::AddressSpace::default());
                                    let cast_ptr = self
                                        .builder
                                        .build_pointer_cast(ptr, void_ptr_type, "cast_aq_ret")
                                        .unwrap();
                                    self.builder
                                        .build_call(acquire_fn, &[cast_ptr.into()], "")
                                        .unwrap();
                                }
                            }
                            Type::Struct(_, _) | Type::UserDefined(_, _) => {
                                // CRITICAL FIX: Unregister struct from scope to transfer ownership to caller.
                                // Without this, exit_scope will free the struct before the caller can use it.
                                // CRITICAL FIX: recursively unregister struct fields (like Tensors)
                                // so they are not freed by exit_scope.
                                self.emit_recursive_unregister(val, &ty)?;
                            }
                            _ => {}
                        }
                    }

                    if uses_sret {
                        // CRITICAL: Copy to sret BEFORE cleanup to avoid stale pointer access
                        // Get the sret pointer (first parameter)
                        let current_fn = self
                            .builder
                            .get_insert_block()
                            .and_then(|b| b.get_parent())
                            .ok_or("No current function")?;
                        let sret_ptr = current_fn
                            .get_nth_param(0)
                            .ok_or("Sret function missing sret parameter")?
                            .into_pointer_value();

                        // Copy struct contents to sret pointer BEFORE cleanup
                        let src_ptr = val.into_pointer_value();
                        self.emit_struct_copy(sret_ptr, src_ptr, &ty)?;
                        self.emit_all_scopes_cleanup();
                        self.builder.build_return(None).map_err(|e| e.to_string())?;
                    } else {
                        // Normal return: cleanup then return value
                        self.emit_all_scopes_cleanup();
                        self.builder
                            .build_return(Some(&val))
                            .map_err(|e| e.to_string())?;
                    }
                } else {
                    // return; (Void return)
                    self.emit_all_scopes_cleanup();
                    self.builder.build_return(None).map_err(|e| e.to_string())?;
                }
                Ok(())
            }
            StmtKind::Assign {
                name,
                indices,
                op,
                value,
            } => {
                if let Some(idxs) = indices {
                    if !idxs.is_empty() {
                        if *op != AssignOp::Assign {
                            return Err(
                                "Only direct assignment (=) supported for indexed assignment"
                                    .into(),
                            );
                        }

                        // 1. Resolve variable
                        let mut found_var_ptr = None;
                        let mut found_var_type = None;
                        for scope in self.variables.iter().rev() {
                            if let Some((v, t, _)) = scope.get(name) {
                                found_var_ptr = Some(*v);
                                found_var_type = Some(t.clone());
                                break;
                            }
                        }
                        let var_ptr =
                            found_var_ptr.ok_or(format!("Variable {} not found", name))?;
                        let var_type =
                            found_var_type.ok_or(format!("Variable {} not found", name))?;

                        // 2. Compile Value
                        let (val_ir, val_ty) = self.compile_expr(value)?;

                        match var_type {
                            Type::ScalarArray(elem_ty, _len) => {
                                if idxs.len() != 1 {
                                    return Err("ScalarArray only supports 1D indexing".into());
                                }
                                // Load Array Pointer
                                let ptr_type =
                                    self.context.ptr_type(inkwell::AddressSpace::default());
                                let array_ptr = self
                                    .builder
                                    .build_load(ptr_type, var_ptr.into_pointer_value(), "arr_ptr")
                                    .map_err(|e| e.to_string())?
                                    .into_pointer_value();

                                // Compile Index
                                let (idx_val, idx_ty) = self.compile_expr(&idxs[0])?;
                                let idx_int = match idx_ty {
                                    Type::I64 => idx_val.into_int_value(),
                                    Type::I32 => self
                                        .builder
                                        .build_int_z_extend(
                                            idx_val.into_int_value(),
                                            self.context.i64_type(),
                                            "zext",
                                        )
                                        .unwrap(),
                                    _ => return Err("Index must be integer".into()),
                                };

                                // GEP
                                let elem_llvm_ty: inkwell::types::BasicTypeEnum =
                                    match elem_ty.as_ref() {
                                        Type::I64 => self.context.i64_type().into(),
                                        Type::F32 => self.context.f32_type().into(),
                                        _ => self.context.i64_type().into(),
                                    };

                                let elem_ptr = unsafe {
                                    self.builder
                                        .build_in_bounds_gep(
                                            elem_llvm_ty,
                                            array_ptr,
                                            &[idx_int],
                                            "elem_ptr",
                                        )
                                        .map_err(|e| e.to_string())?
                                };

                                // Cast logic
                                // (If needed, e.g. i64 -> f32?)
                                // Assume types match for now or basic int/float check above

                                self.builder
                                    .build_store(elem_ptr, val_ir)
                                    .map_err(|e| e.to_string())?;
                                return Ok(());
                            }
                            Type::Tensor(_, _) => {
                                // Compile Indices to Array
                                let i64_type = self.context.i64_type();
                                let idx_array_type = i64_type.array_type(idxs.len() as u32);

                                let current_block = self.builder.get_insert_block().unwrap();
                                let function = current_block.get_parent().unwrap();
                                let entry_block = function.get_first_basic_block().unwrap();
                                let entry_builder = self.context.create_builder();
                                if let Some(first_instr) = entry_block.get_first_instruction() {
                                    entry_builder.position_before(&first_instr);
                                } else {
                                    entry_builder.position_at_end(entry_block);
                                }
                                let idx_ptr_alloca = entry_builder
                                    .build_alloca(idx_array_type, "indices_arr")
                                    .map_err(|e| e.to_string())?;

                                for (i, idx_expr) in idxs.iter().enumerate() {
                                    let (val, ty) = self.compile_expr(idx_expr)?;
                                    let int_val = match ty {
                                        Type::I64 => val.into_int_value(),
                                        Type::I32 => self
                                            .builder
                                            .build_int_z_extend(
                                                val.into_int_value(),
                                                i64_type,
                                                "ext",
                                            )
                                            .unwrap(),
                                        _ => return Err("Index must be int".into()),
                                    };
                                    let ptr = unsafe {
                                        self.builder
                                            .build_in_bounds_gep(
                                                idx_array_type,
                                                idx_ptr_alloca,
                                                &[
                                                    i64_type.const_int(0, false),
                                                    i64_type.const_int(i as u64, false),
                                                ],
                                                "idx_ptr",
                                            )
                                            .map_err(|e| e.to_string())?
                                    };
                                    self.builder
                                        .build_store(ptr, int_val)
                                        .map_err(|e| e.to_string())?;
                                }

                                // Load current tensor ptr
                                let load_type =
                                    self.context.ptr_type(inkwell::AddressSpace::default());
                                let current_tensor = self
                                    .builder
                                    .build_load(load_type, var_ptr.into_pointer_value(), "curr_t")
                                    .unwrap();

                                // Call set fxn
                                let set_fn = self
                                    .module
                                    .get_function("tl_tensor_set_f32_md")
                                    .ok_or("tl_tensor_set_f32_md not found")?;

                                let idx_ptr_cast = self
                                    .builder
                                    .build_pointer_cast(
                                        idx_ptr_alloca,
                                        self.context.ptr_type(inkwell::AddressSpace::default()),
                                        "idx_cast",
                                    )
                                    .unwrap();

                                // Ensure Val is F32
                                let f32_val = match val_ty {
                                    Type::F32 => val_ir.into_float_value(),
                                    Type::I64 => self
                                        .builder
                                        .build_signed_int_to_float(
                                            val_ir.into_int_value(),
                                            self.context.f32_type(),
                                            "f32_cast",
                                        )
                                        .unwrap(),
                                    _ => {
                                        return Err(
                                            "Assignment value must be convertible to f32".into()
                                        )
                                    }
                                };

                                let call = self
                                    .builder
                                    .build_call(
                                        set_fn,
                                        &[
                                            current_tensor.into(),
                                            idx_ptr_cast.into(),
                                            i64_type.const_int(idxs.len() as u64, false).into(),
                                            f32_val.into(),
                                        ],
                                        "new_t",
                                    )
                                    .unwrap();

                                let new_tensor_ptr = match call.try_as_basic_value() {
                                    inkwell::values::ValueKind::Basic(v) => v,
                                    _ => return Err("tl_tensor_set_f32_md returned void".into()),
                                };

                                // Fix: Check if new_tensor_ptr == current_tensor (In-Place Update)
                                // Only free current_tensor if it is DIFFERENT from new_tensor_ptr.
                                let are_diff = self
                                    .builder
                                    .build_int_compare(
                                        inkwell::IntPredicate::NE,
                                        current_tensor.into_pointer_value(),
                                        new_tensor_ptr.into_pointer_value(),
                                        "are_tensors_diff",
                                    )
                                    .map_err(|e| e.to_string())?;

                                let free_block = self.context.append_basic_block(
                                    self.builder
                                        .get_insert_block()
                                        .unwrap()
                                        .get_parent()
                                        .unwrap(),
                                    "free_old_tensor",
                                );
                                let continue_block = self.context.append_basic_block(
                                    self.builder
                                        .get_insert_block()
                                        .unwrap()
                                        .get_parent()
                                        .unwrap(),
                                    "continue_assign",
                                );

                                self.builder
                                    .build_conditional_branch(are_diff, free_block, continue_block)
                                    .map_err(|e| e.to_string())?;

                                // Free Block
                                self.builder.position_at_end(free_block);
                                let free_fn = self
                                    .module
                                    .get_function("tl_tensor_free")
                                    .ok_or("tl_tensor_free not found")?;
                                self.builder
                                    .build_call(free_fn, &[current_tensor.into()], "")
                                    .map_err(|e| e.to_string())?;
                                self.builder
                                    .build_unconditional_branch(continue_block)
                                    .map_err(|e| e.to_string())?;

                                // Continue Block
                                self.builder.position_at_end(continue_block);

                                // Store New Tensor
                                self.builder
                                    .build_store(var_ptr.into_pointer_value(), new_tensor_ptr)
                                    .map_err(|e| e.to_string())?;

                                // Scope Promotion / Registration
                                if !self.is_outer_scope(name) {
                                    self.emit_register_tensor(new_tensor_ptr, &var_type)?;
                                }

                                return Ok(());
                            }
                            _ => {
                                return Err(
                                    "Indexed assignment only supported for Tensor and ScalarArray"
                                        .into(),
                                )
                            }
                        }
                    }
                }

                // Compile value first
                let (val_base, val_type) = self.compile_expr(value)?;

                // Clone if alias (initializing from variable or field) to prevent sharing pointers
                let val = if matches!(
                    &value.inner,
                    ExprKind::Variable(_) | ExprKind::FieldAccess(_, _)
                ) {
                    if let Type::Tensor(_, _) = val_type {
                        let clone_fn = self
                            .module
                            .get_function("tl_tensor_clone")
                            .ok_or("tl_tensor_clone not found")?;
                        let call = self
                            .builder
                            .build_call(clone_fn, &[val_base.into()], "cloned")
                            .map_err(|e| e.to_string())?;

                        self.check_tensor_result(call, "cloned_error")?
                    } else {
                        val_base
                    }
                } else {
                    val_base
                };

                // Lookup variable
                let mut found_var_ptr = None;
                let mut found_var_type = None;
                let mut found_should_free = false;
                for scope in self.variables.iter().rev() {
                    if let Some((v, t, sf)) = scope.get(name) {
                        found_var_ptr = Some(*v);
                        found_var_type = Some(t.clone());
                        found_should_free = *sf;
                        break;
                    }
                }

                let var_ptr = found_var_ptr.ok_or(format!("Variable {} not found", name))?;
                let var_type = found_var_type.ok_or(format!("Variable {} not found", name))?;

                // Fix: Ownership Transfer (Runtime -> Compiler)
                // Unregister the new value (RHS) to take ownership.
                match val_type {
                    Type::Struct(_, _) | Type::UserDefined(_, _) | Type::Tensor(_, _) => {
                        if let Some(unreg_fn) = self.module.get_function("tl_mem_unregister") {
                            let ptr = val.into_pointer_value();
                            let cast_ptr = self
                                .builder
                                .build_pointer_cast(
                                    ptr,
                                    self.context.ptr_type(inkwell::AddressSpace::default()),
                                    "cast_unreg_assign",
                                )
                                .unwrap();
                            self.builder
                                .build_call(unreg_fn, &[cast_ptr.into()], "")
                                .unwrap();
                        }
                    }
                    _ => {}
                }

                if let Some(idxs) = indices {
                    if !idxs.is_empty() {
                        return Err("Indexed assignment not yet supported".into());
                    }
                }

                // Handle assignment operator (e.g., +=, -=, =)
                let final_val = match op {
                    AssignOp::Assign => {
                        // Free old value if it is a Struct OR Tensor
                        if matches!(
                            var_type,
                            Type::Struct(_, _) | Type::UserDefined(_, _) | Type::Tensor(_, _)
                        ) {
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

                            // AND also check if we own it
                            let should_free_val = self
                                .context
                                .bool_type()
                                .const_int(found_should_free as u64, false);

                            // AND check if pointers differ (prevent self-free on return self)
                            let new_ptr = val.into_pointer_value();
                            let are_diff = self
                                .builder
                                .build_int_compare(
                                    inkwell::IntPredicate::NE,
                                    current_val,
                                    new_ptr,
                                    "are_diff",
                                )
                                .map_err(|e| e.to_string())?;

                            let can_free_1 = self
                                .builder
                                .build_and(is_not_null, should_free_val, "can_free_1")
                                .unwrap();
                            let can_free = self
                                .builder
                                .build_and(can_free_1, are_diff, "can_free")
                                .unwrap();

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
                                .build_conditional_branch(can_free, free_block, continue_block)
                                .map_err(|e| e.to_string())?;

                            // Free block
                            self.builder.position_at_end(free_block);

                            // Recursive free fields of the OLD struct
                            self.emit_recursive_free(current_val.into(), &var_type)?;

                            // Also unregister the struct shell itself so Runtime doesn't track it
                            if let Some(unreg_fn) = self.module.get_function("tl_mem_unregister") {
                                let cast_ptr = self
                                    .builder
                                    .build_pointer_cast(
                                        current_val,
                                        self.context.ptr_type(inkwell::AddressSpace::default()),
                                        "cast_unreg_struct",
                                    )
                                    .unwrap();
                                let _ = self.builder.build_call(unreg_fn, &[cast_ptr.into()], "");
                            }

                            // Note: We don't free(malloc) the struct shell. It leaks (small).

                            self.builder
                                .build_unconditional_branch(continue_block)
                                .map_err(|e| e.to_string())?;

                            // Continue block
                            self.builder.position_at_end(continue_block);
                        }

                        // Duplicate Tensor free logic removed

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
                                } else if matches!(var_type, Type::Struct(_, _) | Type::UserDefined(_, _))
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
                        if let Type::Tensor(_, _) = var_type {
                            let load_type = self.context.ptr_type(inkwell::AddressSpace::default());
                            let current_val = self
                                .builder
                                .build_load(
                                    load_type,
                                    var_ptr.into_pointer_value(),
                                    &format!("{}_current", name),
                                )
                                .map_err(|e| e.to_string())?;

                            let sub_assign_fn = if matches!(val_type, Type::Tensor(_, _)) {
                                self.module.get_function("tl_tensor_sub_assign").unwrap()
                            } else {
                                self.module
                                    .get_function("tl_tensor_sub_assign_scalar_f32")
                                    .unwrap()
                            };

                            let val_arg = if matches!(val_type, Type::Tensor(_, _)) {
                                val.into()
                            } else {
                                let scalar_f32: inkwell::values::FloatValue = match val_type {
                                    Type::F32 => val.into_float_value(),
                                    Type::F64 => self
                                        .builder
                                        .build_float_cast(
                                            val.into_float_value(),
                                            self.context.f32_type(),
                                            "f64_to_f32",
                                        )
                                        .map_err(|e| e.to_string())?,
                                    Type::I64 | Type::I32 => self
                                        .builder
                                        .build_signed_int_to_float(
                                            val.into_int_value(),
                                            self.context.f32_type(),
                                            "int_to_f32",
                                        )
                                        .map_err(|e| e.to_string())?,
                                    _ => {
                                        return Err(format!(
                                            "SubAssign: unsupported RHS type {:?}",
                                            val_type
                                        ))
                                    }
                                };
                                scalar_f32.into()
                            };

                            self.builder
                                .build_call(sub_assign_fn, &[current_val.into(), val_arg], "")
                                .map_err(|e| e.to_string())?;

                            return Ok(());
                        } else {
                            // Generic path for primitives
                            let load_type: inkwell::types::BasicTypeEnum = match var_type {
                                Type::I64 => self.context.i64_type().into(),
                                Type::F32 => self.context.f32_type().into(),
                                _ => return Err(format!("Unsupported type for SubAssign: {:?}", var_type)),
                            };
                            let current_val = self.builder.build_load(load_type, var_ptr.into_pointer_value(), "curr").unwrap();
                            let (op_res, _) = self.compile_bin_op(current_val, var_type.clone(), val, val_type, BinOp::Sub)?;
                            op_res
                        }
                    }
                    AssignOp::MulAssign => {
                        if let Type::Tensor(_, _) = var_type {
                            let load_type = self.context.ptr_type(inkwell::AddressSpace::default());
                            let current_val = self
                                .builder
                                .build_load(
                                    load_type,
                                    var_ptr.into_pointer_value(),
                                    &format!("{}_current", name),
                                )
                                .map_err(|e| e.to_string())?;

                            // Check if val is a scalar or Tensor and call appropriate function
                            let mul_assign_fn = if matches!(val_type, Type::Tensor(_, _)) {
                                self.module.get_function("tl_tensor_mul_assign").unwrap()
                            } else {
                                self.module
                                    .get_function("tl_tensor_mul_assign_scalar_f32")
                                    .unwrap()
                            };

                            let val_arg = if matches!(val_type, Type::Tensor(_, _)) {
                                val.into()
                            } else {
                                // Convert to f32 if necessary
                                let scalar_f32: inkwell::values::FloatValue = match val_type {
                                    Type::F32 => val.into_float_value(),
                                    Type::F64 => self
                                        .builder
                                        .build_float_cast(
                                            val.into_float_value(),
                                            self.context.f32_type(),
                                            "f64_to_f32",
                                        )
                                        .map_err(|e| e.to_string())?,
                                    Type::I64 | Type::I32 => self
                                        .builder
                                        .build_signed_int_to_float(
                                            val.into_int_value(),
                                            self.context.f32_type(),
                                            "int_to_f32",
                                        )
                                        .map_err(|e| e.to_string())?,
                                    _ => {
                                        return Err(format!(
                                            "MulAssign: unsupported RHS type {:?}",
                                            val_type
                                        ))
                                    }
                                };
                                scalar_f32.into()
                            };

                            self.builder
                                .build_call(mul_assign_fn, &[current_val.into(), val_arg], "")
                                .map_err(|e| e.to_string())?;

                            return Ok(());
                        } else {
                            // Generic path
                            let load_type: inkwell::types::BasicTypeEnum = match var_type {
                                Type::I64 => self.context.i64_type().into(),
                                Type::F32 => self.context.f32_type().into(),
                                _ => return Err(format!("Unsupported type for MulAssign: {:?}", var_type)),
                            };
                            let current_val = self.builder.build_load(load_type, var_ptr.into_pointer_value(), "curr").unwrap();
                            let (op_res, _) = self.compile_bin_op(current_val, var_type.clone(), val, val_type, BinOp::Mul)?;
                            op_res
                        }
                    }
                    AssignOp::DivAssign => {
                        if let Type::Tensor(_, _) = var_type {
                            let load_type = self.context.ptr_type(inkwell::AddressSpace::default());
                            let current_val = self
                                .builder
                                .build_load(
                                    load_type,
                                    var_ptr.into_pointer_value(),
                                    &format!("{}_current", name),
                                )
                                .map_err(|e| e.to_string())?;

                            let div_assign_fn = if matches!(val_type, Type::Tensor(_, _)) {
                                self.module.get_function("tl_tensor_div_assign").unwrap()
                            } else {
                                self.module
                                    .get_function("tl_tensor_div_assign_scalar_f32")
                                    .unwrap()
                            };

                            let val_arg = if matches!(val_type, Type::Tensor(_, _)) {
                                val.into()
                            } else {
                                let scalar_f32: inkwell::values::FloatValue = match val_type {
                                    Type::F32 => val.into_float_value(),
                                    Type::F64 => self
                                        .builder
                                        .build_float_cast(
                                            val.into_float_value(),
                                            self.context.f32_type(),
                                            "f64_to_f32",
                                        )
                                        .map_err(|e| e.to_string())?,
                                    Type::I64 | Type::I32 => self
                                        .builder
                                        .build_signed_int_to_float(
                                            val.into_int_value(),
                                            self.context.f32_type(),
                                            "int_to_f32",
                                        )
                                        .map_err(|e| e.to_string())?,
                                    _ => {
                                        return Err(format!(
                                            "DivAssign: unsupported RHS type {:?}",
                                            val_type
                                        ))
                                    }
                                };
                                scalar_f32.into()
                            };

                            self.builder
                                .build_call(div_assign_fn, &[current_val.into(), val_arg], "")
                                .map_err(|e| e.to_string())?;

                            return Ok(());
                        } else {
                            // Generic path
                            let load_type: inkwell::types::BasicTypeEnum = match var_type {
                                Type::I64 => self.context.i64_type().into(),
                                Type::F32 => self.context.f32_type().into(),
                                _ => return Err(format!("Unsupported type for DivAssign: {:?}", var_type)),
                            };
                            let current_val = self.builder.build_load(load_type, var_ptr.into_pointer_value(), "curr").unwrap();
                            let (op_res, _) = self.compile_bin_op(current_val, var_type.clone(), val, val_type, BinOp::Div)?;
                            op_res
                        }
                    }
                    AssignOp::ModAssign => {
                        if let Type::Tensor(_, _) = var_type {


                            let load_type = self.context.ptr_type(inkwell::AddressSpace::default());
                            let current_val = self
                                .builder
                                .build_load(
                                    load_type,
                                    var_ptr.into_pointer_value(),
                                    &format!("{}_current", name),
                                )
                                .map_err(|e| e.to_string())?;

                            let mod_assign_fn = if matches!(val_type, Type::Tensor(_, _)) {
                                self.module.get_function("tl_tensor_mod_assign").unwrap()
                            } else {
                                self.module
                                    .get_function("tl_tensor_mod_assign_scalar_f32")
                                    .unwrap()
                            };

                            let val_arg = if matches!(val_type, Type::Tensor(_, _)) {
                                val.into()
                            } else {
                                let scalar_f32: inkwell::values::FloatValue = match val_type {
                                    Type::F32 => val.into_float_value(),
                                    Type::F64 => self
                                        .builder
                                        .build_float_cast(
                                            val.into_float_value(),
                                            self.context.f32_type(),
                                            "f64_to_f32",
                                        )
                                        .map_err(|e| e.to_string())?,
                                    Type::I64 | Type::I32 => self
                                        .builder
                                        .build_signed_int_to_float(
                                            val.into_int_value(),
                                            self.context.f32_type(),
                                            "int_to_f32",
                                        )
                                        .map_err(|e| e.to_string())?,
                                    _ => {
                                        return Err(format!(
                                            "ModAssign: unsupported RHS type {:?}",
                                            val_type
                                        ))
                                    }
                                };
                                scalar_f32.into()
                            };

                            self.builder
                                .build_call(mod_assign_fn, &[current_val.into(), val_arg], "")
                                .map_err(|e| e.to_string())?;

                            return Ok(());
                        } else {
                            // Generic path
                            let load_type: inkwell::types::BasicTypeEnum = match var_type {
                                Type::I64 => self.context.i64_type().into(),
                                Type::F32 => self.context.f32_type().into(),
                                _ => return Err(format!("Unsupported type for ModAssign: {:?}", var_type)),
                            };
                            let current_val = self.builder.build_load(load_type, var_ptr.into_pointer_value(), "curr").unwrap();

                            let (op_res, _) = self.compile_bin_op(current_val, var_type.clone(), val, val_type, BinOp::Mod)?;
                            op_res

                        }

                    }
                    _ => return Err(format!("Unsupported assignment op: {:?}", op)),
                };

                self.builder
                    .build_store(var_ptr.into_pointer_value(), final_val)
                    .map_err(|e| e.to_string())?;
                Ok(())
            }
            StmtKind::For {
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
                let (start_val, end_val, is_tensor_iter) = match &iterator.inner {
                    ExprKind::Range(start, end) => {
                        let (s, _) = self.compile_expr(start)?;
                        let (e, _) = self.compile_expr(end)?;
                        (s.into_int_value(), e.into_int_value(), false)
                    }
                    ExprKind::FnCall(name, args) if name == "range" => {
                        // range(start, end)
                        if args.len() != 2 {
                            return Err("range() requires 2 arguments".into());
                        }
                        let (s, _) = self.compile_expr(&args[0])?;
                        let (e, _) = self.compile_expr(&args[1])?;
                        (s.into_int_value(), e.into_int_value(), false)
                    }
                    ExprKind::Variable(_) | ExprKind::FieldAccess(_, _) => {
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
                            _ => {
                                return Err(
                                    "For loop iterator must be a tensor, array, or range".into()
                                )
                            }
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
                            _ => {
                                return Err(
                                    "For loop iterator must be a tensor, array, or range".into()
                                )
                            }
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
                let loop_latch = self.context.append_basic_block(parent, "for_latch");
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

                // Push loop context for break/continue
                // continue -> latch (to increment index), break -> loop_end
                let loop_depth = self.variables.len();
                self.enter_scope();
                self.loop_stack.push((loop_latch, loop_end, loop_depth));

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
                                            let i_val = self
                                                .builder
                                                .build_float_to_signed_int(
                                                    f_val,
                                                    self.context.i64_type(),
                                                    "f2i",
                                                )
                                                .map_err(|e| e.to_string())?;
                                            (i_val.into(), Type::I64)
                                        }
                                        Type::I32 => {
                                            let i_val = self
                                                .builder
                                                .build_float_to_signed_int(
                                                    f_val,
                                                    self.context.i32_type(),
                                                    "f2i",
                                                )
                                                .map_err(|e| e.to_string())?;
                                            (i_val.into(), Type::I32)
                                        }
                                        _ => (v, Type::F32), // Default/Keep as F32
                                    }
                                }
                                _ => return Err("Invalid tensor_get return".into()),
                            }
                        }
                        Type::ScalarArray(elem_ty, len) => {
                            let llvm_elem_type: inkwell::types::BasicTypeEnum =
                                match elem_ty.as_ref() {
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
                let var_alloca = self.create_entry_block_alloca(parent, loop_var, &loop_var_val.1)?;
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

                // Branch to latch if body didn't terminate (e.g. return/break)
                let body_end_block = self.builder.get_insert_block().unwrap();

                if body_end_block.get_terminator().is_none() {
                    self.builder
                        .build_unconditional_branch(loop_latch)
                        .map_err(|e| e.to_string())?;
                }

                // Latch block: increment index and branch back to header
                self.builder.position_at_end(loop_latch);
                let next_idx = self
                    .builder
                    .build_int_add(
                        phi.as_basic_value().into_int_value(),
                        i64_type.const_int(1, false),
                        "next_idx",
                    )
                    .map_err(|e| e.to_string())?;

                self.builder
                    .build_unconditional_branch(loop_header)
                    .map_err(|e| e.to_string())?;

                // Add PHI incoming edges
                phi.add_incoming(&[(&next_idx, loop_latch)]);
                phi.add_incoming(&[(&start_val, preheader_block)]);

                // Continue at loop end
                self.builder.position_at_end(loop_end);

                // Clean up temporary tensor reference
                if is_tensor_iter {
                    for scope in self.variables.iter_mut().rev() {
                        scope.remove("__for_tensor__");
                    }
                }

                // Pop loop context
                self.loop_stack.pop();

                Ok(())
            }
            StmtKind::While { cond, body } => {
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
                self.enter_scope(); // Condition Scope
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

                self.exit_scope(); // Free condition temps
                self.builder
                    .build_conditional_branch(cond_bool, body_block, end_block)
                    .map_err(|e| e.to_string())?;

                // Compile body
                self.builder.position_at_end(body_block);

                // Push loop context for break/continue
                let loop_depth = self.variables.len();
                self.enter_scope();
                self.loop_stack.push((cond_block, end_block, loop_depth));
                for stmt in body {
                    self.compile_stmt(stmt)?;
                }
                self.exit_scope();

                // Pop loop context
                self.loop_stack.pop();

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
            StmtKind::Loop { body } => {
                let parent = self
                    .builder
                    .get_insert_block()
                    .unwrap()
                    .get_parent()
                    .unwrap();

                let body_block = self.context.append_basic_block(parent, "loop_body");
                let end_block = self.context.append_basic_block(parent, "loop_end");

                // Jump to body from current
                self.builder
                    .build_unconditional_branch(body_block)
                    .map_err(|e| e.to_string())?;

                // Compile body
                self.builder.position_at_end(body_block);

                // Push loop context for break/continue
                // In loop, continue jumps back to the START of the body.
                let loop_depth = self.variables.len();
                self.enter_scope();
                self.loop_stack.push((body_block, end_block, loop_depth));
                for stmt in body {
                    self.compile_stmt(stmt)?;
                }
                self.exit_scope();

                // Pop loop context
                self.loop_stack.pop();

                // Loop back to start of body
                if self
                    .builder
                    .get_insert_block()
                    .unwrap()
                    .get_terminator()
                    .is_none()
                {
                    self.builder
                        .build_unconditional_branch(body_block)
                        .map_err(|e| e.to_string())?;
                }

                // Continue at end
                self.builder.position_at_end(end_block);
                Ok(())
            }
            StmtKind::Expr(expr) => {
                let (val, ty) = self.compile_expr(expr)?;

                // FIX: Handle discarded return values properly to prevent use-after-free bugs.
                // When calling `model.step(lr);` without using the result:
                // - The step method may modify `self` and return a new struct
                // - If we don't capture the return value, the original variable becomes invalid
                // - We need to register the return value as a temporary so it gets freed at scope exit

                match &ty {
                    Type::Struct(_, _)
                    | Type::UserDefined(_, _)
                    | Type::Tensor(_, _)
                    | Type::TensorShaped(_, _)
                    | Type::Enum(_, _)
                    | Type::Vec(_)
                    | Type::Tuple(_) => {
                        // For struct/tensor return values: Register as a temporary variable
                        // This is equivalent to `let _ = expr;`
                        // The value will be properly freed at scope exit
                        static DISCARD_ID: std::sync::atomic::AtomicUsize =
                            std::sync::atomic::AtomicUsize::new(0);
                        let id = DISCARD_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let temp_name = format!("_discard_{}", id);

                        let current_function = self
                            .builder
                            .get_insert_block()
                            .unwrap()
                            .get_parent()
                            .unwrap();

                        let alloca =
                            self.create_entry_block_alloca(current_function, &temp_name, &ty)?;
                        self.builder
                            .build_store(alloca, val)
                            .map_err(|e| e.to_string())?;

                        // Register in current scope with should_free=true
                        // This ensures the struct gets freed when the scope exits
                        self.variables
                            .last_mut()
                            .unwrap()
                            .insert(temp_name, (alloca.into(), ty.clone(), true));
                    }

                    _ => {
                        // Primitive types: no action needed (no memory to manage)
                    }
                }

                Ok(())
            }
            StmtKind::Break => {
                // Cleanup all scopes up to loop entry before jumping
                if let Some((_, break_block, loop_depth)) = self.loop_stack.last() {
                    self.emit_cleanup_to_depth(*loop_depth);
                    self.builder
                        .build_unconditional_branch(*break_block)
                        .map_err(|e| e.to_string())?;
                }
                Ok(())
            }
            StmtKind::Continue => {
                // Cleanup all scopes up to loop entry before jumping
                if let Some((continue_block, _, loop_depth)) = self.loop_stack.last() {
                    self.emit_cleanup_to_depth(*loop_depth);
                    self.builder
                        .build_unconditional_branch(*continue_block)
                        .map_err(|e| e.to_string())?;
                }
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
                    BinOp::Mod => self.builder.build_int_signed_rem(l, r, "modtmp"),
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
                    BinOp::Mod => self
                        .builder
                        .build_float_rem(l, r, "fmodtmp")
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
            (Type::UserDefined(s1, _), Type::UserDefined(s2, _)) if s1 == "String" && s2 == "String" => {
                match op {
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
                        Ok((res_val, Type::UserDefined("String".to_string(), vec![])))
                    }
                    BinOp::Eq | BinOp::Neq => {
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
                            _ => unreachable!(),
                        }
                        .map_err(|e| e.to_string())?;
                        Ok((res.into(), Type::Bool))
                    }
                    _ => Err("Only ==, !=, and + supported for Strings".into()),
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
                    BinOp::Mod => "tl_tensor_rem",
                    BinOp::Eq => "tl_tensor_eq",
                    BinOp::Neq => "tl_tensor_neq",
                    BinOp::Lt => "tl_tensor_lt",
                    BinOp::Gt => "tl_tensor_gt",
                    BinOp::Le => "tl_tensor_le",
                    BinOp::Ge => "tl_tensor_ge",
                    _ => return Err("Unsupported tensor op".into()),
                };


                let fn_val = self
                    .module
                    .get_function(fn_name)
                    .ok_or(format!("Runtime function {} not found", fn_name))?;
                let call = self
                    .builder
                    .build_call(fn_val, &[l.into(), r.into()], "binop_res");

                let res_val =
                    self.check_tensor_result(call.map_err(|e| e.to_string())?, "binop_error")?;
                let res_ptr = res_val.into_pointer_value();
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
                // Create scalar tensor
                let val = rhs.into_float_value();
                let _f32_type = self.context.f32_type();
                let i64_type = self.context.i64_type();

                // 1. Alloca in Entry Block
                let current_block = self.builder.get_insert_block().unwrap();
                let parent_fn = current_block.get_parent().unwrap();

                let data_alloca =
                    self.create_entry_block_alloca(parent_fn, "scalar_data_rhs", &Type::F32)?;
                self.builder
                    .build_store(data_alloca, val)
                    .map_err(|e| e.to_string())?;

                // 2. Shape Alloca (dummy i64)
                let shape_alloca =
                    self.create_entry_block_alloca(parent_fn, "scalar_shape_rhs", &Type::I64)?;

                // 3. New Tensor
                let new_fn = self.module.get_function("tl_tensor_new").unwrap();
                let rank_val = i64_type.const_int(0, false); // Rank 0
                let call = self
                    .builder
                    .build_call(
                        new_fn,
                        &[data_alloca.into(), rank_val.into(), shape_alloca.into()],
                        "scalar_tensor_rhs",
                    )
                    .map_err(|e| e.to_string())?;
                let scalar_tensor = self
                    .check_tensor_result(call, "scalar_tensor_rhs_error")?
                    .into_pointer_value();
                self.emit_register_tensor(scalar_tensor.into(), &Type::Tensor(Box::new(Type::F32), 0))?;


                // 4. Call Op
                let fn_name = match op {
                    BinOp::Add => "tl_tensor_add",
                    BinOp::Mul => "tl_tensor_mul",
                    BinOp::Div => "tl_tensor_div",
                    BinOp::Sub => "tl_tensor_sub",
                    BinOp::Mod => "tl_tensor_rem",
                    BinOp::Eq => "tl_tensor_eq",
                    BinOp::Neq => "tl_tensor_neq",
                    BinOp::Lt => "tl_tensor_lt",
                    BinOp::Gt => "tl_tensor_gt",
                    BinOp::Le => "tl_tensor_le",
                    BinOp::Ge => "tl_tensor_ge",
                    _ => return Err("Unsupported tensor op".into()),
                };

                let fn_val = self
                    .module
                    .get_function(fn_name)
                    .ok_or("Runtime fn not found")?;

                let call = self.builder.build_call(
                    fn_val,
                    &[lhs.into_pointer_value().into(), scalar_tensor.into()],
                    "binop_res",
                );

                let res_val = self.check_tensor_result(
                    call.map_err(|e| e.to_string())?,
                    "binop_scalar_rhs_error",
                )?;
                let res_ptr = res_val.into_pointer_value();
                Ok((res_ptr.into(), lhs_type.clone()))
            }
            (Type::F32, Type::Tensor(inner, _)) if **inner == Type::F32 => {
                // Scalar op Tensor (Broadcasting)
                // Create scalar tensor
                let val = lhs.into_float_value();
                let _f32_type = self.context.f32_type();
                let i64_type = self.context.i64_type();

                // 1. Alloca in Entry Block
                let current_block = self.builder.get_insert_block().unwrap();
                let parent_fn = current_block.get_parent().unwrap();

                let data_alloca =
                    self.create_entry_block_alloca(parent_fn, "scalar_data_lhs", &Type::F32)?;
                self.builder
                    .build_store(data_alloca, val)
                    .map_err(|e| e.to_string())?;

                let shape_alloca =
                    self.create_entry_block_alloca(parent_fn, "scalar_shape_lhs", &Type::I64)?;

                let new_fn = self.module.get_function("tl_tensor_new").unwrap();
                let rank_val = i64_type.const_int(0, false);
                let call = self
                    .builder
                    .build_call(
                        new_fn,
                        &[data_alloca.into(), rank_val.into(), shape_alloca.into()],
                        "scalar_tensor_lhs",
                    )
                    .map_err(|e| e.to_string())?;
                let scalar_tensor = self
                    .check_tensor_result(call, "scalar_tensor_lhs_error")?
                    .into_pointer_value();
                self.emit_register_tensor(scalar_tensor.into(), &Type::Tensor(Box::new(Type::F32), 0))?;


                let fn_name = match op {
                    BinOp::Add => "tl_tensor_add",
                    BinOp::Mul => "tl_tensor_mul",
                    BinOp::Div => "tl_tensor_div",
                    BinOp::Sub => "tl_tensor_sub",
                    BinOp::Mod => "tl_tensor_rem",
                    BinOp::Eq => "tl_tensor_eq",
                    BinOp::Neq => "tl_tensor_neq",
                    BinOp::Lt => "tl_tensor_lt",
                    BinOp::Gt => "tl_tensor_gt",
                    BinOp::Le => "tl_tensor_le",
                    BinOp::Ge => "tl_tensor_ge",
                    _ => return Err("Unsupported tensor op".into()),



                };
                let fn_val = self
                    .module
                    .get_function(fn_name)
                    .ok_or("Runtime fn not found")?;

                let call = self.builder.build_call(
                    fn_val,
                    &[scalar_tensor.into(), rhs.into_pointer_value().into()],
                    "binop_res",
                );

                let res_val = self.check_tensor_result(
                    call.map_err(|e| e.to_string())?,
                    "binop_scalar_lhs_error",
                )?;
                let res_ptr = res_val.into_pointer_value();
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

                        self.check_tensor_result(call, "tensor_from_scalar_arr_error")
                            .map(|v| v.into_pointer_value())
                    };

                let l_tensor =
                    create_tensor(&self.builder, &self.module, lhs.into_pointer_value(), *len1)?;
                self.emit_register_tensor(l_tensor.into(), &Type::Tensor(Box::new(Type::F32), 1))?;

                let r_tensor =
                    create_tensor(&self.builder, &self.module, rhs.into_pointer_value(), *len2)?;
                self.emit_register_tensor(r_tensor.into(), &Type::Tensor(Box::new(Type::F32), 1))?;


                // Now call tensor binary op
                let fn_name = match op {
                    BinOp::Add => "tl_tensor_add",
                    BinOp::Mul => "tl_tensor_mul",
                    BinOp::Div => "tl_tensor_div",
                    BinOp::Sub => "tl_tensor_sub",
                    BinOp::Mod => "tl_tensor_rem",
                    _ => return Err("Unsupported ScalarArray op".into()),

                };

                let fn_val = self
                    .module
                    .get_function(fn_name)
                    .ok_or(format!("Runtime function {} not found", fn_name))?;
                let call = self.builder.build_call(
                    fn_val,
                    &[l_tensor.into(), r_tensor.into()],
                    "binop_res",
                );

                let res_val =
                    self.check_tensor_result(call.map_err(|e| e.to_string())?, "binop_arr_error")?;
                let res_ptr = res_val.into_pointer_value();

                // Return as Tensor (since we converted)
                Ok((res_ptr.into(), Type::Tensor(Box::new(Type::F32), 1)))
            }
            _ => Err(format!(
                "Type mismatch in BinOp {:?}: {:?} vs {:?}",
                op, lhs_type, rhs_type
            )),
        }
    }
    /// Deep clone a value (Tensor or Struct containing Tensors)
    pub(crate) fn emit_deep_clone(
        &self,
        val: inkwell::values::BasicValueEnum<'ctx>,
        ty: &Type,
    ) -> Result<inkwell::values::BasicValueEnum<'ctx>, String> {
        match ty {
            Type::Tensor(_, _) => {
                // Shared Ownership: Acquire reference, return same pointer
                let acquire_fn = self
                    .module
                    .get_function("tl_tensor_acquire")
                    .ok_or("tl_tensor_acquire not found")?;

                // Cast to void ptr for acquire function
                let ptr = val.into_pointer_value();
                let void_ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
                let cast_ptr = self
                    .builder
                    .build_pointer_cast(ptr, void_ptr_type, "cast_tensor_ptr")
                    .unwrap();

                self.builder
                    .build_call(acquire_fn, &[cast_ptr.into()], "")
                    .map_err(|e| e.to_string())?;

                // Return the SAME pointer
                Ok(val)
            }
            Type::Enum(name, _) => {
                let enum_def = self
                    .enum_defs
                    .get(name)
                    .ok_or(format!("Enum {} definition not found", name))?;
                self.emit_enum_deep_clone(val, enum_def)
            }
            Type::Struct(name, _) | Type::UserDefined(name, _) => {
                // Check if it is an Enum
                if let Some(enum_def) = self.enum_defs.get(name) {
                    return self.emit_enum_deep_clone(val, enum_def);
                }

                // HACK: Built-in types (String, File) are opaque pointers
                if name == "String" {
                    // Deep clone string -> strdup (via tl_string_concat("", s) or similar)
                    // We can use tl_string_concat(s, "")
                    let concat_fn = self
                        .module
                        .get_function("tl_string_concat")
                        .ok_or("tl_string_concat not found")?;

                    let s_ptr = val.into_pointer_value();
                    let empty = self
                        .builder
                        .build_global_string_ptr("", "empty_str")
                        .unwrap()
                        .as_pointer_value();

                    let call_site = self
                        .builder
                        .build_call(concat_fn, &[s_ptr.into(), empty.into()], "str_clone")
                        .unwrap();

                    let new_str = match call_site.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err("Failed to clone string".to_string()),
                    };
                    return Ok(new_str);
                } else if name == "File" {
                    // File handle cannot be deeply cloned easily. Return shallow copy (pointer).
                    return Ok(val);
                } else if name == "Path" {
                    // Shallow copy for Path
                    return Ok(val);
                } else if name == "Env" || name == "Http" {
                    // Virtual static classes or opaque
                    return Ok(val);
                }

                let simple_name = if name.contains("::") {
                    name.split("::").last().unwrap()
                } else {
                    name
                };

                let struct_def = self
                    .struct_defs
                    .get(simple_name)
                    .ok_or(format!("Struct {} definition not found", name))?;
                let st_llvm_ty = *self
                    .struct_types
                    .get(simple_name)
                    .ok_or("LLVM Struct type not found")?;

                let new_struct_ptr = self
                    .builder
                    .build_malloc(st_llvm_ty, &format!("copy_{}", name))
                    .map_err(|e| e.to_string())?;

                // Register with MemoryManager (important for nested structs which are not Variables)
                // Actually, if it's a field, it's owned by the parent struct.
                // The parent struct's free will recursively free this.
                // But wait, standard malloc isn't tracked by MemoryManager unless registered.
                // If we use recursive_free for the parent, it calls libc::free on fields.
                // So checking registration is not strictly needed for fields if recursive_free handles it.
                // However, for consistency/debug, we could register? No, let's stick to recursive_free logic.

                let src_ptr = val.into_pointer_value();

                for (i, (field_name, field_ty)) in struct_def.fields.iter().enumerate() {
                    let src_field_ptr = self
                        .builder
                        .build_struct_gep(
                            st_llvm_ty,
                            src_ptr,
                            i as u32,
                            &format!("src_{}", field_name),
                        )
                        .map_err(|e| e.to_string())?;
                    let dst_field_ptr = self
                        .builder
                        .build_struct_gep(
                            st_llvm_ty,
                            new_struct_ptr,
                            i as u32,
                            &format!("dst_{}", field_name),
                        )
                        .map_err(|e| e.to_string())?;

                    let val = match field_ty {
                        Type::Tensor(_, _)
                        | Type::TensorShaped(_, _)
                        | Type::Struct(_, _)
                        | Type::UserDefined(_, _)
                        | Type::Enum(_, _)
                        | Type::Vec(_)
                        | Type::Tuple(_) => {
                            let loaded = self
                                .builder
                                .build_load(
                                    self.context.ptr_type(inkwell::AddressSpace::default()),
                                    src_field_ptr,
                                    "f_val",
                                )
                                .map_err(|e| e.to_string())?;
                            self.emit_deep_clone(loaded, field_ty)?
                        }
                        _ => {
                            let llvm_ty: inkwell::types::BasicTypeEnum = match field_ty {
                                Type::F32 => self.context.f32_type().into(),
                                Type::F64 => self.context.f64_type().into(),
                                Type::I64 => self.context.i64_type().into(),
                                Type::I32 => self.context.i32_type().into(),
                                Type::Bool => self.context.bool_type().into(),
                                _ => {
                                    return Err(format!("Unsupported clone field: {:?}", field_ty))
                                }
                            };
                            self.builder
                                .build_load(llvm_ty, src_field_ptr, "prim_val")
                                .map_err(|e| e.to_string())?
                        }
                    };

                    self.builder
                        .build_store(dst_field_ptr, val)
                        .map_err(|e| e.to_string())?;
                }
                // Return new struct ptr
                Ok(new_struct_ptr.into())
            }
            Type::Tuple(ts) => {
                // 1. Allocate tuple struct
                let mut llvm_types = Vec::new();
                for t in ts {
                    llvm_types.push(self.get_llvm_type(t)?);
                }
                let tuple_struct_type = self.context.struct_type(&llvm_types, false);

                let size = tuple_struct_type
                    .size_of()
                    .ok_or("Cannot get size of tuple")?;
                let malloc_fn = self
                    .module
                    .get_function("malloc")
                    .ok_or("malloc not found")?;
                let call = self
                    .builder
                    .build_call(malloc_fn, &[size.into()], "tuple_malloc")
                    .map_err(|e| e.to_string())?;
                let raw_ptr = match call.try_as_basic_value() {
                    inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
                    _ => return Err("malloc returned invalid value".into()),
                };

                let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
                let tuple_ptr = self
                    .builder
                    .build_pointer_cast(raw_ptr, ptr_type, "tuple_ptr")
                    .unwrap();

                // 2. Deep clone elements
                let src_ptr = val.into_pointer_value(); // Source tuple pointer
                let src_cast = self
                    .builder
                    .build_pointer_cast(src_ptr, ptr_type, "src_tuple_cast")
                    .unwrap();

                for (i, ty) in ts.iter().enumerate() {
                    // Load field from src
                    let field_gep = self
                        .builder
                        .build_struct_gep(tuple_struct_type, src_cast, i as u32, "src_field_gep")
                        .map_err(|e| e.to_string())?;
                    let field_llvm_ty = self.get_llvm_type(ty)?;
                    let field_val = self
                        .builder
                        .build_load(field_llvm_ty, field_gep, "src_field_val")
                        .map_err(|e| e.to_string())?;

                    // RECURSIVE DEEP CLONE
                    let cloned_val = self.emit_deep_clone(field_val, ty)?;

                    // Store into dst
                    let dst_gep = self
                        .builder
                        .build_struct_gep(tuple_struct_type, tuple_ptr, i as u32, "dst_field_gep")
                        .map_err(|e| e.to_string())?;
                    self.builder
                        .build_store(dst_gep, cloned_val)
                        .map_err(|e| e.to_string())?;
                }

                Ok(tuple_ptr.into())
            }
            _ => Ok(val), // Primitives copy by value
        }
    }

    fn emit_enum_deep_clone(
        &self,
        val: BasicValueEnum<'ctx>,
        enum_def: &EnumDef,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        let name = &enum_def.name;
        let enum_ty = *self
            .enum_types
            .get(name)
            .ok_or(format!("Enum type {} not found", name))?;

        let src_ptr = val.into_pointer_value();

        // 1. Allocate new enum instance
        let new_ptr = self
            .builder
            .build_malloc(enum_ty, &format!("copy_{}", name))
            .map_err(|e| e.to_string())?;

        // 2. Load Tag
        let tag_ptr = self
            .builder
            .build_struct_gep(enum_ty, src_ptr, 0, "tag_ptr")
            .map_err(|e| e.to_string())?;
        let tag_val = self
            .builder
            .build_load(self.context.i32_type(), tag_ptr, "tag")
            .map_err(|e| e.to_string())?
            .into_int_value();

        // 3. Store Tag to new instance
        let dst_tag_ptr = self
            .builder
            .build_struct_gep(enum_ty, new_ptr, 0, "dst_tag_ptr")
            .map_err(|e| e.to_string())?;
        let _ = self.builder.build_store(dst_tag_ptr, tag_val);

        // 4. Switch on tag to copy payload
        let current_block = self.builder.get_insert_block().unwrap();
        let func = current_block.get_parent().unwrap();
        let after_switch = self.context.append_basic_block(func, "after_enum_clone");

        let mut cases = vec![];
        for (i, variant) in enum_def.variants.iter().enumerate() {
            let case_block = self
                .context
                .append_basic_block(func, &format!("clone_variant_{}", variant.name));
            cases.push((
                self.context.i32_type().const_int(i as u64, false),
                case_block,
            ));
        }

        let cases_refs: Vec<(inkwell::values::IntValue, inkwell::basic_block::BasicBlock)> =
            cases.iter().map(|(i, b)| (*i, *b)).collect();
        self.builder
            .build_switch(tag_val, after_switch, &cases_refs)
            .map_err(|e| e.to_string())?;

        // Populate cases
        for (i, variant) in enum_def.variants.iter().enumerate() {
            let case_block = cases[i].1;
            self.builder.position_at_end(case_block);

            if !variant.fields.is_empty() {
                // Reconstruct field types for GEP/Load/Store
                let mut field_types: Vec<inkwell::types::BasicTypeEnum> = vec![];
                for (_, ty) in &variant.fields {
                    let llvm_ty = match ty {
                        Type::F32 => self.context.f32_type().into(),
                        Type::I64 => self.context.i64_type().into(),
                        Type::Bool => self.context.bool_type().into(),
                        Type::Tensor(_, _) => self
                            .context
                            .ptr_type(inkwell::AddressSpace::default())
                            .into(),
                        Type::Struct(_, _) | Type::Enum(_, _) | Type::UserDefined(_, _) => self
                            .context
                            .ptr_type(inkwell::AddressSpace::default())
                            .into(),
                        _ => self.context.i64_type().into(),
                    };
                    field_types.push(llvm_ty);
                }
                let variant_struct_ty = self.context.struct_type(&field_types, false);
                let variant_ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());

                // Src Payload
                let src_payload_ptr_raw = self
                    .builder
                    .build_struct_gep(enum_ty, src_ptr, 1, "src_payload_raw")
                    .map_err(|e| e.to_string())?;
                let src_variant_ptr = self
                    .builder
                    .build_pointer_cast(src_payload_ptr_raw, variant_ptr_ty, "src_variant_casted")
                    .unwrap();

                // Dst Payload
                let dst_payload_ptr_raw = self
                    .builder
                    .build_struct_gep(enum_ty, new_ptr, 1, "dst_payload_raw")
                    .map_err(|e| e.to_string())?;
                let dst_variant_ptr = self
                    .builder
                    .build_pointer_cast(dst_payload_ptr_raw, variant_ptr_ty, "dst_variant_casted")
                    .unwrap();

                // Copy Fields
                for (idx, (_, f_ty)) in variant.fields.iter().enumerate() {
                    let src_field_ptr = self
                        .builder
                        .build_struct_gep(variant_struct_ty, src_variant_ptr, idx as u32, "src_f")
                        .map_err(|e| e.to_string())?;
                    let val = self
                        .builder
                        .build_load(field_types[idx], src_field_ptr, "val")
                        .map_err(|e| e.to_string())?;

                    // Recursive Deep Clone
                    let cloned_val = self.emit_deep_clone(val, f_ty)?;

                    let dst_field_ptr = self
                        .builder
                        .build_struct_gep(variant_struct_ty, dst_variant_ptr, idx as u32, "dst_f")
                        .map_err(|e| e.to_string())?;
                    let _ = self.builder.build_store(dst_field_ptr, cloned_val);
                }
            }
            let _ = self.builder.build_unconditional_branch(after_switch);
        }

        self.builder.position_at_end(after_switch);
        Ok(new_ptr.into())
    }
}

fn stmt_trace_tag(stmt: &Stmt) -> &'static str {
    match &stmt.inner {
        StmtKind::Use { .. } => "Use",
        StmtKind::Let { .. } => "Let",
        StmtKind::Assign { .. } => "Assign",
        StmtKind::FieldAssign { .. } => "FieldAssign",
        StmtKind::For { .. } => "For",
        StmtKind::While { .. } => "While",
        StmtKind::Loop { .. } => "Loop",
        StmtKind::Return(_) => "Return",
        StmtKind::Break => "Break",
        StmtKind::Continue => "Continue",
        StmtKind::Expr(_) => "Expr",
        StmtKind::TensorDecl { .. } => "TensorDecl",
    }
}
