use super::CodeGenerator;
use crate::compiler::ast::*;
use inkwell::types::BasicType;
use inkwell::values::*;
use std::collections::HashMap;

impl<'ctx> CodeGenerator<'ctx> {
    pub(crate) fn gen_save_struct(
        &self,
        map: inkwell::values::BasicValueEnum<'ctx>,
        struct_ptr: inkwell::values::BasicValueEnum<'ctx>,
        struct_name: &str,
        prefix: String,
    ) -> Result<(), String> {
        let def = self
            .struct_defs
            .get(struct_name)
            .ok_or(format!("Struct definition '{}' not found", struct_name))?;

        let struct_ty = *self
            .struct_types
            .get(struct_name)
            .ok_or("Struct LLVM type not found")?;

        for (i, (field_name, field_type)) in def.fields.iter().enumerate() {
            let full_key = if prefix.is_empty() {
                field_name.clone()
            } else {
                format!("{}.{}", prefix, field_name)
            };

            let ptr = struct_ptr.into_pointer_value();
            let field_ptr = self
                .builder
                .build_struct_gep(struct_ty, ptr, i as u32, field_name)
                .map_err(|e| e.to_string())?;

            match field_type {
                Type::Tensor(_, _) => {
                    // Save Tensor
                    let tensor_ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
                    let t_val = self
                        .builder
                        .build_load(tensor_ptr_ty, field_ptr, field_name)
                        .map_err(|e| e.to_string())?;
                    let key_ptr = self
                        .builder
                        .build_global_string_ptr(&full_key, "key_str")
                        .map_err(|e| e.to_string())?;

                    let i8_ptr = self
                        .builder
                        .build_pointer_cast(
                            key_ptr.as_pointer_value(),
                            self.context.ptr_type(inkwell::AddressSpace::default()),
                            "key_cast",
                        )
                        .map_err(|e| e.to_string())?;

                    let insert_fn = self
                        .module
                        .get_function("tl_tensor_map_insert")
                        .ok_or("tl_tensor_map_insert not found")?;
                    let _ = self
                        .builder
                        .build_call(insert_fn, &[map.into(), i8_ptr.into(), t_val.into()], "")
                        .map_err(|e| e.to_string())?;
                }
                Type::UserDefined(sub_name) if sub_name != "String" => {
                    // Recurse
                    let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
                    let sub_val = self
                        .builder
                        .build_load(ptr_ty, field_ptr, "sub_ptr")
                        .map_err(|e| e.to_string())?;
                    self.gen_save_struct(map, sub_val, sub_name, full_key)?;
                }
                _ => {
                    // Skip primitives
                }
            }
        }
        Ok(())
    }

    pub(crate) fn gen_load_struct(
        &self,
        map: inkwell::values::BasicValueEnum<'ctx>,
        struct_ptr: inkwell::values::BasicValueEnum<'ctx>,
        struct_name: &str,
        prefix: String,
    ) -> Result<(), String> {
        let def = self
            .struct_defs
            .get(struct_name)
            .ok_or(format!("Struct definition '{}' not found", struct_name))?;

        let struct_ty = *self
            .struct_types
            .get(struct_name)
            .ok_or("Struct LLVM type not found")?;

        for (i, (field_name, field_type)) in def.fields.iter().enumerate() {
            let full_key = if prefix.is_empty() {
                field_name.clone()
            } else {
                format!("{}.{}", prefix, field_name)
            };

            let ptr = struct_ptr.into_pointer_value();
            let field_ptr = self
                .builder
                .build_struct_gep(struct_ty, ptr, i as u32, field_name)
                .map_err(|e| e.to_string())?;

            match field_type {
                Type::Tensor(_, _) => {
                    // Load Tensor
                    let key_ptr = self
                        .builder
                        .build_global_string_ptr(&full_key, "key_str")
                        .map_err(|e| e.to_string())?;

                    let i8_ptr = self
                        .builder
                        .build_pointer_cast(
                            key_ptr.as_pointer_value(),
                            self.context.ptr_type(inkwell::AddressSpace::default()),
                            "key_cast",
                        )
                        .map_err(|e| e.to_string())?;

                    let get_fn = self
                        .module
                        .get_function("tl_tensor_map_get")
                        .ok_or("tl_tensor_map_get not found")?;
                    let call = self
                        .builder
                        .build_call(get_fn, &[map.into(), i8_ptr.into()], "t_val")
                        .map_err(|e| e.to_string())?;

                    let t_val = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err("tl_tensor_map_get returned inst/void".into()),
                    };

                    self.builder
                        .build_store(field_ptr, t_val)
                        .map_err(|e| e.to_string())?;
                }
                Type::UserDefined(sub_name) if sub_name != "String" => {
                    // Recurse: load the pointer to inner struct
                    let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
                    let sub_val = self
                        .builder
                        .build_load(ptr_ty, field_ptr, "sub_ptr")
                        .map_err(|e| e.to_string())?;
                    self.gen_load_struct(map, sub_val, sub_name, full_key)?;
                }
                _ => {
                    // Skip primitives
                }
            }
        }
        Ok(())
    }

    pub(crate) fn gen_register_params(
        &self,
        struct_ptr: inkwell::values::BasicValueEnum<'ctx>,
        struct_name: &str,
        prefix: String,
    ) -> Result<(), String> {
        let def = self
            .struct_defs
            .get(struct_name)
            .ok_or(format!("Struct definition '{}' not found", struct_name))?;

        let struct_ty = *self
            .struct_types
            .get(struct_name)
            .ok_or("Struct LLVM type not found")?;

        for (i, (field_name, field_type)) in def.fields.iter().enumerate() {
            let full_key = if prefix.is_empty() {
                field_name.clone()
            } else {
                format!("{}.{}", prefix, field_name)
            };

            let ptr = struct_ptr.into_pointer_value();
            let field_ptr = self
                .builder
                .build_struct_gep(struct_ty, ptr, i as u32, field_name)
                .map_err(|e| e.to_string())?;

            match field_type {
                Type::Tensor(_, _) => {
                    // Register Tensor: tl_add_parameter(name, tensor)
                    let tensor_ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
                    let t_val = self
                        .builder
                        .build_load(tensor_ptr_ty, field_ptr, field_name)
                        .map_err(|e| e.to_string())?;

                    let key_ptr = self
                        .builder
                        .build_global_string_ptr(&full_key, "key_str")
                        .map_err(|e| e.to_string())?;

                    let i8_ptr = self
                        .builder
                        .build_pointer_cast(
                            key_ptr.as_pointer_value(),
                            self.context.ptr_type(inkwell::AddressSpace::default()),
                            "key_cast",
                        )
                        .map_err(|e| e.to_string())?;

                    let add_fn = self
                        .module
                        .get_function("tl_add_parameter")
                        .ok_or("tl_add_parameter not found")?;

                    self.builder
                        .build_call(add_fn, &[i8_ptr.into(), t_val.into()], "")
                        .map_err(|e| e.to_string())?;
                }
                Type::UserDefined(sub_name) if sub_name != "String" => {
                    // Recurse
                    let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
                    let sub_val = self
                        .builder
                        .build_load(ptr_ty, field_ptr, "sub_ptr")
                        .map_err(|e| e.to_string())?;
                    self.gen_register_params(sub_val, sub_name, full_key)?;
                }
                Type::Struct(sub_name) => {
                    // Recurse for Type::Struct as well (e.g. from generic instantiation)
                    let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
                    let sub_val = self
                        .builder
                        .build_load(ptr_ty, field_ptr, "sub_ptr")
                        .map_err(|e| e.to_string())?;
                    self.gen_register_params(sub_val, sub_name, full_key)?;
                }
                _ => {
                    // Skip primitives
                }
            }
        }
        Ok(())
    }

    pub(crate) fn compile_expr(
        &mut self,
        expr: &Expr,
    ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
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
            Expr::StringLiteral(s) => {
                // Create a global string constant and return pointer to it
                let global_str = self
                    .builder
                    .build_global_string_ptr(s, "str_literal")
                    .map_err(|e| e.to_string())?;
                Ok((
                    global_str.as_pointer_value().into(),
                    Type::UserDefined("String".to_string()),
                ))
            }
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

                if obj_val.is_pointer_value() {
                    let ptr = obj_val.into_pointer_value();
                    let st_llvm_ty = self.struct_types.get(&struct_name).unwrap();

                    let field_ptr = self
                        .builder
                        .build_struct_gep(
                            *st_llvm_ty,
                            ptr,
                            field_idx as u32,
                            &format!("ptr_{}", field),
                        )
                        .map_err(|e| e.to_string())?;

                    let llvm_ty: inkwell::types::BasicTypeEnum = match field_ty {
                        Type::I64 => self.context.i64_type().into(),
                        Type::F32 => self.context.f32_type().into(),
                        Type::Bool => self.context.bool_type().into(),
                        Type::Tensor(_, _) => self
                            .context
                            .ptr_type(inkwell::AddressSpace::default())
                            .into(),
                        Type::Struct(_) | Type::UserDefined(_) => self
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
                } else if obj_val.is_struct_value() {
                    let struct_val = obj_val.into_struct_value();
                    let extracted = self
                        .builder
                        .build_extract_value(struct_val, field_idx as u32, field)
                        .map_err(|e| e.to_string())?;
                    Ok((extracted, field_ty.clone()))
                } else {
                    Err("Cannot access field of non-pointer and non-struct value".into())
                }
            }

            Expr::Variable(name) => {
                for scope in self.variables.iter().rev() {
                    if let Some((val, ty, _)) = scope.get(name) {
                        if val.is_pointer_value() {
                            let ptr = val.into_pointer_value();

                            // ScalarArray: load the pointer from alloca (it stores ptr to global)
                            if let Type::ScalarArray(_, _) = ty {
                                let ptr_type =
                                    self.context.ptr_type(inkwell::AddressSpace::default());
                                let loaded_ptr = self
                                    .builder
                                    .build_load(ptr_type, ptr, &format!("{}_ptr", name))
                                    .map_err(|e| e.to_string())?;
                                return Ok((loaded_ptr, ty.clone()));
                            }

                            let llvm_ty: inkwell::types::BasicTypeEnum = match ty {
                                Type::I64 => self.context.i64_type().into(),
                                Type::F32 => self.context.f32_type().into(),
                                Type::Bool => self.context.bool_type().into(),
                                Type::Tensor(_, _) => self
                                    .context
                                    .ptr_type(inkwell::AddressSpace::default())
                                    .into(),
                                Type::Struct(_) | Type::UserDefined(_) => self
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
                            return Ok((*val, ty.clone()));
                        }
                    }
                }
                Err(format!("Variable {} not found in scopes", name))
            }
            Expr::StructInit(name, fields) => self.compile_struct_init(name, fields),
            Expr::StaticMethodCall(type_name, method_name, args) => {
                self.compile_static_method_call(type_name, method_name, args)
            }
            Expr::BinOp(lhs, op, rhs) => {
                let left = self.compile_expr(lhs)?;
                let right = self.compile_expr(rhs)?;
                self.compile_bin_op(left.0, left.1, right.0, right.1, op.clone())
            }
            Expr::TensorComprehension { indices, body } => {
                // Generate a unique name for the temporary result
                static NEXT_ID: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(0);
                let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let temp_name = format!("_comp_res_{}", id);

                // Compile as a tensor equation: let temp[indices] = body;
                // Since compile_tensor_equation expects &str names for indices when passed,
                // and accepts Expr for body.
                // However, compile_tensor_equation logic assumes it's creating a NEW variable 'temp_name'
                // and expects free_indices to be passed to it.
                // The indices in comprehension become the "free indices" of the equation (the output dimensions).

                self.compile_tensor_equation(&temp_name, indices, body)
                    .map_err(|e| e.to_string())?;

                // After compilation, the tensor 'temp_name' is registered in the scope.
                // We need to load it to return it as an expression value.
                // It should be in the current scope.

                let (val_enum, val_ty, _) = self
                    .variables
                    .last()
                    .unwrap()
                    .get(&temp_name)
                    .ok_or(format!("Failed to retrieve temporary tensor {}", temp_name))?
                    .clone();

                if let Type::Tensor(_, _) = val_ty {
                    // It's a pointer to the tensor struct
                    if val_enum.is_pointer_value() {
                        let ptr_to_ptr = val_enum.into_pointer_value();
                        let void_ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
                        let ptr = self
                            .builder
                            .build_load(void_ptr_type, ptr_to_ptr, "tensor_ptr")
                            .map_err(|e| e.to_string())?;
                        Ok((ptr, val_ty))
                    } else {
                        Err("Tensor variable should be a pointer".into())
                    }
                } else {
                    Err("Comprehension result must be a tensor".into())
                }
            }
            Expr::TensorLiteral(elements) => self.compile_tensor_literal(elements),
            Expr::TensorConstLiteral(elements) => self.compile_tensor_const_literal(elements),
            Expr::MethodCall(obj, method, args) => self.compile_method_call(obj, method, args),
            Expr::FnCall(name, args) => self.compile_fn_call(name, args),
            Expr::IndexAccess(target, indices) => {
                let (val, val_type) = self.compile_expr(target)?;
                match val_type {
                    // OPTIMIZATION: ScalarArray direct access (no runtime call)
                    Type::ScalarArray(elem_type, len) => {
                        if indices.len() != 1 {
                            return Err("ScalarArray only supports 1D index".into());
                        }

                        let llvm_elem_type: inkwell::types::BasicTypeEnum = match elem_type.as_ref()
                        {
                            Type::I64 => self.context.i64_type().into(),
                            Type::I32 => self.context.i32_type().into(),
                            Type::F32 => self.context.f32_type().into(),
                            _ => self.context.f32_type().into(),
                        };
                        let i64_type = self.context.i64_type();
                        let _array_type = llvm_elem_type.array_type(len as u32);
                        let array_ptr = val.into_pointer_value();

                        let (idx_val, idx_ty) = self.compile_expr(&indices[0])?;
                        let idx_int = match idx_ty {
                            Type::I64 => idx_val.into_int_value(),
                            Type::I32 => self
                                .builder
                                .build_int_z_extend(idx_val.into_int_value(), i64_type, "zext")
                                .map_err(|e| e.to_string())?,
                            _ => return Err("Index must be integer".into()),
                        };

                        // Direct GEP into array
                        let elem_ptr = unsafe {
                            self.builder
                                .build_in_bounds_gep(
                                    llvm_elem_type,
                                    array_ptr,
                                    &[idx_int],
                                    "scalar_elem_ptr",
                                )
                                .map_err(|e| e.to_string())?
                        };

                        let loaded = self
                            .builder
                            .build_load(llvm_elem_type, elem_ptr, "scalar_elem")
                            .map_err(|e| e.to_string())?;

                        Ok((loaded, *elem_type))
                    }
                    Type::Tensor(_, _) => {
                        // Prepare indices array
                        let rank = indices.len();
                        let i64_type = self.context.i64_type();

                        // Create array on stack in the ENTRY block to avoid stack overflow in loops
                        let array_type = i64_type.array_type(rank as u32);

                        let current_block = self.builder.get_insert_block().unwrap();
                        let function = current_block.get_parent().unwrap();
                        let entry_block = function.get_first_basic_block().unwrap();

                        let entry_builder = self.context.create_builder();
                        if let Some(first_instr) = entry_block.get_first_instruction() {
                            entry_builder.position_before(&first_instr);
                        } else {
                            entry_builder.position_at_end(entry_block);
                        }

                        let array_alloca = entry_builder
                            .build_alloca(array_type, "idx_arr")
                            .map_err(|e| e.to_string())?;

                        for (i, idx_expr) in indices.iter().enumerate() {
                            let (compiled_idx, ty) = self.compile_expr(idx_expr)?;

                            // Ensure index is integer or float (cast if needed)
                            let idx_val = match ty {
                                Type::I64 => compiled_idx.into_int_value(),
                                Type::I32 => self
                                    .builder
                                    .build_int_z_extend(
                                        compiled_idx.into_int_value(),
                                        i64_type,
                                        "zext",
                                    )
                                    .map_err(|e| e.to_string())?,
                                Type::F64 | Type::F32 => self
                                    .builder
                                    .build_float_to_signed_int(
                                        compiled_idx.into_float_value(),
                                        i64_type,
                                        "f2i",
                                    )
                                    .map_err(|e| e.to_string())?,
                                _ => return Err(format!("Invalid index type {:?}", ty)),
                            };
                            let idx_val = inkwell::values::BasicValueEnum::IntValue(idx_val);

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
                    _ => Err("Index access only on Tensor or ScalarArray".into()),
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
                let function = self
                    .builder
                    .get_insert_block()
                    .unwrap()
                    .get_parent()
                    .unwrap();

                // Convert range to Tensor
                let range_val = self.ensure_tensor_v2(range, 1)?;
                let _range_ty = Type::Tensor(Box::new(Type::F32), 1); // Simplified

                // Get length
                let len_fn = self
                    .module
                    .get_function("tl_tensor_len")
                    .ok_or("tl_tensor_len not found")?;
                let call = self
                    .builder
                    .build_call(len_fn, &[range_val.into()], "len")
                    .map_err(|e| e.to_string())?;
                let loop_count = match call.try_as_basic_value() {
                    ValueKind::Basic(v) => v.into_int_value(),
                    _ => return Err("Failed to get tensor length".into()),
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

                // Determine element type from original range expression
                let (_, orig_range_ty) = self.compile_expr(range)?;
                let elem_ty = match orig_range_ty {
                    Type::Tensor(ref t, _) => *t.clone(),
                    Type::ScalarArray(ref t, _) => *t.clone(),
                    _ => Type::F32,
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
                    Type::I32 => self
                        .builder
                        .build_float_to_signed_int(val_f32, self.context.i32_type(), "cast_i32")
                        .map_err(|e| e.to_string())?
                        .into(),
                    _ => val_f32.into(),
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

            Expr::IfExpr(cond, then_stmts, else_stmts) => {
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
                    .map_err(|e| e.to_string())?;

                let then_bb = self.context.append_basic_block(parent, "if_then");
                let else_bb = self.context.append_basic_block(parent, "if_else");
                let merge_bb = self.context.append_basic_block(parent, "if_merge");

                self.builder
                    .build_conditional_branch(cond_int, then_bb, else_bb)
                    .map_err(|e| e.to_string())?;

                // Then branch
                self.builder.position_at_end(then_bb);
                self.enter_scope();
                let mut then_val: Option<(BasicValueEnum<'ctx>, Type)> = None;
                for (i, stmt) in then_stmts.iter().enumerate() {
                    if i == then_stmts.len() - 1 {
                        if let Stmt::Expr(e) = stmt {
                            then_val = Some(self.compile_expr(e)?);
                        } else {
                            self.compile_stmt(stmt)?;
                        }
                    } else {
                        self.compile_stmt(stmt)?;
                    }
                }
                // Default value if no expression
                let then_result = then_val.unwrap_or((
                    self.context.i64_type().const_int(0, false).into(),
                    Type::Void,
                ));
                let then_end_bb = self.builder.get_insert_block().unwrap();
                if then_end_bb.get_terminator().is_none() {
                    self.builder
                        .build_unconditional_branch(merge_bb)
                        .map_err(|e| e.to_string())?;
                }
                self.exit_scope();

                // Else branch
                self.builder.position_at_end(else_bb);
                self.enter_scope();
                let mut else_val: Option<(BasicValueEnum<'ctx>, Type)> = None;
                if let Some(else_body) = else_stmts {
                    for (i, stmt) in else_body.iter().enumerate() {
                        if i == else_body.len() - 1 {
                            if let Stmt::Expr(e) = stmt {
                                else_val = Some(self.compile_expr(e)?);
                            } else {
                                self.compile_stmt(stmt)?;
                            }
                        } else {
                            self.compile_stmt(stmt)?;
                        }
                    }
                }
                let else_result = else_val.unwrap_or((
                    self.context.i64_type().const_int(0, false).into(),
                    Type::Void,
                ));
                let else_end_bb = self.builder.get_insert_block().unwrap();
                if else_end_bb.get_terminator().is_none() {
                    self.builder
                        .build_unconditional_branch(merge_bb)
                        .map_err(|e| e.to_string())?;
                }
                self.exit_scope();

                // Merge block with PHI
                self.builder.position_at_end(merge_bb);

                // Only create PHI if both branches return non-void values
                if !matches!(then_result.1, Type::Void) && !matches!(else_result.1, Type::Void) {
                    let llvm_ty: inkwell::types::BasicTypeEnum = match &then_result.1 {
                        Type::I64 => self.context.i64_type().into(),
                        Type::F32 => self.context.f32_type().into(),
                        Type::Bool => self.context.bool_type().into(),
                        Type::Tensor(_, _) | Type::Struct(_) | Type::UserDefined(_) => self
                            .context
                            .ptr_type(inkwell::AddressSpace::default())
                            .into(),
                        _ => self.context.i64_type().into(),
                    };

                    let phi = self
                        .builder
                        .build_phi(llvm_ty, "if_result")
                        .map_err(|e| e.to_string())?;
                    phi.add_incoming(&[
                        (&then_result.0, then_end_bb),
                        (&else_result.0, else_end_bb),
                    ]);

                    Ok((phi.as_basic_value(), then_result.1))
                } else {
                    Ok((
                        self.context.i64_type().const_int(0, false).into(),
                        Type::Void,
                    ))
                }
            }
        }
    }

    fn compile_struct_init(
        &mut self,
        name: &str,
        fields: &[(String, Expr)],
    ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        let struct_type = *self
            .struct_types
            .get(name)
            .ok_or(format!("Struct type {} not found in codegen", name))?;

        let struct_def = self
            .struct_defs
            .get(name)
            .ok_or(format!("Struct definition {} not found", name))?
            .clone();

        // Determine allocation strategy: Arena or Heap
        let size = struct_type
            .size_of()
            .ok_or(format!("Cannot determine size of struct {}", name))?;

        // 1. Check if Arena is active
        let is_active_fn = self
            .module
            .get_function("tl_arena_is_active")
            .ok_or("tl_arena_is_active not found")?;
        let is_active_call = self
            .builder
            .build_call(is_active_fn, &[], "is_arena_active")
            .map_err(|e| e.to_string())?;
        let is_active_val = match is_active_call.try_as_basic_value() {
            inkwell::values::ValueKind::Basic(v) => v.into_int_value(),
            _ => return Err("tl_arena_is_active returned void".into()),
        };
        let is_active_bool = self
            .builder
            .build_int_compare(
                inkwell::IntPredicate::NE,
                is_active_val,
                self.context.bool_type().const_zero(),
                "is_active_bool",
            )
            .map_err(|e| e.to_string())?;

        // 2. Setup blocks
        let current_block = self.builder.get_insert_block().unwrap();
        let function = current_block.get_parent().unwrap();
        let arena_block = self.context.append_basic_block(function, "alloc_arena");
        let heap_block = self.context.append_basic_block(function, "alloc_heap");
        let merge_block = self.context.append_basic_block(function, "alloc_merge");

        self.builder
            .build_conditional_branch(is_active_bool, arena_block, heap_block)
            .map_err(|e| e.to_string())?;

        // 3. Heap Allocation (Legacy path)
        self.builder.position_at_end(heap_block);
        let malloc_fn = self
            .module
            .get_function("malloc")
            .ok_or("malloc not found (declare in builtins)")?;
        let call = self
            .builder
            .build_call(malloc_fn, &[size.into()], "struct_malloc")
            .map_err(|e| e.to_string())?;
        let raw_ptr_heap = match call.try_as_basic_value() {
            inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
            _ => return Err("malloc returned invalid value".into()),
        };
        // Register with MemoryManager
        if let Some(register_fn) = self.module.get_function("tl_mem_register_struct") {
            let cast_ptr = self
                .builder
                .build_pointer_cast(
                    raw_ptr_heap,
                    self.context.ptr_type(inkwell::AddressSpace::default()),
                    "cast_ptr",
                )
                .unwrap();
            self.builder
                .build_call(register_fn, &[cast_ptr.into()], "")
                .map_err(|e| e.to_string())?;
        }
        self.builder
            .build_unconditional_branch(merge_block)
            .unwrap();
        let heap_end_block = self.builder.get_insert_block().unwrap();

        // 4. Arena Allocation
        self.builder.position_at_end(arena_block);
        let arena_alloc_fn = self
            .module
            .get_function("tl_arena_alloc")
            .ok_or("tl_arena_alloc not found")?;
        let alloc_call = self
            .builder
            .build_call(arena_alloc_fn, &[size.into()], "struct_arena_alloc")
            .map_err(|e| e.to_string())?;
        let raw_ptr_arena = match alloc_call.try_as_basic_value() {
            inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
            _ => return Err("tl_arena_alloc returned invalid value".into()),
        };
        // NO registration for arena pointers!
        self.builder
            .build_unconditional_branch(merge_block)
            .unwrap();
        let arena_end_block = self.builder.get_insert_block().unwrap();

        // 5. Merge
        self.builder.position_at_end(merge_block);
        let ptr_phi = self
            .builder
            .build_phi(
                self.context.ptr_type(inkwell::AddressSpace::default()),
                "struct_ptr_phi",
            )
            .unwrap();
        ptr_phi.add_incoming(&[
            (&raw_ptr_heap, heap_end_block),
            (&raw_ptr_arena, arena_end_block),
        ]);
        let raw_ptr = ptr_phi.as_basic_value().into_pointer_value();

        // Cast to Struct Pointer (opaque pointer in modern LLVM, but typed for GEP)
        let struct_ptr = self
            .builder
            .build_pointer_cast(
                raw_ptr,
                self.context.ptr_type(inkwell::AddressSpace::default()),
                "struct_ptr",
            )
            .map_err(|e| e.to_string())?;

        for (field_name, field_expr) in fields {
            let field_idx = struct_def
                .fields
                .iter()
                .position(|(n, _)| n == field_name)
                .ok_or(format!("Field {} not found in struct {}", field_name, name))?;

            let (val, _ty) = self.compile_expr(field_expr)?;

            let field_ptr = self
                .builder
                .build_struct_gep(
                    struct_type,
                    struct_ptr,
                    field_idx as u32,
                    &format!("{}.{}", name, field_name),
                )
                .map_err(|e| e.to_string())?;

            self.builder
                .build_store(field_ptr, val)
                .map_err(|e| e.to_string())?;
        }

        // Return the pointer directly (no load)
        Ok((struct_ptr.into(), Type::Struct(name.to_string())))
    }

    fn compile_static_method_call(
        &mut self,
        type_name: &str,
        method_name: &str,
        args: &[Expr],
    ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        // 1. Resolve Mangled Name
        let mangled_name = format!("tl_{}_{}", type_name, method_name);
        let stdlib_name = format!("tl_{}_{}", type_name.to_lowercase(), method_name);

        // 2. Lookup Function
        let (func, actual_name) = if let Some(f) = self.module.get_function(&mangled_name) {
            (f, mangled_name)
        } else if let Some(f) = self.module.get_function(&stdlib_name) {
            (f, stdlib_name)
        } else {
            return Err(format!(
                "Static method {}::{} not found (checked {} and {})",
                type_name, method_name, mangled_name, stdlib_name
            ));
        };

        // 3. Compile Args
        let mut compiled_args = Vec::new();
        for arg in args {
            let (val, _) = self.compile_expr(arg)?;
            compiled_args.push(val.into());
        }

        // 4. Call
        let call = self
            .builder
            .build_call(func, &compiled_args, "static_call")
            .map_err(|e| e.to_string())?;

        // 5. Return Value
        let ret_ty = self
            .fn_return_types
            .get(&actual_name)
            .cloned()
            .unwrap_or(Type::Void);

        match call.try_as_basic_value() {
            inkwell::values::ValueKind::Basic(v) => Ok((v, ret_ty)),
            _ => Ok((
                self.context.i64_type().const_int(0, false).into(),
                Type::Void,
            )),
        }
    }

    fn compile_tensor_literal(
        &mut self,
        elements: &[Expr],
    ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        let f32_type = self.context.f32_type();
        let i64_type = self.context.i64_type();

        // Check if all elements are static literals (for optimized path)
        // or if we need dynamic compilation
        fn count_elements(exprs: &[Expr]) -> (usize, Vec<usize>) {
            if exprs.is_empty() {
                return (0, vec![0]);
            }

            let is_nested = matches!(exprs[0], Expr::TensorLiteral(_));
            if is_nested {
                let mut total = 0;
                let mut first_shape = None;
                for e in exprs {
                    if let Expr::TensorLiteral(children) = e {
                        let (count, shape) = count_elements(children);
                        total += count;
                        if first_shape.is_none() {
                            first_shape = Some(shape);
                        }
                    }
                }
                let mut shape = vec![exprs.len()];
                if let Some(s) = first_shape {
                    shape.extend(s);
                }
                (total, shape)
            } else {
                (exprs.len(), vec![exprs.len()])
            }
        }

        let (total_elements, shape) = count_elements(elements);
        let rank = shape.len();

        // Allocate buffer for elements
        // Use entry block allocation to prevent stack explosion in loops
        let current_block = self.builder.get_insert_block().unwrap();
        let function = current_block.get_parent().unwrap();
        let entry_block = function.get_first_basic_block().unwrap();

        // Use calloc for data to ensure alignment and support broad sizes (matches tensor.rs)
        let calloc_fn = self
            .module
            .get_function("calloc")
            .expect("calloc not found");
        let call_idx = self
            .builder
            .build_call(
                calloc_fn,
                &[
                    i64_type.const_int(total_elements as u64, false).into(),
                    i64_type.const_int(4, false).into(), // sizeof(f32)
                ],
                "buf_void",
            )
            .map_err(|e| e.to_string())?;

        let data_alloca = match call_idx.try_as_basic_value() {
            inkwell::values::ValueKind::Basic(inkwell::values::BasicValueEnum::PointerValue(v)) => {
                v
            }
            _ => return Err("Invalid calloc return".to_string()),
        };

        // Helper to flatten and compile elements
        fn flatten_exprs(exprs: &[Expr], result: &mut Vec<Expr>) {
            for e in exprs {
                if let Expr::TensorLiteral(children) = e {
                    flatten_exprs(children, result);
                } else {
                    result.push(e.clone());
                }
            }
        }

        let mut flat_exprs = Vec::new();
        flatten_exprs(elements, &mut flat_exprs);

        // Compile each element and store to buffer
        for (i, expr) in flat_exprs.iter().enumerate() {
            let (val, val_ty) = self.compile_expr(expr)?;

            // Convert to f32 if necessary
            let f32_val = match val_ty {
                Type::F32 => val.into_float_value(),
                Type::I64 => self
                    .builder
                    .build_signed_int_to_float(val.into_int_value(), f32_type, "i2f")
                    .map_err(|e| e.to_string())?,
                _ => return Err(format!("Tensor element must be numeric, got {:?}", val_ty)),
            };

            // Get pointer to element position
            // data_alloca is Pointer to Array (or just ptr due to array decay? build_array_alloca returns ptr)
            // build_array_alloca returns pointer to allocated type (float*).
            // So we treat it as float*. Indices: [i]
            let ptr = unsafe {
                self.builder
                    .build_in_bounds_gep(
                        f32_type,
                        data_alloca,
                        &[i64_type.const_int(i as u64, false)],
                        "elem_ptr",
                    )
                    .map_err(|e| e.to_string())?
            };

            // Store value
            self.builder
                .build_store(ptr, f32_val)
                .map_err(|e| e.to_string())?;
        }

        // Allocate and fill shape buffer
        // Move to entry block again for shape
        let current_block_2 = self.builder.get_insert_block().unwrap(); // Should be same
        if let Some(first_instr) = entry_block.get_first_instruction() {
            self.builder.position_before(&first_instr);
        } else {
            self.builder.position_at_end(entry_block);
        }

        let shape_array_type = i64_type.array_type(rank as u32);
        let shape_alloca = self
            .builder
            .build_alloca(shape_array_type, "tensor_shape_arr")
            .map_err(|e| e.to_string())?;

        // Move back
        self.builder.position_at_end(current_block_2);

        for (i, dim) in shape.iter().enumerate() {
            let ptr = unsafe {
                self.builder
                    .build_in_bounds_gep(
                        shape_array_type,
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
                .build_store(ptr, i64_type.const_int(*dim as u64, false))
                .map_err(|e| e.to_string())?;
        }

        // Call tl_tensor_new
        let new_fn = self
            .module
            .get_function("tl_tensor_new")
            .ok_or("tl_tensor_new not found")?;

        let shape_ptr_cast = self
            .builder
            .build_pointer_cast(
                shape_alloca,
                self.context.ptr_type(inkwell::AddressSpace::default()),
                "shap_ptr_cast",
            )
            .map_err(|e| e.to_string())?;

        let call = self
            .builder
            .build_call(
                new_fn,
                &[
                    data_alloca.into(),
                    i64_type.const_int(rank as u64, false).into(),
                    shape_ptr_cast.into(),
                ],
                "new_tensor",
            )
            .map_err(|e| e.to_string())?;

        let res = match call.try_as_basic_value() {
            ValueKind::Basic(v) => v,
            _ => return Err("Invalid tl_tensor_new return".into()),
        };

        Ok((res, Type::Tensor(Box::new(Type::F32), rank)))
    }

    fn compile_tensor_const_literal(
        &mut self,
        elements: &[Expr],
    ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        // Optimized path for constant tensor literals - static extraction
        fn flatten_const(exprs: &[Expr]) -> Result<(Vec<f64>, Vec<usize>, bool), String> {
            if exprs.is_empty() {
                return Ok((vec![], vec![0], false));
            }

            let is_nested = matches!(
                exprs[0],
                Expr::TensorConstLiteral(_) | Expr::TensorLiteral(_)
            );
            if is_nested {
                let mut flat_data = Vec::new();
                let mut first_shape = None;
                let mut all_ints = true;

                for e in exprs {
                    let (children, shape, ints) = match e {
                        Expr::TensorConstLiteral(c) | Expr::TensorLiteral(c) => flatten_const(c)?,
                        _ => return Err("Mixed types in const tensor".into()),
                    };
                    if let Some(ref s) = first_shape {
                        if s != &shape {
                            return Err("Ragged tensors not supported".into());
                        }
                    } else {
                        first_shape = Some(shape.clone());
                    }
                    flat_data.extend(children);
                    all_ints &= ints;
                }

                let mut shape = vec![exprs.len()];
                if let Some(s) = first_shape {
                    shape.extend(s);
                }
                Ok((flat_data, shape, all_ints))
            } else {
                let mut data = Vec::new();
                let mut all_ints = true;
                for e in exprs {
                    match e {
                        Expr::Float(f) => {
                            data.push(*f);
                            all_ints = false;
                        }
                        Expr::Int(i) => data.push(*i as f64),
                        _ => return Err("Const tensor must contain only literals".into()),
                    }
                }
                Ok((data, vec![exprs.len()], all_ints))
            }
        }

        let (flat_data, shape, all_ints) = flatten_const(elements)?;
        let rank = shape.len();
        let len = flat_data.len();

        // OPTIMIZATION: For small 1D constant tensors (≤8 elements), use heap-based scalar array
        if rank == 1 && len <= 8 && len > 0 {
            let (elem_ty, llvm_elem_type): (Type, inkwell::types::BasicTypeEnum) = if all_ints {
                (Type::I64, self.context.i64_type().into())
            } else {
                (Type::F32, self.context.f32_type().into())
            };

            let i64_type = self.context.i64_type();

            // Allocate on HEAP
            let malloc_fn = self
                .module
                .get_function("malloc")
                .ok_or("malloc not found")?;
            let size_elem = match elem_ty {
                Type::I64 => self.context.i64_type().size_of(),
                _ => self.context.f32_type().size_of(),
            };
            let size = self
                .builder
                .build_int_mul(
                    size_elem,
                    i64_type.const_int(len as u64, false),
                    "malloc_size",
                )
                .map_err(|e| e.to_string())?;
            let call = self
                .builder
                .build_call(malloc_fn, &[size.into()], "arr_malloc")
                .map_err(|e| e.to_string())?;
            let raw_ptr = match call.try_as_basic_value() {
                ValueKind::Basic(v) => v.into_pointer_value(),
                _ => return Err("malloc failed".into()),
            };

            // Register with memory manager for automatic free
            if let Some(reg_fn) = self.module.get_function("tl_mem_register_struct") {
                self.builder
                    .build_call(reg_fn, &[raw_ptr.into()], "")
                    .map_err(|e| e.to_string())?;
            }

            // Populate array
            for (i, val) in flat_data.iter().enumerate() {
                let v: inkwell::values::BasicValueEnum = if all_ints {
                    i64_type.const_int(*val as u64, true).into()
                } else {
                    self.context.f32_type().const_float(*val).into()
                };
                let elem_ptr = unsafe {
                    self.builder
                        .build_in_bounds_gep(
                            llvm_elem_type,
                            raw_ptr,
                            &[self.context.i64_type().const_int(i as u64, false)],
                            "elem_ptr",
                        )
                        .map_err(|e| e.to_string())?
                };
                self.builder
                    .build_store(elem_ptr, v)
                    .map_err(|e| e.to_string())?;
            }

            return Ok((
                inkwell::values::BasicValueEnum::PointerValue(raw_ptr),
                Type::ScalarArray(Box::new(elem_ty), len),
            ));
        }

        // Fall back to standard tensor creation for larger tensors
        let len = len as u64;
        let f32_type = self.context.f32_type();
        let i64_type = self.context.i64_type();

        // CRITICAL FIX: Use HEAP allocation (malloc) instead of STACK (alloca)
        // to prevent stack overflow with many tensor literals

        // Get malloc and free functions
        let malloc_fn = self
            .module
            .get_function("malloc")
            .ok_or("malloc not found")?;
        let free_fn = self.module.get_function("free").ok_or("free not found")?;

        // Allocate data buffer on HEAP
        let data_size_bytes = len * 4; // f32 = 4 bytes
        let malloc_call = self
            .builder
            .build_call(
                malloc_fn,
                &[i64_type.const_int(data_size_bytes, false).into()],
                "temp_data_heap",
            )
            .map_err(|e| e.to_string())?;
        let data_ptr = match malloc_call.try_as_basic_value() {
            ValueKind::Basic(v) => v.into_pointer_value(),
            _ => return Err("malloc returned non-pointer".into()),
        };

        // Allocate shape buffer on HEAP
        let shape_size_bytes = rank as u64 * 8; // i64 = 8 bytes
        let shape_malloc_call = self
            .builder
            .build_call(
                malloc_fn,
                &[i64_type.const_int(shape_size_bytes, false).into()],
                "temp_shape_heap",
            )
            .map_err(|e| e.to_string())?;
        let shape_ptr = match shape_malloc_call.try_as_basic_value() {
            ValueKind::Basic(v) => v.into_pointer_value(),
            _ => return Err("malloc returned non-pointer".into()),
        };

        // Populate data buffer
        for (i, val) in flat_data.iter().enumerate() {
            let float_val = f32_type.const_float(*val);
            let elem_ptr = unsafe {
                self.builder
                    .build_in_bounds_gep(
                        f32_type,
                        data_ptr,
                        &[i64_type.const_int(i as u64, false)],
                        "data_elem",
                    )
                    .map_err(|e| e.to_string())?
            };
            self.builder
                .build_store(elem_ptr, float_val)
                .map_err(|e| e.to_string())?;
        }

        // Populate shape buffer
        for (i, dim) in shape.iter().enumerate() {
            let elem_ptr = unsafe {
                self.builder
                    .build_in_bounds_gep(
                        i64_type,
                        shape_ptr,
                        &[i64_type.const_int(i as u64, false)],
                        "shape_elem",
                    )
                    .map_err(|e| e.to_string())?
            };
            self.builder
                .build_store(elem_ptr, i64_type.const_int(*dim as u64, false))
                .map_err(|e| e.to_string())?;
        }

        // Call tl_tensor_new with heap-allocated buffers
        let new_fn = self
            .module
            .get_function("tl_tensor_new")
            .ok_or("tl_tensor_new not found")?;

        let call = self
            .builder
            .build_call(
                new_fn,
                &[
                    data_ptr.into(),
                    i64_type.const_int(rank as u64, false).into(),
                    shape_ptr.into(),
                ],
                "new_const_tensor",
            )
            .map_err(|e| e.to_string())?;

        let res = match call.try_as_basic_value() {
            ValueKind::Basic(v) => v,
            _ => return Err("Invalid tl_tensor_new return".into()),
        };

        // FREE heap-allocated buffers immediately after tl_tensor_new
        // (tl_tensor_new copies the data internally via Candle's from_slice)
        self.builder
            .build_call(free_fn, &[data_ptr.into()], "")
            .map_err(|e| e.to_string())?;
        self.builder
            .build_call(free_fn, &[shape_ptr.into()], "")
            .map_err(|e| e.to_string())?;

        Ok((res, Type::Tensor(Box::new(Type::F32), rank)))
    }

    pub(crate) fn create_entry_block_alloca(
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
            Type::Tensor(_, _)
            | Type::UserDefined(_)
            | Type::Struct(_)
            | Type::ScalarArray(_, _) => self
                .context
                .ptr_type(inkwell::AddressSpace::default())
                .into(),
            _ => self.context.i64_type().into(),
        };

        let alloca = builder.build_alloca(llvm_type, name).unwrap();
        if let Some(instr) = alloca.as_instruction_value() {
            // Force 16-byte alignment to satisfy SIMD/slice requirements
            instr.set_alignment(16).ok();
        }
        alloca
    }

    // Debug method to print IR
    pub fn dump_llvm_ir(&self) {
        self.module.print_to_stderr();
    }

    pub(crate) fn extract_index_bounds(
        &mut self,
        expr: &Expr,
        bounds: &mut HashMap<String, inkwell::values::IntValue<'ctx>>,
    ) -> Result<(), String> {
        match expr {
            Expr::IndexAccess(target, indices) => {
                // Target should be Expr::Ident for variable access
                // Instead of compiling, look up the variable directly
                let (tensor_ptr, is_scalar_array, array_len) = match target.as_ref() {
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
                                (loaded, false, 0)
                            }
                            Type::ScalarArray(_, len) => {
                                // For ScalarArray, val is pointer to alloca storing pointer to array
                                // We don't need the runtime pointer for bounds, just the length
                                (val.into_pointer_value(), true, len)
                            }

                            _ => return Err("Expected tensor variable".into()),
                        }
                    }
                    _ => {
                        return Err("Complex index target not supported in bounds extraction".into())
                    }
                };

                let dim_fn = self.module.get_function("tl_tensor_dim").unwrap();
                for (i, idx_expr) in indices.iter().enumerate() {
                    match idx_expr {
                        Expr::Int(_) | Expr::Float(_) => continue,
                        Expr::Variable(name) => {
                            if !bounds.contains_key(name) {
                                let dim_size = if is_scalar_array {
                                    if i == 0 {
                                        self.context.i64_type().const_int(array_len as u64, false)
                                    } else {
                                        return Err("ScalarArray only has 1 dimension".into());
                                    }
                                } else {
                                    let dim_idx_val =
                                        self.context.i64_type().const_int(i as u64, false);
                                    let call_result = self
                                        .builder
                                        .build_call(
                                            dim_fn,
                                            &[tensor_ptr.into(), dim_idx_val.into()],
                                            "dim_size",
                                        )
                                        .map_err(|e| e.to_string())?;
                                    match call_result.try_as_basic_value() {
                                        ValueKind::Basic(v) => v.into_int_value(),
                                        _ => return Err("Invalid dim return".into()),
                                    }
                                };
                                bounds.insert(name.clone(), dim_size);
                            }
                        }
                        _ => continue,
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
    fn compile_method_call(
        &mut self,
        obj: &Expr,
        method: &str,
        args: &[Expr],
    ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        let (obj_val, obj_ty) = self.compile_expr(obj)?;

        let maybe_struct_name = match &obj_ty {
            Type::Struct(name) => Some(name.clone()),
            Type::UserDefined(name) => Some(name.clone()),
            _ => None,
        };

        if let Some(struct_name) = maybe_struct_name {
            // Try exact mangling first: tl_{Struct}_{Method}
            let mangled_name = format!("tl_{}_{}", struct_name, method);
            // Fallback to lowercase for stdlib compatibility (e.g. tl_file_open?)
            // Actually stdlib uses lowercase.
            let stdlib_name = format!("tl_{}_{}", struct_name.to_lowercase(), method);

            let (func_val, final_name) = if let Some(f) = self.module.get_function(&mangled_name) {
                (f, mangled_name)
            } else if let Some(f) = self.module.get_function(&stdlib_name) {
                (f, stdlib_name)
            } else {
                return Err(format!(
                    "Method {} not found in struct {} (checked {} and {})",
                    method, struct_name, mangled_name, stdlib_name
                ));
            };

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
                .get(&final_name)
                .unwrap_or(&Type::Void)
                .clone();
            if let Type::Void = ret_ty {
                Ok((
                    self.context.i64_type().const_int(0, false).into(),
                    Type::Void,
                ))
            } else {
                match call.try_as_basic_value() {
                    ValueKind::Basic(v) => {
                        // FIX: User-defined methods unregister return value, so we must register it
                        // to prevent memory leaks (and keep graph alive for backprop).
                        if let Type::Tensor(_, _) = ret_ty {
                            // Only register if it's a tensor
                            // Assumes user-defined methods always unregister.
                            // Runtime methods are handled in the 'else' block below or specific matches.
                            if let Some(reg_fn) = self.module.get_function("tl_mem_register_tensor")
                            {
                                let ptr = v.into_pointer_value();
                                // Ensure pointer type match
                                let cast_ptr = self
                                    .builder
                                    .build_pointer_cast(
                                        ptr,
                                        self.context.ptr_type(inkwell::AddressSpace::default()),
                                        "cast_reg",
                                    )
                                    .map_err(|e| e.to_string())?;

                                self.builder
                                    .build_call(reg_fn, &[cast_ptr.into()], "")
                                    .map_err(|e| e.to_string())?;
                            }
                        }
                        Ok((v, ret_ty))
                    }
                    _ => Err("Invalid return value".into()),
                }
            }
        } else {
            match method {
                "get" => {
                    if args.len() != 1 {
                        return Err("get requires 1 argument".into());
                    }
                    let (idx_val, idx_ty) = self.compile_expr(&args[0])?;

                    // Ensure index is i64
                    let idx_i64 = match idx_ty {
                        Type::I64 => idx_val.into_int_value(),
                        Type::I32 => self
                            .builder
                            .build_int_z_extend(
                                idx_val.into_int_value(),
                                self.context.i64_type(),
                                "idx_ext",
                            )
                            .map_err(|e| e.to_string())?,
                        _ => return Err("Index must be integer".into()),
                    };

                    let fn_val = self.module.get_function("tl_tensor_get").unwrap();
                    let call = self
                        .builder
                        .build_call(fn_val, &[obj_val.into(), idx_i64.into()], "get_res")
                        .map_err(|e| e.to_string())?;

                    let res = match call.try_as_basic_value() {
                        ValueKind::Basic(v) => v,
                        _ => return Err("Invalid get return".into()),
                    };
                    Ok((res, Type::F32))
                }
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
                    let fn_val = self.module.get_function("tl_tensor_detach").unwrap();
                    // Optional arg: req_grad (bool). Default to false.
                    let req_grad = if !args.is_empty() {
                        let (arg_val, _arg_ty) = self.compile_expr(&args[0])?;
                        arg_val.into_int_value()
                    } else {
                        self.context.bool_type().const_int(0, false)
                    };

                    let call = self
                        .builder
                        .build_call(fn_val, &[obj_val.into(), req_grad.into()], "detach_res")
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

                    match call.try_as_basic_value() {
                        ValueKind::Basic(v) => Ok((v, obj_ty)),
                        _ => Err("Invalid grad return".into()),
                    }
                }
                "save" => {
                    let fn_val = self.module.get_function("tl_tensor_save").unwrap();
                    let (path_val, _) = self.compile_expr(&args[0])?;

                    // tl_tensor_save(path, tensor)
                    self.builder
                        .build_call(fn_val, &[path_val.into(), obj_val.into()], "save_call")
                        .map_err(|e| e.to_string())?;

                    Ok((
                        self.context.i64_type().const_int(0, false).into(),
                        Type::Void,
                    ))
                }

                "reshape" => {
                    if args.len() != 1 {
                        return Err("reshape method requires 1 argument (shape)".into());
                    }
                    let (s_val, _) = self.compile_expr(&args[0])?;
                    let reshape_fn = self.module.get_function("tl_tensor_reshape").unwrap();
                    let call = self
                        .builder
                        .build_call(reshape_fn, &[obj_val.into(), s_val.into()], "reshape_res")
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
                "slice" => {
                    if args.len() != 2 {
                        return Err("slice requires 2 arguments".into());
                    }

                    // slice() only works on Tensors, not ScalarArrays
                    if matches!(obj_ty, Type::ScalarArray(_, _)) {
                        return Err("slice() does not support ScalarArray. Convert to Tensor first using Tensor::new() or similar".into());
                    }

                    let (start_val, _) = self.compile_expr(&args[0])?;
                    let (len_val, _) = self.compile_expr(&args[1])?;

                    let fn_val = self.module.get_function("tl_tensor_slice").unwrap();
                    let call = self
                        .builder
                        .build_call(
                            fn_val,
                            &[obj_val.into(), start_val.into(), len_val.into()],
                            "slice_res",
                        )
                        .map_err(|e| e.to_string())?;
                    let res = match call.try_as_basic_value() {
                        ValueKind::Basic(v) => v,
                        _ => return Err("Invalid slice return".into()),
                    };
                    Ok((res, obj_ty))
                }
                "add_assign" | "sub_assign" | "mul_assign" | "div_assign" => {
                    if args.len() != 1 {
                        return Err(format!("{} requires 1 argument", method));
                    }
                    // Must use ensure_tensor for RHS
                    let rhs_val = self.ensure_tensor_v2(&args[0], 0)?;

                    let fn_name = match method {
                        "add_assign" => "tl_tensor_add_assign",
                        "sub_assign" => "tl_tensor_sub_assign",
                        "mul_assign" => "tl_tensor_mul_assign",
                        "div_assign" => "tl_tensor_div_assign",
                        _ => unreachable!(),
                    };

                    let fn_val = self
                        .module
                        .get_function(fn_name)
                        .ok_or(format!("Runtime fn {} not found", fn_name))?;

                    self.builder
                        .build_call(fn_val, &[obj_val.into(), rhs_val.into()], "assign_res")
                        .map_err(|e| e.to_string())?;

                    Ok((
                        self.context.i64_type().const_int(0, false).into(),
                        Type::Void,
                    ))
                }
                _ => {
                    // Generic method dispatch for UserDefined types calling runtime functions
                    // e.g. File.read_string -> tl_file_read_string
                    if let Type::UserDefined(type_name) = &obj_ty {
                        let method_name = method; // e.g. read_string
                        let runtime_fn_name =
                            format!("tl_{}_{}", type_name.to_lowercase(), method_name);

                        let fn_val = self.module.get_function(&runtime_fn_name).ok_or(format!(
                            "Method {} not found on type {} (checked {})",
                            method, type_name, runtime_fn_name
                        ))?;

                        // Prepend object to args
                        let mut compiled_args = Vec::with_capacity(args.len() + 1);
                        compiled_args.push(obj_val.into());
                        for arg in args {
                            let (val, _) = self.compile_expr(arg)?;
                            compiled_args.push(val.into());
                        }

                        let call = self
                            .builder
                            .build_call(fn_val, &compiled_args, "method_res")
                            .map_err(|e| e.to_string())?;

                        // Determine return type from fn_return_types map
                        let ret_type = self
                            .fn_return_types
                            .get(&runtime_fn_name)
                            .cloned()
                            .unwrap_or(Type::Void);

                        match call.try_as_basic_value() {
                            ValueKind::Basic(v) => Ok((v, ret_type)),
                            _ => Ok((
                                self.context.i64_type().const_int(0, false).into(),
                                Type::Void,
                            )),
                        }
                    } else {
                        Err(format!("Unknown method: {} on type {:?}", method, obj_ty))
                    }
                }
            }
        }
    }

    fn compile_fn_call(
        &mut self,
        name: &str,
        args: &[Expr],
    ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        if let Some(struct_def) = self.struct_defs.get(name).cloned() {
            let st_llvm_ty = *self.struct_types.get(name).unwrap();
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
                    .build_struct_gep(st_llvm_ty, struct_ptr, i as u32, "init_field")
                    .map_err(|e| e.to_string())?;
                self.builder
                    .build_store(field_ptr, val)
                    .map_err(|e| e.to_string())?;
            }
            return Ok((struct_ptr.into(), Type::Struct(name.to_string())));
        }
        match name {
            "register_modules" => {
                if args.len() != 1 {
                    return Err("register_modules requires 1 argument (struct)".into());
                }
                let (val, ty) = self.compile_expr(&args[0])?;
                match ty {
                    Type::Struct(sname) | Type::UserDefined(sname) => {
                        self.gen_register_params(val, &sname, "".to_string())?;
                        return Ok((self.context.i64_type().const_zero().into(), Type::Void));
                    }
                    _ => return Err("register_modules expects a struct argument".into()),
                }
            }
            "print" | "println" => {
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
                    Type::UserDefined(s) if s == "String" => {
                        let fn_val = self.module.get_function("tl_print_string");
                        if let Some(f) = fn_val {
                            self.builder
                                .build_call(f, &[arg_val.into()], "print_call")
                                .map_err(|e| e.to_string())?;
                        } else {
                            // If not declared, try to declare it (lazy) or error.
                            // Better to return error if not found, but it should be found if declared.
                            // For now, assume declared or error.
                            return Err("tl_print_string not found (add to init)".into());
                        }
                    }
                    Type::ScalarArray(ref elem_type, len) => {
                        let i64_type = self.context.i64_type();
                        let f32_type = self.context.f32_type();

                        let (llvm_elem_type, print_fn_name): (inkwell::types::BasicTypeEnum, &str) =
                            match elem_type.as_ref() {
                                Type::I64 => (i64_type.into(), "tl_print_i64"),
                                _ => (f32_type.into(), "tl_print_f32"),
                            };

                        let print_fn = self
                            .module
                            .get_function(print_fn_name)
                            .ok_or(format!("{} not found", print_fn_name))?;

                        let array_ptr = arg_val.into_pointer_value();

                        for i in 0..len {
                            let idx = i64_type.const_int(i as u64, false);
                            let elem_ptr = unsafe {
                                self.builder
                                    .build_in_bounds_gep(
                                        llvm_elem_type,
                                        array_ptr,
                                        &[idx],
                                        "elem_ptr",
                                    )
                                    .map_err(|e| e.to_string())?
                            };
                            let elem_val = self
                                .builder
                                .build_load(llvm_elem_type, elem_ptr, "elem_val")
                                .map_err(|e| e.to_string())?;
                            self.builder
                                .build_call(print_fn, &[elem_val.into()], "print_elem")
                                .map_err(|e| e.to_string())?;
                        }
                    }
                    _ => return Err(format!("Cannot print type {:?}", arg_type)),
                }
                Ok((
                    self.context.i64_type().const_int(0, false).into(),
                    Type::Void,
                ))
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
            "save_weights" => {
                if args.len() != 2 {
                    return Err("save_weights requires 2 arguments: tensor/struct, path".into());
                }
                let (t_val, t_ty) = self.compile_expr(&args[0])?;
                let (path_val, path_ty) = self.compile_expr(&args[1])?;

                if !matches!(path_ty, Type::UserDefined(s) if s == "String") {
                    return Err("Second argument to save_weights must be a String (path)".into());
                }

                match t_ty {
                    Type::Tensor(_, _) => {
                        let fn_val = self
                            .module
                            .get_function("tl_tensor_save")
                            .ok_or("tl_tensor_save not found")?;
                        self.builder
                            .build_call(fn_val, &[t_val.into(), path_val.into()], "")
                            .map_err(|e| e.to_string())?;
                    }
                    Type::UserDefined(struct_name) | Type::Struct(struct_name)
                        if struct_name != "String" =>
                    {
                        // Struct serialization
                        let new_fn = self
                            .module
                            .get_function("tl_tensor_map_new")
                            .ok_or("tl_tensor_map_new not found")?;
                        let map_call = self
                            .builder
                            .build_call(new_fn, &[], "map")
                            .map_err(|e| e.to_string())?;
                        let map_val = match map_call.try_as_basic_value() {
                            inkwell::values::ValueKind::Basic(v) => v,
                            _ => return Err("tl_tensor_map_new returned void".into()),
                        };

                        self.gen_save_struct(map_val, t_val, &struct_name, "".to_string())?;

                        let save_fn = self
                            .module
                            .get_function("tl_tensor_map_save")
                            .ok_or("tl_tensor_map_save not found")?;
                        self.builder
                            .build_call(save_fn, &[map_val.into(), path_val.into()], "")
                            .map_err(|e| e.to_string())?;

                        let free_fn = self
                            .module
                            .get_function("tl_tensor_map_free")
                            .ok_or("tl_tensor_map_free not found")?;
                        self.builder
                            .build_call(free_fn, &[map_val.into()], "")
                            .map_err(|e| e.to_string())?;
                    }
                    _ => {
                        return Err(format!(
                        "First argument to save_weights must be a tensor or struct. Found: {:?}",
                        t_ty
                    ))
                    }
                }

                Ok((
                    self.context.i64_type().const_int(0, false).into(),
                    Type::Void,
                ))
            }
            "load_weights" => {
                if args.len() == 1 {
                    let (path_val, path_ty) = self.compile_expr(&args[0])?;
                    if !matches!(path_ty, Type::UserDefined(s) if s == "String") {
                        return Err("Argument to load_weights must be a String (path)".into());
                    }

                    let fn_val = self
                        .module
                        .get_function("tl_tensor_load")
                        .ok_or("tl_tensor_load not found")?;
                    let call = self
                        .builder
                        .build_call(fn_val, &[path_val.into()], "load_res")
                        .map_err(|e| e.to_string())?;

                    let res = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err("Invalid load_weights return".into()),
                    };
                    Ok((res, Type::Tensor(Box::new(Type::F32), 0)))
                } else if args.len() == 2 {
                    // Struct load
                    let (struct_val, s_ty) = self.compile_expr(&args[0])?;
                    let (path_val, path_ty) = self.compile_expr(&args[1])?;
                    if !matches!(path_ty, Type::UserDefined(s) if s == "String") {
                        return Err(
                            "Second argument to load_weights must be a String (path)".into()
                        );
                    }

                    let struct_name_opt = match &s_ty {
                        Type::UserDefined(s) => Some(s.clone()),
                        Type::Struct(s) => Some(s.clone()),
                        _ => None,
                    };

                    if let Some(struct_name) = struct_name_opt {
                        if struct_name == "String" {
                            return Err("Cannot load weights into String".into());
                        }

                        let load_fn = self
                            .module
                            .get_function("tl_tensor_map_load")
                            .ok_or("tl_tensor_map_load not found")?;
                        let map_call = self
                            .builder
                            .build_call(load_fn, &[path_val.into()], "map")
                            .map_err(|e| e.to_string())?;
                        let map_val = match map_call.try_as_basic_value() {
                            inkwell::values::ValueKind::Basic(v) => v,
                            _ => return Err("tl_tensor_map_load returned void".into()),
                        };

                        self.gen_load_struct(map_val, struct_val, &struct_name, "".to_string())?;

                        let free_fn = self
                            .module
                            .get_function("tl_tensor_map_free")
                            .ok_or("tl_tensor_map_free not found")?;
                        self.builder
                            .build_call(free_fn, &[map_val.into()], "")
                            .map_err(|e| e.to_string())?;

                        Ok((
                            self.context.i64_type().const_int(0, false).into(),
                            Type::Void,
                        ))
                    } else {
                        Err(format!(
                            "First argument to load_weights (2 args) must be a struct. Found: {:?}",
                            s_ty
                        ))
                    }
                } else {
                    Err("load_weights requires 1 or 2 arguments".into())
                }
            }
            "reshape" => {
                if args.len() < 2 {
                    return Err("reshape requires at least tensor and 1 dimension".into());
                }
                let t_val = self.ensure_tensor_v2(&args[0], 0)?;

                // Check if 2nd arg is Tensor/Array (Old behavior)
                let (_, arg1_ty) = self.compile_expr(&args[1])?;
                if (matches!(arg1_ty, Type::Tensor(_, _))
                    || matches!(arg1_ty, Type::ScalarArray(_, _)))
                    && args.len() == 2
                {
                    let s_val = self.ensure_tensor_v2(&args[1], 1)?;
                    let reshape_fn = self
                        .module
                        .get_function("tl_tensor_reshape")
                        .ok_or("tl_tensor_reshape not found")?;
                    let call = self
                        .builder
                        .build_call(reshape_fn, &[t_val.into(), s_val.into()], "reshape_res")
                        .map_err(|e| e.to_string())?;
                    match call.try_as_basic_value() {
                        ValueKind::Basic(v) => {
                            return Ok((v, Type::Tensor(Box::new(Type::Void), 0)))
                        }
                        _ => return Err("Invalid reshape return".into()),
                    }
                }
                // New behavior: Varargs dims (arg 1..N are ints)
                let fn_val = self.module.get_function("tl_tensor_reshape_dims").unwrap();
                let num_dims = args.len() - 1;
                let i64_type = self.context.i64_type();
                // Allocate array for dims
                let dims_array_type = i64_type.array_type(num_dims as u32);
                let dims_alloca = self
                    .builder
                    .build_alloca(dims_array_type, "dims_alloca")
                    .map_err(|e| e.to_string())?;
                // Store dims
                for (i, arg) in args[1..].iter().enumerate() {
                    let (val, val_ty) = self.compile_expr(arg)?;
                    let val_int = if val_ty == Type::I32 {
                        self.builder
                            .build_int_z_extend(val.into_int_value(), i64_type, "ext")
                            .map_err(|e| e.to_string())?
                    } else {
                        val.into_int_value()
                    };
                    unsafe {
                        let gep = self
                            .builder
                            .build_gep(
                                dims_array_type,
                                dims_alloca,
                                &[
                                    i64_type.const_int(0, false),
                                    i64_type.const_int(i as u64, false),
                                ],
                                "dim_ptr",
                            )
                            .map_err(|e| e.to_string())?;
                        self.builder
                            .build_store(gep, val_int)
                            .map_err(|e| e.to_string())?;
                    }
                }
                // Pass pointer to first element
                let first_elem_ptr = unsafe {
                    self.builder
                        .build_gep(
                            dims_array_type,
                            dims_alloca,
                            &[i64_type.const_int(0, false), i64_type.const_int(0, false)],
                            "dims_ptr",
                        )
                        .map_err(|e| e.to_string())?
                };
                let call = self
                    .builder
                    .build_call(
                        fn_val,
                        &[
                            t_val.into(),
                            first_elem_ptr.into(),
                            i64_type.const_int(num_dims as u64, false).into(),
                        ],
                        "reshape_dims_res",
                    )
                    .map_err(|e| e.to_string())?;
                let res = match call.try_as_basic_value() {
                    ValueKind::Basic(v) => v,
                    _ => return Err("Invalid reshape_dims return".into()),
                };
                Ok((res, Type::Tensor(Box::new(Type::Void), 0)))
            }
            "softmax" => {
                if args.len() != 2 {
                    return Err("softmax requires 2 arguments".into());
                }
                let arg0_val = self.ensure_tensor_v2(&args[0], 0)?;
                let arg0_ty = Type::Tensor(Box::new(Type::F32), 0); // Simplified
                let (arg1_val, _arg1_ty) = self.compile_expr(&args[1])?; // dim

                // arg1 must be i64 (dim)
                let fn_val = self
                    .module
                    .get_function("tl_tensor_softmax")
                    .ok_or("tl_tensor_softmax not found")?;
                let call = self
                    .builder
                    .build_call(fn_val, &[arg0_val.into(), arg1_val.into()], "softmax_res")
                    .map_err(|e| e.to_string())?;
                let res = match call.try_as_basic_value() {
                    ValueKind::Basic(v) => v,
                    _ => return Err("Invalid softmax return".into()),
                };
                Ok((res, arg0_ty))
            }
            "cross_entropy" => {
                if args.len() != 2 {
                    return Err("cross_entropy requires 2 arguments".into());
                }
                let arg0_val = self.ensure_tensor_v2(&args[0], 0)?;
                let arg1_val = self.ensure_tensor_v2(&args[1], 0)?;

                let fn_val = self
                    .module
                    .get_function("tl_tensor_cross_entropy")
                    .ok_or("tl_tensor_cross_entropy not found")?;
                let call = self
                    .builder
                    .build_call(fn_val, &[arg0_val.into(), arg1_val.into()], "ce_res")
                    .map_err(|e| e.to_string())?;
                let res = match call.try_as_basic_value() {
                    ValueKind::Basic(v) => v,
                    _ => return Err("Invalid cross_entropy return".into()),
                };
                // Returns scalar tensor (float)
                Ok((res, Type::Tensor(Box::new(Type::F32), 0)))
            }
            "exp" => {
                if args.len() != 1 {
                    return Err("exp requires 1 argument".into());
                }
                let arg_val = self.ensure_tensor_v2(&args[0], 0)?;
                let arg_ty = Type::Tensor(Box::new(Type::F32), 0);

                let fn_val = self.module.get_function("tl_tensor_exp").unwrap();
                let call = self
                    .builder
                    .build_call(fn_val, &[arg_val.into()], "exp_res")
                    .map_err(|e| e.to_string())?;
                let res = match call.try_as_basic_value() {
                    ValueKind::Basic(v) => v,
                    _ => return Err("Invalid exp return".into()),
                };
                Ok((res, arg_ty))
            }
            "log" => {
                if args.len() != 1 {
                    return Err("log requires 1 argument".into());
                }
                let arg_val = self.ensure_tensor_v2(&args[0], 0)?;
                let arg_ty = Type::Tensor(Box::new(Type::F32), 0);

                let fn_val = self.module.get_function("tl_tensor_log").unwrap();
                let call = self
                    .builder
                    .build_call(fn_val, &[arg_val.into()], "log_res")
                    .unwrap();
                let res = match call.try_as_basic_value() {
                    ValueKind::Basic(v) => v,
                    _ => return Err("Invalid log return".into()),
                };
                Ok((res, arg_ty))
            }
            "len" => {
                if args.len() != 1 {
                    return Err("len requires 1 argument".into());
                }
                let arg_val = self.ensure_tensor_v2(&args[0], 0)?;

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
                Ok((res, Type::I64))
            }
            "sqrt" => {
                if args.len() != 1 {
                    return Err("sqrt requires 1 argument".into());
                }
                let arg_val = self.ensure_tensor_v2(&args[0], 0)?;
                let arg_ty = Type::Tensor(Box::new(Type::F32), 0);

                let fn_val = self.module.get_function("tl_tensor_sqrt").unwrap();
                let call = self
                    .builder
                    .build_call(fn_val, &[arg_val.into()], "sqrt_res")
                    .map_err(|e| e.to_string())?;
                let res = match call.try_as_basic_value() {
                    ValueKind::Basic(v) => v,
                    _ => return Err("Invalid sqrt return".into()),
                };
                Ok((res, arg_ty))
            }
            "matmul" => {
                if args.len() != 2 {
                    return Err("matmul requires 2 arguments".into());
                }
                let lhs_val = self.ensure_tensor_v2(&args[0], 0)?;
                let rhs_val = self.ensure_tensor_v2(&args[1], 0)?;

                let fn_val = self.module.get_function("tl_tensor_matmul").unwrap();
                let call = self
                    .builder
                    .build_call(fn_val, &[lhs_val.into(), rhs_val.into()], "matmul_res")
                    .map_err(|e| e.to_string())?;
                let res = match call.try_as_basic_value() {
                    ValueKind::Basic(v) => v,
                    _ => return Err("Invalid matmul return".into()),
                };
                Ok((res, Type::Tensor(Box::new(Type::F32), 0)))
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
                Ok((res, arg_ty))
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
                Ok((
                    self.context.i64_type().const_int(0, false).into(),
                    Type::Void,
                ))
            }
            "varbuilder_get" => {
                if args.len() < 2 {
                    return Err(
                        "varbuilder_get requires at least 2 arguments (name and dimensions)".into(),
                    );
                }
                let (name_val, name_ty) = self.compile_expr(&args[0])?;
                if !matches!(name_ty, Type::UserDefined(ref s) if s == "String") {
                    return Err(format!(
                        "varbuilder_get expects String as first argument, found {:?}",
                        name_ty
                    ));
                }
                let name_ptr = name_val.into_pointer_value();

                // Handle both TensorLiteral/ScalarArray and Varargs dims
                let (rank, shape_ptr) = if args.len() == 2
                    && matches!(
                        self.compile_expr(&args[1])?.1,
                        Type::Tensor(_, _) | Type::ScalarArray(_, _)
                    ) {
                    let (shape_val, arg1_ty) = self.compile_expr(&args[1])?;
                    let (num_elements, shape_vals) = match &arg1_ty {
                        Type::Tensor(_, _) => {
                            let len_fn = self
                                .module
                                .get_function("tl_tensor_len")
                                .ok_or("tl_tensor_len not found")?;
                            let call = self
                                .builder
                                .build_call(len_fn, &[shape_val.into()], "len")
                                .map_err(|e| e.to_string())?;
                            let _len = match call.try_as_basic_value() {
                                ValueKind::Basic(v) => v.into_int_value(),
                                _ => return Err("Invalid len return".into()),
                            };

                            // We need the values too. For now, assume it's a literal we can inspect or use runtime loop.
                            // But CodeGen for varbuilder_get currently expects to build a static shape array.
                            // If it's a literal, we can flatten it.
                            match &args[1] {
                                Expr::TensorLiteral(elements)
                                | Expr::TensorConstLiteral(elements) => (
                                    elements.len(),
                                    elements
                                        .iter()
                                        .map(|e| {
                                            let (val, _) = self.compile_expr(e)?;
                                            Ok(val)
                                        })
                                        .collect::<Result<Vec<_>, String>>()?,
                                ),
                                _ => {
                                    return Err(
                                        "varbuilder_get shape must be a literal array".into()
                                    )
                                }
                            }
                        }
                        Type::ScalarArray(_, l) => match &args[1] {
                            Expr::TensorLiteral(elements) | Expr::TensorConstLiteral(elements) => (
                                *l,
                                elements
                                    .iter()
                                    .map(|e| {
                                        let (val, _) = self.compile_expr(e)?;
                                        Ok(val)
                                    })
                                    .collect::<Result<Vec<_>, String>>()?,
                            ),
                            _ => return Err("varbuilder_get shape must be a literal array".into()),
                        },
                        _ => unreachable!(),
                    };

                    let i64_type = self.context.i64_type();
                    let shape_alloca = self
                        .builder
                        .build_alloca(i64_type.array_type(num_elements as u32), "shape_arr")
                        .unwrap();
                    for (i, val) in shape_vals.iter().enumerate() {
                        let idx = self.context.i64_type().const_int(i as u64, false);
                        let ptr = unsafe {
                            self.builder
                                .build_in_bounds_gep(
                                    i64_type.array_type(num_elements as u32),
                                    shape_alloca,
                                    &[self.context.i64_type().const_zero(), idx],
                                    "shptr",
                                )
                                .unwrap()
                        };
                        self.builder.build_store(ptr, val.into_int_value()).unwrap();
                    }
                    (num_elements, shape_alloca)
                } else {
                    // Varargs mode: args[1..] are dims
                    let num_dims = args.len() - 1;
                    let i64_type = self.context.i64_type();
                    let shape_alloca = self
                        .builder
                        .build_alloca(i64_type.array_type(num_dims as u32), "shape_arr")
                        .unwrap();
                    for (i, arg) in args[1..].iter().enumerate() {
                        let (val, _) = self.compile_expr(arg)?;
                        let idx = i64_type.const_int(i as u64, false);
                        let ptr = unsafe {
                            self.builder
                                .build_in_bounds_gep(
                                    i64_type.array_type(num_dims as u32),
                                    shape_alloca,
                                    &[i64_type.const_zero(), idx],
                                    "shptr",
                                )
                                .unwrap()
                        };
                        self.builder.build_store(ptr, val.into_int_value()).unwrap();
                    }
                    (num_dims, shape_alloca)
                };

                let fn_val = self.module.get_function("tl_varbuilder_get").unwrap();
                let call = self
                    .builder
                    .build_call(
                        fn_val,
                        &[
                            name_ptr.into(),
                            self.context.i64_type().const_int(rank as u64, false).into(),
                            shape_ptr.into(),
                        ],
                        "varbuilder_get_result",
                    )
                    .unwrap();
                let res = match call.try_as_basic_value() {
                    ValueKind::Basic(v) => v,
                    _ => return Err("Invalid varbuilder_get return".into()),
                };
                Ok((res, Type::Tensor(Box::new(Type::F32), 0)))
            }
            "update_all_params" => {
                if args.len() != 1 {
                    return Err("update_all_params requires 1 argument".into());
                }
                let (lr_val, _) = self.compile_expr(&args[0])?;
                let fn_val = self.module.get_function("tl_update_all_params").unwrap();
                self.builder
                    .build_call(fn_val, &[lr_val.into()], "")
                    .unwrap();
                Ok((
                    self.context.i64_type().const_int(0, false).into(),
                    Type::Void,
                ))
            }
            "varbuilder_grad" => {
                if args.len() != 1 {
                    return Err("varbuilder_grad requires 1 argument".into());
                }
                let (name_val, name_ty) = self.compile_expr(&args[0])?;
                if !matches!(name_ty, Type::UserDefined(ref s) if s == "String") {
                    return Err(format!(
                        "varbuilder_grad expects String as argument, found {:?}",
                        name_ty
                    ));
                }
                let name_ptr = name_val.into_pointer_value();

                let fn_val = self.module.get_function("tl_varbuilder_grad").unwrap();
                let call = self
                    .builder
                    .build_call(fn_val, &[name_ptr.into()], "varbuilder_grad_result")
                    .unwrap();
                let res = match call.try_as_basic_value() {
                    ValueKind::Basic(v) => v,
                    _ => return Err("Invalid varbuilder_grad return".into()),
                };
                Ok((res, Type::Tensor(Box::new(Type::F32), 0)))
            }
            "sum" => {
                if args.len() == 1 {
                    // Global sum
                    let arg_val = self.ensure_tensor_v2(&args[0], 0)?;
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
                    Ok((res, Type::Tensor(Box::new(Type::F32), 0)))
                } else if args.len() == 2 {
                    // Sum over dim: sum(t, dim)
                    let t_val = self.ensure_tensor_v2(&args[0], 0)?;
                    let (dim_val, dim_ty) = self.compile_expr(&args[1])?;
                    // Convert dim to i64 (usize)
                    let dim_int = match dim_ty {
                        Type::I64 => dim_val.into_int_value(),
                        Type::I32 => self
                            .builder
                            .build_int_z_extend(
                                dim_val.into_int_value(),
                                self.context.i64_type(),
                                "dim_ext",
                            )
                            .map_err(|e| e.to_string())?,
                        _ => return Err("sum dimension must be integer".into()),
                    };
                    // keep_dim = false (hardcoded for now, or could be optional arg)
                    let keep_dim = self.context.bool_type().const_int(0, false);
                    let fn_val = self.module.get_function("tl_tensor_sum_dim").unwrap();
                    let call = self
                        .builder
                        .build_call(
                            fn_val,
                            &[t_val.into(), dim_int.into(), keep_dim.into()],
                            "sum_dim_res",
                        )
                        .map_err(|e| e.to_string())?;
                    let res = match call.try_as_basic_value() {
                        ValueKind::Basic(v) => v,
                        _ => return Err("Invalid sum return".into()),
                    };
                    Ok((res, Type::Tensor(Box::new(Type::F32), 0)))
                } else {
                    Err("sum requires 1 or 2 arguments".into())
                }
            }
            "sin" | "cos" | "relu" | "gelu" => {
                if args.len() != 1 {
                    return Err(format!("{} requires 1 argument", name));
                }
                let (arg_val, _arg_ty) = self.compile_expr(&args[0])?;
                let func_name = format!("tl_tensor_{}", name);
                let fn_val = self
                    .module
                    .get_function(&func_name)
                    .ok_or(format!("Function {} not found", func_name))?;
                let call = self
                    .builder
                    .build_call(fn_val, &[arg_val.into()], &format!("{}_res", name))
                    .map_err(|e| e.to_string())?;
                let res = match call.try_as_basic_value() {
                    ValueKind::Basic(v) => v,
                    _ => return Err(format!("Invalid {} return", name)),
                };
                Ok((res, Type::Tensor(Box::new(Type::F32), 1)))
            }
            "tril" => {
                if args.len() != 2 {
                    return Err("tril requires 2 arguments".into());
                }
                let t_val = self.ensure_tensor_v2(&args[0], 0)?;
                let (diag_val, diag_ty) = self.compile_expr(&args[1])?;
                // Cast diag to i32
                let diag_i32 = match diag_ty {
                    Type::I64 => {
                        // Truncate i64 -> i32
                        self.builder
                            .build_int_cast(
                                diag_val.into_int_value(),
                                self.context.i32_type(),
                                "diag_cast",
                            )
                            .map_err(|e| e.to_string())?
                    }
                    Type::I32 => diag_val.into_int_value(), // Should be this based on semantics
                    _ => return Err("tril diagonal must be integer".into()),
                };
                let fn_val = self.module.get_function("tl_tensor_tril").unwrap();
                let call = self
                    .builder
                    .build_call(fn_val, &[t_val.into(), diag_i32.into()], "tril_res")
                    .map_err(|e| e.to_string())?;
                let res = match call.try_as_basic_value() {
                    ValueKind::Basic(v) => v,
                    _ => return Err("Invalid tril return".into()),
                };
                Ok((res, Type::Tensor(Box::new(Type::F32), 1)))
            }
            "embedding" => {
                if args.len() != 2 {
                    return Err("embedding requires 2 arguments".into());
                }
                let idx_val = self.ensure_tensor_v2(&args[0], 0)?;
                let w_val = self.ensure_tensor_v2(&args[1], 0)?;
                let fn_val = self.module.get_function("tl_tensor_embedding").unwrap();
                let call = self
                    .builder
                    .build_call(fn_val, &[idx_val.into(), w_val.into()], "emb_res")
                    .map_err(|e| e.to_string())?;
                let res = match call.try_as_basic_value() {
                    ValueKind::Basic(v) => v,
                    _ => return Err("Invalid embedding return".into()),
                };
                Ok((res, Type::Tensor(Box::new(Type::F32), 1)))
            }
            "pow" => {
                if args.len() != 2 {
                    return Err("pow requires 2 arguments".into());
                }
                // Use ensure_tensor for both args
                let base_val = self.ensure_tensor_v2(&args[0], 0)?;
                let exp_val = self.ensure_tensor_v2(&args[1], 0)?;
                let fn_val = self.module.get_function("tl_tensor_pow").unwrap();
                let call = self
                    .builder
                    .build_call(fn_val, &[base_val.into(), exp_val.into()], "pow_res")
                    .map_err(|e| e.to_string())?;
                let res = match call.try_as_basic_value() {
                    ValueKind::Basic(v) => v,
                    _ => return Err("Invalid pow return".into()),
                };
                Ok((res, Type::Tensor(Box::new(Type::F32), 0)))
            }
            _ => {
                // Generic function call logic
                let llvm_func_name = match name {
                    "slice" => "tl_tensor_slice",
                    "sum" => "tl_tensor_sum",
                    "load_tensor" => {
                        let fn_val = self.module.get_function("tl_tensor_load").unwrap();
                        let (path_val, _) = self.compile_expr(&args[0])?;
                        let call = self
                            .builder
                            .build_call(fn_val, &[path_val.into()], "load_res")
                            .map_err(|e| e.to_string())?;
                        let res = match call.try_as_basic_value() {
                            ValueKind::Basic(v) => v,
                            _ => return Err("Invalid load return".into()),
                        };
                        return Ok((res, Type::Tensor(Box::new(Type::F32), 1)));
                    }
                    "save_all_params" => {
                        let fn_val = self.module.get_function("tl_save_all_params").unwrap();
                        let path_val = if args.len() == 2 {
                            // Arg 0: Struct/UserDefined
                            let (struct_val, struct_ty) = self.compile_expr(&args[0])?;
                            let struct_name = match struct_ty {
                                Type::Struct(s) | Type::UserDefined(s) => s,
                                _ => return Err("Expected struct as first arg".into()),
                            };
                            self.gen_register_params(struct_val, &struct_name, "".to_string())?;

                            // Arg 1: String
                            let (path, _) = self.compile_expr(&args[1])?;
                            path
                        } else {
                            // Arg 0: String
                            let (path, _) = self.compile_expr(&args[0])?;
                            path
                        };

                        self.builder
                            .build_call(fn_val, &[path_val.into()], "save_all_res")
                            .map_err(|e| e.to_string())?;
                        return Ok((
                            self.context.i64_type().const_int(0, false).into(),
                            Type::Void,
                        ));
                    }
                    "add_parameter" => {
                        let fn_val = self.module.get_function("tl_add_parameter").unwrap();
                        let (name_val, _) = self.compile_expr(&args[0])?;
                        let (tensor_val, _) = self.compile_expr(&args[1])?;
                        self.builder
                            .build_call(fn_val, &[name_val.into(), tensor_val.into()], "")
                            .unwrap();
                        return Ok((
                            self.context.i64_type().const_int(0, false).into(),
                            Type::Void,
                        ));
                    }
                    "load_all_params" => {
                        let fn_val = self.module.get_function("tl_load_all_params").unwrap();
                        let path_val = if args.len() == 2 {
                            // Arg 0: Struct/UserDefined
                            let (struct_val, struct_ty) = self.compile_expr(&args[0])?;
                            let struct_name = match struct_ty {
                                Type::Struct(s) | Type::UserDefined(s) => s,
                                _ => return Err("Expected struct as first arg".into()),
                            };
                            self.gen_register_params(struct_val, &struct_name, "".to_string())?;

                            // Arg 1: String
                            let (path, _) = self.compile_expr(&args[1])?;
                            path
                        } else {
                            // Arg 0: String
                            let (path, _) = self.compile_expr(&args[0])?;
                            path
                        };

                        self.builder
                            .build_call(fn_val, &[path_val.into()], "load_all_res")
                            .map_err(|e| e.to_string())?;
                        return Ok((
                            self.context.i64_type().const_int(0, false).into(),
                            Type::Void,
                        ));
                    }
                    "randn" => {
                        // randn(shape, requires_grad)
                        if args.is_empty() {
                            return Err("randn requires shape".into());
                        }
                        let shape_expr = &args[0];
                        let (rank, shape_vals) = match shape_expr {
                            Expr::TensorLiteral(el) | Expr::TensorConstLiteral(el) => {
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
                            }
                            _ => {
                                // Dynamic shape tensor
                                let shape_tensor = self.ensure_tensor_v2(shape_expr, 1)?;
                                let len_fn = self
                                    .module
                                    .get_function("tl_tensor_len")
                                    .ok_or("tl_tensor_len not found")?;
                                let call = self
                                    .builder
                                    .build_call(len_fn, &[shape_tensor.into()], "len")
                                    .map_err(|e| e.to_string())?;
                                let _ = match call.try_as_basic_value() {
                                    ValueKind::Basic(v) => v.into_int_value(),
                                    _ => return Err("Invalid len return".into()),
                                };

                                // This is hard because we need the actual values at compile time to build the alloca?
                                // No, we can use a dynamic alloca or just a fixed large enough one?
                                // For now, only support literals for shape in randn.
                                return Err(
                                    "randn currently requires array literal [dim, ...] for shape"
                                        .into(),
                                );
                            }
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

                        // Entry block alloca for shape array
                        let current_block = self.builder.get_insert_block().unwrap();
                        let function = current_block.get_parent().unwrap();
                        let entry_block = function.get_first_basic_block().unwrap();
                        let entry_builder = self.context.create_builder();
                        if let Some(first_instr) = entry_block.get_first_instruction() {
                            entry_builder.position_before(&first_instr);
                        } else {
                            entry_builder.position_at_end(entry_block);
                        }

                        let shape_array_type = i64_type.array_type(rank as u32);
                        let shape_alloca = entry_builder
                            .build_alloca(shape_array_type, "shape_arr")
                            .map_err(|e| e.to_string())?;

                        // Fix alignment
                        shape_alloca
                            .as_instruction_value()
                            .unwrap()
                            .set_alignment(16)
                            .map_err(|e| e.to_string())?;

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
                    "parameter" => {
                        if args.len() != 1 {
                            return Err("parameter requires 1 argument".into());
                        }
                        let (arg_val, arg_ty) = self.compile_expr(&args[0])?;
                        let fn_val = self
                            .module
                            .get_function("tl_register_parameter")
                            .ok_or("tl_register_parameter not found")?;
                        let call = self
                            .builder
                            .build_call(fn_val, &[arg_val.into()], "param_reg")
                            .map_err(|e| e.to_string())?;
                        let res = match call.try_as_basic_value() {
                            ValueKind::Basic(v) => v,
                            _ => return Err("Invalid parameter return".into()),
                        };
                        return Ok((res, arg_ty));
                    }
                    _ => name,
                };
                // Handle static method syntax: Type::method -> tl_type_method
                let resolved_name = if llvm_func_name.contains("::") {
                    let parts: Vec<&str> = llvm_func_name.split("::").collect();
                    if parts.len() == 2 {
                        let type_name = parts[0];
                        let method = parts[1];
                        format!("tl_{}_{}", type_name.to_lowercase(), method)
                    } else {
                        llvm_func_name.to_string()
                    }
                } else {
                    llvm_func_name.to_string()
                };
                let func = self.module.get_function(&resolved_name).ok_or(format!(
                    "Function {} not found (resolved: {})",
                    name, resolved_name
                ))?;
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
                let lookup_name = resolved_name.as_str();
                let ret_type = self
                    .fn_return_types
                    .get(lookup_name)
                    .cloned()
                    .unwrap_or(Type::Void);
                match call.try_as_basic_value() {
                    ValueKind::Basic(v) => {
                        // FIX: User-defined functions unregister return value, so we must register it
                        if let Type::Tensor(_, _) = ret_type {
                            // Assume user functions always unregister.
                            // Runtime functions usually matched above, but if any slip through and are registered,
                            // we might double-register.
                            // Ideally we check if it is a runtime function (starts with tl_tensor_).
                            // But usually user functions don't start with tl_tensor_.
                            // And source name is used here.
                            if let Some(reg_fn) = self.module.get_function("tl_mem_register_tensor")
                            {
                                let ptr = v.into_pointer_value();
                                let cast_ptr = self
                                    .builder
                                    .build_pointer_cast(
                                        ptr,
                                        self.context.ptr_type(inkwell::AddressSpace::default()),
                                        "cast_reg",
                                    )
                                    .map_err(|e| e.to_string())?;

                                self.builder
                                    .build_call(reg_fn, &[cast_ptr.into()], "")
                                    .map_err(|e| e.to_string())?;
                            }
                        }
                        Ok((v, ret_type))
                    }
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
}
