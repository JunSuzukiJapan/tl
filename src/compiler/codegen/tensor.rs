use super::CodeGenerator;
use crate::compiler::ast::*;
use inkwell::values::*;
use std::collections::HashMap;

impl<'ctx> CodeGenerator<'ctx> {
    pub(crate) fn compile_tensor_equation(
        &mut self,
        name: &str,
        lhs_indices: &[String],
        value: &Expr,
    ) -> Result<(), String> {
        let i64_type = self.context.i64_type();
        let f32_type = self.context.f32_type();

        eprintln!(
            "compile_tensor_equation: lhs={:?}, name={}",
            lhs_indices, name
        );

        // 1. Try MatMul Optimization
        if let Some(res_ptr) = self.try_compile_matmul(lhs_indices, value)? {
            eprintln!("MatMul Optimization SUCCEEDED");
            // Optimization successful, store result and return
            // Register variable
            let val = res_ptr;

            // Correctly use ast::Type for alloca
            let tensor_type = Type::Tensor(Box::new(Type::F32), lhs_indices.len());

            let alloca = self.create_entry_block_alloca(
                self.builder
                    .get_insert_block()
                    .unwrap()
                    .get_parent()
                    .unwrap(),
                name,
                &tensor_type,
            );
            self.builder
                .build_store(alloca, val)
                .map_err(|e| e.to_string())?;

            self.variables
                .last_mut()
                .unwrap()
                .insert(name.to_string(), (alloca.into(), tensor_type, false));
            return Ok(());
        }

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
            let limit = index_bounds.get(idx).unwrap();
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
                .build_store(elem_ptr, *limit)
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

    // FIXED Helper to ensure an expression evaluates to a Tensor, converting scalars if needed
    // FIXED Helper to ensure an expression evaluates to a Tensor, converting scalars if needed
    // Returns (Value, Type) where Type is the Tensor type (e.g. Tensor<f32, rank>)
    pub(crate) fn ensure_tensor_v2(
        &mut self,
        expr: &Expr,
        _expected_dims: usize,
    ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        let (val, ty) = self.compile_expr(expr)?;

        match ty {
            Type::Tensor(_, _) => Ok((val, ty)),
            Type::ScalarArray(elem_ty, len) => {
                let i64_type = self.context.i64_type();
                let f32_type = self.context.f32_type();
                let rank_val = i64_type.const_int(1, false);

                let (data_ptr, tensor_type, func_name) = match elem_ty.as_ref() {
                    Type::I64 => {
                        // Cast [N x i64]* to i64*
                        let ptr = val.into_pointer_value();
                        let i64_ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
                        let cast_ptr = self
                            .builder
                            .build_pointer_cast(ptr, i64_ptr_type, "i64_ptr_cast")
                            .unwrap();

                        (
                            cast_ptr,
                            Type::Tensor(Box::new(Type::I64), 1),
                            "tl_tensor_new_i64",
                        )
                    }
                    _ => {
                        // Convert to F32, create F32 tensor
                        // Current function for alloca
                        let current_block = self.builder.get_insert_block().unwrap();
                        let parent_fn = current_block.get_parent().unwrap();
                        let f32_array_type = f32_type.array_type(len as u32);
                        // Use manual alloca for array
                        let entry_builder = self.context.create_builder();
                        let entry = parent_fn.get_first_basic_block().unwrap();
                        if let Some(fi) = entry.get_first_instruction() {
                            entry_builder.position_before(&fi);
                        } else {
                            entry_builder.position_at_end(entry);
                        }
                        let new_buf = entry_builder
                            .build_alloca(f32_array_type, "conv_buf")
                            .unwrap();

                        // Copy and convert
                        for i in 0..len {
                            let idx = i64_type.const_int(i as u64, false);
                            let src_ptr = unsafe {
                                self.builder
                                    .build_in_bounds_gep(
                                        i64_type, // Assuming src is i64
                                        val.into_pointer_value(),
                                        &[idx],
                                        "src",
                                    )
                                    .unwrap()
                            };
                            let loaded = self.builder.build_load(i64_type, src_ptr, "l").unwrap();
                            let f_val = self
                                .builder
                                .build_signed_int_to_float(loaded.into_int_value(), f32_type, "c")
                                .unwrap();
                            let dst_ptr = unsafe {
                                self.builder
                                    .build_in_bounds_gep(f32_type, new_buf, &[idx], "dst")
                                    .unwrap()
                            };
                            self.builder.build_store(dst_ptr, f_val).unwrap();
                        }
                        (
                            new_buf,
                            Type::Tensor(Box::new(Type::F32), 1),
                            "tl_tensor_new",
                        )
                    }
                };

                // Shape array on stack
                let shape_array_type = i64_type.array_type(1);
                let shape_alloca = self
                    .builder
                    .build_alloca(shape_array_type, "shape_arr")
                    .map_err(|e| e.to_string())?;
                let shape_elem_ptr = unsafe {
                    self.builder
                        .build_in_bounds_gep(
                            shape_array_type,
                            shape_alloca,
                            &[i64_type.const_int(0, false), i64_type.const_int(0, false)],
                            "shape_ptr",
                        )
                        .map_err(|e| e.to_string())?
                };
                self.builder
                    .build_store(shape_elem_ptr, i64_type.const_int(len as u64, false))
                    .map_err(|e| e.to_string())?;

                let tensor_new_fn = self
                    .module
                    .get_function(func_name)
                    .ok_or(format!("{} not found", func_name))?;
                let shape_ptr_cast = self
                    .builder
                    .build_pointer_cast(
                        shape_alloca,
                        self.context.ptr_type(inkwell::AddressSpace::default()),
                        "shape_ptr_cast",
                    )
                    .map_err(|e| e.to_string())?;

                let call = self
                    .builder
                    .build_call(
                        tensor_new_fn,
                        &[data_ptr.into(), rank_val.into(), shape_ptr_cast.into()],
                        "converted_tensor",
                    )
                    .map_err(|e| e.to_string())?;

                match call.try_as_basic_value() {
                    ValueKind::Basic(v) => Ok((v, tensor_type)),
                    _ => Err(format!("{} returned void", func_name)),
                }
            }
            Type::F32 | Type::I64 => {
                let f32_type = self.context.f32_type();
                let i64_type = self.context.i64_type();

                // 1. Alloca for data
                let current_block = self.builder.get_insert_block().unwrap();
                let parent_fn = current_block.get_parent().unwrap();
                let data_ptr = self.create_entry_block_alloca(parent_fn, "scalar_data", &Type::F32);

                // 2. Determine value to store
                let cast_val = if let Type::I64 = ty {
                    self.builder
                        .build_signed_int_to_float(val.into_int_value(), f32_type, "cast_i64_f32")
                        .map_err(|e| e.to_string())?
                } else {
                    val.into_float_value()
                };

                // 3. Store val
                self.builder
                    .build_store(data_ptr, cast_val)
                    .map_err(|e| e.to_string())?;

                // 4. Call tl_tensor_new(data, rank=0, shape=dummy_ptr)
                let rank_val = i64_type.const_int(0, false);
                // Allocate a dummy shape (i64) to provide non-null pointer for slice::from_raw_parts
                // Use entry block alloca for shape too
                let shape_ptr =
                    self.create_entry_block_alloca(parent_fn, "scalar_shape", &Type::I64);

                let tensor_new_fn = self
                    .module
                    .get_function("tl_tensor_new")
                    .ok_or("tl_tensor_new not found")?;

                let call_site_value = self
                    .builder
                    .build_call(
                        tensor_new_fn,
                        &[data_ptr.into(), rank_val.into(), shape_ptr.into()],
                        "scalar_tensor",
                    )
                    .map_err(|e| e.to_string())?;

                let tensor_ptr = match call_site_value.try_as_basic_value() {
                    ValueKind::Basic(v) => v,
                    _ => return Err("tl_tensor_new failed".into()),
                };

                Ok((tensor_ptr, Type::Tensor(Box::new(Type::F32), 0)))
            }
            _ => Err(format!("Cannot convert {:?} to Tensor", ty)),
        }
    }
    // MatMul Optimization Helper
    pub(crate) fn try_compile_matmul(
        &mut self,
        _lhs_indices: &[String],
        _value: &Expr,
    ) -> Result<Option<inkwell::values::PointerValue<'ctx>>, String> {
        Ok(None)
    }
    #[allow(dead_code)]
    pub(crate) fn lookup_variable_ptr(
        &self,
        name: &str,
    ) -> Result<inkwell::values::PointerValue<'ctx>, String> {
        for scope in self.variables.iter().rev() {
            if let Some((val, Type::Tensor(_, _), _)) = scope.get(name) {
                // val is the alloca (pointer to pointer for Tensor)
                // We need to load the pointer
                if val.is_pointer_value() {
                    // Load the pointer from the alloca
                    let ptr_to_ptr = val.into_pointer_value();
                    let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
                    let loaded = self
                        .builder
                        .build_load(ptr_type, ptr_to_ptr, "load_tensor_ptr")
                        .map_err(|e| e.to_string())?;
                    return Ok(loaded.into_pointer_value());
                }
            }
        }
        Err(format!("Variable {} not found or not a tensor", name))
    }
}
