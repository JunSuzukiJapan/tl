use super::expr::cast_value_to_i64;
use super::CodeGenerator;
use crate::compiler::ast::*;
use inkwell::basic_block::BasicBlock;

use inkwell::values::*;
use std::collections::HashMap;

impl<'ctx> CodeGenerator<'ctx> {
    pub(crate) fn compile_tensor_equation(
        &mut self,
        name: &str,
        indices: &[String],
        clauses: &[ComprehensionClause],
        body: Option<&Expr>,
    ) -> Result<(), String> {
        let i64_type = self.context.i64_type();
        let f32_type = self.context.f32_type();

        // 0. Synthesize Implicit Body
        let synthesized_body_expr;
        let final_body = if let Some(b) = body {
            b
        } else {
            if indices.len() == 1 {
                synthesized_body_expr = Spanned::dummy(ExprKind::Variable(indices[0].clone()));
            } else {
                // Vector body: [i, j]
                synthesized_body_expr = Spanned::dummy(ExprKind::TensorLiteral(
                    indices
                        .iter()
                        .map(|idx| Spanned::dummy(ExprKind::Variable(idx.clone())))
                        .collect(),
                ));
            }
            &synthesized_body_expr
        };

        // 1. Analyze Body Shape (Simple Pre-analysis)
        // We only support inferring shape from TensorLiteral (Rank 1) or assuming Scalar (Rank 0) for now.
        // Nested TensorLiterals (Rank > 1) could be supported recursively but sticking to simple vector for now.
        let (body_dims, body_elem_count) = match &final_body.inner {
            ExprKind::TensorLiteral(elems) => (vec![elems.len()], elems.len()),
            _ => (vec![], 1), // Default to scalar
        };
        let body_rank = body_dims.len();

        // 2. Determine Bounds
        let mut index_bounds = HashMap::new();
        self.extract_index_bounds(final_body, &mut index_bounds)?;
        for clause in clauses {
            if let ComprehensionClause::Condition(cond) = clause {
                self.extract_index_bounds(cond, &mut index_bounds)?;
            }
        }

        // 3. Process Generators to get Bounds
        let mut loop_ranges = HashMap::new(); // idx -> (start, end)
        let mut bound_vars = std::collections::HashSet::new();

        for clause in clauses {
            if let ComprehensionClause::Generator { name, range } = clause {
                match &range.inner {
                    ExprKind::Range(start, end) => {
                        let (start_val, _) = self.compile_expr(start)?;
                        let (end_val, _) = self.compile_expr(end)?;
                        let start_i64 = cast_value_to_i64(self, start_val, &Type::I64)?;
                        let end_i64 = cast_value_to_i64(self, end_val, &Type::I64)?;
                        loop_ranges.insert(name.clone(), (start_i64, end_i64));
                        bound_vars.insert(name.clone());

                        let count = self
                            .builder
                            .build_int_sub(end_i64, start_i64, "count")
                            .unwrap();
                        index_bounds.insert(name.clone(), count);
                    }
                    _ => return Err("Generator range must be start..end".into()),
                }
            }
        }

        // 4. Identify Implicit Loops
        let mut loops_to_generate = Vec::new();

        // Implicit LHS Indices
        for idx in indices {
            if !bound_vars.contains(idx) {
                if !index_bounds.contains_key(idx) {
                    return Err(format!("Implicit bound not found for index {}", idx));
                }
                let limit = *index_bounds.get(idx).unwrap();
                loop_ranges.insert(idx.clone(), (i64_type.const_int(0, false), limit));
                loops_to_generate.push(idx.clone());
            }
        }

        // Implicit Reduction: Free vars in body
        // Simplified: Iterate through known bounds. If var is distinct and not bound, it's reduction.
        let mut candidates: Vec<String> = index_bounds.keys().cloned().collect();
        candidates.sort(); // Deterministic order

        for var in candidates {
            if !indices.contains(&var) && !bound_vars.contains(&var) {
                let limit = *index_bounds.get(&var).unwrap();
                loop_ranges.insert(var.clone(), (i64_type.const_int(0, false), limit));
                loops_to_generate.push(var.clone());
            }
        }

        // 5. Calculate Output Size
        let mut total_size = i64_type.const_int(body_elem_count as u64, false); // Initialize with body size
        for idx in indices {
            let (start, end) = loop_ranges
                .get(idx)
                .ok_or(format!("Missing range for output index {}", idx))?;
            let dim_size = self.builder.build_int_sub(*end, *start, "dim_sz").unwrap();
            total_size = self
                .builder
                .build_int_mul(total_size, dim_size, "sz_acc")
                .unwrap();
        }

        // Allocate Buffer
        let calloc_fn = self.module.get_function("calloc").unwrap();
        let call_result = self
            .builder
            .build_call(
                calloc_fn,
                &[total_size.into(), i64_type.const_int(4, false).into()],
                "buf_void",
            )
            .unwrap();

        let buffer_void = match call_result.try_as_basic_value() {
            inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
            _ => return Err("calloc returned void".into()),
        };

        let buffer_ptr = self
            .builder
            .build_pointer_cast(
                buffer_void,
                self.context.ptr_type(inkwell::AddressSpace::default()),
                "buf_f32",
            )
            .unwrap();

        let parent_fn = self
            .builder
            .get_insert_block()
            .unwrap()
            .get_parent()
            .unwrap();

        // 6. Generate Loops
        let mut current_bb = self.builder.get_insert_block().unwrap();
        self.enter_scope();

        let mut active_loops: Vec<(String, BasicBlock, BasicBlock, PhiValue)> = Vec::new();

        enum LoopItem<'a> {
            Clause(&'a ComprehensionClause),
            Implicit(&'a String),
        }

        let mixed_items: Vec<LoopItem> = clauses
            .iter()
            .map(LoopItem::Clause)
            .chain(loops_to_generate.iter().map(LoopItem::Implicit))
            .collect();

        let mut loop_skip_stack: Vec<Vec<BasicBlock>> = Vec::new();

        for item in mixed_items {
            match item {
                LoopItem::Clause(c) => match c {
                    ComprehensionClause::Generator { name, .. } => {
                        let (start, end) = loop_ranges.get(name).unwrap();
                        let (cond, body_bb, aft, phi, alloca) = self
                            .build_loop_start(parent_fn, current_bb, name, *start, *end)
                            .map_err(|e| e.to_string())?;
                        current_bb = body_bb;
                        active_loops.push((name.clone(), cond, aft, phi));
                        loop_skip_stack.push(Vec::new());

                        self.variables
                            .last_mut()
                            .unwrap()
                            .insert(name.clone(), (alloca.into(), Type::I64, super::CLEANUP_NONE));
                    }
                    ComprehensionClause::Condition(cond_expr) => {
                        let (true_bb, false_bb) = self
                            .build_condition(parent_fn, current_bb, cond_expr)
                            .map_err(|e| e.to_string())?;
                        current_bb = true_bb;
                        if let Some(skips) = loop_skip_stack.last_mut() {
                            skips.push(false_bb);
                        } else {
                            return Err("Condition found outside of any loop context".into());
                        }
                    }
                },
                LoopItem::Implicit(name) => {
                    let (start, end) = loop_ranges.get(name).unwrap();
                    let (cond, body_bb, aft, phi, alloca) = self
                        .build_loop_start(parent_fn, current_bb, name, *start, *end)
                        .map_err(|e| e.to_string())?;
                    current_bb = body_bb;
                    active_loops.push((name.clone(), cond, aft, phi));
                    loop_skip_stack.push(Vec::new());

                    self.variables
                        .last_mut()
                        .unwrap()
                        .insert(name.clone(), (alloca.into(), Type::I64, super::CLEANUP_NONE));
                }
            }
        }

        // Body Computation
        self.builder.position_at_end(current_bb);

        let (rhs_val, rhs_ty) = if let ExprKind::Block(_) = &final_body.inner {
            self.compile_expr(final_body)?
        } else {
            // If body is not a block, wrap it in a block to ensure scope is created
            // and intermediate tensors are freed.
            let block_wrapper = Spanned::dummy(ExprKind::Block(vec![Spanned::dummy(
                StmtKind::Expr(final_body.clone()),
            )]));
            self.compile_expr(&block_wrapper)?
        };

        // Compute element(s) to store
        // We might need to store multiple elements if body is a vector/tensor
        let elements_to_store: Vec<BasicValueEnum> = match rhs_ty {
            Type::ScalarArray(_, len) => {
                // For TensorLiteral, compile_expr returns a pointer to an array/alloca
                // We need to load elements one by one
                if len != body_elem_count {
                    return Err(format!(
                        "Body size mismatch: inferred {} but got {}",
                        body_elem_count, len
                    ));
                }
                let ptr = rhs_val.into_pointer_value();
                let mut elems = Vec::new();
                for k in 0..len {
                    let gep = unsafe {
                        self.builder.build_gep(
                            f32_type,
                            ptr,
                            &[i64_type.const_int(k as u64, false)],
                            "elem_ptr",
                        )
                    }
                    .unwrap();
                    let val = self
                        .builder
                        .build_load(f32_type, gep, "elem_val")
                        .map_err(|e| e.to_string())?;
                    elems.push(val);
                }
                elems
            }
            Type::Tensor(_inner_ty, _rank) => {
                // If the body returns a full Tensor, we use tl_tensor_get to extract scalars.
                // This handles device transfer and flattening automatically.
                let get_fn = self
                    .module
                    .get_function("tl_tensor_get")
                    .expect("tl_tensor_get not found");

                let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
                let tensor_ptr = self
                    .builder
                    .build_pointer_cast(rhs_val.into_pointer_value(), ptr_type, "t_ptr")
                    .map_err(|e| e.to_string())?;

                let mut elems = Vec::new();
                for k in 0..body_elem_count {
                    let idx_val = i64_type.const_int(k as u64, false);
                    let call_res = self
                        .builder
                        .build_call(get_fn, &[tensor_ptr.into(), idx_val.into()], "val_call")
                        .map_err(|e| e.to_string())?;
                    let val = match call_res.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err("tl_tensor_get returned void?".into()),
                    };
                    elems.push(val);
                }
                elems
            }
            Type::F32 => vec![rhs_val],
            Type::I64 | Type::I32 => {
                let i_val = cast_value_to_i64(self, rhs_val, &rhs_ty)?;
                let f_val = self
                    .builder
                    .build_signed_int_to_float(i_val, self.context.f32_type(), "f32_cast")
                    .map_err(|e| e.to_string())?;
                vec![f_val.into()]
            }
            _ => return Err(format!("Unsupported body type: {:?}", rhs_ty)),
        };

        let mut offset = i64_type.const_int(0, false);
        let mut stride = i64_type.const_int(body_elem_count as u64, false); // Base stride is body size

        for idx_name in indices.iter().rev() {
            let (start, end) = loop_ranges.get(idx_name).unwrap();
            let limit = self.builder.build_int_sub(*end, *start, "lim").unwrap();

            // Find phi for this index
            let (_, _, _, phi) = active_loops
                .iter()
                .find(|(n, _, _, _)| n == idx_name)
                .unwrap();
            let iv = phi.as_basic_value().into_int_value();
            let relative_idx = self.builder.build_int_sub(iv, *start, "rel_idx").unwrap(); // i - start

            let term = self
                .builder
                .build_int_mul(relative_idx, stride, "term")
                .unwrap();
            offset = self.builder.build_int_add(offset, term, "off").unwrap();

            stride = self
                .builder
                .build_int_mul(stride, limit, "new_str")
                .unwrap();
        }

        // Store Elements
        for (k, val) in elements_to_store.iter().enumerate() {
            let elem_offset = self
                .builder
                .build_int_add(offset, i64_type.const_int(k as u64, false), "elem_off")
                .unwrap();

            let ptr_bound = unsafe {
                self.builder
                    .build_gep(f32_type, buffer_ptr, &[elem_offset], "ptr_bound")
            }
            .unwrap();

            // Reduction Logic (if stride < total loop space? No, reduction is handled by repeatedly writing to same offset?)
            // Wait, for reduction, 'offset' should NOT include reduction variables.
            // Our offset calculation only iterates over 'indices'.
            // So reduction vars do not affect 'offset'.
            // Thus, we write to the same address multiple times.
            // We need LOAD + ADD + STORE.

            let current_val = self
                .builder
                .build_load(f32_type, ptr_bound, "curr_val")
                .map_err(|e| e.to_string())?
                .into_float_value();

            let new_val = self
                .builder
                .build_float_add(current_val, val.into_float_value(), "accum")
                .unwrap();

            self.builder.build_store(ptr_bound, new_val).unwrap();
        }

        // Loop End Logic (no changes needed)

        // Merge logic
        let mut last_processed_bb = self.builder.get_insert_block().unwrap();

        for (loop_name, cond_bb, aft_bb, phi) in active_loops.iter().rev() {
            let latch_bb = self
                .context
                .append_basic_block(parent_fn, &format!("loop_latch_{}", loop_name));

            self.builder.position_at_end(last_processed_bb);
            self.builder.build_unconditional_branch(latch_bb).unwrap();

            let skips = loop_skip_stack.pop().unwrap();
            for skip_bb in skips {
                self.builder.position_at_end(skip_bb);
                self.builder.build_unconditional_branch(latch_bb).unwrap();
            }

            self.builder.position_at_end(latch_bb);
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
            phi.add_incoming(&[(&next, latch_bb)]);
            self.builder
                .build_unconditional_branch(*cond_bb)
                .map_err(|e| e.to_string())?;

            last_processed_bb = *aft_bb;
        }
        let after_bb = last_processed_bb;
        self.builder.position_at_end(after_bb);
        self.exit_scope();

        let new_fn = self.module.get_function("tl_tensor_new").unwrap();
        let loop_rank = indices.len();
        let rank = loop_rank + body_rank;
        let shape_alloca = self
            .builder
            .build_alloca(i64_type.array_type(rank as u32), "shape")
            .map_err(|e| e.to_string())?;

        // Fill Loop Dimensions
        for (i, idx) in indices.iter().enumerate() {
            let (start, end) = loop_ranges.get(idx).unwrap();
            let dim_size = self.builder.build_int_sub(*end, *start, "dim").unwrap();

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
                .build_store(elem_ptr, dim_size)
                .map_err(|e| e.to_string())?;
        }

        // Fill Body Dimensions
        for (i, dim) in body_dims.iter().enumerate() {
            let elem_ptr = unsafe {
                self.builder
                    .build_gep(
                        i64_type.array_type(rank as u32),
                        shape_alloca,
                        &[
                            i64_type.const_int(0, false),
                            i64_type.const_int((loop_rank + i) as u64, false),
                        ],
                        "body_shape_ptr",
                    )
                    .map_err(|e| e.to_string())?
            };
            self.builder
                .build_store(elem_ptr, i64_type.const_int(*dim as u64, false))
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
        let tptr = self.check_tensor_result(call_result, "tl_tensor_new")?;

        let v_alloca = self.create_entry_block_alloca(
            parent_fn,
            name,
            &Type::Tensor(Box::new(Type::F32), rank),
        )?;
        self.builder
            .build_store(v_alloca, tptr)
            .map_err(|e| e.to_string())?;
        self.variables.last_mut().unwrap().insert(
            name.to_string(),
            (
                v_alloca.into(),
                Type::Tensor(Box::new(Type::F32), rank),
                super::CLEANUP_FULL,
            ),
        );

        Ok(())
    }

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
                        let current_block = self.builder.get_insert_block().unwrap();
                        let parent_fn = current_block.get_parent().unwrap();
                        let f32_array_type = f32_type.array_type(len as u32);
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

                        for i in 0..len {
                            let idx = i64_type.const_int(i as u64, false);
                            let src_ptr = unsafe {
                                self.builder
                                    .build_in_bounds_gep(
                                        i64_type,
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

                let res_val = self.check_tensor_result(call, &func_name)?;
                Ok((res_val, tensor_type))
            }
            Type::F32 | Type::I64 => {
                let f32_type = self.context.f32_type();
                let current_block = self.builder.get_insert_block().unwrap();
                let parent_fn = current_block.get_parent().unwrap();
                let data_ptr = self.create_entry_block_alloca(parent_fn, "scalar_data", &Type::F32)?;

                let cast_val = if let Type::I64 = ty {
                    self.builder
                        .build_signed_int_to_float(val.into_int_value(), f32_type, "cast_i64_f32")
                        .map_err(|e| e.to_string())?
                } else {
                    val.into_float_value()
                };

                self.builder
                    .build_store(data_ptr, cast_val)
                    .map_err(|e| e.to_string())?;

                let rank_val = self.context.i64_type().const_int(0, false);
                let shape_ptr =
                    self.create_entry_block_alloca(parent_fn, "scalar_shape", &Type::I64)?;

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

                let tensor_ptr = self.check_tensor_result(call_site_value, "tl_tensor_new")?;

                Ok((tensor_ptr, Type::Tensor(Box::new(Type::F32), 0)))
            }
            _ => Err(format!("Cannot convert {:?} to Tensor", ty)),
        }
    }

    #[allow(dead_code)]
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
                if val.is_pointer_value() {
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

    fn build_loop_start(
        &self,
        parent_fn: FunctionValue<'ctx>,
        current_bb: BasicBlock<'ctx>,
        name: &str,
        start: IntValue<'ctx>,
        end: IntValue<'ctx>,
    ) -> Result<
        (
            BasicBlock<'ctx>,
            BasicBlock<'ctx>,
            BasicBlock<'ctx>,
            PhiValue<'ctx>,
            PointerValue<'ctx>,
        ),
        String,
    > {
        let i64_type = self.context.i64_type();

        let cond_bb = self
            .context
            .append_basic_block(parent_fn, &format!("loop_cond_{}", name));
        let body_bb = self
            .context
            .append_basic_block(parent_fn, &format!("loop_body_{}", name));
        let aft_bb = self
            .context
            .append_basic_block(parent_fn, &format!("loop_aft_{}", name));

        self.builder.position_at_end(current_bb);
        self.builder.build_unconditional_branch(cond_bb).unwrap();

        self.builder.position_at_end(cond_bb);
        let phi = self.builder.build_phi(i64_type, name).unwrap();
        phi.add_incoming(&[(&start, current_bb)]);

        let iv = phi.as_basic_value().into_int_value();
        let cmp = self
            .builder
            .build_int_compare(inkwell::IntPredicate::SLT, iv, end, "cmp")
            .unwrap();
        self.builder
            .build_conditional_branch(cmp, body_bb, aft_bb)
            .unwrap();

        self.builder.position_at_end(body_bb);
        let alloca = self.create_entry_block_alloca(parent_fn, name, &Type::I64)?;
        self.builder.build_store(alloca, iv).unwrap();

        Ok((cond_bb, body_bb, aft_bb, phi, alloca))
    }

    fn build_condition(
        &mut self,
        parent_fn: FunctionValue<'ctx>,
        current_bb: BasicBlock<'ctx>,
        cond_expr: &Expr,
    ) -> Result<(BasicBlock<'ctx>, BasicBlock<'ctx>), String> {
        let check_bb = self.context.append_basic_block(parent_fn, "check_cond");
        let true_bb = self.context.append_basic_block(parent_fn, "cond_true");
        let false_bb = self.context.append_basic_block(parent_fn, "cond_false");

        self.builder.position_at_end(current_bb);
        self.builder.build_unconditional_branch(check_bb).unwrap();
        self.builder.position_at_end(check_bb);

        let (cond_val, _) = self.compile_expr(cond_expr)?;
        let cond_bool = cond_val.into_int_value();

        self.builder
            .build_conditional_branch(cond_bool, true_bb, false_bb)
            .unwrap();

        Ok((true_bb, false_bb))
    }
}
