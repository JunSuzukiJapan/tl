use crate::compiler::error::{TlError, CodegenErrorKind};
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
        reduction_indices: &[String],
        clauses: &[ComprehensionClause],
        body: Option<&Expr>,
    ) -> Result<(), TlError> {
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
                    ExprKind::Range(Some(start), Some(end)) => {
                        let (start_val, _) = self.compile_expr(start)?;
                        let (end_val, _) = self.compile_expr(end)?;
                        let start_i64 = cast_value_to_i64(self, start_val, &Type::I64)?;
                        let end_i64 = cast_value_to_i64(self, end_val, &Type::I64)?;
                        loop_ranges.insert(name.clone(), (start_i64, end_i64));
                        bound_vars.insert(name.clone());

                        let count = self
                            .builder
                            .build_int_sub(end_i64, start_i64, "count")
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        index_bounds.insert(name.clone(), count);
                    }
                    ExprKind::Range(_, _) => return Err(CodegenErrorKind::Internal("Generator range must be a closed range (start..end)".to_string()).into()),
                    _ => return Err(CodegenErrorKind::Internal("Generator range must be start..end".to_string()).into()),
                }
            }
        }

        // 4. Identify Implicit Loops
        let mut output_loops_to_generate = Vec::new();

        // Implicit LHS Indices (Output Dimensions)
        for idx in indices {
            if !bound_vars.contains(idx) {
                if !index_bounds.contains_key(idx) {
                    return Err(TlError::from(CodegenErrorKind::Internal(format!("Implicit bound not found for index {}", idx))));
                }
                let limit = *index_bounds.get(idx).expect("index_bounds key was just verified");
                loop_ranges.insert(idx.clone(), (i64_type.const_int(0, false), limit));
                output_loops_to_generate.push(idx.clone());
            }
        }

        // Implicit Reduction Indices
        let mut reduction_loops_to_generate = Vec::new();
        for idx in reduction_indices {
             if !bound_vars.contains(idx) {
                if !index_bounds.contains_key(idx) {
                    // Try to infer from explicit intersection analysis from stmt.rs
                    // If not found in index_bounds (which comes from body traversal), it's an error.
                     return Err(TlError::from(CodegenErrorKind::Internal(format!("Implicit bound not found for reduction index {}", idx))));
                }
                let limit = *index_bounds.get(idx).expect("index_bounds key was just verified");
                loop_ranges.insert(idx.clone(), (i64_type.const_int(0, false), limit));
                reduction_loops_to_generate.push(idx.clone());
            }
        }

        // 5. Calculate Output Size
        let mut total_size = i64_type.const_int(body_elem_count as u64, false); // Initialize with body size
        for idx in indices {
            let (start, end) = loop_ranges
                .get(idx)
                .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Missing range for output index {}", idx))))?;
            let dim_size = self.builder.build_int_sub(*end, *start, "dim_sz").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            total_size = self
                .builder
                .build_int_mul(total_size, dim_size, "sz_acc")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        }

        // Allocate Buffer
        let calloc_fn = self.get_fn("calloc")?;
        let call_result = self
            .builder
            .build_call(
                calloc_fn,
                &[total_size.into(), i64_type.const_int(4, false).into()],
                "buf_void",
            )
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        let buffer_void = match call_result.try_as_basic_value() {
            inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
            _ => return Err(CodegenErrorKind::Internal("calloc returned void".to_string()).into()),
        };

        let buffer_ptr = self
            .builder
            .build_pointer_cast(
                buffer_void,
                self.context.ptr_type(inkwell::AddressSpace::default()),
                "buf_f32",
            )
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        let parent_fn = self.current_function()?;

        // 6. Generate Output Loops
        let mut current_bb = self.current_block()?;
        self.enter_scope();

        let mut active_loops: Vec<(String, BasicBlock, BasicBlock, PhiValue)> = Vec::new();

        enum LoopItem<'a> {
            Clause(&'a ComprehensionClause),
            Implicit(&'a String),
        }

        // Mix Output loops and Generators. Generators usually bind vars used in body.
        // Assuming Generators are for Output logic if they are in 'indices'?
        // Actually, 'clauses' are explicit generators. 'indices' are LHS.
        // If a generator variable is in 'indices', it's an output loop.
        // If NOT in 'indices', it is a reduction loop (explicit reduction).
        // For now, let's treat explicit clauses as occurring before implicit output loops?
        // Or should we respect the order?
        // Simple strategy:
        // 1. Output Loops (Implicit LHS)
        // 2. Reduction Loops (Implicit + Explicit Generators not in LHS)

        // Correction: 'indices' defines the output nesting order.
        // We should generate loops for 'indices' in order.
        // If an index is bound by a Generator, use that generator's range.
        // If not, use implicit range.

        let mut loop_skip_stack: Vec<Vec<BasicBlock>> = Vec::new();

        // 6a. Generate Output Loops (Iterate over LHS indices)
        for idx in indices {
             // Check if explicit generator exists for this index
             if let Some(clause) = clauses.iter().find(|c| matches!(c, ComprehensionClause::Generator{name, ..} if name == idx)) {
                 if let ComprehensionClause::Generator { name, range: _ } = clause {
                        let (start, end) = loop_ranges.get(name).expect("loop_ranges key was just inserted for generator");
                        let (cond, body_bb, aft, phi, alloca) = self
                            .build_loop_start(parent_fn, current_bb, name, *start, *end)
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        current_bb = body_bb;
                        active_loops.push((name.clone(), cond, aft, phi));
                        loop_skip_stack.push(Vec::new());

                        self.variables
                            .last_mut()
                            .expect("variables scope always non-empty")
                            .insert(name.clone(), (alloca.into(), Type::I64, super::CLEANUP_NONE));
                 }
             } else {
                 // Implicit Output Loop
                 let (start, end) = loop_ranges.get(idx).expect("loop_ranges key was just inserted for implicit output loop");
                 let (cond, body_bb, aft, phi, alloca) = self
                        .build_loop_start(parent_fn, current_bb, idx, *start, *end)
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                 current_bb = body_bb;
                 active_loops.push((idx.clone(), cond, aft, phi));
                 loop_skip_stack.push(Vec::new());

                 self.variables
                        .last_mut()
                        .expect("variables scope always non-empty")
                        .insert(idx.clone(), (alloca.into(), Type::I64, super::CLEANUP_NONE));
             }
        }

        // Pre-Calculation of Output Offset (Optimization: calculate once inside output loops)
        // Used to determine where to store/accumulate result.
        self.builder.position_at_end(current_bb);
        let mut offset = i64_type.const_int(0, false);
        let mut stride = i64_type.const_int(body_elem_count as u64, false); // Base stride is body size

        for idx_name in indices.iter().rev() {
            let (start, end) = loop_ranges.get(idx_name).expect("loop_ranges key exists for output index");
            let limit = self.builder.build_int_sub(*end, *start, "lim").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            // Find phi for this index
            let (_, _, _, phi) = active_loops
                .iter()
                .find(|(n, _, _, _)| n == idx_name)
                .expect("active_loops entry exists for each output index");
            let iv = phi.as_basic_value().into_int_value();
            let relative_idx = self.builder.build_int_sub(iv, *start, "rel_idx").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?; // i - start

            let term = self
                .builder
                .build_int_mul(relative_idx, stride, "term")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            offset = self.builder.build_int_add(offset, term, "off").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            stride = self
                .builder
                .build_int_mul(stride, limit, "new_str")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        }

        // Initialize Accumulator (if reduction exists)
        // For scalar body: 0.0
        // For vector body: [0.0, ...]
        // We initialize memory at 'offset' to 0.0 directly in the buffer.
        // This avoids needing complex phi nodes for vector accumulators.
        let has_reduction = !reduction_indices.is_empty() || clauses.iter().any(|c| matches!(c, ComprehensionClause::Generator{name, ..} if !indices.contains(name)));

        if has_reduction {
            for k in 0..body_elem_count {
                 let elem_offset = self
                    .builder
                    .build_int_add(offset, i64_type.const_int(k as u64, false), "elem_off")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let ptr_bound = unsafe {
                    self.builder
                        .build_gep(f32_type, buffer_ptr, &[elem_offset], "ptr_init")
                }
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(ptr_bound, f32_type.const_float(0.0)).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            }
        }

        // 6b. Generate Reduction Loops
        // Explicit Generators NOT in LHS
        let explicit_reductions: Vec<&ComprehensionClause> = clauses.iter().filter(|c| {
            matches!(c, ComprehensionClause::Generator{name, ..} if !indices.contains(name))
        }).collect();

        let reduction_items: Vec<LoopItem> = explicit_reductions
            .iter()
            .map(|c| LoopItem::Clause(c))
            .chain(reduction_loops_to_generate.iter().map(LoopItem::Implicit))
            .collect();

        for item in reduction_items {
            match item {
                LoopItem::Clause(c) => {
                     if let ComprehensionClause::Generator { name, range: _ } = c {
                        let (start, end) = loop_ranges.get(name).expect("loop_ranges key was just inserted for explicit reduction generator");
                        let (cond, body_bb, aft, phi, alloca) = self
                            .build_loop_start(parent_fn, current_bb, name, *start, *end)
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        current_bb = body_bb;
                        active_loops.push((name.clone(), cond, aft, phi));
                        loop_skip_stack.push(Vec::new());

                         self.variables
                            .last_mut()
                            .expect("variables scope always non-empty")
                            .insert(name.clone(), (alloca.into(), Type::I64, super::CLEANUP_NONE));
                     }
                },
                LoopItem::Implicit(name) => {
                    let (start, end) = loop_ranges.get(name).expect("loop_ranges key was just inserted for implicit reduction loop");
                    let (cond, body_bb, aft, phi, alloca) = self
                        .build_loop_start(parent_fn, current_bb, name, *start, *end)
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    current_bb = body_bb;
                    active_loops.push((name.clone(), cond, aft, phi));
                    loop_skip_stack.push(Vec::new());

                    self.variables
                        .last_mut()
                        .expect("variables scope always non-empty")
                        .insert(name.clone(), (alloca.into(), Type::I64, super::CLEANUP_NONE));
                }
            }
        }

        // Body Computation
        self.builder.position_at_end(current_bb);

        let conditions: Vec<&Expr> = clauses.iter().filter_map(|c| {
            if let ComprehensionClause::Condition(cond) = c { Some(cond) } else { None }
        }).collect();

        if !conditions.is_empty() {
            let mut final_cond = None;
            for cond in conditions {
                let (c_val, c_ty) = self.compile_expr(cond)?;
                let c_bool = if let Type::Bool = c_ty {
                    c_val.into_int_value()
                } else {
                    return Err(TlError::from(CodegenErrorKind::Internal(format!("Condition must be bool, found {:?}", c_ty))));
                };
                final_cond = match final_cond {
                    None => Some(c_bool),
                    Some(prev) => Some(self.builder.build_and(prev, c_bool, "cond_and").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?),
                };
            }
            if let Some(cond_val) = final_cond {
                let skip_bb = self.context.append_basic_block(parent_fn, "cond_fail_skip");
                let body_eval_bb = self.context.append_basic_block(parent_fn, "body_eval");
                self.builder.build_conditional_branch(cond_val, body_eval_bb, skip_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // Add skip_bb to the INNERMOST loop's skip stack
                if let Some(skips) = loop_skip_stack.last_mut() {
                    skips.push(skip_bb);
                } else {
                    // No loops active, but we have a condition.
                    // If condition fails, we should jump to end.
                    // But for scalar comprehension, skip_stack is empty.
                    // Let's just create a dummy end block if needed, but it's an edge case.
                    return Err(TlError::from(CodegenErrorKind::Internal("Condition outside of any loop generator in tensor comprehension".to_string())));
                }

                self.builder.position_at_end(body_eval_bb);
            }
        }

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
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                let mut elems = Vec::new();
                for k in 0..body_elem_count {
                    let idx_val = i64_type.const_int(k as u64, false);
                    let call_res = self
                        .builder
                        .build_call(get_fn, &[tensor_ptr.into(), idx_val.into()], "val_call")
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    let val = match call_res.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err(CodegenErrorKind::Internal("tl_tensor_get returned void?".to_string()).into()),
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
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                vec![f_val.into()]
            }
            _ => return Err(TlError::from(CodegenErrorKind::UnsupportedOperation(format!("Unsupported body type: {:?}", rhs_ty)))),
        };

        // Store Elements (Accumulate or Overwrite)
        for (k, val) in elements_to_store.iter().enumerate() {
            let elem_offset = self
                .builder
                .build_int_add(offset, i64_type.const_int(k as u64, false), "elem_off")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            let ptr_bound = unsafe {
                self.builder
                    .build_gep(f32_type, buffer_ptr, &[elem_offset], "ptr_bound")
            }
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            let val_to_store = if has_reduction {
                // Load Accumulator
                let current_acc = self
                    .builder
                    .build_load(f32_type, ptr_bound, "acc_load")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                    .into_float_value();

                // Add
                self.builder
                    .build_float_add(current_acc, val.into_float_value(), "acc_add")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
            } else {
                // Direct Store (Overwrite)
                val.into_float_value()
            };

            self.builder.build_store(ptr_bound, val_to_store).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        }

        // Loop End Logic (no changes needed)

        // Merge logic
        let mut last_processed_bb = self.current_block()?;

        for (loop_name, cond_bb, aft_bb, phi) in active_loops.iter().rev() {
            let latch_bb = self
                .context
                .append_basic_block(parent_fn, &format!("loop_latch_{}", loop_name));

            self.builder.position_at_end(last_processed_bb);
            self.builder.build_unconditional_branch(latch_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            let skips = loop_skip_stack.pop().expect("stack always has matching push");
            for skip_bb in skips {
                self.builder.position_at_end(skip_bb);
                self.builder.build_unconditional_branch(latch_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            }

            self.builder.position_at_end(latch_bb);
            let (alloca_val, _) = self.lookup_variable(loop_name).expect("loop variable exists in scope");
            let iv = self
                .builder
                .build_load(i64_type, alloca_val.into_pointer_value(), "iv")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                .into_int_value();
            let next = self
                .builder
                .build_int_add(iv, i64_type.const_int(1, false), "next")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            phi.add_incoming(&[(&next, latch_bb)]);
            self.builder
                .build_unconditional_branch(*cond_bb)
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            last_processed_bb = *aft_bb;
        }
        let after_bb = last_processed_bb;
        self.builder.position_at_end(after_bb);
        self.exit_scope();

        let new_fn = self.get_fn("tl_tensor_new")?;
        let loop_rank = indices.len();
        let rank = loop_rank + body_rank;
        let shape_alloca = self
            .builder
            .build_alloca(i64_type.array_type(rank as u32), "shape")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        // Fill Loop Dimensions
        for (i, idx) in indices.iter().enumerate() {
            let (start, end) = loop_ranges.get(idx).expect("loop_ranges key exists for output index");
            let dim_size = self.builder.build_int_sub(*end, *start, "dim").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

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
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
            };
            self.builder
                .build_store(elem_ptr, dim_size)
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
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
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
            };
            self.builder
                .build_store(elem_ptr, i64_type.const_int(*dim as u64, false))
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        }

        let shape_ptr = self
            .builder
            .build_pointer_cast(
                shape_alloca,
                self.context.ptr_type(inkwell::AddressSpace::default()),
                "sptr",
            )
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
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
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        let tptr = self.check_tensor_result(call_result, "tl_tensor_new")?;

        let v_alloca = self.create_entry_block_alloca(
            parent_fn,
            name,
            &Type::Tensor(Box::new(Type::F32), rank),
        )?;
        self.builder
            .build_store(v_alloca, tptr)
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        self.variables.last_mut().expect("variables scope always non-empty").insert(
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
    ) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
        let (val, ty) = self.compile_expr(expr)?;

        match ty {
            Type::Tensor(_, _) => Ok((val, ty)),

            Type::F32 | Type::I64 => {
                let f32_type = self.context.f32_type();
                let parent_fn = self.current_function()?;
                let data_ptr = self.create_entry_block_alloca(parent_fn, "scalar_data", &Type::F32)?;

                let cast_val = if let Type::I64 = ty {
                    self.builder
                        .build_signed_int_to_float(val.into_int_value(), f32_type, "cast_i64_f32")
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                } else {
                    val.into_float_value()
                };

                self.builder
                    .build_store(data_ptr, cast_val)
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                let rank_val = self.context.i64_type().const_int(0, false);
                let shape_ptr =
                    self.create_entry_block_alloca(parent_fn, "scalar_shape", &Type::I64)?;

                let tensor_new_fn = self
                    .module
                    .get_function("tl_tensor_new")
                    .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_tensor_new not found".to_string())))?;

                let call_site_value = self
                    .builder
                    .build_call(
                        tensor_new_fn,
                        &[data_ptr.into(), rank_val.into(), shape_ptr.into()],
                        "scalar_tensor",
                    )
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                let tensor_ptr = self.check_tensor_result(call_site_value, "tl_tensor_new")?;

                Ok((tensor_ptr, Type::Tensor(Box::new(Type::F32), 0)))
            }
            _ => Err(TlError::from(CodegenErrorKind::Internal(format!("Cannot convert {:?} to Tensor", ty)))),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn try_compile_matmul(
        &mut self,
        _lhs_indices: &[String],
        _value: &Expr,
    ) -> Result<Option<inkwell::values::PointerValue<'ctx>>, TlError> {
        Ok(None)
    }

    #[allow(dead_code)]
    pub(crate) fn lookup_variable_ptr(
        &self,
        name: &str,
    ) -> Result<inkwell::values::PointerValue<'ctx>, TlError> {
        for scope in self.variables.iter().rev() {
            if let Some((val, Type::Tensor(_, _), _)) = scope.get(name) {
                if val.is_pointer_value() {
                    let ptr_to_ptr = val.into_pointer_value();
                    let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
                    let loaded = self
                        .builder
                        .build_load(ptr_type, ptr_to_ptr, "load_tensor_ptr")
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    return Ok(loaded.into_pointer_value());
                }
            }
        }
        Err(TlError::from(CodegenErrorKind::VariableNotFound(format!("Variable {} not found or not a tensor", name))))
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
        TlError,
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
        self.builder.build_unconditional_branch(cond_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        self.builder.position_at_end(cond_bb);
        let phi = self.builder.build_phi(i64_type, name).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        phi.add_incoming(&[(&start, current_bb)]);

        let iv = phi.as_basic_value().into_int_value();
        let cmp = self
            .builder
            .build_int_compare(inkwell::IntPredicate::SLT, iv, end, "cmp")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        self.builder
            .build_conditional_branch(cmp, body_bb, aft_bb)
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        self.builder.position_at_end(body_bb);
        let alloca = self.create_entry_block_alloca(parent_fn, name, &Type::I64)?;
        self.builder.build_store(alloca, iv).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        Ok((cond_bb, body_bb, aft_bb, phi, alloca))
    }


}
