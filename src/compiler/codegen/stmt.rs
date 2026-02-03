use super::CodeGenerator;
use crate::compiler::ast::*;

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



    /// Copy struct contents from src to dst pointer (used for sret)
    pub(crate) fn emit_struct_copy(
        &self,
        dst: inkwell::values::PointerValue<'ctx>,
        src: inkwell::values::PointerValue<'ctx>,
        ty: &Type,
    ) -> Result<(), String> {
        match ty {
            Type::Struct(name, generics) => {
                let mangled_name = if generics.is_empty() {
                    name.clone()
                } else {
                    self.mangle_type_name(name, generics)
                };

                let struct_def = self
                    .struct_defs
                    .get(&mangled_name)
                    .ok_or(format!("Struct {} not found ({})", name, mangled_name))?
                    .clone();
                let st_llvm_ty = *self
                    .struct_types
                    .get(&mangled_name)
                    .ok_or(format!("LLVM struct type {} not found", mangled_name))?;

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
                        Type::Tensor(_, _) => self
                            .context
                            .ptr_type(inkwell::AddressSpace::default())
                            .into(),
                        Type::Struct(name, _) => {
                             // Check for ZST
                            let simple_name = if name.contains("::") {
                                name.split("::").last().unwrap()
                            } else {
                                name.as_str()
                            };

                            let _is_zst = if let Some(def) = self.struct_defs.get(simple_name) {
                                def.fields.is_empty()
                            } else {
                                false
                            };

                            // ZST Strategy V3.2: Even ZSTs are treated as Pointers (NULL).
                            self.context.ptr_type(inkwell::AddressSpace::default()).into()
                        }
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
                            | Type::Enum(_, _)
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
        &mut self,
        val: BasicValueEnum<'ctx>,
        ty: &Type,
        mode: u8,
    ) -> Result<(), String> {
        if mode == super::CLEANUP_NONE {
             return Ok(());
        }

        if std::env::var("TL_DEBUG_FREE").is_ok() {
             println!("[DEBUG_FREE] emit_recursive_free: {:?}", ty);
        }
        match ty {
            Type::Ref(_) => {},
            Type::Enum(name, generics) => {
                let mangled_name = if generics.is_empty() {
                    name.clone()
                } else {
                    self.mangle_type_name(name, generics)
                };

                let enum_def = self
                    .enum_defs
                    .get(&mangled_name)
                    .ok_or(format!("Enum def {} not found ({})", name, mangled_name))?
                    .clone();
                let enum_ty = *self
                    .enum_types
                    .get(&mangled_name)
                    .ok_or(format!("Enum type {} not found ({})", name, mangled_name))?;

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

                    let field_types_list = match &variant.kind {
                        crate::compiler::ast::VariantKind::Unit => vec![],
                        crate::compiler::ast::VariantKind::Tuple(types) => types.clone(),
                        crate::compiler::ast::VariantKind::Struct(fields) => fields.iter().map(|(_, t)| t.clone()).collect(),
                    };

                    if !field_types_list.is_empty() {
                        // Cast Payload (Element 1 is [i8 x N])
                        let payload_ptr_raw = self
                            .builder
                            .build_struct_gep(enum_ty, ptr, 1, "payload_ptr_raw")
                            .map_err(|e| e.to_string())?;

                        // Reconstruct Variant Struct Type for GEP
                        let mut field_types: Vec<inkwell::types::BasicTypeEnum> = vec![];
                        for ty in &field_types_list {
                            let llvm_ty = match ty {
                                Type::F32 => self.context.f32_type().into(),
                                Type::I64 => self.context.i64_type().into(),
                                Type::Bool => self.context.bool_type().into(),
                                Type::Tensor(_, _) => self
                                    .context
                                    .ptr_type(inkwell::AddressSpace::default())
                                    .into(),
                                Type::Struct(_, _) | Type::Enum(_, _) => self
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

                        for (idx, f_ty) in field_types_list.iter().enumerate() {
                            if matches!(
                                f_ty,
                                Type::Tensor(_, _)
                                    | Type::TensorShaped(_, _)
                                    | Type::Struct(_, _)
                                    | Type::Enum(_, _)
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

                                // Recursive calls use DEFAULT (FULL) cleanup for contents
                                self.emit_recursive_free(f_val, f_ty, super::CLEANUP_FULL)?;
                            }
                        }
                    }
                    // After recursive calls, branch from current position to after_switch
                    if self.builder.get_insert_block().unwrap().get_terminator().is_none() {
                         self.builder.build_unconditional_branch(after_switch).unwrap();
                    }
                }

                self.builder.position_at_end(after_switch);
                if self.builder.get_insert_block().unwrap().get_terminator().is_none() {
                     self.builder.build_unconditional_branch(merge_block).unwrap();
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

                if mode == super::CLEANUP_FINALIZE {
                    // Call tl_tensor_finalize (Drop content, Keep struct)
                    let finalize_fn = self
                        .module
                        .get_function("tl_tensor_finalize")
                        .or_else(|| {
                             // Declare if missing
                             let void_ty = self.context.void_type();
                             let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
                             let ft = void_ty.fn_type(&[ptr_ty.into()], false);
                             Some(self.module.add_function("tl_tensor_finalize", ft, None))
                        })
                        .ok_or("tl_tensor_finalize not found")?;
                    
                    self.builder.build_call(finalize_fn, &[val.into()], "").map_err(|e| e.to_string())?;
                } else {
                    // Call tl_tensor_release (Standard RefCount check + Free)
                    let free_fn = self
                        .module
                        .get_function("tl_tensor_release")
                        .ok_or("tl_tensor_release not found")?;

                    // LOG FREE
                    self.emit_log_free(val)?;

                    self.builder
                        .build_call(free_fn, &[val.into()], "")
                        .map_err(|e| e.to_string())?;
                }

                self.builder
                    .build_unconditional_branch(merge_block)
                    .map_err(|e| e.to_string())?;
                self.builder.position_at_end(merge_block);
            }
            Type::Struct(name, generic_args) => {
                // Check if it's a pointer before treating it as a heap-allocated struct
                // Primitives like u8 might be represented as Struct("u8") but are IntValue
                if !val.is_pointer_value() {
                    return Ok(());
                }

                
                // Define Basic Blocks Early
                let current_block = self.builder.get_insert_block().unwrap();
                let func = current_block.get_parent().unwrap();
                let merge_block = self.context.append_basic_block(func, "after_free");

                // Unregister from Runtime Scope (Safety against double free)
                let ptr = val.into_pointer_value();
                if let Some(unreg_fn) = self.module.get_function("tl_mem_unregister") {
                     if std::env::var("TL_DEBUG_FREE").is_ok() {
                          // Call debug printf or just use rust println for invalid ptr?
                          // Rust println runs at compile time logic.
                          // But we want runtime pointer value.
                          // We can't print runtime pointer easily from JIT without building a format string.
                          // But we CAN print the TYPE at JIT time.
                     }
                     let void_ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
                     let cast = self.builder.build_pointer_cast(ptr, void_ptr_ty, "unreg_cast").unwrap();
                     self.builder.build_call(unreg_fn, &[cast.into()], "").ok(); 
                }

                // 1. Stack Cleanup Mode (Jump directly to structural cleanup)
                if mode == super::CLEANUP_STACK {
                     // We need struct def for this
                     // Let's defer this check until we have struct def? 
                     // Or just handle it separately.
                     // The original code handled it after null check.
                }

                // 2. RefCount Check (DecRef) BEFORE calling destructor
                // Call tl_ptr_dec_ref(ptr) -> i32 (1=True/Free, 0=False/Keep)
                let dec_ref_fn = self
                    .module
                    .get_function("tl_ptr_dec_ref")
                    .ok_or("tl_ptr_dec_ref not found")?;
                    
                let cast_void = self.builder.build_pointer_cast(
                    ptr, 
                    self.context.ptr_type(inkwell::AddressSpace::default()), 
                    "cast_void"
                ).unwrap();

                let call = self
                    .builder
                    .build_call(dec_ref_fn, &[cast_void.into()], "should_free")
                    .map_err(|e| e.to_string())?;

                let should_free_val = match call.try_as_basic_value() {
                    inkwell::values::ValueKind::Basic(v) => v.into_int_value(),
                    _ => return Err("tl_ptr_dec_ref returned void/invalid".to_string()),
                };
                    
                let should_free = self.builder.build_int_compare(
                    inkwell::IntPredicate::NE,
                    should_free_val,
                    self.context.i32_type().const_int(0, false),
                    "should_free_bool"
                ).map_err(|e| e.to_string())?;
                
                let recurse_block = self.context.append_basic_block(func, "recurse_free");
                self.builder.build_conditional_branch(should_free, recurse_block, merge_block)
                    .map_err(|e| e.to_string())?;

                // --- Recurse Block (Destruction) ---
                self.builder.position_at_end(recurse_block);

                // 3. Generic Destructor Lookup (Static Method "free")
                // We MUST trigger monomorphization if it's a generic struct to ensure the function exists.
                
                let simple_name = if name.contains("::") {
                    name.split("::").last().unwrap()
                } else {
                    name.as_str()
                };

                // Try monomorphization (instantiates if needed and returns mangled name)
                let runtime_name_res = if !generic_args.is_empty() {
                     self.monomorphize_method(simple_name, "free", generic_args)
                } else {
                     self.monomorphize_method(simple_name, "free", generic_args)
                };

                // Fallback removed.
                // **静的型付け言語において、フォールバックは百害あって一利なしなので、絶対にフォールバックはいれないで。**

                if let Ok(runtime_name) = runtime_name_res {
                    if let Some(fn_val) = self.module.get_function(&runtime_name) {
                         // Call it: fn free(self)
                         self.builder.build_call(fn_val, &[val.into()], "").map_err(|e| e.to_string())?;
                         self.builder.build_unconditional_branch(merge_block).unwrap();
                         self.builder.position_at_end(merge_block);
                         return Ok(());
                    }
                } else {
                    // Legacy / Non-Generic resolver fallback (for runtime builtins not in generic_impls)
                    let legacy_name = crate::compiler::codegen::builtin_types::resolver::resolve_static_method_name(
                         name, "free", generic_args
                    );
                     if let Some(fn_val) = self.module.get_function(&legacy_name) {
                         self.builder.build_call(fn_val, &[val.into()], "").map_err(|e| e.to_string())?;
                         self.builder.build_unconditional_branch(merge_block).unwrap();
                         self.builder.position_at_end(merge_block);
                         return Ok(());
                     }
                 }

                // 4. Specialized Tensor Handling (Legacy but maintained for now)
                if name == "Tensor" {
                    // Note: Recurse free for tensor internal buffer? `emit_recursive_free` for tensor handles release.
                    // But we are ALREADY releasing the wrapper struct here.
                    // Tensor contents are managed by `tl_tensor_release`.
                    // If we just free wrapper, we LEAK tensor content if wrapper owned it!
                    // Wait, `Tensor` struct IS the handle.
                    // If we are freeing a `Tensor` STRUCT (wrapper), we might need to release the internal one?
                    // But `Tensor` is Opaque Pointer mostly.
                    return self.emit_recursive_free(val, &Type::Tensor(Box::new(Type::F32), 1), mode);
                }

                // 3. Fallback: Structural Cleanup (Member-wise)
                // Use mangled name if generic
                let mangled_name = if generic_args.is_empty() {
                    let simple_name = if name.contains("::") {
                         name.split("::").last().unwrap()
                    } else {
                         name.as_str()
                    };
                    simple_name.to_string()
                } else {
                    self.mangle_type_name(name, generic_args)
                };

                // Check if struct def exists
                if !self.struct_defs.contains_key(&mangled_name) {
                     return Ok(());
                }

                let struct_def = self
                    .struct_defs
                    .get(&mangled_name)
                    .ok_or(format!("Struct def {} not found", mangled_name))?
                    .clone();
                let struct_ty = *self
                    .struct_types
                    .get(&mangled_name)
                    .ok_or(format!("Struct type {} not found", mangled_name))?;
                let ptr = val.into_pointer_value();

                // Runtime Null Check
                let free_block = self.context.append_basic_block(func, "free_struct");
                // merge_block already defined
                
                let is_null = self
                    .builder
                    .build_is_null(ptr, "is_null")
                    .map_err(|e| e.to_string())?;
                self.builder
                    .build_conditional_branch(is_null, merge_block, free_block)
                    .map_err(|e| e.to_string())?;

                // Stack Cleanup Mode
                if mode == super::CLEANUP_STACK {
                     self.builder.position_at_end(free_block);

                     for (i, (_name, f_ty)) in struct_def.fields.iter().enumerate() {
                         // Check ZST optimization compatibility
                         if let Type::Struct(s_name, _) = f_ty {
                              // If field struct is ZST, we didn't store a pointer, so don't try to load/free it.
                              let simple_s_name = if s_name.contains("::") { s_name.split("::").last().unwrap() } else { s_name.as_str() };
                              if let Some(def) = self.struct_defs.get(simple_s_name) {
                                   if def.fields.is_empty() {
                                        continue;
                                   }
                              }
                         }

                          }
                     }

                     match f_ty {
                            Type::Tensor(_, _)
                            | Type::TensorShaped(_, _)
                            | Type::Struct(_, _)
                            | Type::Enum(_, _)
                            | Type::Tuple(_) => {
                                let f_ptr = self
                                    .builder
                                    .build_struct_gep(struct_ty, ptr, i as u32, "field_gep_stack")
                                    .map_err(|e| e.to_string())?;
                                let f_val = self
                                    .builder
                                    .build_load(
                                        self.context.ptr_type(inkwell::AddressSpace::default()),
                                        f_ptr,
                                        "field_load_stack",
                                    )
                                    .map_err(|e| e.to_string())?;
                                // Recursively free content (Clean FULL)
                                self.emit_recursive_free(f_val, f_ty, super::CLEANUP_FULL)?;
                            }
                            _ => {}
                        }
                    }
                    self.builder.build_unconditional_branch(merge_block).map_err(|e| e.to_string())?;
                    self.builder.position_at_end(merge_block);
                    return Ok(());
                }

                self.builder.position_at_end(free_block);



                // (DecRef Logic Moved UP)
                
                // --- Recurse Block Continued ---

                for (i, (_, f_ty)) in struct_def.fields.iter().enumerate() {
                     // Check ZST optimization compatibility
                     if let Type::Struct(s_name, _) = f_ty {
                          // If field struct is ZST, we didn't store a pointer, so don't try to load/free it.
                          let simple_s_name = if s_name.contains("::") { s_name.split("::").last().unwrap() } else { s_name.as_str() };
                          if let Some(def) = self.struct_defs.get(simple_s_name) {
                               if def.fields.is_empty() {
                                    continue;
                               }
                          }
                     }

                    match f_ty {
                        Type::Tensor(_, _)
                        | Type::TensorShaped(_, _)
                        | Type::Struct(_, _)
                        | Type::Enum(_, _)
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
                            // Recursively free content (Clean FULL)
                            self.emit_recursive_free(f_val, f_ty, super::CLEANUP_FULL)?;
                        }
                        _ => {}
                    }
                }
                
                // --- FIX: Free the Struct itself ---
                // Only if refcount dropped to 0 (which it did if we are here)
                let mem_free_fn = self.module.get_function("tl_mem_free")
                     .ok_or("tl_mem_free not found")?;
                
                // Trace Free
                self.emit_log_free(val)?;

                self.builder.build_call(mem_free_fn, &[cast_void.into()], "")
                    .map_err(|e| e.to_string())?;

                self.builder
                     .build_unconditional_branch(merge_block)
                     .map_err(|e| e.to_string())?;
                
                self.builder.position_at_end(merge_block);
            }
            Type::String(_) => {
                  let free_fn = self
                        .module
                        .get_function("tl_string_free")
                        .or_else(|| {
                             let void_ty = self.context.void_type();
                             let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
                             let ft = void_ty.fn_type(&[ptr_ty.into()], false);
                             Some(self.module.add_function("tl_string_free", ft, None))
                        })
                        .ok_or("tl_string_free not found")?;
                  
                  let ptr = val.into_pointer_value();
                  
                  // Runtime null check
                  let current_block = self.builder.get_insert_block().unwrap();
                  let func = current_block.get_parent().unwrap();
                  let free_block = self.context.append_basic_block(func, "free_string");
                  let merge_block = self.context.append_basic_block(func, "after_string_free");

                  let is_null = self.builder.build_is_null(ptr, "is_null_str").map_err(|e| e.to_string())?;
                  self.builder.build_conditional_branch(is_null, merge_block, free_block).map_err(|e| e.to_string())?;

                  self.builder.position_at_end(free_block);
                  // Cast to opaque pointer if needed (though StringStruct* is ptr)
                  let cast_ptr = self.builder.build_pointer_cast(ptr, self.context.ptr_type(inkwell::AddressSpace::default()), "cast_str").unwrap();
                  self.builder.build_call(free_fn, &[cast_ptr.into()], "").map_err(|e| e.to_string())?;
                  self.builder.build_unconditional_branch(merge_block).unwrap();

                  self.builder.position_at_end(merge_block);
                  return Ok(());
            }

            Type::Tuple(elem_types) => {
                 // Check if any element needs freeing
                 let needs_free = elem_types.iter().any(|t| matches!(t, Type::Tensor(_, _) | Type::Struct(_, _)));
                 if !needs_free {
                     return Ok(());
                 }
                 
                 let ptr = val.into_pointer_value();

                 // Reconstruct LLVM struct type for GEP
                 let mut llvm_types = Vec::new();
                 for ty in elem_types {
                     // Need get_llvm_type accessible or inline mapping
                     // Using simpler match map since self.get_llvm_type might be restricted?
                     // Actually self.context...
                     llvm_types.push(match ty {
                         Type::F32 => self.context.f32_type().into(),
                         Type::I64 => self.context.i64_type().into(),
                         Type::I32 => self.context.i32_type().into(),
                         Type::Bool => self.context.bool_type().into(),
                            Type::Tensor(_, _) | Type::Struct(_, _) | Type::Enum(_, _) | Type::Tuple(_) => self.context.ptr_type(inkwell::AddressSpace::default()).into(),
                          Type::Void => self.context.i8_type().into(),
                         _ => self.context.i64_type().into(), // Placeholder, potentially unsafe if not matching
                     });
                 }
                 let struct_ty = self.context.struct_type(&llvm_types, false);
                 
                 for (i, ty) in elem_types.iter().enumerate() {
                      if matches!(ty, Type::Tensor(_, _) | Type::String(_) | Type::Struct(_, _)) {
                           let f_ptr = self.builder.build_struct_gep(struct_ty, ptr, i as u32, "tup_elem").unwrap();
                           let load = self.builder.build_load(self.context.ptr_type(inkwell::AddressSpace::default()), f_ptr, "load").unwrap();
                           self.emit_recursive_free(load, ty, super::CLEANUP_FULL)?;
                      }
                 }
                 // Only free content. The tuple struct itself?
                 // Usually tuple struct is Alloc'd. We should free it if it's heap?
                 // But tuples in TL might be fully structural?
                 // Current impl allocates tuples on Heap (malloc).
                 // So we should free the tuple pointer too.
                 // Assuming CLEANUP_FULL.
                 if mode == super::CLEANUP_FULL {
                      let free = self.module.get_function("free").or_else(|| self.module.get_function("libc_free"));
                      if let Some(f) = free {
                            self.builder.build_call(f, &[ptr.into()], "").unwrap();
                      }
                 }
            }
            _ => {}
        }
        Ok(())
    }

    pub(crate) fn emit_recursive_unregister(
        &self,
        val: BasicValueEnum<'ctx>,
        ty: &Type,
    ) -> Result<(), String> {
        match ty {
            Type::Tensor(_, _) | Type::TensorShaped(_, _) => {
                // Check if pointer
                if !val.is_pointer_value() {
                    return Ok(()); // Should not happen for tensor
                }
                let ptr = val.into_pointer_value();
                let unreg_fn = self.module.get_function("tl_mem_unregister")
                    .ok_or("tl_mem_unregister not found")?;
                
                // Check Null?
                // Logic usually assumes non-null if registered. But let's build null check for safety?
                // Runtime handles null check.
                
                let cast_ptr = self.builder.build_pointer_cast(
                    ptr,
                    self.context.ptr_type(inkwell::AddressSpace::default()),
                    "cast_unreg_tens"
                ).unwrap();
                
                self.builder.build_call(unreg_fn, &[cast_ptr.into()], "").map_err(|e| e.to_string())?;
            }
            Type::Struct(name, _) => {
                if name == "Tensor" {
                    return self.emit_recursive_unregister(val, &Type::Tensor(Box::new(Type::F32), 1));
                }
                
                let simple_name = if name.contains("::") {
                    name.split("::").last().unwrap()
                } else {
                    name.as_str()
                };

                // Some structs might be opaque or non-existent (e.g. String wrapper)
                if simple_name == "String" {
                     // String is a struct but generally treated as scalar resource.
                     // But if it is registered, we should unregister it.
                     let ptr = val.into_pointer_value();
                     let unreg_fn = self.module.get_function("tl_mem_unregister")
                        .ok_or("tl_mem_unregister not found")?;
                     let cast_ptr = self.builder.build_pointer_cast(ptr, self.context.ptr_type(inkwell::AddressSpace::default()), "cast_unreg_str").unwrap();
                     self.builder.build_call(unreg_fn, &[cast_ptr.into()], "").map_err(|e| e.to_string())?;
                     return Ok(());
                }

                let struct_def = self.struct_defs.get(simple_name)
                    .ok_or(format!("Struct def {} not found", name))?
                    .clone();
                let struct_ty = *self.struct_types.get(simple_name)
                    .ok_or(format!("Struct type {} not found", name))?;
                
                let ptr = val.into_pointer_value();

                // 1. Unregister the Struct itself
                let unreg_fn = self.module.get_function("tl_mem_unregister")
                    .ok_or("tl_mem_unregister not found")?;
                
                let cast_ptr = self.builder.build_pointer_cast(
                    ptr,
                    self.context.ptr_type(inkwell::AddressSpace::default()),
                    "cast_unreg_struct"
                ).unwrap();

                self.builder.build_call(unreg_fn, &[cast_ptr.into()], "").map_err(|e| e.to_string())?;

                // 2. Recurse fields
                for (i, (_, f_ty)) in struct_def.fields.iter().enumerate() {
                    match f_ty {
                        Type::Tensor(_, _) 
                        | Type::TensorShaped(_, _) 
                        | Type::Struct(_, _) 
                        | Type::Enum(_, _) 
                        | Type::Tuple(_) => {
                            let f_ptr = self.builder.build_struct_gep(struct_ty, ptr, i as u32, "field_gep_unreg")
                                .map_err(|e| e.to_string())?;
                            let f_val = self.builder.build_load(
                                self.context.ptr_type(inkwell::AddressSpace::default()),
                                f_ptr,
                                "field_val_unreg"
                            ).map_err(|e| e.to_string())?;
                            
                            self.emit_recursive_unregister(f_val, f_ty)?;
                        }
                        _ => {}
                    }
                }
            }
            Type::Tuple(types) => {
                 let ptr = val.into_pointer_value();
                 let unreg_fn = self.module.get_function("tl_mem_unregister")
                    .ok_or("tl_mem_unregister not found")?;
                 let cast_ptr = self.builder.build_pointer_cast(ptr, self.context.ptr_type(inkwell::AddressSpace::default()), "cast_unreg_tup").unwrap();
                 self.builder.build_call(unreg_fn, &[cast_ptr.into()], "").map_err(|e| e.to_string())?;

                 // Need LLVM type for GEP
                 let llvm_types: Vec<_> = types.iter().map(|t| {
                      match t {
                         Type::F32 => self.context.f32_type().into(),
                         Type::I64 => self.context.i64_type().into(),
                         Type::I32 => self.context.i32_type().into(),
                         Type::Bool => self.context.bool_type().into(),
                         Type::Tensor(_, _) | Type::Struct(_, _) | Type::Enum(_, _) | Type::Tuple(_) | Type::String(_) => self.context.ptr_type(inkwell::AddressSpace::default()).into(),
                         Type::Void => self.context.i8_type().into(),
                         _ => self.context.i64_type().into(),
                      }
                 }).collect();
                 let struct_ty = self.context.struct_type(&llvm_types, false);

                 for (i, elem_ty) in types.iter().enumerate() {
                      if matches!(elem_ty, Type::Tensor(_, _) | Type::Struct(_, _) | Type::Tuple(_)) {
                           let f_ptr = self.builder.build_struct_gep(struct_ty, ptr, i as u32, "tup_gep_unreg").unwrap();
                           let f_val = self.builder.build_load(self.context.ptr_type(inkwell::AddressSpace::default()), f_ptr, "tup_val_unreg").unwrap();
                           self.emit_recursive_unregister(f_val, elem_ty)?;
                      }
                 }
            }

            Type::Enum(_name, _) => {
                 let ptr = val.into_pointer_value();
                 let unreg_fn = self.module.get_function("tl_mem_unregister")
                    .ok_or("tl_mem_unregister not found")?;
                 let cast_ptr = self.builder.build_pointer_cast(ptr, self.context.ptr_type(inkwell::AddressSpace::default()), "cast_unreg_enum").unwrap();
                 self.builder.build_call(unreg_fn, &[cast_ptr.into()], "").map_err(|e| e.to_string())?;
                 // TODO: Deep unregister for Enums
            }
            _ => {}
        }
        Ok(())
    }

    pub(crate) fn compile_stmt(&mut self, stmt: &Stmt) -> Result<(), String> {
        self.current_time += 1;
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

                        let (res, _) = self.compile_bin_op(current_val, field_type.clone(), val, val_ty.clone(), bin_op)?;
                        res
                    }
                };

                // Free old value if it's a structural type (Tensor/Struct)
                if matches!(
                    field_type,
                    Type::Tensor(_, _) | Type::Struct(_, _)
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
                    self.emit_recursive_free(current_val.into(), &field_type, super::CLEANUP_FULL)?;
                    self.builder.build_unconditional_branch(skip_block).unwrap();
                    self.builder.position_at_end(skip_block);
                }

                self.builder.build_store(field_ptr, final_val).map_err(|e| e.to_string())?;

                // Prevent compiler from freeing the temporary now that it's stored in a field
                // RefCount Logic:
                // If it was a temporary, we remove it (Move). RefCount unchanged (1->1).
                // If it was L-Value, we must IncRef. RefCount (1->2).
                let mut moved = false;
                if let Some(temps) = self.temporaries.last_mut() {
                    if let Some(idx) = temps.iter().position(|(v, _, _)| *v == val) {
                        temps.remove(idx);
                        moved = true;
                    }
                }
                
                if !moved {
                     match val_ty {
                        | Type::Struct(_, _) 
                        | Type::Enum(_, _) => {
                            let inc_fn = self.module.get_function("tl_ptr_inc_ref")
                                .or_else(|| {
                                    let void_ty = self.context.void_type();
                                    let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
                                    let ft = void_ty.fn_type(&[ptr_ty.into()], false);
                                    Some(self.module.add_function("tl_ptr_inc_ref", ft, None))
                                })
                                .expect("tl_ptr_inc_ref decl failed");
                                
                            let ptr = val.into_pointer_value();
                            let void_ptr = self.builder.build_pointer_cast(
                                ptr,
                                self.context.ptr_type(inkwell::AddressSpace::default()),
                                "void_cast_inc_fassign"
                            ).unwrap();
                            self.builder.build_call(inc_fn, &[void_ptr.into()], "").unwrap();
                        }
                        _ => {}
                     }
                }

                // Ownership transfer
                if let Some(f) = self.module.get_function("tl_mem_unregister") {
                    let should_unregister = match &field_type {
                        Type::Tensor(_, _) | Type::Struct(_, _) => true,
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
                let def_time = self.current_time;

                if let Some(expr) = init {
                    let (val_ir, _inferred_ty) = self.ensure_tensor_v2(expr, 0)?;
                    let val_ty = if matches!(type_annotation, Type::Tensor(_, _)) {
                        type_annotation.clone()
                    } else {
                        // tensor name: f32 means Tensor<f32, 0>
                        Type::Tensor(Box::new(type_annotation.clone()), 0)
                    };

                    // NOTE: Removed clone to preserve gradients

                    if self.variables.last().unwrap().contains_key(name) {
                        // Start of double-free fix logic
                        let (_var_val, _, cleanup_mode) = &self.variables.last().unwrap()[name];

                        if *cleanup_mode != super::CLEANUP_NONE {
                            // Restore Free Logic for RefCounting
                            self.emit_recursive_free(*_var_val, &val_ty, *cleanup_mode)?;
                        }

                        let ptr = self.variables.last().unwrap()[name].0.into_pointer_value();
                        self.builder
                            .build_store(ptr, val_ir)
                            .map_err(|e| e.to_string())?;

                        // Update variable map to mark as owned (should_free = true)
                        self.variables
                            .last_mut()
                            .unwrap()
                            .insert(name.clone(), (ptr.into(), val_ty, super::CLEANUP_FULL));
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

                        // Consumed by variable
                        self.consume_temp(val_ir);

                        self.variables
                            .last_mut()
                            .unwrap()
                            .insert(name.clone(), (ptr.into(), val_ty, super::CLEANUP_FULL));
                    }
                } else {
                     // Alloc but no init? (Current code path seems to assume init is Some for now based on AST?)
                     // If init is None, we might just alloc.
                }

                // Register Liveness
                let last_use = if let Some(analysis) = &self.function_analysis {
                    match analysis.last_use_times.get(&def_time) {
                         Some(&t) => t,
                         None => 0
                    }
                } else {
                    0
                };
                if let Some(scope) = self.variable_liveness.last_mut() {
                    scope.insert(name.clone(), last_use);
                }

                Ok(())
            }
            StmtKind::Let {
                name,
                type_annotation: _, // TODO: Use this
                value,
                mutable: _, // TODO: Use this
            } => {
                let def_time = self.current_time;

                // 1. Analyze value for Free Indices (Implicit Tensor Equation)
                let free_indices = self.infer_free_indices(value);

                if !free_indices.is_empty() {
                    let clauses: Vec<ComprehensionClause> = Vec::new();
                    return self
                        .compile_tensor_equation(name, &free_indices, &clauses, Some(value))
                        .map_err(|e| e.to_string());
                }

                let is_slot_backed = false;
                let dps_result = None;

                if let ExprKind::FnCall(fn_name, _args) = &value.inner {
                    // Check if function returns Tensor
                    // We need to resolve name properly if it's imported (simplified check for now)
                    let simple_name = fn_name.split("::").last().unwrap_or(fn_name);
                    let lookup_name = if self.module.get_function(fn_name).is_some() {
                         fn_name
                    } else if self.module.get_function(simple_name).is_some() {
                         simple_name
                    } else {
                         fn_name
                    };

                    if let Some(func) = self.module.get_function(lookup_name) {
                         let ret_ty = self.get_return_type_from_signature(func);
                         if matches!(ret_ty, Type::Tensor(_, _)) {
                             // Check if we have a slot for this variable
                             // FIX: Disabled DPS for Tensors for now.
                             // Most runtime functions (tl_tensor_new, add, etc) return a NEW pointer (Box::into_raw).
                             // They do not support writing to a caller-provided OpaqueTensor* (Slot).
                             // Attempting DPS here causes the Return Value to be ignored (Leak)
                             // and the Slot Buffer (uninitialized) to be used (Crash/UB) and finalized (Double Free/Bad Free).
                             
                             /*
                             if let Some(analysis) = &self.function_analysis {
                                 if let Some(&slot_id) = analysis.slots.get(name) {
                                      // DO DPS
                                      // Get Buffer from Slot
                                      let buf_fn_name = "tl_mem_get_buffer";
                                      let buf_fn = if let Some(f) = self.module.get_function(buf_fn_name) {
                                           f
                                      } else {
                                           let i64_ty = self.context.i64_type();
                                           let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
                                           let ft = ptr_ty.fn_type(&[i64_ty.into(), i64_ty.into()], false);
                                           self.module.add_function(buf_fn_name, ft, None)
                                      };

                                      // Assuming 96 bytes for Tensor struct
                                      let size = self.context.i64_type().const_int(96, false);
                                      let slot = self.context.i64_type().const_int(slot_id as u64, false);
                                      let call = self.builder.build_call(buf_fn, &[slot.into(), size.into()], "slot_buf").unwrap();
                                      let raw_ptr = match call.try_as_basic_value() {
                                          inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
                                          _ => panic!("tl_mem_get_buffer returned non-basic value"),
                                      };
                                      let cast_ptr = self.builder.build_pointer_cast(raw_ptr, self.context.ptr_type(inkwell::AddressSpace::default()), "cast_slot").unwrap();
                                      
                                      // Call compile_fn_call_dps
                                      let res = self.compile_fn_call_dps(fn_name, args, Some(cast_ptr.into()))?;
                                      dps_result = Some(res);
                                      is_slot_backed = true;
                                 }
                             }
                             */
                         }
                    }
                }

                let (mut val_ir, val_ty) = if let Some((r, t)) = dps_result {
                    (r, t)
                } else {
                    // Standard expression compilation
                    self.compile_expr(value)?
                };
                
                // Ownership: Shared. The temporary (value) remains in scope and will be released at scope exit.
                // The variable (name) acquires a NEW reference via deep_clone below.
                // We do NOT unregister the temporary. Ref 1 (Temp) + Ref 1 (Var) = 2.
                // Temp Scope Exit -> -1. Var Scope Exit -> -1. Total 0. Safe.

                // Removed legacy hardcoded type fixups for HashMap/Vec/Option.
                // Rely on correct type checking in semantics phase.

                // Variable Assignment: Deep Clone (Struct Copy + Tensor Acquire)
                // Optimization: R-value Move Semantics
                // If the value is a temporary (FnCall, BinOp, etc), we take ownership (Move).
                // If the value is an L-value (Variable, FieldAccess), we must Copy (Acquire/Clone).

                let mut is_last_use_move = false;
                if let ExprKind::Variable(vname) = &value.inner {
                    if self.is_last_use(vname) {
                        is_last_use_move = true;
                        // Mark source as moved (transfer ownership)
                        for scope in self.variables.iter_mut().rev() {
                            if let Some(entry) = scope.get_mut(vname) {
                                entry.2 = super::CLEANUP_NONE;
                                break;
                            }
                        }
                    }
                }

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
                ) || is_last_use_move;

                let should_deep_clone = match &val_ty {
                    Type::Tensor(_, _) | Type::TensorShaped(_, _) => !is_rvalue, // Clone only if L-value
                    Type::Struct(_, _) | Type::Enum(_, _) | Type::Tuple(_) => {
                        // Structs/UserDefined/Enum/Vec/Tuple: Pointer copy vs Deep Clone
                        // If R-value, we own the pointer. Move.
                        !is_rvalue
                    }
                    _ => false,
                };

                if should_deep_clone {
                    val_ir = self.emit_deep_clone(val_ir, &val_ty)?;
                } else if is_rvalue {
                    // Move Semantics:
                    // If it's an R-value (temporary), we skip deep_clone (ownership transfer).
                    // BUT we must remove it from the temporary list so it's not freed 
                    // when the temporary scope ends. The variable now owns it.
                    self.try_consume_temp(val_ir);
                }

                let current_function = self
                    .builder
                    .get_insert_block()
                    .unwrap()
                    .get_parent()
                    .unwrap();

                // Check for shadowing in CURRENT scope
                let shadow_info = if let Some(scope) = self.variables.last() {
                    if let Some((old_ptr, old_ty, cleanup_mode)) = scope.get(name) {
                        if *cleanup_mode != super::CLEANUP_NONE {
                            Some((*old_ptr, old_ty.clone(), *cleanup_mode))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };

                if let Some((old_ptr_val, old_ty, old_mode)) = shadow_info {
                    // Load the actual pointer value from the alloca
                    let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
                    let old_value = self
                        .builder
                        .build_load(ptr_type, old_ptr_val.into_pointer_value(), "old_shadowed")
                        .map_err(|e| e.to_string())?;

                    self.emit_recursive_free(old_value, &old_ty, old_mode)?;
                }

                let alloca = self.create_entry_block_alloca(current_function, name, &val_ty)?;
                self.builder
                    .build_store(alloca, val_ir)
                    .map_err(|e| e.to_string())?;

                // Keep ownership in the variable (if val_ir was a temporary)
                // self.mark_temp_no_cleanup(val_ir);
                
                // RefCount Logic: Clone on Copy
                let mut moved = false;
                let mut cleanup_mode = super::CLEANUP_FULL;
                
                if let Some(temps) = self.temporaries.last_mut() {
                    if let Some(idx) = temps.iter().position(|(v, _, _)| *v == val_ir) {
                        let (_, _, mode) = temps.remove(idx);
                        cleanup_mode = mode;
                        moved = true;
                    }
                }
                
                if !moved {
                     // L-Value copy -> IncRef
                     match val_ty {
                        Type::Tensor(_, _) 
                        | Type::Enum(_, _) => {
                            let inc_fn = self.module.get_function("tl_ptr_inc_ref")
                                .or_else(|| {
                                    let void_ty = self.context.void_type();
                                    let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
                                    let ft = void_ty.fn_type(&[ptr_ty.into()], false);
                                    Some(self.module.add_function("tl_ptr_inc_ref", ft, None))
                                })
                                .expect("tl_ptr_inc_ref decl failed");

                            let ptr = val_ir.into_pointer_value();
                            let void_ptr = self.builder.build_pointer_cast(
                                ptr,
                                self.context.ptr_type(inkwell::AddressSpace::default()),
                                "void_cast_inc_let"
                            ).unwrap();
                            self.builder.build_call(inc_fn, &[void_ptr.into()], "").unwrap();
                        }
                        Type::Struct(ref name, _) if name != "String" && name != "File" && name != "Path" && name != "Map" && name != "Tokenizer" && name != "KVCache" => {
                            // Only inc_ref if it's a pointer (skip ZSTs)
                            if val_ir.is_pointer_value() {
                                let inc_fn = self.module.get_function("tl_ptr_inc_ref")
                                    .or_else(|| {
                                        let void_ty = self.context.void_type();
                                        let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
                                        let ft = void_ty.fn_type(&[ptr_ty.into()], false);
                                        Some(self.module.add_function("tl_ptr_inc_ref", ft, None))
                                    })
                                    .expect("tl_ptr_inc_ref decl failed");

                                let ptr = val_ir.into_pointer_value();
                                let void_ptr = self.builder.build_pointer_cast(
                                    ptr,
                                    self.context.ptr_type(inkwell::AddressSpace::default()),
                                    "void_cast_inc_let_st"
                                ).unwrap();
                                self.builder.build_call(inc_fn, &[void_ptr.into()], "").unwrap();
                            }
                        }
                        _ => {}
                     }
                }

                self.variables
                    .last_mut()
                    .unwrap()
                    .insert(
                        name.clone(),
                        (
                            alloca.into(),
                            val_ty.clone(),
                            if is_slot_backed {
                                super::CLEANUP_FINALIZE
                            } else {
                                cleanup_mode
                            },
                        ),
                    ); // Store pointer and type

                // Register Liveness
                let last_use = if let Some(analysis) = &self.function_analysis {
                    match analysis.last_use_times.get(&def_time) {
                        Some(&t) => t,
                        None => 0
                    }
                } else {
                    0
                };
                if let Some(scope) = self.variable_liveness.last_mut() {
                    scope.insert(name.clone(), last_use);
                }

                // DEBUG TRACE: Log assignment to variable
                match &val_ty {
                    Type::Struct(sname, _) => {
                         if sname.contains("GPT") {
                             let size_val = self.context.i64_type().const_int(0, false);
                             self.emit_log_alloc(val_ir, size_val).ok();
                         }
                    }
                    _ => {}
                }
                Ok(())
            }
            StmtKind::Return(expr_opt) => {
                if let Some(expr) = expr_opt {
                    let (val, ty) = self.compile_expr(expr)?;

                    // If returning a variable, mark it as moved (should_free = false)
                    if let ExprKind::Variable(name) = &expr.inner {
                        for scope in self.variables.iter_mut().rev() {
                            if let Some(entry) = scope.get_mut(name) {
                                entry.2 = super::CLEANUP_NONE;
                            }
                        }
                    }
                    // FIX: Ensure the returned value is NOT cleaned up by emit_all_scopes_cleanup.
                    // This is handled for Variables above, but temporaries (like StructInit result)
                    // might be in self.temporaries (via add_temp or implicit) and need to be marked.
                    // We consume it from the temporary list so cleanup logic skips it.
                    self.mark_temp_no_cleanup(val);

                    if val.is_pointer_value() {
                        let ptr = val.into_pointer_value();
                        for scope in self.variables.iter_mut().rev() {
                            // Find precise match for pointer value
                            for (_, (v, _, cleanup)) in scope.iter_mut() {
                                if v.is_pointer_value() && v.into_pointer_value() == ptr {
                                    *cleanup = super::CLEANUP_NONE;
                                }
                            }
                        }
                    }

                    // Check if this is a struct return (uses sret)
                    let uses_sret = self.current_sret_dest.is_some();

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
                            Type::Struct(_, _) => {
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
                        // CRITICAL: Copy to sret BEFORE cleanup to avoid stale pointer access
                        // Get the sret pointer
                        let sret_ptr = self.current_sret_dest.unwrap();

                        // Copy struct contents to sret pointer BEFORE cleanup
                        let src_ptr = val.into_pointer_value();
                        self.emit_struct_copy(sret_ptr, src_ptr, &ty)?;
                        self.emit_all_scopes_cleanup();
                        if let Some(exit_fn) = self.module.get_function("tl_mem_function_exit") {
                             self.builder.build_call(exit_fn, &[], "").unwrap();
                        }
                        self.builder.build_return(None).map_err(|e| e.to_string())?;
                    } else {
                        // Normal return: cleanup then return value
                        self.emit_all_scopes_cleanup();
                        if let Some(exit_fn) = self.module.get_function("tl_mem_function_exit") {
                             self.builder.build_call(exit_fn, &[], "").unwrap();
                        }
                        let ret_instr = self.builder
                            .build_return(Some(&val));
                        ret_instr.map_err(|e| e.to_string())?;
                    }

                } else {
                    // return; (Void return)
                    self.emit_all_scopes_cleanup();
                    if let Some(exit_fn) = self.module.get_function("tl_mem_function_exit") {
                            self.builder.build_call(exit_fn, &[], "").unwrap();
                    }
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
                            Type::Struct(_, _) => {
                                // Generic structural assignment via .set(index, value) method
                                // Support arr[i] = val for any struct implementing set(i64, T)
                                if idxs.len() != 1 {
                                    return Err("Struct indexing assignment only supports 1D indexing currently".into());
                                }
                                let (idx_val, _idx_ty) = self.compile_expr(&idxs[0])?;
                                
                                // Load the struct value (pointer)
                                let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
                                let obj_val = self.builder.build_load(ptr_type, var_ptr.into_pointer_value(), "obj_val").map_err(|e| e.to_string())?;
                                
                                let method_name = "set";
   
                                // Monomorphize 'set'
                                let generics = if let Type::Struct(_, g) = &var_type { g.clone() } else { vec![] };
                                
                                let mangled_name = self.monomorphize_method(
                                    &var_type.get_base_name(), 
                                    method_name, 
                                    &generics
                                )?;
                                
                                let set_fn = self.module.get_function(&mangled_name).ok_or(format!("Method {} not found check impl", mangled_name))?;
                                
                                // Call it
                                let mut call_args: Vec<inkwell::values::BasicMetadataValueEnum> = Vec::new();
                                call_args.push(obj_val.into()); // Self
                                call_args.push(idx_val.into()); // Index
                                call_args.push(val_ir.into());  // Value
                                
                                self.builder.build_call(set_fn, &call_args, "").map_err(|e| e.to_string())?;
                                
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
                                    "Indexed assignment only supported for Tensor".into(),
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
                let mut found_should_free = super::CLEANUP_NONE;
                for scope in self.variables.iter().rev() {
                    if let Some((v, t, mode)) = scope.get(name) {
                        found_var_ptr = Some(*v);
                        found_var_type = Some(t.clone());
                        found_should_free = *mode;
                        break;
                    }
                }

                let var_ptr = found_var_ptr.ok_or(format!("Variable {} not found", name))?;
                let var_type = found_var_type.ok_or(format!("Variable {} not found", name))?;

                // If val_base (RHS) was a temporary, we take ownership (Move).
                // If it was L-value, compile_expr returns non-temp, consume_temp does nothing.
                // RefCount Logic: Clone on Copy
                // Use `val` (actual value to be stored)
                let mut moved = false;
                if let Some(temps) = self.temporaries.last_mut() {
                     if let Some(idx) = temps.iter().position(|(v, _, _)| *v == val) {
                         temps.remove(idx);
                         moved = true;
                     }
                }
                
                if !moved {
                     // L-Value copy -> IncRef
                     match val_type {
                         Type::Tensor(_, _) 
                         | Type::Struct(_, _) 
                         | Type::Enum(_, _) => {
                             let inc_fn = self.module.get_function("tl_ptr_inc_ref")
                                .or_else(|| {
                                    let void_ty = self.context.void_type();
                                    let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
                                    let ft = void_ty.fn_type(&[ptr_ty.into()], false);
                                    Some(self.module.add_function("tl_ptr_inc_ref", ft, None))
                                })
                                .expect("tl_ptr_inc_ref decl failed");

                             // GUARD: Only inc_ref if it's a pointer (skip ZST)
                             if val.is_pointer_value() {
                                 let ptr = val.into_pointer_value();
                                 let void_ptr = self.builder.build_pointer_cast(
                                     ptr,
                                     self.context.ptr_type(inkwell::AddressSpace::default()),
                                     "void_cast_inc_assign"
                                 ).unwrap();
                                 self.builder.build_call(inc_fn, &[void_ptr.into()], "").unwrap();
                             }
                         }
                         _ => {}
                     }
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
                            Type::Struct(_, _) | Type::Tensor(_, _)
                        ) {
                             if found_should_free != super::CLEANUP_NONE && val.is_pointer_value() {
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

                                 let can_free = self
                                     .builder
                                     .build_and(is_not_null, are_diff, "can_free")
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

                                 // Recursive free fields of the OLD struct (or Tensor content)
                                 self.emit_recursive_free(current_val.into(), &var_type, found_should_free)?;

                                 // Also unregister the struct shell itself so Runtime doesn't track it
                                 if found_should_free == super::CLEANUP_FULL {
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
                                 }

                                 self.builder
                                     .build_unconditional_branch(continue_block)
                                     .map_err(|e| e.to_string())?;

                                 // Continue block
                                 self.builder.position_at_end(continue_block);
                             }
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
                                } else if matches!(var_type, Type::Struct(_, _))
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

                            let add_assign_fn = if matches!(val_type, Type::Tensor(_, _)) {
                                self.module.get_function("tl_tensor_add_assign").unwrap()
                            } else {
                                self.module
                                    .get_function("tl_tensor_add_assign_scalar_f32")
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
                                            "AddAssign: unsupported RHS type {:?}",
                                            val_type
                                        ))
                                    }
                                };
                                scalar_f32.into()
                            };

                            self.builder
                                .build_call(add_assign_fn, &[current_val.into(), val_arg], "")
                                .map_err(|e| e.to_string())?;

                            return Ok(());
                        } else {
                            // Generic path
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
                let def_time = self.current_time;
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
                            (tensor_alloca.into(), tensor_ty.clone(), super::CLEANUP_NONE),
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
                            (tensor_alloca.into(), iter_ty.clone(), super::CLEANUP_NONE),
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
                    .insert(loop_var.clone(), (var_alloca.into(), loop_var_val.1, super::CLEANUP_NONE));

                // Register Liveness for loop variable
                let last_use = if let Some(analysis) = &self.function_analysis {
                    match analysis.last_use_times.get(&def_time) {
                         Some(&t) => t,
                         None => 0
                    }
                } else {
                    0
                };
                if let Some(scope) = self.variable_liveness.last_mut() {
                    scope.insert(loop_var.clone(), last_use);
                }

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
                    | Type::Tensor(_, _)
                    | Type::TensorShaped(_, _)
                    | Type::Enum(_, _)

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
                            .insert(temp_name, (alloca.into(), ty.clone(), super::CLEANUP_FULL));
                    }

                    _ => {
                        // Primitive types: no action needed (no memory to manage)
                    }
                }

                Ok(())
            }
            StmtKind::Break => {
                // Cleanup all scopes up to loop entry before jumping
                let target = self.loop_stack.last().map(|(_, bb, depth)| (*bb, *depth));
                if let Some((break_block, loop_depth)) = target {
                    self.emit_cleanup_to_depth(loop_depth);
                    self.builder
                        .build_unconditional_branch(break_block)
                        .map_err(|e| e.to_string())?;
                }
                Ok(())
            }
            StmtKind::Continue => {
                // Cleanup all scopes up to loop entry before jumping
                let target = self.loop_stack.last().map(|(bb, _, depth)| (*bb, *depth));
                if let Some((continue_block, loop_depth)) = target {
                    self.emit_cleanup_to_depth(loop_depth);
                    self.builder
                        .build_unconditional_branch(continue_block)
                        .map_err(|e| e.to_string())?;
                }
                Ok(())
            }
        }
    }

    fn compile_tensor_scalar_op(
        &self,
        lhs: BasicValueEnum<'ctx>,
        lhs_type: Type,
        rhs: BasicValueEnum<'ctx>,
        rhs_type: Type,
        op: BinOp,
        scalar_is_rhs: bool,
    ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        let (scalar_val, tensor_val, tensor_ty) = if scalar_is_rhs {
            (rhs, lhs, lhs_type)
        } else {
            (lhs, rhs, rhs_type)
        };

        let val_f32 = if scalar_val.is_int_value() {
            let v = scalar_val.into_int_value();
            self.builder
                .build_signed_int_to_float(v, self.context.f32_type(), "cast_scalar")
                .map_err(|e| e.to_string())?
        } else {
            scalar_val.into_float_value()
        };

        // 1. Alloca in Entry Block
        let current_block = self.builder.get_insert_block().unwrap();
        let parent_fn = current_block.get_parent().unwrap();

        let data_alloca =
            self.create_entry_block_alloca(parent_fn, "scalar_data", &Type::F32)?;
        self.builder
            .build_store(data_alloca, val_f32)
            .map_err(|e| e.to_string())?;

        // 2. Shape Alloca (dummy i64)
        let shape_alloca =
            self.create_entry_block_alloca(parent_fn, "scalar_shape", &Type::I64)?;

        // 3. New Tensor
        let new_fn = self.module.get_function("tl_tensor_new").unwrap();
        let rank_val = self.context.i64_type().const_int(0, false);
        let call = self
            .builder
            .build_call(
                new_fn,
                &[data_alloca.into(), rank_val.into(), shape_alloca.into()],
                "scalar_tensor",
            )
            .map_err(|e| e.to_string())?;
        let scalar_tensor = self
            .check_tensor_result(call, "scalar_tensor_error")?
            .into_pointer_value();

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
            .ok_or(format!("Runtime function {} not found", fn_name))?;

        let (arg1, arg2) = if scalar_is_rhs {
            (
                tensor_val.into_pointer_value().into(),
                scalar_tensor.into(),
            )
        } else {
            (
                scalar_tensor.into(),
                tensor_val.into_pointer_value().into(),
            )
        };

        let call = self.builder.build_call(fn_val, &[arg1, arg2], "binop_res");

        let res_val =
            self.check_tensor_result(call.map_err(|e| e.to_string())?, "binop_scalar_error")?;

        // Free temporary scalar tensor
        let free_fn = self
            .module
            .get_function("tl_tensor_free")
            .ok_or("tl_tensor_free not found")?;
        self.builder
            .build_call(free_fn, &[scalar_tensor.into()], "")
            .map_err(|e| e.to_string())?;

        let res_ptr = res_val.into_pointer_value();
        Ok((res_ptr.into(), tensor_ty))
    }

    fn compile_string_bin_op(
        &self,
        lhs: BasicValueEnum<'ctx>,
        rhs: BasicValueEnum<'ctx>,
        op: BinOp,
    ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
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
                    inkwell::values::ValueKind::Basic(v) => v,
                    _ => return Err("Invalid string concat return".into()),
                };
                Ok((res_val, Type::String("String".to_string())))
            }
            BinOp::Eq | BinOp::Neq => {
                let streq_fn = self
                    .module
                    .get_function("tl_string_eq")
                    .ok_or("tl_string_eq not found")?;
                let cmp = self
                    .builder
                    .build_call(streq_fn, &[lhs.into(), rhs.into()], "streq_res")
                    .map_err(|e| e.to_string())?;

                let cmp_val = match cmp.try_as_basic_value() {
                    inkwell::values::ValueKind::Basic(v) => v.into_int_value(),
                    _ => return Err("Invalid tl_string_eq return".into()),
                };
                
                let res = match op {
                    BinOp::Eq => cmp_val,
                    BinOp::Neq => self.builder
                        .build_not(cmp_val, "strneq")
                        .map_err(|e| e.to_string())?,
                    _ => unreachable!(),
                };
                Ok((res.into(), Type::Bool))
            }
            _ => Err("Only ==, !=, and + supported for Strings".into()),
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
            // String Concatenation
            (Type::String(_), Type::String(_)) if op == BinOp::Add => {
                let concat_fn = self
                    .module
                    .get_function("tl_string_concat")
                    .ok_or("tl_string_concat not found")?;
                let call = self
                    .builder
                    .build_call(concat_fn, &[lhs.into(), rhs.into()], "str_concat")
                    .map_err(|e| e.to_string())?;
                let res = match call.try_as_basic_value() {
                    inkwell::values::ValueKind::Basic(v) => v,
                    _ => return Err("tl_string_concat returned void".into()),
                };
                Ok((res, Type::String("String".to_string())))
            }
            (Type::String(_), Type::Char(_)) if op == BinOp::Add => {
                 let char_to_str_fn = self.module.get_function("tl_string_from_char").ok_or("tl_string_from_char missing")?;
                 let call_c = self.builder.build_call(char_to_str_fn, &[rhs.into()], "char_str").map_err(|e| e.to_string())?;
                 let rhs_str = match call_c.try_as_basic_value() {
                    inkwell::values::ValueKind::Basic(v) => v,
                    _ => return Err("tl_string_from_char returned void".into()),
                 };
                 
                let concat_fn = self
                    .module
                    .get_function("tl_string_concat")
                    .ok_or("tl_string_concat not found")?;
                let call = self
                    .builder
                    .build_call(concat_fn, &[lhs.into(), rhs_str.into()], "str_concat")
                    .map_err(|e| e.to_string())?;
                 let res = match call.try_as_basic_value() {
                    inkwell::values::ValueKind::Basic(v) => v,
                    _ => return Err("tl_string_concat returned void".into()),
                 };
                 Ok((res, Type::String("String".to_string())))
            }
            (Type::Char(_), Type::String(_)) if op == BinOp::Add => {
                 let char_to_str_fn = self.module.get_function("tl_string_from_char").ok_or("tl_string_from_char missing")?;
                 let call_c = self.builder.build_call(char_to_str_fn, &[lhs.into()], "char_str").map_err(|e| e.to_string())?;
                 let lhs_str = match call_c.try_as_basic_value() {
                    inkwell::values::ValueKind::Basic(v) => v,
                    _ => return Err("tl_string_from_char returned void".into()),
                 };
                 
                let concat_fn = self
                    .module
                    .get_function("tl_string_concat")
                    .ok_or("tl_string_concat not found")?;
                let call = self
                    .builder
                    .build_call(concat_fn, &[lhs_str.into(), rhs.into()], "str_concat")
                    .map_err(|e| e.to_string())?;
                 let res = match call.try_as_basic_value() {
                    inkwell::values::ValueKind::Basic(v) => v,
                    _ => return Err("tl_string_concat returned void".into()),
                 };
                 Ok((res, Type::String("String".to_string())))
            }
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
            (Type::String(_), Type::String(_)) => self.compile_string_bin_op(lhs, rhs, op),

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
            (Type::Char(_), Type::Char(_)) => {
                let l = lhs.into_int_value();
                let r = rhs.into_int_value();
                let res = match op {
                    BinOp::Eq => {
                        self.builder
                            .build_int_compare(inkwell::IntPredicate::EQ, l, r, "eqtmp")
                    }
                    BinOp::Neq => {
                        self.builder
                            .build_int_compare(inkwell::IntPredicate::NE, l, r, "neqtmp")
                    }
                    _ => return Err("Unsupported char op".into()),
                }
                .map_err(|e| e.to_string())?;
                Ok((res.into(), Type::Bool))
            }
            (
                Type::Tensor(_, _)
                | Type::Struct(..),
                Type::Tensor(_, _)
                | Type::Struct(..),
            ) if (matches!(lhs_type, Type::Tensor(_, _))
                || (matches!(&lhs_type, Type::Struct(n, _) if n == "Tensor"))
                    || (matches!(&rhs_type, Type::Struct(n, _) if n == "Tensor"))) =>
            {
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
                 self.compile_tensor_scalar_op(lhs, lhs_type, rhs, rhs_type, op, true)
            }
            (Type::Struct(name, _), Type::F32) if name == "Tensor" => {
                 self.compile_tensor_scalar_op(lhs, lhs_type, rhs, rhs_type, op, true)
            }
            (Type::F32, Type::Tensor(inner, _)) if **inner == Type::F32 => {
                 self.compile_tensor_scalar_op(lhs, lhs_type, rhs, rhs_type, op, false)
            }
            (Type::F32, Type::Struct(name, _)) if name == "Tensor" => {
                 self.compile_tensor_scalar_op(lhs, lhs_type, rhs, rhs_type, op, false)
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
            Type::String(_) => {
                // String Deep Clone
                // Uses tl_string_new("") + tl_string_concat(val, empty)
                // 1. Create empty string object
                let empty_cstr = self.builder.build_global_string_ptr("", "empty_cstr").unwrap();
                let string_new_fn = self.module.get_function("tl_string_new")
                    .or_else(|| {
                         // Declare if missing
                         let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
                         let fn_type = ptr_ty.fn_type(&[ptr_ty.into()], false);
                         Some(self.module.add_function("tl_string_new", fn_type, None))
                    })
                    .ok_or("tl_string_new not found")?;
                    
                let empty_call = self.builder.build_call(string_new_fn, &[empty_cstr.as_pointer_value().into()], "empty_str").map_err(|e| e.to_string())?;
                let empty_str_obj = match empty_call.try_as_basic_value() {
                    inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
                    _ => return Err("tl_string_new returned void".into()),
                };

                // 2. Deep Clone via Concat
                let string_concat_fn = self.module.get_function("tl_string_concat")
                     .ok_or("tl_string_concat not found")?;
                let val_ptr = val.into_pointer_value();
                let clone_call = self.builder.build_call(string_concat_fn, &[val_ptr.into(), empty_str_obj.into()], "str_clone").map_err(|e| e.to_string())?;
                let clone_res = match clone_call.try_as_basic_value() {
                    inkwell::values::ValueKind::Basic(v) => v,
                    _ => return Err("tl_string_concat returned void".into()),
                };

                // 3. Free empty string object
                let string_free_fn = self.module.get_function("tl_string_free")
                    .ok_or("tl_string_free not found")?;
                self.builder.build_call(string_free_fn, &[empty_str_obj.into()], "").map_err(|e| e.to_string())?;

                Ok(clone_res)
            }
            Type::Enum(name, generics) => {
                let mangled_name = if generics.is_empty() {
                    name.clone()
                } else {
                    self.mangle_type_name(name, generics)
                };

                let enum_def = self
                    .enum_defs
                    .get(&mangled_name)
                    .ok_or(format!("Enum {} definition not found ({})", name, mangled_name))?;
                self.emit_enum_deep_clone(val, enum_def)
            }
            Type::Struct(name, generics) => {
                let mangled_name = if generics.is_empty() {
                    name.clone()
                } else {
                    self.mangle_type_name(name, generics)
                };

                // Check if it is an Enum
                if let Some(enum_def) = self.enum_defs.get(&mangled_name) {
                    return self.emit_enum_deep_clone(val, enum_def);
                }
                
                // Handle String struct
                if name == "String" {
                    return self.emit_deep_clone(val, &Type::String(name.clone()));
                }

                // HACK: Built-in types (File) are opaque pointers
                if name == "File" {
                    // File handle cannot be deeply cloned easily. Return shallow copy (pointer).
                    return Ok(val);
                } else if name == "Path" {
                    // Shallow copy for Path
                    return Ok(val);
                } else if name == "Env" || name == "Http" {
                    // Virtual static classes or opaque
                    return Ok(val);
                }

                // Reference Semantics for Structs (Shared Ownership)
                // Instead of deep copying (malloc + loop), we treat structs like RefCounted objects (like Tensors).
                // We acquire a reference and return the same pointer.
                
                // Check if it is a ZST (Value Type)
                if !val.is_pointer_value() {
                    return Ok(val);
                }

                // 1. Acquire reference
                let acquire_fn = self
                    .module
                    .get_function("tl_ptr_acquire")
                    .ok_or("tl_ptr_acquire not found")?;

                let ptr = val.into_pointer_value();
                let void_ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
                let cast_ptr = self
                    .builder
                    .build_pointer_cast(ptr, void_ptr_type, "cast_struct_ptr")
                    .unwrap();

                 self.builder
                    .build_call(acquire_fn, &[cast_ptr.into()], "")
                    .map_err(|e| e.to_string())?;

                // 2. Return SAME pointer
                return Ok(val);

                /* 
                 * DEPRECATED: Deep Copy Logic (removed)
                 * This caused massive leaks & trace traps because copies were unregistered.
                 */
                #[allow(unreachable_code)]
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

                // Calculate size manually for correct alignment/type (i64)
                let size_ptr = unsafe {
                    self.builder.build_gep(
                        st_llvm_ty,
                        self.context.ptr_type(inkwell::AddressSpace::default()).const_null(),
                        &[self.context.i64_type().const_int(1, false)],
                        "size_ptr",
                    ).map_err(|e| e.to_string())?
                };
                let size = self.builder
                    .build_ptr_to_int(size_ptr, self.context.i64_type(), "size")
                    .map_err(|e| e.to_string())?;

                let malloc_fn = self.module.get_function("malloc").ok_or("malloc not found")?;
                let new_struct_ptr_val = match self.builder
                    .build_call(malloc_fn, &[size.into()], &format!("copy_{}", name))
                    .map_err(|e| e.to_string())?
                    .try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
                        _ => return Err("malloc returned void".into()),
                    };

                // Cast if necessary (malloc returns i8* usually/void* so cast to struct*)
                let new_struct_ptr = new_struct_ptr_val; // Inkwell pointer is typed/untyped depending on version but GEP needs type

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
                        | Type::Enum(_, _)
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
                // Ensure size is i64
                let size = if size.get_type() == self.context.i32_type() {
                    self.builder.build_int_z_extend(size, self.context.i64_type(), "size_i64").unwrap()
                } else {
                    size
                };

                let malloc_fn = self
                    .module
                    .get_function("malloc")
                    .ok_or("malloc not found")?;
                let new_tuple_ptr_val = self
                    .builder
                    .build_call(malloc_fn, &[size.into()], "tuple_malloc")
                    .map_err(|e| e.to_string())?;
                let raw_ptr = match new_tuple_ptr_val.try_as_basic_value() {
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
            // Manual malloc(i64)
            let size_ptr = unsafe {
                self.builder.build_gep(
                    enum_ty,
                    self.context.ptr_type(inkwell::AddressSpace::default()).const_null(),
                    &[self.context.i64_type().const_int(1, false)],
                    "size_ptr",
                ).map_err(|e| e.to_string())?
            };
            let size = self.builder
                .build_ptr_to_int(size_ptr, self.context.i64_type(), "size")
                .map_err(|e| e.to_string())?;

            let malloc_fn = self.module.get_function("malloc").ok_or("malloc not found")?;
            let new_ptr = match self.builder
                .build_call(malloc_fn, &[size.into()], &format!("copy_{}", name))
                .map_err(|e| e.to_string())?
                .try_as_basic_value() {
                    inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
                    _ => return Err("malloc returned void".into()),
                };

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

            let field_types_list = match &variant.kind {
                crate::compiler::ast::VariantKind::Unit => vec![],
                crate::compiler::ast::VariantKind::Tuple(types) => types.clone(),
                crate::compiler::ast::VariantKind::Struct(fields) => fields.iter().map(|(_, t)| t.clone()).collect(),
            };

            if !field_types_list.is_empty() {
                // Reconstruct field types for GEP/Load/Store
                let mut field_types: Vec<inkwell::types::BasicTypeEnum> = vec![];
                for ty in &field_types_list {
                    let llvm_ty = match ty {
                        Type::F32 => self.context.f32_type().into(),
                        Type::I64 => self.context.i64_type().into(),
                        Type::Bool => self.context.bool_type().into(),
                        Type::Tensor(_, _) => self
                            .context
                            .ptr_type(inkwell::AddressSpace::default())
                            .into(),
                        Type::Struct(_, _) | Type::Enum(_, _) | Type::Tuple(_) | Type::String(_) => self
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
                for (idx, f_ty) in field_types_list.iter().enumerate() {
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
