use super::CodeGenerator;
use crate::compiler::ast::*;
use inkwell::values::*;
use std::collections::HashMap;

impl<'ctx> CodeGenerator<'ctx> {
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

                if !obj_val.is_pointer_value() {
                    return Err("Cannot access field of non-pointer struct".into());
                }
                let ptr = obj_val.into_pointer_value();
                let st_llvm_ty = self.struct_types.get(&struct_name).unwrap();

                let field_ptr = self
                    .builder
                    .build_struct_gep(
                        st_llvm_ty.clone(),
                        ptr,
                        field_idx as u32,
                        &format!("ptr_{}", field),
                    )
                    .map_err(|e| e.to_string())?;

                let llvm_ty: inkwell::types::BasicTypeEnum = match field_ty {
                    Type::I64 => self.context.i64_type().into(),
                    Type::F32 => self.context.f32_type().into(),
                    Type::Bool => self.context.bool_type().into(),
                    Type::Tensor(_, _) | Type::Struct(_) | Type::UserDefined(_) => self
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
            }

            Expr::Variable(name) => {
                for scope in self.variables.iter().rev() {
                    if let Some((val, ty, _)) = scope.get(name) {
                        if val.is_pointer_value() {
                            let ptr = val.into_pointer_value();
                            let llvm_ty: inkwell::types::BasicTypeEnum = match ty {
                                Type::I64 => self.context.i64_type().into(),
                                Type::F32 => self.context.f32_type().into(),
                                Type::Bool => self.context.bool_type().into(),
                                Type::Tensor(_, _) | Type::Struct(_) | Type::UserDefined(_) => self
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
            Expr::StaticMethodCall(type_name, method_name, args) => {
                // 1. Resolve Mangled Name
                let mangled_name = format!("tl_{}_{}", type_name, method_name);

                // 2. Lookup Function
                let func = self
                    .module
                    .get_function(&mangled_name)
                    .or_else(|| {
                        // Try fallback for stdlib mappings (lowercase)
                        self.module.get_function(&format!(
                            "tl_{}_{}",
                            type_name.to_lowercase(),
                            method_name
                        ))
                    })
                    .ok_or(format!(
                        "Static method {}::{} not found (mangled: {})",
                        type_name, method_name, mangled_name
                    ))?;

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
                // Look up return type from semantic info?
                // Or simplified: Just check what `try_as_basic_value` gives.
                // But we need the Type enum for further compilation.
                // We should really get this from the FunctionDef/StructDef.
                // For now, let's rely on declared return types map or infer?
                // `semantics.rs` already checked it.
                // We need to know specific return Type to return (BasicValueEnum, Type).

                // Hack: Look up in `self.fn_return_types` using mangled name?
                // We should register these in `compile_module` when processing impls.
                let ret_ty = self
                    .fn_return_types
                    .get(&mangled_name)
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
            Expr::BinOp(lhs, op, rhs) => {
                let left = self.compile_expr(lhs)?;
                let right = self.compile_expr(rhs)?;
                self.compile_bin_op(left.0, left.1, right.0, right.1, op.clone())
            }
            Expr::TensorLiteral(elements) => {
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
                let data_alloca = self
                    .builder
                    .build_array_alloca(
                        f32_type,
                        i64_type.const_int(total_elements as u64, false),
                        "tensor_data",
                    )
                    .map_err(|e| e.to_string())?;

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
                        _ => {
                            return Err(format!("Tensor element must be numeric, got {:?}", val_ty))
                        }
                    };

                    // Get pointer to element position
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
                let shape_alloca = self
                    .builder
                    .build_array_alloca(
                        i64_type,
                        i64_type.const_int(rank as u64, false),
                        "tensor_shape",
                    )
                    .map_err(|e| e.to_string())?;

                for (i, dim) in shape.iter().enumerate() {
                    let ptr = unsafe {
                        self.builder
                            .build_in_bounds_gep(
                                i64_type,
                                shape_alloca,
                                &[i64_type.const_int(i as u64, false)],
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

                let call = self
                    .builder
                    .build_call(
                        new_fn,
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
                    _ => return Err("Invalid tl_tensor_new return".into()),
                };

                Ok((res, Type::Tensor(Box::new(Type::F32), rank)))
            }
            Expr::TensorConstLiteral(elements) => {
                // Optimized path for constant tensor literals - static extraction
                fn flatten_const(exprs: &[Expr]) -> Result<(Vec<f64>, Vec<usize>), String> {
                    if exprs.is_empty() {
                        return Ok((vec![], vec![0]));
                    }

                    let is_nested = matches!(
                        exprs[0],
                        Expr::TensorConstLiteral(_) | Expr::TensorLiteral(_)
                    );
                    if is_nested {
                        let mut flat_data = Vec::new();
                        let mut first_shape = None;

                        for e in exprs {
                            let (children, shape) = match e {
                                Expr::TensorConstLiteral(c) | Expr::TensorLiteral(c) => {
                                    let (data, s) = flatten_const(c)?;
                                    (data, s)
                                }
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
                        }

                        let mut shape = vec![exprs.len()];
                        if let Some(s) = first_shape {
                            shape.extend(s);
                        }
                        Ok((flat_data, shape))
                    } else {
                        let mut data = Vec::new();
                        for e in exprs {
                            match e {
                                Expr::Float(f) => data.push(*f),
                                Expr::Int(i) => data.push(*i as f64),
                                _ => return Err("Const tensor must contain only literals".into()),
                            }
                        }
                        Ok((data, vec![exprs.len()]))
                    }
                }

                let (flat_data, shape) = flatten_const(elements)?;
                let rank = shape.len();
                let len = flat_data.len() as u64;

                let f32_type = self.context.f32_type();
                let i64_type = self.context.i64_type();

                // Allocate and store static data
                let data_alloca = self
                    .builder
                    .build_array_alloca(
                        f32_type,
                        i64_type.const_int(len, false),
                        "const_tensor_data",
                    )
                    .map_err(|e| e.to_string())?;

                for (i, val) in flat_data.iter().enumerate() {
                    let float_val = f32_type.const_float(*val);
                    let ptr = unsafe {
                        self.builder
                            .build_in_bounds_gep(
                                f32_type,
                                data_alloca,
                                &[i64_type.const_int(i as u64, false)],
                                "const_elem_ptr",
                            )
                            .map_err(|e| e.to_string())?
                    };
                    self.builder
                        .build_store(ptr, float_val)
                        .map_err(|e| e.to_string())?;
                }

                // Allocate and store shape
                let shape_alloca = self
                    .builder
                    .build_array_alloca(
                        i64_type,
                        i64_type.const_int(rank as u64, false),
                        "const_tensor_shape",
                    )
                    .map_err(|e| e.to_string())?;

                for (i, dim) in shape.iter().enumerate() {
                    let ptr = unsafe {
                        self.builder
                            .build_in_bounds_gep(
                                i64_type,
                                shape_alloca,
                                &[i64_type.const_int(i as u64, false)],
                                "const_shape_ptr",
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

                let call = self
                    .builder
                    .build_call(
                        new_fn,
                        &[
                            data_alloca.into(),
                            i64_type.const_int(rank as u64, false).into(),
                            shape_alloca.into(),
                        ],
                        "new_const_tensor",
                    )
                    .map_err(|e| e.to_string())?;

                let res = match call.try_as_basic_value() {
                    ValueKind::Basic(v) => v,
                    _ => return Err("Invalid tl_tensor_new return".into()),
                };

                Ok((res, Type::Tensor(Box::new(Type::F32), rank)))
            }
            Expr::MethodCall(obj, method, args) => {
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

                    let (func_val, final_name) =
                        if let Some(f) = self.module.get_function(&mangled_name) {
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
                            ValueKind::Basic(v) => Ok((v, ret_ty)),
                            _ => Err("Invalid return value".into()),
                        }
                    }
                } else {
                    match method.as_str() {
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
                            let fn_val = self
                                .module
                                .get_function("tl_tensor_detach")
                                .ok_or("Runtime fn tl_tensor_detach not found")?;

                            let mut compiled_args = Vec::with_capacity(args.len() + 1);
                            compiled_args.push(obj_val.into());

                            // Check args. If arg provided, use it for req_grad. Default true for params?
                            // detach() -> requires_grad=false.
                            // detach(true) -> requires_grad=true.
                            // But method sig in parser puts args in `args`.
                            // If `args` has 1 element, use it.
                            if args.len() >= 1 {
                                let (val, _) = self.compile_expr(&args[0])?;
                                compiled_args.push(val.into());
                            } else {
                                // Default false? step uses `(w - g*lr).detach()`.
                                // We want the new weight to require grad for NEXT iter.
                                // So default true?
                                // Candle: `detach()` creates new tensor sharing storage. `requires_grad` is false.
                                // We want to update parameter `self.W`.
                                // Next iter `backward` needs `self.W` to require grad.
                                // So we MUST set `req_grad=true`.
                                // Let's make it mandatory or check logic.
                                // In `train_add.tl`: `(self.W - g*lr).detach()`.
                                // We likely want `detach(true)`.

                                // Let's default to `true` for now in `tl` or enforce explicit?
                                // Let's default to `true` because it's mostly used for param update.
                                // Wait, usually `detach()` means "stop gradient".
                                // But here we use it to "reset graph but keep leaf status".
                                // So `true` makes sense for parameters.
                                let true_val = self.context.bool_type().const_int(1, false);
                                compiled_args.push(true_val.into());
                            }

                            let call = self
                                .builder
                                .build_call(fn_val, &compiled_args, "detach_res")
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
                            let res = match call.try_as_basic_value() {
                                ValueKind::Basic(v) => v,
                                _ => return Err("Invalid grad return".into()),
                            };
                            // Assuming grad has same rank as obj, but for now just opaque tensor
                            Ok((res, obj_ty))
                        }
                        "reshape" => {
                            if args.len() != 1 {
                                return Err("reshape method requires 1 argument (shape)".into());
                            }
                            let (s_val, _) = self.compile_expr(&args[0])?;
                            let reshape_fn = self.module.get_function("tl_tensor_reshape").unwrap();
                            let call = self
                                .builder
                                .build_call(
                                    reshape_fn,
                                    &[obj_val.into(), s_val.into()],
                                    "reshape_res",
                                )
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
                        "add_assign" | "sub_assign" | "mul_assign" | "div_assign" => {
                            if args.len() != 1 {
                                return Err(format!("{} requires 1 argument", method));
                            }
                            // Must use ensure_tensor for RHS
                            let rhs_val = self.ensure_tensor_v2(&args[0], 0)?;

                            let fn_name = match method.as_str() {
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
                                let method_name = method.clone(); // e.g. read_string
                                let runtime_fn_name =
                                    format!("tl_{}_{}", type_name.to_lowercase(), method_name);

                                let fn_val =
                                    self.module.get_function(&runtime_fn_name).ok_or(format!(
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
            Expr::FnCall(name, args) => {
                if let Some(struct_def) = self.struct_defs.get(name).cloned() {
                    let st_llvm_ty = self.struct_types.get(name).unwrap().clone();
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
                            .build_struct_gep(
                                st_llvm_ty.clone(),
                                struct_ptr,
                                i as u32,
                                "init_field",
                            )
                            .map_err(|e| e.to_string())?;
                        self.builder
                            .build_store(field_ptr, val)
                            .map_err(|e| e.to_string())?;
                    }

                    return Ok((struct_ptr.into(), Type::Struct(name.clone())));
                }

                match name.as_str() {
                    "print" => {
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
                            _ => return Err(format!("Cannot print type {:?}", arg_type)),
                        }
                        return Ok((
                            self.context.i64_type().const_int(0, false).into(),
                            Type::Void,
                        ));
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
                    "reshape" => {
                        if args.len() < 2 {
                            return Err("reshape requires at least tensor and 1 dimension".into());
                        }

                        let (t_val, t_ty) = self.compile_expr(&args[0])?;
                        if !matches!(t_ty, Type::Tensor(_, _)) {
                            return Err("First argument to reshape must be a tensor".into());
                        }

                        // Check if 2nd arg is Tensor (Old behavior)
                        let (_, arg1_ty) = self.compile_expr(&args[1])?;
                        if matches!(arg1_ty, Type::Tensor(_, _)) && args.len() == 2 {
                            let (s_val, _) = self.compile_expr(&args[1])?;
                            let reshape_fn = self
                                .module
                                .get_function("tl_tensor_reshape")
                                .ok_or("tl_tensor_reshape not found")?;
                            let call = self
                                .builder
                                .build_call(
                                    reshape_fn,
                                    &[t_val.into(), s_val.into()],
                                    "reshape_res",
                                )
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
                        return Ok((res, Type::Tensor(Box::new(Type::Void), 0)));
                    }
                    "softmax" => {
                        if args.len() != 2 {
                            return Err("softmax requires 2 arguments".into());
                        }
                        let (arg0_val, arg0_ty) = self.compile_expr(&args[0])?;
                        let (arg1_val, _arg1_ty) = self.compile_expr(&args[1])?; // dim

                        if !matches!(arg0_ty, Type::Tensor(_, _)) {
                            return Err("softmax arg0 must be tensor".into());
                        }
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
                        let (arg0_val, arg0_ty) = self.compile_expr(&args[0])?;
                        let (arg1_val, arg1_ty) = self.compile_expr(&args[1])?;

                        if !matches!(arg0_ty, Type::Tensor(_, _)) {
                            return Err("cross_entropy arg0 must be tensor".into());
                        }
                        if !matches!(arg1_ty, Type::Tensor(_, _)) {
                            return Err("cross_entropy arg1 must be tensor".into());
                        }

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
                        let (arg_val, arg_ty) = self.compile_expr(&args[0])?;
                        if !matches!(arg_ty, Type::Tensor(_, _)) {
                            return Err("exp requires a tensor".into());
                        }
                        let fn_val = self.module.get_function("tl_tensor_exp").unwrap();
                        let call = self
                            .builder
                            .build_call(fn_val, &[arg_val.into()], "exp_res")
                            .map_err(|e| e.to_string())?;
                        let res = match call.try_as_basic_value() {
                            ValueKind::Basic(v) => v,
                            _ => return Err("Invalid exp return".into()),
                        };
                        return Ok((res, arg_ty));
                    }
                    "log" => {
                        if args.len() != 1 {
                            return Err("log requires 1 argument".into());
                        }
                        let (arg_val, arg_ty) = self.compile_expr(&args[0])?;
                        let fn_val = self.module.get_function("tl_tensor_log").unwrap();
                        let call = self
                            .builder
                            .build_call(fn_val, &[arg_val.into()], "log_res")
                            .unwrap();
                        let res = match call.try_as_basic_value() {
                            ValueKind::Basic(v) => v,
                            _ => return Err("Invalid log return".into()),
                        };
                        return Ok((res, arg_ty));
                    }
                    "len" => {
                        if args.len() != 1 {
                            return Err("len requires 1 argument".into());
                        }
                        let (arg_val, arg_ty) = self.compile_expr(&args[0])?;
                        if !matches!(arg_ty, Type::Tensor(_, _)) {
                            return Err("len requires a tensor".into());
                        }
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
                        return Ok((res, Type::I64));
                    }

                    "sqrt" => {
                        if args.len() != 1 {
                            return Err("sqrt requires 1 argument".into());
                        }
                        let (arg_val, arg_ty) = self.compile_expr(&args[0])?;
                        if !matches!(arg_ty, Type::Tensor(_, _)) {
                            return Err("sqrt requires a tensor".into());
                        }
                        let fn_val = self.module.get_function("tl_tensor_sqrt").unwrap();
                        let call = self
                            .builder
                            .build_call(fn_val, &[arg_val.into()], "sqrt_res")
                            .map_err(|e| e.to_string())?;
                        let res = match call.try_as_basic_value() {
                            ValueKind::Basic(v) => v,
                            _ => return Err("Invalid sqrt return".into()),
                        };
                        return Ok((res, arg_ty));
                    }
                    "matmul" => {
                        if args.len() != 2 {
                            return Err("matmul requires 2 arguments".into());
                        }
                        let (lhs_val, lhs_ty) = self.compile_expr(&args[0])?;
                        let (rhs_val, rhs_ty) = self.compile_expr(&args[1])?;
                        if !matches!(lhs_ty, Type::Tensor(_, _))
                            || !matches!(rhs_ty, Type::Tensor(_, _))
                        {
                            return Err("matmul requires tensors".into());
                        }
                        let fn_val = self.module.get_function("tl_tensor_matmul").unwrap();
                        let call = self
                            .builder
                            .build_call(fn_val, &[lhs_val.into(), rhs_val.into()], "matmul_res")
                            .map_err(|e| e.to_string())?;
                        let res = match call.try_as_basic_value() {
                            ValueKind::Basic(v) => v,
                            _ => return Err("Invalid matmul return".into()),
                        };
                        // Only supporting basic tensor type propagation for now
                        return Ok((res, lhs_ty));
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
                        return Ok((res, arg_ty));
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
                        return Ok((
                            self.context.i64_type().const_int(0, false).into(),
                            Type::Void,
                        ));
                    }
                    "sum" => {
                        if args.len() == 1 {
                            // Global sum
                            let (arg_val, _arg_ty) = self.compile_expr(&args[0])?;
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
                            return Ok((res, Type::Tensor(Box::new(Type::F32), 1)));
                        } else if args.len() == 2 {
                            // Sum over dim: sum(t, dim)
                            let (t_val, _t_ty) = self.compile_expr(&args[0])?;
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
                            return Ok((res, Type::Tensor(Box::new(Type::F32), 1)));
                        } else {
                            return Err("sum requires 1 or 2 arguments".into());
                        }
                    }
                    "sin" | "cos" | "relu" | "gelu" => {
                        if args.len() != 1 {
                            return Err(format!("{} requires 1 argument", name).into());
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
                            _ => return Err(format!("Invalid {} return", name).into()),
                        };
                        return Ok((res, Type::Tensor(Box::new(Type::F32), 1)));
                    }
                    "tril" => {
                        if args.len() != 2 {
                            return Err("tril requires 2 arguments".into());
                        }
                        let (t_val, _) = self.compile_expr(&args[0])?;
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
                        return Ok((res, Type::Tensor(Box::new(Type::F32), 1)));
                    }
                    "embedding" => {
                        if args.len() != 2 {
                            return Err("embedding requires 2 arguments".into());
                        }
                        let (idx_val, _) = self.compile_expr(&args[0])?;
                        let (w_val, _) = self.compile_expr(&args[1])?;

                        let fn_val = self.module.get_function("tl_tensor_embedding").unwrap();
                        let call = self
                            .builder
                            .build_call(fn_val, &[idx_val.into(), w_val.into()], "emb_res")
                            .map_err(|e| e.to_string())?;
                        let res = match call.try_as_basic_value() {
                            ValueKind::Basic(v) => v,
                            _ => return Err("Invalid embedding return".into()),
                        };
                        return Ok((res, Type::Tensor(Box::new(Type::F32), 1)));
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
                        let llvm_func_name = match name.as_str() {
                            "slice" => "tl_tensor_slice",
                            "sum" => "tl_tensor_sum",
                            "randn" => {
                                // randn(shape, requires_grad)
                                // Handle specially to pass rank/shape pointer and bool
                                // args[0] must be tensor (shape) or literal?
                                // Actually, user might pass [10, 10].
                                // But `compile_expr` for TensorLiteral returns a Tensor pointer.
                                // We need shape as generic array?
                                // Existing `tl_tensor_new` logic handled parsing TensorLiteral manually to create C-array.
                                // But here `args[0]` is an Expr.
                                // If it's a TensorLiteral, we can do similar logic.
                                // If it's a variable, it is a Tensor*.
                                // `tl_tensor_randn` needs `rank, shape_ptr`.
                                // Let's support only Literal Shape for now for simplicity, OR
                                // use a version of randn that takes a shape TENSOR.
                                // To match `tl_tensor_new`, we need raw shape.
                                // Let's assume usage: let x = randn([10, 20], true);
                                // The parser gives `TensorLiteral`.
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
                                        return Err(
                                            "randn currently requires array literal [dim, ...] for shape".into(),
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
                                let usize_type = self.context.i64_type(); // usize is 64-bit

                                // Stack allocate shape array
                                let current_block = self.builder.get_insert_block().unwrap();
                                let entry = current_block
                                    .get_parent()
                                    .unwrap()
                                    .get_first_basic_block()
                                    .unwrap();
                                self.builder.position_at_end(entry);

                                let shape_array_type = usize_type.array_type(rank as u32);
                                let shape_alloca = self
                                    .builder
                                    .build_alloca(shape_array_type, "shape_arr")
                                    .map_err(|e| e.to_string())?;

                                self.builder.position_at_end(current_block);

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

            Expr::Aggregation {
                op,
                expr,
                var,
                range,
                condition,
            } => {
                // For now, implement a simple version:
                // Assume range is a tensor/array and we iterate over its length
                // sum(arr[i] for i in arr) -> loop over arr indices

                let function = self
                    .builder
                    .get_insert_block()
                    .unwrap()
                    .get_parent()
                    .unwrap();

                // Get range expression to determine loop bounds
                let (range_val, range_ty) = self.compile_expr(range)?;

                // For Tensor types, get the length from the first dimension
                let loop_count = match &range_ty {
                    Type::Tensor(_, _) => {
                        let len_fn = self
                            .module
                            .get_function("tl_tensor_len")
                            .ok_or("tl_tensor_len not found")?;
                        let call = self
                            .builder
                            .build_call(len_fn, &[range_val.into()], "len")
                            .map_err(|e| e.to_string())?;
                        match call.try_as_basic_value() {
                            ValueKind::Basic(v) => v.into_int_value(),
                            _ => return Err("Failed to get tensor length".into()),
                        }
                    }
                    _ => return Err("Aggregation range must be a tensor".into()),
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

                // Determine element type from range_ty
                let elem_ty = match &range_ty {
                    Type::Tensor(t, _) => *t.clone(),
                    _ => Type::I64, // Fallback (should be checked earlier)
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
                    Type::F32 => val_f32.into(),
                    _ => return Err("Unsupported tensor element type for aggregation".into()),
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
            Type::Tensor(_, _) | Type::UserDefined(_) | Type::Struct(_) => self
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

    pub(crate) fn extract_index_bounds(
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
}
