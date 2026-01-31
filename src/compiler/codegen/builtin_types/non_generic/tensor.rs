use crate::compiler::codegen::type_manager::{CodeGenType, TypeManager};
use crate::compiler::codegen::CodeGenerator;
use crate::compiler::ast::{Type, Expr, ExprKind};
use inkwell::values::{BasicValueEnum, ValueKind, BasicValue};



pub fn register_tensor_types(manager: &mut TypeManager) {
    let mut tensor = CodeGenType::new("Tensor");
    
    // Unevaluated static methods (for literal optimizations)
    tensor.register_unevaluated_static_method("zeros", compile_tensor_zeros);
    tensor.register_unevaluated_static_method("ones", compile_ones);
    tensor.register_unevaluated_static_method("randn", compile_randn);
    
    // Evaluated static methods
    tensor.register_evaluated_static_method("load", compile_load_tensor);
    tensor.register_evaluated_static_method("clear_grads", compile_clear_grads);
    tensor.register_evaluated_static_method("from_vec_u8", compile_from_vec_u8);

    manager.register_type(tensor);
}

fn compile_clear_grads<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if !args.is_empty() { return Err("Tensor::clear_grads takes no arguments".into()); }
    let fn_val = codegen.module.get_function("tl_clear_grads").ok_or("tl_clear_grads not found")?;
    codegen.builder.build_call(fn_val, &[], "clear_grads").map_err(|e| e.to_string())?;
    Ok((codegen.context.i64_type().const_int(0, false).into(), Type::Void))
}

fn compile_from_vec_u8<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
     // Tensor::from_vec_u8(vec, offset, shape, rank) -> Tensor
    // Arguments are already evaluated by GenericResolver (evaluated args mode)
    // args[0]: vec (Vec<U8>)
    // args[1]: offset (I64)
    // args[2]: shape (ScalarArray I64)
    // args[3]: rank (I64)
    
    if args.len() != 4 {
        return Err("Tensor::from_vec_u8 requires 4 arguments (vec, offset, shape, rank)".into());
    }
    
    // Check types?
    // args[0] is Vec<U8> (pointer)
    // args[1] is I64
    // args[2] is ScalarArray pointer
    // args[3] is I64
    
    let vec_val = args[0].0;
    let offset_val = args[1].0;
    let shape_val = args[2].0;
    let rank_val = args[3].0;
    
    let fn_val = codegen.module.get_function("tl_tensor_from_vec_u8").ok_or("tl_tensor_from_vec_u8 not found")?;
    
    let call = codegen.builder.build_call(fn_val, &[
        vec_val.into(),
        offset_val.into(),
        shape_val.into(),
        rank_val.into()
    ], "from_vec_res").map_err(|e| e.to_string())?;
    
    let v = codegen.check_tensor_result(call, "from_vec_error")?;
    
    // We don't know the exact rank at compile time easily unless we parse the literal.
    // However, if we assume generic Tensor type for return, we need a Type::Tensor(Box::new(Type::F32), rank).
    // The previous implementation used args[3] (rank expression) to determine compile-time rank?
    // Actually the previous implementation in compile_static_method_call was accessing args[3] via compile_expr.
    // Here we have evaluated values. We can't easily extract compile-time constant from runtime value without optimization.
    // BUT, usually signatures like from_vec_u8 are used in specific contexts where we might rely on Type::Tensor(..., 0) as dynamic?
    // Or we should trust the user provided rank?
    // Wait, GenericResolver evaluates args. If rank is constant, we might not see it here as constant easily.
    // However, TypeManager return type inference usually relies on consistent types.
    // For now, let's assume unknown rank (0) or try to adhere to signature?
    // The previous code returned `Type::Tensor(Box::new(Type::F32), rank_val_as_int)`?
    // No, previous code was evaluating expressions.
    // Let's assume dynamic rank (0) or 1 for now, as from_vec_u8 is low-level.
    // Correct approach: Return Type::Tensor(F32, 0) which means "Any Rank" or "Dynamic"?
    // Or maybe we should improve this later. For now, 0 or 1.
    Ok((v, Type::Tensor(Box::new(Type::F32), 0)))
}

fn compile_load_tensor<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen.module.get_function("tl_tensor_load").ok_or("tl_tensor_load not found")?;
    let (path_val, _) = &args[0];
    let call = codegen
        .builder
        .build_call(fn_val, &[(*path_val).into()], "load_res")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid load return".into()),
    };
    // Return rank 1? Or dynamic? Most loaded tensors have specific valid ranks.
    // Existing code returned rank 1.
    Ok((res, Type::Tensor(Box::new(Type::F32), 1)))
}

fn compile_tensor_creation_helper<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
    runtime_func_name: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.is_empty() {
        return Err(format!("{} requires shape", runtime_func_name));
    }
    let shape_expr = &args[0];
    let (rank, shape_vals) = match &shape_expr.inner {
        ExprKind::TensorLiteral(el) | ExprKind::TensorConstLiteral(el) => {
            let mut vals = Vec::new();
            for e in el {
                let (v, t) = codegen.compile_expr(e)?;
                let int_val = match t {
                    Type::I64 => v.into_int_value(),
                    Type::I32 => codegen
                        .builder
                        .build_int_z_extend(
                            v.into_int_value(),
                            codegen.context.i64_type(),
                            "dim_ext",
                        )
                        .map_err(|e| e.to_string())?,
                    _ => return Err(format!("Dimension must be integer, found {:?}", t)),
                };
                vals.push(int_val);
            }
            (el.len(), vals)
        }
        _ => {
            return Err(format!(
                "{} currently requires array literal [dim, ...] for shape",
                runtime_func_name
            ));
        }
    };
    let requires_grad = if args.len() > 1 {
        match &args[1].inner {
            ExprKind::Bool(b) => *b,
            _ => false,
        }
    } else {
        false
    };
    let i64_type = codegen.context.i64_type();

    let current_block = codegen.builder.get_insert_block().unwrap();
    let function = current_block.get_parent().unwrap();
    let entry_block = function.get_first_basic_block().unwrap();
    let entry_builder = codegen.context.create_builder();
    if let Some(first_instr) = entry_block.get_first_instruction() {
        entry_builder.position_before(&first_instr);
    } else {
        entry_builder.position_at_end(entry_block);
    }

    let shape_array_type = i64_type.array_type(rank as u32);
    let shape_alloca = entry_builder
        .build_alloca(shape_array_type, "shape_arr")
        .map_err(|e| e.to_string())?;

    shape_alloca
        .as_instruction_value()
        .unwrap()
        .set_alignment(16)
        .map_err(|e| e.to_string())?;

    for (i, val) in shape_vals.iter().enumerate() {
        let ptr = unsafe {
            codegen.builder.build_in_bounds_gep(
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
        codegen
            .builder
            .build_store(ptr, *val)
            .map_err(|e| e.to_string())?;
    }
    let req_grad_val = codegen
        .context
        .bool_type()
        .const_int(if requires_grad { 1 } else { 0 }, false);
    let f = codegen
        .module
        .get_function(runtime_func_name)
        .ok_or(format!("{} not found", runtime_func_name))?;

    let zero = i64_type.const_int(0, false);
    let first_elem_ptr = unsafe {
        codegen.builder.build_in_bounds_gep(
            shape_array_type,
            shape_alloca,
            &[zero, zero],
            "first_elem_ptr",
        )
    }
    .map_err(|e| e.to_string())?;

    let call = codegen
        .builder
        .build_call(
            f,
            &[
                i64_type.const_int(rank as u64, false).into(),
                first_elem_ptr.into(),
                req_grad_val.into(),
            ],
            "creation_res",
        )
        .map_err(|e| e.to_string())?;
    let res = codegen.check_tensor_result(call, "creation_error")?;

    Ok((res, Type::Tensor(Box::new(Type::F32), rank)))
}

fn compile_randn<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_creation_helper(codegen, args, "tl_tensor_randn_debug")
}

fn compile_ones<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_creation_helper(codegen, args, "tl_tensor_ones")
}

fn compile_tensor_zeros<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.is_empty() {
        return Err("Tensor::zeros requires shape argument".into());
    }

    let elements_ref = if let ExprKind::TensorLiteral(el) = &args[0].inner {
        Some(el)
    } else if let ExprKind::TensorConstLiteral(el) = &args[0].inner {
        Some(el)
    } else {
        None
    };

    if let Some(el) = elements_ref {
        let i64_type = codegen.context.i64_type();
        let mut vals = Vec::new();
        for e in el {
            let (v, t) = codegen.compile_expr(e)?;
            let int_val = match t {
                Type::I64 => v.into_int_value(),
                Type::I32 => codegen
                    .builder
                    .build_int_z_extend(v.into_int_value(), i64_type, "ext")
                    .map_err(|e| e.to_string())?,
                _ => return Err(format!("Dimension must be integer, found {:?}", t)),
            };
            vals.push(int_val);
        }

        let rank = el.len();
        let shape_array_type = i64_type.array_type(rank as u32);
        let shape_alloca = codegen
            .builder
            .build_alloca(shape_array_type, "shape_arr")
            .map_err(|e| e.to_string())?;

        for (i, val) in vals.iter().enumerate() {
            let ptr = unsafe {
                codegen
                    .builder
                    .build_in_bounds_gep(
                        shape_array_type,
                        shape_alloca,
                        &[
                            i64_type.const_int(0, false),
                            i64_type.const_int(i as u64, false),
                        ],
                        "tmp",
                    )
                    .map_err(|e| e.to_string())?
            };
            codegen
                .builder
                .build_store(ptr, *val)
                .map_err(|e| e.to_string())?;
        }

        let req_grad = if args.len() > 1 {
            let (v, _) = codegen.compile_expr(&args[1])?;
            v.into_int_value()
        } else {
            codegen.context.bool_type().const_int(0, false)
        };

        let f = codegen
            .module
            .get_function("tl_tensor_zeros")
            .ok_or("tl_tensor_zeros not found")?;
        let call = codegen
            .builder
            .build_call(
                f,
                &[
                    i64_type.const_int(rank as u64, false).into(),
                    shape_alloca.into(),
                    req_grad.into(),
                ],
                "zeros_res",
            )
            .map_err(|e| e.to_string())?;

        let v = codegen.check_tensor_result(call, "zeros_error")?;
        let result_ty = Type::Tensor(Box::new(Type::F32), rank);
        // codegen.emit_register_tensor(v, &result_ty)?;
        return Ok((v, result_ty));
    }

    Err("Generic Tensor::zeros (non-literal shape) not yet ported. Please use literal shape [d1, d2] for now.".into())
}
