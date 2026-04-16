//! codegen/expr/tensor_methods.rs
//!
//! Tensor 操作のコンパイル関数群。
//! compile_tensor_get, compile_tensor_backward, compile_tensor_sum 等。
use crate::compiler::error::TlError;

use inkwell::values::*;

use crate::compiler::ast::*;
use crate::compiler::codegen::CodeGenerator;


pub(super) fn compile_tensor_get<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    // args: index1, index2, ...
    if args.is_empty() {
        return Err("get requires at least 1 argument (index...)".into());
    }
    let rank = args.len();

    // Create array of indices on stack
    // indices: *const i64
    let i64_type = codegen.context.i64_type();
    let index_array_type = i64_type.array_type(rank as u32);

    // Move to entry block for alloca to avoid stack overflow in loops
    let current_block = codegen.current_block()?;
    let func = current_block.get_parent().ok_or_else(|| "block has no parent function".to_string())?;
    let entry_block = func.get_first_basic_block().ok_or_else(|| "function has no entry block".to_string())?;

    if let Some(first_inst) = entry_block.get_first_instruction() {
        codegen.builder.position_before(&first_inst);
    } else {
        codegen.builder.position_at_end(entry_block);
    }

    let index_array_ptr = codegen
        .builder
        .build_alloca(index_array_type, "index_array")
        .map_err(|e| e.to_string())?;

    // Move back to current block
    codegen.builder.position_at_end(current_block);

    for i in 0..rank {
        let (idx_val, idx_ty) = args[i].clone();
        let idx_i64 = match idx_ty {
            crate::compiler::ast::Type::I64 => idx_val.into_int_value(),
            crate::compiler::ast::Type::I32 => codegen
                .builder
                .build_int_z_extend(idx_val.into_int_value(), i64_type, "idx_ext")
                .map_err(|e| e.to_string())?,
            _ => return Err(format!("Index {} must be integer", i).into()),
        };

        let ptr = unsafe {
            codegen
                .builder
                .build_gep(
                    index_array_type,
                    index_array_ptr,
                    &[
                        codegen.context.i64_type().const_int(0, false),
                        codegen.context.i64_type().const_int(i as u64, false),
                    ],
                    "idx_ptr",
                )
                .map_err(|e| e.to_string())?
        };
        codegen
            .builder
            .build_store(ptr, idx_i64)
            .map_err(|e| e.to_string())?;
    }

    // Cast array ptr to i64 ptr (use i8 ptr generic)
    let indices_ptr = codegen
        .builder
        .build_pointer_cast(
            index_array_ptr,
            codegen.context.ptr_type(inkwell::AddressSpace::default()),
            "indices_ptr_cast",
        )
        .map_err(|e| e.to_string())?;

    let fn_val = codegen
        .module
        .get_function("tl_tensor_get_f32_md")
        .ok_or("tl_tensor_get_f32_md not found")?;

    let rank_val = codegen.context.i64_type().const_int(rank as u64, false);

    let call = codegen
        .builder
        .build_call(
            fn_val,
            &[obj_val.into(), indices_ptr.into(), rank_val.into()],
            "get_res",
        )
        .map_err(|e| e.to_string())?;

    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid get return".into()),
    };

    Ok((res, crate::compiler::ast::Type::F32))
}

pub(super) fn compile_tensor_backward<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = codegen.get_fn("tl_tensor_backward")?;
    codegen
        .builder
        .build_call(fn_val, &[obj_val.into()], "backward_call")
        .map_err(|e| e.to_string())?;

    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

pub(super) fn compile_tensor_clone<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = codegen.get_fn("tl_tensor_clone")?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_val.into()], "clone_res")
        .map_err(|e| e.to_string())?;

    let res = codegen.check_tensor_result(call, "clone_error")?;

    // Runtime already registers the tensor (Ref=1). We just track it as a temporary.
    Ok((res, obj_ty))
}

pub(super) fn compile_tensor_detach<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = codegen.get_fn("tl_tensor_detach")?;
    // Optional arg: req_grad (bool). Default to false.
    let req_grad = if !args.is_empty() {
        let (arg_val, _) = args[0].clone();
        arg_val.into_int_value()
    } else {
        codegen.context.bool_type().const_int(0, false)
    };

    let call = codegen
        .builder
        .build_call(fn_val, &[obj_val.into(), req_grad.into()], "detach_res")
        .map_err(|e| e.to_string())?;

    let res = codegen.check_tensor_result(call, "detach_error")?;

    // codegen.emit_register_tensor(res, &obj_ty)?;
    Ok((res, obj_ty))
}

pub(super) fn compile_tensor_grad<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = codegen.get_fn("tl_tensor_grad")?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_val.into()], "grad_res")
        .map_err(|e| e.to_string())?;

    let res = codegen.check_tensor_result(call, "grad_error")?;

    // codegen.emit_register_tensor(res, &obj_ty)?;
    Ok((res, obj_ty))
}

pub(super) fn compile_tensor_contiguous<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = codegen.get_fn("tl_tensor_contiguous")?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_val.into()], "contiguous_res")
        .map_err(|e| e.to_string())?;

    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid contiguous return".into()),
    };

    // codegen.emit_register_tensor(res, &obj_ty)?;
    Ok((res, obj_ty))
}

pub(super) fn compile_tensor_save<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 {
        return Err("save requires 1 argument (path)".into());
    }
    let fn_val = codegen.get_fn("tl_tensor_save")?;
    let (path_val, _) = args[0].clone();

    codegen
        .builder
        .build_call(fn_val, &[path_val.into(), obj_val.into()], "save_call")
        .map_err(|e| e.to_string())?;

    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

pub(super) fn compile_tensor_sum<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.is_empty() {
        // Standard sum
        let fn_val = codegen
            .module
            .get_function("tl_tensor_sum")
            .ok_or("tl_tensor_sum not found")?;
        let call = codegen
            .builder
            .build_call(fn_val, &[obj_val.into()], "sum_res");

        let res = codegen.check_tensor_result(call.map_err(|e| e.to_string())?, "sum_error")?;
        Ok((
            res,
            crate::compiler::ast::Type::Tensor(Box::new(crate::compiler::ast::Type::F32), 0),
        ))
    } else {
        // Sum with dim
        if args.len() != 1 {
            return Err("sum takes at most 1 argument".into());
        }
        let (dim_val, dim_ty) = args[0].clone();
        let dim_i64 = match dim_ty {
            crate::compiler::ast::Type::I64 => dim_val.into_int_value(),
            crate::compiler::ast::Type::I32 => codegen
                .builder
                .build_int_z_extend(
                    dim_val.into_int_value(),
                    codegen.context.i64_type(),
                    "dim_ext",
                )
                .map_err(|e| e.to_string())?,
            _ => return Err("Dimension must be integer".into()),
        };

        let fn_val = codegen
            .module
            .get_function("tl_tensor_sum_dim")
            .ok_or("tl_tensor_sum_dim not found")?;

        let call = codegen.builder.build_call(
            fn_val,
            &[
                obj_val.into(),
                dim_i64.into(),
                codegen.context.bool_type().const_zero().into(),
            ],
            "sum_dim_res",
        );

        let res = codegen.check_tensor_result(call.map_err(|e| e.to_string())?, "sum_dim_error")?;
        // Ideally we subtract 1 from rank, but 0 is safe generic guess for now if we don't track rank strictly
        Ok((
            res,
            crate::compiler::ast::Type::Tensor(Box::new(crate::compiler::ast::Type::F32), 0),
        ))
    }
}
/// .slice(start, len) — グローバル関数版
/// FFI: tl_tensor_slice(t, dim=0, start, end=start+len, step=1)
pub(super) fn compile_tensor_slice2<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let i64_ty = codegen.context.i64_type();

    let dim = i64_ty.const_int(0, false);
    let start = args[0].0.into_int_value();
    let len = args[1].0.into_int_value();
    let end = codegen.builder.build_int_add(start, len, "slice_end")
        .map_err(|e| e.to_string())?;
    let step = i64_ty.const_int(1, false);

    let fn_val = codegen.get_fn("tl_tensor_slice")?;
    let call = codegen.builder
        .build_call(fn_val, &[obj_val.into(), dim.into(), start.into(), end.into(), step.into()], "slice_res")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid slice return".into()),
    };
    Ok((res, obj_ty))
}

/// .slice(dim, start, len) — グローバル関数版
/// FFI: tl_tensor_slice(t, dim, start, end=start+len, step=1)
#[allow(dead_code)]
pub(super) fn compile_tensor_slice3<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let i64_ty = codegen.context.i64_type();

    let dim = args[0].0;
    let start = args[1].0.into_int_value();
    let len = args[2].0.into_int_value();
    let end = codegen.builder.build_int_add(start, len, "slice_end")
        .map_err(|e| e.to_string())?;
    let step = i64_ty.const_int(1, false);

    let fn_val = codegen.get_fn("tl_tensor_slice")?;
    let call = codegen.builder
        .build_call(fn_val, &[obj_val.into(), dim.into(), start.into(), end.into(), step.into()], "slice_res")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid slice return".into()),
    };
    Ok((res, obj_ty))
}




pub(super) fn compile_tensor_to<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 {
        return Err("to/to_device requires 1 argument (device name string)".into());
    }
    let (dev_val, dev_ty) = args[0].clone();
    if !matches!(&dev_ty, Type::String(_)) {
        return Err("Device name must be a string".into());
    }

    let fn_val = codegen
        .module
        .get_function("tl_tensor_to_device")
        .ok_or("Runtime fn tl_tensor_to_device not found")?;

    let call = codegen
        .builder
        .build_call(fn_val, &[obj_val.into(), dev_val.into()], "to_dev_res")
        .map_err(|e| e.to_string())?;

    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from to_device".into()),
    };

    Ok((res, obj_ty))
}

/// `add_assign` / `sub_assign` / `mul_assign` / `div_assign` の共通実装。
///
/// - テンソル同士: `tl_tensor_{op}_assign(obj, rhs)` を呼ぶ。
/// - スカラー:   rhs を f32 に変換して `tl_tensor_{op}_assign_scalar_f32(obj, scalar)` を呼ぶ。
pub(super) fn compile_tensor_assign_op<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    op: &str, // "add" | "sub" | "mul" | "div"
    obj_val: BasicValueEnum<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 {
        return Err(format!("{}_assign requires 1 argument", op).into());
    }
    let (rhs_val, rhs_ty) = args[0].clone();

    if matches!(rhs_ty, Type::Tensor(_, _) | Type::GradTensor(_, _)) {
        let ffi_name = format!("tl_tensor_{}_assign", op);
        let fn_val = codegen
            .module
            .get_function(&ffi_name)
            .ok_or_else(|| format!("{} not found in module", ffi_name))?;
        codegen
            .builder
            .build_call(fn_val, &[obj_val.into(), rhs_val.into()], "assign_res")
            .map_err(|e| e.to_string())?;
    } else if matches!(rhs_ty, Type::F32 | Type::F64 | Type::I64 | Type::I32) {
        let scalar_f32 = match rhs_ty {
            Type::F32 => rhs_val.into_float_value(),
            Type::F64 => codegen
                .builder
                .build_float_cast(
                    rhs_val.into_float_value(),
                    codegen.context.f32_type(),
                    "f64_to_f32",
                )
                .map_err(|e| e.to_string())?,
            Type::I64 | Type::I32 => codegen
                .builder
                .build_signed_int_to_float(
                    rhs_val.into_int_value(),
                    codegen.context.f32_type(),
                    "int_to_f32",
                )
                .map_err(|e| e.to_string())?,
            _ => return Err(format!("{}_assign scalar: unsupported type {:?}", op, rhs_ty).into()),
        };
        let ffi_name = format!("tl_tensor_{}_assign_scalar_f32", op);
        let fn_val = codegen
            .module
            .get_function(&ffi_name)
            .ok_or_else(|| format!("{} not found in module", ffi_name))?;
        codegen
            .builder
            .build_call(fn_val, &[obj_val.into(), scalar_f32.into()], "assign_res")
            .map_err(|e| e.to_string())?;
    } else {
        return Err(format!(
            "{}_assign requires Tensor or scalar argument, got {:?}",
            op, rhs_ty
        ).into());
    }

    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

pub(super) fn compile_tensor_add_assign<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    compile_tensor_assign_op(codegen, "add", obj_val, args)
}

pub(super) fn compile_tensor_sub_assign<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    compile_tensor_assign_op(codegen, "sub", obj_val, args)
}

pub(super) fn compile_tensor_mul_assign<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    compile_tensor_assign_op(codegen, "mul", obj_val, args)
}

pub(super) fn compile_tensor_div_assign<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    compile_tensor_assign_op(codegen, "div", obj_val, args)
}

pub(super) fn compile_tensor_pow<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 {
        return Err("pow requires 1 argument (exponent)".into());
    }
    let (exp_val, exp_ty) = args[0].clone();

    // Check if exponent is Tensor or Scalar
    if let Type::Tensor(_, _) = exp_ty {
        // Tensor exponent
        let fn_val = codegen
            .module
            .get_function("tl_tensor_pow")
            .ok_or("tl_tensor_pow not found")?;
        let call = codegen
            .builder
            .build_call(fn_val, &[obj_val.into(), exp_val.into()], "pow_res");

        let res = codegen.check_tensor_result(call.map_err(|e| e.to_string())?, "pow_error")?;
        codegen.emit_register_tensor(res, &obj_ty)?;
        Ok((res, obj_ty))
    } else {
        // Scalar exponent (assume f32 or convert to f32)
        let exp_f32 = match exp_ty {
            Type::F32 => exp_val.into_float_value(),
            Type::I64 => codegen
                .builder
                .build_signed_int_to_float(
                    exp_val.into_int_value(),
                    codegen.context.f32_type(),
                    "exp_i64_to_f32",
                )
                .map_err(|e| e.to_string())?,
            Type::I32 => codegen
                .builder
                .build_signed_int_to_float(
                    exp_val.into_int_value(),
                    codegen.context.f32_type(),
                    "exp_i32_to_f32",
                )
                .map_err(|e| e.to_string())?,
            _ => {
                return Err(format!(
                    "pow exponent must be Tensor or Number, got {:?}",
                    exp_ty
                ).into())
            }
        };

        let fn_val = codegen
            .module
            .get_function("tl_tensor_pow_scalar")
            .ok_or("tl_tensor_pow_scalar not found")?;
        let call =
            codegen
                .builder
                .build_call(fn_val, &[obj_val.into(), exp_f32.into()], "pow_scalar_res");

        let res =
            codegen.check_tensor_result(call.map_err(|e| e.to_string())?, "pow_scalar_error")?;
        
        // Fix: Result is always a Tensor (Rank 0), even if invoked on Scalar
        let res_ty = Type::Tensor(Box::new(Type::F32), 0);
        codegen.emit_register_tensor(res, &res_ty)?;
        Ok((res, res_ty))
    }
}

pub(super) fn compile_tensor_transpose<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    // Prepends receiver to args and calls standard transpose
    let mut new_args = Vec::with_capacity(args.len() + 1);
    new_args.push((obj_val, obj_ty));
    new_args.extend(args);
    compile_transpose(codegen, new_args)
}

pub(super) fn compile_transpose<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    // transpose(tensor, d0, d1)
    if args.len() != 3 {
        return Err("transpose requires 3 arguments: tensor, dim0, dim1".into());
    }
    let (t_val, t_ty) = &args[0];
    let (d0_val, _) = &args[1];
    let (d1_val, _) = &args[2];
    let is_tensor = matches!(t_ty, Type::Tensor(_, _)) 
        || matches!(t_ty, Type::TensorShaped(_, _))
        || matches!(t_ty, Type::Struct(name, _) if name == "Tensor");
        
    if !is_tensor {
        return Err(format!("First argument to transpose must be a tensor. Found: {:?}", t_ty).into());
    }
    let transpose_fn = codegen
        .module
        .get_function("tl_tensor_transpose")
        .ok_or("tl_tensor_transpose not found")?;

    let t_arg = if let Type::Struct(name, _) = t_ty {
        if name == "Tensor" {
            // Extract handle (i64) from struct
            let handle_i64 = if t_val.is_pointer_value() {
                let ptr = t_val.into_pointer_value();
                let i64_type = codegen.context.i64_type();
                let cast_ptr = codegen.builder.build_pointer_cast(ptr, codegen.context.ptr_type(inkwell::AddressSpace::default()), "cast_tensor_handle").map_err(|e| e.to_string())?;
                codegen.builder.build_load(i64_type, cast_ptr, "tensor_handle").map_err(|e| e.to_string())?.into_int_value()
            } else if t_val.is_struct_value() {
                codegen.builder.build_extract_value(t_val.into_struct_value(), 0, "tensor_handle").map_err(|e| e.to_string())?.into_int_value()
            } else {
                return Err(format!("Unexpected value kind for Struct Tensor: {:?}", t_val).into());
            };
            
            // Cast i64 handle to Pointer
            codegen.builder.build_int_to_ptr(handle_i64, codegen.context.ptr_type(inkwell::AddressSpace::default()), "handle_ptr").map_err(|e| e.to_string())?.into()
        } else {
            *t_val
        }
    } else {
        *t_val
    };

    let call = codegen
        .builder
        .build_call(
            transpose_fn,
            &[t_arg.into(), (*d0_val).into(), (*d1_val).into()],
            "transpose_res",
        )
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid transpose return".into()),
    };

    if let Type::Struct(name, _) = t_ty {
        if name == "Tensor" {
            // Wrap primitive result (ptr) into Struct { handle: i64 }
            let current_block = codegen.current_block()?;
            let parent_fn = current_block.get_parent().ok_or_else(|| "block has no parent function".to_string())?;
            
            // Alloca struct
            let i64_type = codegen.context.i64_type();
            let struct_type = codegen.context.struct_type(&[i64_type.into()], false);
            
            // Manual entry block alloca
            let entry = parent_fn.get_first_basic_block().ok_or_else(|| "function has no entry block".to_string())?;
            let builder = codegen.context.create_builder();
            if let Some(first_instr) = entry.get_first_instruction() {
                builder.position_before(&first_instr);
            } else {
                builder.position_at_end(entry);
            }
            let struct_alloca = builder.build_alloca(struct_type, "tensor_struct_res").map_err(|e| e.to_string())?;
            
            // Convert ptr -> i64
            let handle_i64 = codegen.builder.build_ptr_to_int(res.into_pointer_value(), i64_type, "handle_i64").map_err(|e| e.to_string())?;
            
            // Store handle (field 0)
            let handle_ptr = codegen.builder.build_struct_gep(struct_type, struct_alloca, 0, "handle_ptr").map_err(|e| e.to_string())?;
            codegen.builder.build_store(handle_ptr, handle_i64).map_err(|e| e.to_string())?;
            
            // Return pointer to struct
            return Ok((struct_alloca.into(), t_ty.clone()));
        }
    }

    Ok((res, t_ty.clone())) // Returns same type (Tensor)
}


pub(super) fn compile_tensor_reshape_uneval<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: &Expr,
    _method: &str,
    args: &[Expr],
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 {
        return Err("reshape method requires exactly 1 argument (shape array)".into());
    }

    // 1. Compile Receiver
    let (obj_val, obj_ty) = codegen.compile_expr(obj)?;

    // 2. Inspect shape argument for static rank inference
    let shape_expr = &args[0];
    let new_rank = match &shape_expr.inner {
        ExprKind::TensorLiteral(elements) | ExprKind::TensorConstLiteral(elements) => {
            elements.len()
        }
        _ => 0, // Unknown rank
    };

    // 3. Compile shape argument
    let (s_val, _) = codegen.compile_expr(shape_expr)?;

    // 4. Call runtime function
    let reshape_fn = codegen
        .module
        .get_function("tl_tensor_reshape_new")
        .ok_or("tl_tensor_reshape_new not found")?;
    let call = codegen
        .builder
        .build_call(reshape_fn, &[obj_val.into(), s_val.into()], "reshape_res")
        .map_err(|e| e.to_string())?;

    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid reshape return".into()),
    };

    // 5. Construct return type
    let new_ty = if new_rank > 0 {
        if let Type::Tensor(inner, _) = obj_ty {
            Type::Tensor(inner, new_rank)
        } else {
            Type::Tensor(Box::new(Type::F32), new_rank)
        }
    } else {
        // Unknown rank - preserve original
        obj_ty
    };

    Ok((res, new_ty))
}

