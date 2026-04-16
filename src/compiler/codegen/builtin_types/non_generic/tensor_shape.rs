use crate::compiler::error::TlError;
use crate::compiler::codegen::CodeGenerator;
use crate::compiler::ast::Type;
use inkwell::values::{BasicValueEnum, ValueKind};

/// Tensor の shape 操作メソッドを compile する共通ヘルパー。
/// パターン: tl_tensor_{op}(tensor, args...) -> tensor
fn compile_tensor_shape_op<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    op_name: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_name = format!("tl_tensor_{}", op_name);
    let fn_val = codegen
        .module
        .get_function(&fn_name)
        .ok_or(format!("{} not found", fn_name))?;
    
    let mut call_args: Vec<inkwell::values::BasicMetadataValueEnum> = Vec::with_capacity(args.len() + 1);
    call_args.push(obj.into());
    for (val, _) in &args {
        call_args.push((*val).into());
    }
    
    let call = codegen
        .builder
        .build_call(fn_val, &call_args, &format!("{}_res", op_name))
        .map_err(|e| e.to_string())?;
    let res = codegen.check_tensor_result(call, &format!("{}_error", op_name))?;
    Ok((res, obj_ty))
}

/// Void を返す shape 操作
#[allow(dead_code)]
fn compile_tensor_shape_void<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    op_name: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_name = format!("tl_tensor_{}", op_name);
    let fn_val = codegen
        .module
        .get_function(&fn_name)
        .ok_or(format!("{} not found", fn_name))?;
    
    let mut call_args: Vec<inkwell::values::BasicMetadataValueEnum> = Vec::with_capacity(args.len() + 1);
    call_args.push(obj.into());
    for (val, _) in &args {
        call_args.push((*val).into());
    }
    
    codegen
        .builder
        .build_call(fn_val, &call_args, &format!("{}_res", op_name))
        .map_err(|e| e.to_string())?;
    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

/// i64 を返す shape query
fn compile_tensor_shape_i64<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    op_name: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_name = format!("tl_tensor_{}", op_name);
    let fn_val = codegen
        .module
        .get_function(&fn_name)
        .ok_or(format!("{} not found", fn_name))?;
    
    let mut call_args: Vec<inkwell::values::BasicMetadataValueEnum> = Vec::with_capacity(args.len() + 1);
    call_args.push(obj.into());
    for (val, _) in &args {
        call_args.push((*val).into());
    }
    
    let call = codegen
        .builder
        .build_call(fn_val, &call_args, &format!("{}_res", op_name))
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(format!("Invalid return from {}()", op_name).into()),
    };
    Ok((res, Type::I64))
}

// ---- squeeze(dim) -> Tensor ----
pub fn compile_squeeze<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    compile_tensor_shape_op(c, o, t, a, "squeeze")
}

// ---- unsqueeze(dim) -> Tensor ----
pub fn compile_unsqueeze<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    compile_tensor_shape_op(c, o, t, a, "unsqueeze")
}

// ---- flatten(dim) -> Tensor ----
pub fn compile_flatten<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    compile_tensor_shape_op(c, o, t, a, "flatten")
}

// ---- gather(indices) -> Tensor ----
pub fn compile_gather<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    compile_tensor_shape_op(c, o, t, a, "gather")
}

// ---- permute(dims: Vec<i64>) -> Tensor ----
pub fn compile_permute<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    compile_tensor_shape_op(c, o, t, a, "permute")
}

// ---- cat(other: Tensor) -> Tensor ----
pub fn compile_cat<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_name = "tl_tensor_cat_4d";
    let fn_val = c.module.get_function(fn_name).ok_or("tl_tensor_cat_4d not found")?;
    
    let mut call_args: Vec<inkwell::values::BasicMetadataValueEnum> = Vec::with_capacity(3);
    call_args.push(o.into());
    for (val, _) in &a {
        call_args.push((*val).into());
    }
    call_args.push(c.context.i64_type().const_int(0, false).into());
    
    let call = c.builder.build_call(fn_val, &call_args, "cat_res").map_err(|e| e.to_string())?;
    let res = c.check_tensor_result(call, "cat_error")?;
    Ok((res, t))
}

// ---- len() -> i64 ----
pub fn compile_len<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    compile_tensor_shape_i64(c, o, t, a, "len")
}

// ---- dim(d) -> i64 ----
pub fn compile_dim<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    compile_tensor_shape_i64(c, o, t, a, "dim")
}

// ---- ndim() -> i64 ----
pub fn compile_ndim<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    compile_tensor_shape_i64(c, o, t, a, "ndim")
}

// ---- get_shape() -> Vec<i64> ----
pub fn compile_get_shape<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = codegen
        .module
        .get_function("tl_tensor_get_shape")
        .ok_or("tl_tensor_get_shape not found")?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj.into()], "get_shape_res")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from get_shape()".into()),
    };
    Ok((res, Type::Struct("Vec".into(), vec![Type::I64])))
}
