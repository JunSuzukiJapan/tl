use crate::compiler::error::{TlError, CodegenErrorKind};
use crate::compiler::codegen::CodeGenerator;
use crate::compiler::ast::Type;
use inkwell::values::BasicValueEnum;

/// print() -> Void
pub fn compile_print<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = codegen
        .module
        .get_function("tl_tensor_print")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_tensor_print not found".to_string())))?;
    codegen
        .builder
        .build_call(fn_val, &[obj.into()], "print_call")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

/// print_1() / print_2() / print_3() -> Void
/// These are rank-specific print aliases that all call tl_tensor_print (same behavior).
pub fn compile_print_n<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = codegen
        .module
        .get_function("tl_tensor_print")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_tensor_print not found".to_string())))?;
    codegen
        .builder
        .build_call(fn_val, &[obj.into()], "print_n_call")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

/// debug_ptr() -> Void
pub fn compile_debug_ptr<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = codegen
        .module
        .get_function("tl_metal_debug_tensor")
        .or_else(|| {
            let ptr_ty = codegen.context.ptr_type(inkwell::AddressSpace::default());
            let fn_type = codegen.context.void_type().fn_type(&[ptr_ty.into()], false);
            Some(codegen.module.add_function("tl_metal_debug_tensor", fn_type, None))
        })
        .expect("or_else always provides Some");
    codegen
        .builder
        .build_call(fn_val, &[obj.into()], "debug_ptr_call")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

/// display() -> Void
pub fn compile_display<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = codegen
        .module
        .get_function("tl_tensor_display")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_tensor_display not found".to_string())))?;
    codegen
        .builder
        .build_call(fn_val, &[obj.into()], "display_call")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

/// .slice(start, len) → FFI: tl_tensor_slice(t, dim=0, start, end=start+len, step=1)
pub fn compile_slice2<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = codegen
        .module
        .get_function("tl_tensor_slice")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_tensor_slice not found".to_string())))?;
    let i64_ty = codegen.context.i64_type();

    let dim = i64_ty.const_int(0, false);
    let start = args[0].0.into_int_value();
    let len = args[1].0.into_int_value();
    let end = codegen.builder.build_int_add(start, len, "slice_end")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let step = i64_ty.const_int(1, false);

    let call = codegen.builder
        .build_call(fn_val, &[obj.into(), dim.into(), start.into(), end.into(), step.into()], "slice_res")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = codegen.check_tensor_result(call, "slice_error")?;
    Ok((res, obj_ty))
}

/// .slice(dim, start, len) → FFI: tl_tensor_slice(t, dim, start, end=start+len, step=1)
pub fn compile_slice3<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = codegen
        .module
        .get_function("tl_tensor_slice")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_tensor_slice not found".to_string())))?;
    let i64_ty = codegen.context.i64_type();

    let dim = args[0].0;
    let start = args[1].0.into_int_value();
    let len = args[2].0.into_int_value();
    let end = codegen.builder.build_int_add(start, len, "slice_end")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let step = i64_ty.const_int(1, false);

    let call = codegen.builder
        .build_call(fn_val, &[obj.into(), dim.into(), start.into(), end.into(), step.into()], "slice_res")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = codegen.check_tensor_result(call, "slice_error")?;
    Ok((res, obj_ty))
}

/// .slice(dim, start, end, step) → FFI: tl_tensor_slice(t, dim, start, end, step)
pub fn compile_slice4<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = codegen
        .module
        .get_function("tl_tensor_slice")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_tensor_slice not found".to_string())))?;

    let call = codegen.builder
        .build_call(fn_val, &[obj.into(), args[0].0.into(), args[1].0.into(), args[2].0.into(), args[3].0.into()], "slice_res")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = codegen.check_tensor_result(call, "slice_error")?;
    Ok((res, obj_ty))
}

