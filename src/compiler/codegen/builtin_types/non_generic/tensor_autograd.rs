use crate::compiler::error::{TlError, CodegenErrorKind};
use crate::compiler::codegen::CodeGenerator;
use crate::compiler::ast::Type;
use inkwell::values::BasicValueEnum;

/// backward() -> Void
pub fn compile_backward<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = codegen
        .module
        .get_function("tl_tensor_backward")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_tensor_backward not found".to_string())))?;
    codegen
        .builder
        .build_call(fn_val, &[obj.into()], "backward_call")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

/// enable_grad() -> Void (in-place void FFI)
pub fn compile_enable_grad<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = codegen
        .module
        .get_function("tl_tensor_enable_grad")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_tensor_enable_grad not found".to_string())))?;
    // enable_grad is a void FFI: it enables grad in-place, returns nothing.
    codegen
        .builder
        .build_call(fn_val, &[obj.into()], "enable_grad")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    // §6.5: void メソッドは Void を返す。GradTensor を返すと、
    // スコープ cleanup で self ポインタが release され、データが破壊される。
    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

