use crate::compiler::error::TlError;
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
        .ok_or("tl_tensor_backward not found")?;
    codegen
        .builder
        .build_call(fn_val, &[obj.into()], "backward_call")
        .map_err(|e| e.to_string())?;
    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

/// enable_grad() -> GradTensor (void FFI, returns obj as GradTensor type)
pub fn compile_enable_grad<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = codegen
        .module
        .get_function("tl_tensor_enable_grad")
        .ok_or("tl_tensor_enable_grad not found")?;
    // enable_grad is a void FFI: it enables grad in-place, returns nothing.
    codegen
        .builder
        .build_call(fn_val, &[obj.into()], "enable_grad")
        .map_err(|e| e.to_string())?;
    // Return the same pointer, but with GradTensor type
    let grad_ty = match &obj_ty {
        Type::Tensor(inner, rank) => Type::GradTensor(inner.clone(), *rank),
        Type::GradTensor(_, _) => obj_ty.clone(),
        _ => obj_ty.clone(),
    };
    Ok((obj, grad_ty))
}

