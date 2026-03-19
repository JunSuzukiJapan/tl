use crate::compiler::codegen::CodeGenerator;
use crate::compiler::ast::Type;
use inkwell::values::BasicValueEnum;

/// print() -> Void
pub fn compile_print<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen
        .module
        .get_function("tl_tensor_print")
        .ok_or("tl_tensor_print not found")?;
    codegen
        .builder
        .build_call(fn_val, &[obj.into()], "print_call")
        .map_err(|e| e.to_string())?;
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
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen
        .module
        .get_function("tl_metal_debug_tensor")
        .or_else(|| {
            let ptr_ty = codegen.context.ptr_type(inkwell::AddressSpace::default());
            let fn_type = codegen.context.void_type().fn_type(&[ptr_ty.into()], false);
            Some(codegen.module.add_function("tl_metal_debug_tensor", fn_type, None))
        })
        .unwrap();
    codegen
        .builder
        .build_call(fn_val, &[obj.into()], "debug_ptr_call")
        .map_err(|e| e.to_string())?;
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
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen
        .module
        .get_function("tl_tensor_display")
        .ok_or("tl_tensor_display not found")?;
    codegen
        .builder
        .build_call(fn_val, &[obj.into()], "display_call")
        .map_err(|e| e.to_string())?;
    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

/// slice(dim, start, end, step) -> Tensor
pub fn compile_slice<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen
        .module
        .get_function("tl_tensor_slice")
        .ok_or("tl_tensor_slice not found")?;
    
    let mut call_args: Vec<inkwell::values::BasicMetadataValueEnum> = Vec::with_capacity(args.len() + 1);
    call_args.push(obj.into());
    for (val, _) in &args {
        call_args.push((*val).into());
    }
    
    let call = codegen
        .builder
        .build_call(fn_val, &call_args, "slice_res")
        .map_err(|e| e.to_string())?;
    let res = codegen.check_tensor_result(call, "slice_error")?;
    Ok((res, obj_ty))
}
