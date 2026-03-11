use crate::compiler::codegen::CodeGenerator;
use crate::compiler::ast::Type;
use inkwell::values::{BasicValueEnum, BasicValue, ValueKind};

/// String.print() -> Void
pub fn compile_print<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen
        .module
        .get_function("tl_print_string")
        .ok_or("tl_print_string not found")?;
    codegen
        .builder
        .build_call(fn_val, &[obj.into()], "")
        .map_err(|e| e.to_string())?;
    Ok((
        codegen.context.i64_type().const_zero().into(),
        Type::Void,
    ))
}

/// String.display() -> Void
pub fn compile_display<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen
        .module
        .get_function("tl_display_string")
        .ok_or("tl_display_string not found")?;
    codegen
        .builder
        .build_call(fn_val, &[obj.into()], "")
        .map_err(|e| e.to_string())?;
    Ok((
        codegen.context.i64_type().const_zero().into(),
        Type::Void,
    ))
}

/// String.len() -> i64
/// StringStruct = { ptr: *i8, len: i64 } — extract len field directly
pub fn compile_len<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let str_struct_ty = codegen.context.struct_type(&[
        codegen.context.ptr_type(inkwell::AddressSpace::default()).into(),
        codegen.context.i64_type().into(),
    ], false);
    let ptr = obj.into_pointer_value();
    let len_ptr = codegen.builder.build_struct_gep(str_struct_ty, ptr, 1, "len_ptr")
        .map_err(|_| "Failed to GEP String len")?;
    let len_val = codegen.builder.build_load(codegen.context.i64_type(), len_ptr, "len")
        .map_err(|e| e.to_string())?;
    Ok((len_val, Type::I64))
}

/// String.contains(other: String) -> Bool
pub fn compile_contains<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 { return Err("String.contains requires 1 argument".into()); }
    let fn_val = codegen.module.get_function("tl_string_contains")
        .ok_or("tl_string_contains not found")?;
    let call = codegen.builder.build_call(fn_val, &[obj.into(), args[0].0.into()], "contains_res")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from String.contains".into()),
    };
    Ok((res, Type::Bool))
}

/// String.concat(other: String) -> String
pub fn compile_concat<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 { return Err("String.concat requires 1 argument".into()); }
    let fn_val = codegen.module.get_function("tl_string_concat")
        .ok_or("tl_string_concat not found")?;
    let call = codegen.builder.build_call(fn_val, &[obj.into(), args[0].0.into()], "concat_res")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from String.concat".into()),
    };
    Ok((res, Type::String("String".to_string())))
}

/// String.char_at(index: i64) -> Char
pub fn compile_char_at<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 { return Err("String.char_at requires 1 argument".into()); }
    let fn_val = codegen.module.get_function("tl_string_char_at")
        .ok_or("tl_string_char_at not found")?;
    let call = codegen.builder.build_call(fn_val, &[obj.into(), args[0].0.into()], "char_at_res")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from String.char_at".into()),
    };
    Ok((res, Type::Char("Char".to_string())))
}

/// String.to_i64() -> i64
pub fn compile_to_i64<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen.module.get_function("tl_string_to_i64")
        .ok_or("tl_string_to_i64 not found")?;
    let call = codegen.builder.build_call(fn_val, &[obj.into()], "to_i64_res")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from String.to_i64".into()),
    };
    Ok((res, Type::I64))
}
