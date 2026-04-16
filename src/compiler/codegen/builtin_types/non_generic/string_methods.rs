use crate::compiler::error::TlError;
use crate::compiler::codegen::CodeGenerator;
use crate::compiler::ast::Type;
use inkwell::values::{BasicValueEnum, ValueKind};

/// String.print() -> Void
pub fn compile_print<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
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
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
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
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
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
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
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
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
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
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
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
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
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

/// String.trim() -> String
pub fn compile_trim<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = codegen.module.get_function("tl_string_trim")
        .ok_or("tl_string_trim not found")?;
    let call = codegen.builder.build_call(fn_val, &[obj.into()], "trim_res")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from String.trim".into()),
    };
    Ok((res, Type::String("String".to_string())))
}

/// String.starts_with(prefix: String) -> bool
pub fn compile_starts_with<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 { return Err("String.starts_with requires 1 argument".into()); }
    let fn_val = codegen.module.get_function("tl_string_starts_with")
        .ok_or("tl_string_starts_with not found")?;
    let call = codegen.builder.build_call(fn_val, &[obj.into(), args[0].0.into()], "starts_with_res")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from String.starts_with".into()),
    };
    Ok((res, Type::Bool))
}

/// String.ends_with(suffix: String) -> bool
pub fn compile_ends_with<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 { return Err("String.ends_with requires 1 argument".into()); }
    let fn_val = codegen.module.get_function("tl_string_ends_with")
        .ok_or("tl_string_ends_with not found")?;
    let call = codegen.builder.build_call(fn_val, &[obj.into(), args[0].0.into()], "ends_with_res")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from String.ends_with".into()),
    };
    Ok((res, Type::Bool))
}

/// String.replace(from: String, to: String) -> String
pub fn compile_replace<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 2 { return Err("String.replace requires 2 arguments".into()); }
    let fn_val = codegen.module.get_function("tl_string_replace")
        .ok_or("tl_string_replace not found")?;
    let call = codegen.builder.build_call(fn_val, &[obj.into(), args[0].0.into(), args[1].0.into()], "replace_res")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from String.replace".into()),
    };
    Ok((res, Type::String("String".to_string())))
}

/// String.substring(start: i64, len: i64) -> String
pub fn compile_substring<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 2 { return Err("String.substring requires 2 arguments".into()); }
    let fn_val = codegen.module.get_function("tl_string_substring")
        .ok_or("tl_string_substring not found")?;
    let call = codegen.builder.build_call(fn_val, &[obj.into(), args[0].0.into(), args[1].0.into()], "substring_res")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from String.substring".into()),
    };
    Ok((res, Type::String("String".to_string())))
}

/// String.is_empty() -> bool
pub fn compile_is_empty<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = codegen.module.get_function("tl_string_is_empty")
        .ok_or("tl_string_is_empty not found")?;
    let call = codegen.builder.build_call(fn_val, &[obj.into()], "is_empty_res")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from String.is_empty".into()),
    };
    Ok((res, Type::Bool))
}

/// String.to_uppercase() -> String
pub fn compile_to_uppercase<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = codegen.module.get_function("tl_string_to_uppercase")
        .ok_or("tl_string_to_uppercase not found")?;
    let call = codegen.builder.build_call(fn_val, &[obj.into()], "to_upper_res")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from String.to_uppercase".into()),
    };
    Ok((res, Type::String("String".to_string())))
}

/// String.to_lowercase() -> String
pub fn compile_to_lowercase<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = codegen.module.get_function("tl_string_to_lowercase")
        .ok_or("tl_string_to_lowercase not found")?;
    let call = codegen.builder.build_call(fn_val, &[obj.into()], "to_lower_res")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from String.to_lowercase".into()),
    };
    Ok((res, Type::String("String".to_string())))
}

/// String.index_of(needle: String) -> i64
pub fn compile_index_of<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 { return Err("String.index_of requires 1 argument".into()); }
    let fn_val = codegen.module.get_function("tl_string_index_of")
        .ok_or("tl_string_index_of not found")?;
    let call = codegen.builder.build_call(fn_val, &[obj.into(), args[0].0.into()], "index_of_res")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from String.index_of".into()),
    };
    Ok((res, Type::I64))
}

/// String.split(sep: String) -> Vec<String>
pub fn compile_split<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 { return Err("String.split requires 1 argument".into()); }
    let fn_val = codegen.module.get_function("tl_string_split")
        .ok_or("tl_string_split not found")?;
    let call = codegen.builder.build_call(fn_val, &[obj.into(), args[0].0.into()], "split_res")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from String.split".into()),
    };
    Ok((res, Type::Struct("Vec".to_string(), vec![Type::String("String".to_string())])))
}

/// String.to_f64() -> f64
pub fn compile_to_f64<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = codegen.module.get_function("tl_string_to_f64")
        .ok_or("tl_string_to_f64 not found")?;
    let call = codegen.builder.build_call(fn_val, &[obj.into()], "to_f64_res")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from String.to_f64".into()),
    };
    Ok((res, Type::F64))
}

/// String.repeat(n: i64) -> String
pub fn compile_repeat<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 { return Err("String.repeat requires 1 argument".into()); }
    let fn_val = codegen.module.get_function("tl_string_repeat")
        .ok_or("tl_string_repeat not found")?;
    let call = codegen.builder.build_call(fn_val, &[obj.into(), args[0].0.into()], "repeat_res")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from String.repeat".into()),
    };
    Ok((res, Type::String("String".to_string())))
}

/// String.chars() -> Vec<i64>
pub fn compile_chars<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = codegen.module.get_function("tl_string_chars")
        .ok_or("tl_string_chars not found")?;
    let call = codegen.builder.build_call(fn_val, &[obj.into()], "chars_res")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from String.chars".into()),
    };
    Ok((res, Type::Struct("Vec".to_string(), vec![Type::I64])))
}

/// String::from_chars(chars: Vec<i64>) -> String
pub fn compile_from_chars<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _hint: Option<&Type>
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 { return Err("String::from_chars requires 1 argument".into()); }
    let fn_val = codegen.module.get_function("tl_string_from_chars")
        .ok_or("tl_string_from_chars not found")?;
    let call = codegen.builder.build_call(fn_val, &[args[0].0.into()], "from_chars_res")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from String::from_chars".into()),
    };
    Ok((res, Type::String("String".to_string())))
}

/// String.to_bytes() -> Vec<u8>
pub fn compile_to_bytes<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = codegen.module.get_function("tl_string_to_bytes")
        .ok_or("tl_string_to_bytes not found")?;
    let call = codegen.builder.build_call(fn_val, &[obj.into()], "to_bytes_res")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from String.to_bytes".into()),
    };
    Ok((res, Type::Struct("Vec".to_string(), vec![Type::U8])))
}

/// String::from_utf8(bytes: Vec<u8>) -> String
pub fn compile_from_utf8<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _hint: Option<&Type>
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 { return Err("String::from_utf8 requires 1 argument".into()); }
    let fn_val = codegen.module.get_function("tl_string_from_utf8")
        .ok_or("tl_string_from_utf8 not found")?;
    let call = codegen.builder.build_call(fn_val, &[args[0].0.into()], "from_utf8_res")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from String::from_utf8".into()),
    };
    Ok((res, Type::String("String".to_string())))
}
