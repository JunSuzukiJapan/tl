use crate::compiler::error::{TlError, CodegenErrorKind};
use crate::compiler::codegen::type_manager::{CodeGenType, TypeManager};
use crate::compiler::codegen::CodeGenerator;
use crate::compiler::ast::Type;
use inkwell::values::{BasicValueEnum, ValueKind};

pub fn register_regex_types(manager: &mut TypeManager) {
    let string_type = Type::String("String".to_string());
    
    let mut regex_type = CodeGenType::new("Regex");
    
    // Regex::new(pattern: String) -> Option<Regex>
    // TL側での Option は Struct("Option", [Type::Struct("Regex", [])]) となる。
    // しかし、LLVMレベルでは i64 型 (id) または構造体 { i64 } を返す。
    // 単純に `Regex::new(pattern: String) -> Regex` とし、内部IDが -1 ならエラーとする方が
    // Opaqueな実装としては簡単。まずは `Regex::new` を定義する。
    regex_type.register_evaluated_static_method(
        "new", 
        compile_regex_new, 
        vec![string_type.clone()],
        Type::Struct("Regex".to_string(), vec![])
    );
    
    // Regex.is_valid() -> Bool (IDが -1 以外か)
    regex_type.register_evaluated_instance_method(
        "is_valid", 
        compile_regex_is_valid, 
        vec![],
        Type::Bool
    );
    
    // Regex.is_match(text: String) -> Bool
    regex_type.register_evaluated_instance_method(
        "is_match", 
        compile_regex_is_match, 
        vec![string_type.clone()],
        Type::Bool
    );
    
    // Regex.replace(text: String, replacement: String) -> String
    regex_type.register_evaluated_instance_method(
        "replace", 
        compile_regex_replace, 
        vec![string_type.clone(), string_type.clone()],
        string_type.clone()
    );
    
    // Regex.release() -> Void
    // NOTE: We don't use 'free' as the method name or the generated C function name (tl_regex_free)
    // because the compiler implicitly tries to generate/call a global `tl_<struct>_free(ptr)`
    // which leads to an LLVM signature mismatch (id as i64 vs ptr).
    regex_type.register_evaluated_instance_method(
        "release", 
        compile_regex_release, 
        vec![],
        Type::Void
    );
    
    manager.register_type(regex_type);
}

// ---------------------------------------------------------
// Regex Methods
// ---------------------------------------------------------

/// Struct "Regex" receives { id: i64 } from codegen side.
/// For simplicity, we just pass the struct pointer to extract `id`.
fn extract_regex_id<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
) -> Result<inkwell::values::IntValue<'ctx>, TlError> {
    // Assuming `obj` is a pointer to the Regex struct which has a single i64 field.
    let struct_ty = codegen.context.struct_type(&[
        codegen.context.i64_type().into()
    ], false);
    
    let ptr = obj.into_pointer_value();
    let id_ptr = codegen.builder.build_struct_gep(struct_ty, ptr, 0, "regex_id_ptr")
        .map_err(|_| TlError::from(CodegenErrorKind::Internal("Failed to GEP Regex ID".to_string())))?;
    let id_val = codegen.builder.build_load(codegen.context.i64_type(), id_ptr, "regex_id")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value();
    
    Ok(id_val)
}

pub fn compile_regex_new<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _hint: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = codegen.module.get_function("tl_regex_new")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_regex_new not found".to_string())))?;
        
    let call = codegen.builder.build_call(fn_val, &[args[0].0.into()], "regex_id_raw")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        
    let id_val = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(CodegenErrorKind::Internal("Invalid return from tl_regex_new".to_string()).into()),
    };
    
    // Allocate a Regex struct { i64 }
    let struct_ty = codegen.context.struct_type(&[
        codegen.context.i64_type().into()
    ], false);
    
    let regex_ptr = codegen.builder.build_alloca(struct_ty, "regex_struct")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        
    let id_ptr = codegen.builder.build_struct_gep(struct_ty, regex_ptr, 0, "id_field")
        .map_err(|_| TlError::from(CodegenErrorKind::Internal("Failed to GEP".to_string())))?;
    codegen.builder.build_store(id_ptr, id_val)
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        
    Ok((regex_ptr.into(), Type::Struct("Regex".to_string(), vec![])))
}

pub fn compile_regex_is_valid<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let id_val = extract_regex_id(codegen, obj)?;
    let zero = codegen.context.i64_type().const_zero();
    
    let is_valid = codegen.builder.build_int_compare(
        inkwell::IntPredicate::SGE,
        id_val,
        zero,
        "is_valid"
    ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    
    Ok((is_valid.into(), Type::Bool))
}

pub fn compile_regex_is_match<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let id_val = extract_regex_id(codegen, obj)?;
    
    let fn_val = codegen.module.get_function("tl_regex_is_match")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_regex_is_match not found".to_string())))?;
        
    let call = codegen.builder.build_call(fn_val, &[id_val.into(), args[0].0.into()], "is_match_res")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(CodegenErrorKind::Internal("Invalid return from tl_regex_is_match".to_string()).into()),
    };
    
    Ok((res, Type::Bool))
}

pub fn compile_regex_replace<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let id_val = extract_regex_id(codegen, obj)?;
    
    let fn_val = codegen.module.get_function("tl_regex_replace")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_regex_replace not found".to_string())))?;
        
    let call = codegen.builder.build_call(fn_val, &[id_val.into(), args[0].0.into(), args[1].0.into()], "replace_res")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(CodegenErrorKind::Internal("Invalid return from tl_regex_replace".to_string()).into()),
    };
    
    Ok((res, Type::String("String".to_string())))
}

/// NOTE: Avoid naming this `compile_regex_free` and calling `tl_regex_free` to prevent 
/// collision with the compiler's implicit cleanup hooks which expect a pointer argument.
pub fn compile_regex_release<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let id_val = extract_regex_id(codegen, obj)?;
    
    let fn_val = codegen.module.get_function("tl_regex_release")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_regex_release not found".to_string())))?;
        
    codegen.builder.build_call(fn_val, &[id_val.into()], "")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        
    Ok((codegen.context.i64_type().const_zero().into(), Type::Void))
}
