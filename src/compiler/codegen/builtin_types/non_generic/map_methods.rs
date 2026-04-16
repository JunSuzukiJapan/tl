use crate::compiler::error::{TlError, CodegenErrorKind};
use crate::compiler::codegen::CodeGenerator;
use crate::compiler::ast::Type;
use inkwell::values::{BasicValueEnum, ValueKind};

/// Map の get 系メソッドの共通ヘルパー
fn compile_map_get_impl<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    fn_name: &str,
    ret_ty: Type,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 {
        return Err(TlError::from(CodegenErrorKind::Internal(format!("Map::{} requires 1 argument", fn_name))));
    }
    let (key_val, key_ty) = &args[0];
    if !matches!(key_ty, Type::String(_)) {
        return Err(TlError::from(CodegenErrorKind::Internal(format!("Map::{} expects String argument, got {:?}", fn_name, key_ty))));
    }
    let fn_val = codegen
        .module
        .get_function(fn_name)
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("{} not found", fn_name))))?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj.into(), (*key_val).into()], "map_get")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(TlError::from(CodegenErrorKind::Internal(format!("Invalid return from {}", fn_name)))),
    };
    Ok((res, ret_ty))
}

/// Map.get(key: String) -> Tensor
pub fn compile_get<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    compile_map_get_impl(c, o, a, "tl_tensor_map_get", Type::Tensor(Box::new(Type::F32), 0))
}

/// Map.get_1d(key: String) -> Tensor<F32, 1>
pub fn compile_get_1d<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    compile_map_get_impl(c, o, a, "tl_tensor_map_get_1d", Type::Tensor(Box::new(Type::F32), 1))
}

/// Map.get_quantized(key: String) -> Tensor<I8, 2>
pub fn compile_get_quantized<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    compile_map_get_impl(c, o, a, "tl_tensor_map_get_quantized", Type::Tensor(Box::new(Type::I8), 2))
}

/// Map.set(key: String, value: Tensor) -> Void
pub fn compile_set<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 2 {
        return Err(TlError::from(CodegenErrorKind::Internal("Map.set requires 2 arguments".to_string())));
    }
    let fn_val = codegen.module.get_function("tl_tensor_map_insert")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_tensor_map_insert not found".to_string())))?;
    codegen.builder.build_call(fn_val, &[obj.into(), args[0].0.into(), args[1].0.into()], "map_set")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((codegen.context.i64_type().const_zero().into(), Type::Void))
}

/// Map.metadata() -> Map
pub fn compile_metadata<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = codegen.module.get_function("tl_tensor_map_metadata")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_tensor_map_metadata not found".to_string())))?;
    let call = codegen.builder.build_call(fn_val, &[obj.into()], "map_metadata")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(CodegenErrorKind::Internal("Invalid return from Map.metadata".to_string()).into()),
    };
    Ok((res, Type::Struct("Map".to_string(), vec![])))
}

/// Map.get_i64(key: String) -> i64
pub fn compile_get_i64<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    compile_map_get_impl(c, o, a, "tl_tensor_map_get_i64", Type::I64)
}

/// Map.get_string(key: String) -> String
pub fn compile_get_string<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    compile_map_get_impl(c, o, a, "tl_tensor_map_get_string", Type::String("String".to_string()))
}

/// Map.get_vec_string(key: String) -> Vec<String>
pub fn compile_get_vec_string<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    compile_map_get_impl(c, o, a, "tl_tensor_map_get_vec_string", Type::Struct("Vec".to_string(), vec![Type::String("String".to_string())]))
}
