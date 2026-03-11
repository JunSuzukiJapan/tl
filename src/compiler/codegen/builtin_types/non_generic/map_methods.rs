use crate::compiler::codegen::CodeGenerator;
use crate::compiler::ast::Type;
use inkwell::values::{BasicValueEnum, BasicValue, ValueKind};

/// Map の get 系メソッドの共通ヘルパー
fn compile_map_get_impl<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    fn_name: &str,
    ret_ty: Type,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err(format!("Map::{} requires 1 argument", fn_name));
    }
    let (key_val, key_ty) = &args[0];
    if !matches!(key_ty, Type::String(_)) {
        return Err(format!("Map::{} expects String argument, got {:?}", fn_name, key_ty));
    }
    let fn_val = codegen
        .module
        .get_function(fn_name)
        .ok_or(format!("{} not found", fn_name))?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj.into(), (*key_val).into()], "map_get")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(format!("Invalid return from {}", fn_name)),
    };
    Ok((res, ret_ty))
}

/// Map.get(key: String) -> Tensor
pub fn compile_get<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_map_get_impl(c, o, a, "tl_tensor_map_get", Type::Tensor(Box::new(Type::F32), 0))
}

/// Map.get_1d(key: String) -> Tensor<F32, 1>
pub fn compile_get_1d<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_map_get_impl(c, o, a, "tl_tensor_map_get_1d", Type::Tensor(Box::new(Type::F32), 1))
}

/// Map.get_quantized(key: String) -> Tensor<I8, 2>
pub fn compile_get_quantized<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_map_get_impl(c, o, a, "tl_tensor_map_get_quantized", Type::Tensor(Box::new(Type::I8), 2))
}

/// Map.set(key: String, value: Tensor) -> Void
pub fn compile_set<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 2 {
        return Err("Map.set requires 2 arguments".into());
    }
    let fn_val = codegen.module.get_function("tl_tensor_map_insert")
        .ok_or("tl_tensor_map_insert not found")?;
    codegen.builder.build_call(fn_val, &[obj.into(), args[0].0.into(), args[1].0.into()], "map_set")
        .map_err(|e| e.to_string())?;
    Ok((codegen.context.i64_type().const_zero().into(), Type::Void))
}

/// Map.metadata() -> Map
pub fn compile_metadata<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen.module.get_function("tl_tensor_map_metadata")
        .ok_or("tl_tensor_map_metadata not found")?;
    let call = codegen.builder.build_call(fn_val, &[obj.into()], "map_metadata")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from Map.metadata".into()),
    };
    Ok((res, Type::Struct("Map".into(), vec![])))
}

/// Map.get_i64(key: String) -> i64
pub fn compile_get_i64<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_map_get_impl(c, o, a, "tl_tensor_map_get_i64", Type::I64)
}

/// Map.get_string(key: String) -> String
pub fn compile_get_string<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_map_get_impl(c, o, a, "tl_tensor_map_get_string", Type::String("String".to_string()))
}

/// Map.get_vec_string(key: String) -> Vec<String>
pub fn compile_get_vec_string<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, _t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_map_get_impl(c, o, a, "tl_tensor_map_get_vec_string", Type::Struct("Vec".into(), vec![Type::String("String".to_string())]))
}
