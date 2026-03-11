use crate::compiler::codegen::CodeGenerator;
use crate::compiler::ast::Type;
use inkwell::values::{BasicValueEnum, BasicValue, ValueKind};

/// LLM 専用メソッドの共通ヘルパー。
/// パターン: tl_tensor_{op}(tensor, args...) -> tensor
fn compile_tensor_llm_op<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    op_name: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
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

/// i64 を返す LLM メソッド
fn compile_tensor_llm_i64<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    op_name: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
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
        _ => return Err(format!("Invalid return from {}()", op_name)),
    };
    Ok((res, Type::I64))
}

// ---- scale(f32) -> Tensor ----
pub fn compile_scale<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_llm_op(c, o, t, a, "scale")
}

// ---- add_4d(other) -> Tensor ----
pub fn compile_add_4d<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_llm_op(c, o, t, a, "add_4d")
}

// ---- matmul_4d(other) -> Tensor ----
pub fn compile_matmul_4d<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_llm_op(c, o, t, a, "matmul_4d")
}

// ---- cat_4d(other, dim) -> Tensor ----
pub fn compile_cat_4d<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_llm_op(c, o, t, a, "cat_4d")
}

// ---- rms_norm(weight, eps) -> Tensor ----
pub fn compile_rms_norm<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_llm_op(c, o, t, a, "rms_norm")
}

// ---- sample(temp, top_p) -> Tensor ----
pub fn compile_sample<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    // sample returns Tensor<f32, 1> but we keep obj_ty for now
    let fn_name = "tl_tensor_sample";
    let fn_val = c.module.get_function(fn_name).ok_or(format!("{} not found", fn_name))?;
    let mut call_args: Vec<inkwell::values::BasicMetadataValueEnum> = Vec::with_capacity(a.len() + 1);
    call_args.push(o.into());
    for (val, _) in &a { call_args.push((*val).into()); }
    let call = c.builder.build_call(fn_val, &call_args, "sample_res").map_err(|e| e.to_string())?;
    let res = c.check_tensor_result(call, "sample_error")?;
    Ok((res, Type::Tensor(Box::new(Type::F32), 1)))
}

// ---- repeat_interleave(repeats, dim) -> Tensor ----
pub fn compile_repeat_interleave<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_llm_op(c, o, t, a, "repeat_interleave")
}

// ---- apply_rope(cos_cache, sin_cache) -> Tensor ----
pub fn compile_apply_rope<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_llm_op(c, o, t, a, "apply_rope")
}

// ---- narrow(dim, start, len) -> Tensor ----
pub fn compile_narrow<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_llm_op(c, o, t, a, "narrow")
}

// ---- transpose_2d(d0, d1) -> Tensor ----
pub fn compile_transpose_2d<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_llm_op(c, o, t, a, "transpose_2d")
}

// ---- item_i64() -> i64 ----
pub fn compile_item_i64<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_llm_i64(c, o, t, a, "item_i64")
}

// ---- cat_i64(other, dim) -> Tensor ----
pub fn compile_cat_i64<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_llm_op(c, o, t, a, "cat_i64")
}
