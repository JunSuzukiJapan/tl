use crate::compiler::codegen::CodeGenerator;
use crate::compiler::ast::Type;
use inkwell::values::BasicValueEnum;

/// Tensor の elementwise メソッドを compile する共通ヘルパー。
/// パターン: tl_tensor_{op}(tensor) -> tensor
fn compile_tensor_unary_ffi<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
    op_name: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_name = format!("tl_tensor_{}", op_name);
    let fn_val = codegen
        .module
        .get_function(&fn_name)
        .ok_or(format!("{} not found", fn_name))?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj.into()], "unary_res")
        .map_err(|e| e.to_string())?;
    let res = codegen.check_tensor_result(call, &format!("{}_error", op_name))?;
    Ok((res, obj_ty))
}

// ---- 13 elementwise unary ops ----

pub fn compile_abs<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_unary_ffi(c, o, t, a, "abs")
}

pub fn compile_relu<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_unary_ffi(c, o, t, a, "relu")
}

pub fn compile_gelu<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_unary_ffi(c, o, t, a, "gelu")
}

pub fn compile_silu<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_unary_ffi(c, o, t, a, "silu")
}

pub fn compile_exp<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_unary_ffi(c, o, t, a, "exp")
}

pub fn compile_log<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_unary_ffi(c, o, t, a, "log")
}

pub fn compile_sqrt<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_unary_ffi(c, o, t, a, "sqrt")
}

pub fn compile_sin<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_unary_ffi(c, o, t, a, "sin")
}

pub fn compile_cos<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_unary_ffi(c, o, t, a, "cos")
}

pub fn compile_tanh<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_unary_ffi(c, o, t, a, "tanh")
}

pub fn compile_sigmoid<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_unary_ffi(c, o, t, a, "sigmoid")
}

pub fn compile_neg<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_unary_ffi(c, o, t, a, "neg")
}

pub fn compile_tan<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_unary_ffi(c, o, t, a, "tan")
}
