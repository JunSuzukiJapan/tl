use crate::compiler::codegen::CodeGenerator;
use crate::compiler::ast::Type;
use inkwell::values::BasicValueEnum;

/// Tensor の reduction / activation (1引数) メソッドの共通ヘルパー。
/// パターン: tl_tensor_{op}(tensor, args...) -> tensor
fn compile_tensor_reduce_op<'ctx>(
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

// ---- softmax(dim) -> Tensor ----
pub fn compile_softmax<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_reduce_op(c, o, t, a, "softmax")
}

// ---- log_softmax(dim) -> Tensor ----
pub fn compile_log_softmax<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_reduce_op(c, o, t, a, "log_softmax")
}

// ---- matmul(other) -> Tensor ----
pub fn compile_matmul<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_reduce_op(c, o, t, a, "matmul")
}

// ---- cross_entropy(target) -> Tensor ----
pub fn compile_cross_entropy<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_reduce_op(c, o, t, a, "cross_entropy")
}

// ---- pow(exp) -> Tensor ----
// FFI: tl_tensor_pow(ptr, ptr) for tensor, tl_tensor_pow_scalar(ptr, f32) for scalar
pub fn compile_pow<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if a.len() != 1 { return Err("pow requires 1 argument".into()); }
    let (arg_val, arg_ty) = &a[0];
    
    match arg_ty {
        Type::Tensor(_, _) => {
            // Tensor exponent: use tl_tensor_pow(ptr, ptr)
            compile_tensor_reduce_op(c, o, t, a, "pow")
        }
        Type::I64 | Type::I32 => {
            // Integer scalar: convert to f32, use tl_tensor_pow_scalar
            let f32_val = c.builder.build_signed_int_to_float(
                arg_val.into_int_value(),
                c.context.f32_type(),
                "pow_i64_to_f32",
            ).map_err(|e| e.to_string())?;
            let fn_val = c.module.get_function("tl_tensor_pow_scalar")
                .ok_or("tl_tensor_pow_scalar not found")?;
            let call = c.builder.build_call(fn_val, &[o.into(), f32_val.into()], "pow_res")
                .map_err(|e| e.to_string())?;
            let res = c.check_tensor_result(call, "pow_error")?;
            Ok((res, t))
        }
        Type::F32 => {
            // f32 scalar: use tl_tensor_pow_scalar directly
            let fn_val = c.module.get_function("tl_tensor_pow_scalar")
                .ok_or("tl_tensor_pow_scalar not found")?;
            let call = c.builder.build_call(fn_val, &[o.into(), (*arg_val).into()], "pow_res")
                .map_err(|e| e.to_string())?;
            let res = c.check_tensor_result(call, "pow_error")?;
            Ok((res, t))
        }
        Type::F64 => {
            // f64 scalar: cast to f32, use tl_tensor_pow_scalar
            let f32_val = c.builder.build_float_cast(
                arg_val.into_float_value(),
                c.context.f32_type(),
                "pow_f64_to_f32",
            ).map_err(|e| e.to_string())?;
            let fn_val = c.module.get_function("tl_tensor_pow_scalar")
                .ok_or("tl_tensor_pow_scalar not found")?;
            let call = c.builder.build_call(fn_val, &[o.into(), f32_val.into()], "pow_res")
                .map_err(|e| e.to_string())?;
            let res = c.check_tensor_result(call, "pow_error")?;
            Ok((res, t))
        }
        _ => Err(format!("pow: unsupported argument type {:?}", arg_ty)),
    }
}


// ---- sum_dim(dim, keepdim) -> Tensor ----
pub fn compile_sum_dim<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_reduce_op(c, o, t, a, "sum_dim")
}

// ---- mean_dim(dim, keepdim) -> Tensor ----
pub fn compile_mean_dim<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_reduce_op(c, o, t, a, "mean_dim")
}

// ---- max_dim(dim) -> Tensor ----
pub fn compile_max_dim<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_reduce_op(c, o, t, a, "max_dim")
}

// ---- min_dim(dim) -> Tensor ----
pub fn compile_min_dim<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_reduce_op(c, o, t, a, "min_dim")
}

// ---- mod(divisor) -> Tensor ----
pub fn compile_mod<'ctx>(c: &mut CodeGenerator<'ctx>, o: BasicValueEnum<'ctx>, t: Type, a: Vec<(BasicValueEnum<'ctx>, Type)>) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_reduce_op(c, o, t, a, "mod")
}
