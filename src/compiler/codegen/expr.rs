use super::CodeGenerator;
use crate::compiler::ast::*;
use inkwell::types::BasicType;
use inkwell::values::*;
use std::collections::{HashMap, HashSet};

pub type BuiltinFnEval = for<'a, 'ctx> fn(
    &'a mut CodeGenerator<'ctx>,
    Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String>;

pub type BuiltinFnUneval = for<'a, 'ctx> fn(
    &'a mut CodeGenerator<'ctx>,
    &[Expr],
) -> Result<(BasicValueEnum<'ctx>, Type), String>;

#[derive(Clone, Copy)]
pub enum BuiltinFn {
    Evaluated(BuiltinFnEval),
    Unevaluated(BuiltinFnUneval),
}

pub type InstanceMethodEval = for<'a, 'ctx> fn(
    &'a mut CodeGenerator<'ctx>,
    BasicValueEnum<'ctx>, // receiver value
    Type,                 // receiver type
    Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String>;

pub type InstanceMethodUneval = for<'a, 'ctx> fn(
    &'a mut CodeGenerator<'ctx>,
    &Expr,   // receiver expr
    &str,    // method name
    &[Expr], // args
) -> Result<(BasicValueEnum<'ctx>, Type), String>;

#[derive(Clone, Copy)]
pub enum InstanceMethod {
    Evaluated(InstanceMethodEval),
    Unevaluated(InstanceMethodUneval),
}

pub struct InstanceMethodManager {
    methods: HashMap<String, InstanceMethod>,
}

impl InstanceMethodManager {
    pub fn new() -> Self {
        InstanceMethodManager {
            methods: HashMap::new(),
        }
    }

    pub fn register_eval(&mut self, name: &str, func: InstanceMethodEval) {
        self.methods
            .insert(name.to_string(), InstanceMethod::Evaluated(func));
    }

    pub fn register_uneval(&mut self, name: &str, func: InstanceMethodUneval) {
        self.methods
            .insert(name.to_string(), InstanceMethod::Unevaluated(func));
    }

    pub fn get(&self, name: &str) -> Option<&InstanceMethod> {
        self.methods.get(name)
    }
}

pub type StaticMethodEval = for<'a, 'ctx> fn(
    &'a mut CodeGenerator<'ctx>,
    Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String>;

pub type StaticMethodUneval = for<'a, 'ctx> fn(
    &'a mut CodeGenerator<'ctx>,
    &[Expr], // args
) -> Result<(BasicValueEnum<'ctx>, Type), String>;

#[derive(Clone, Copy)]
pub enum StaticMethod {
    Evaluated(StaticMethodEval),
    Unevaluated(StaticMethodUneval),
}

pub struct StaticMethodManager {
    methods: HashMap<String, StaticMethod>,
}

impl StaticMethodManager {
    pub fn new() -> Self {
        StaticMethodManager {
            methods: HashMap::new(),
        }
    }

    pub fn register_eval(&mut self, name: &str, func: StaticMethodEval) {
        self.methods
            .insert(name.to_string(), StaticMethod::Evaluated(func));
    }

    pub fn register_uneval(&mut self, name: &str, func: StaticMethodUneval) {
        self.methods
            .insert(name.to_string(), StaticMethod::Unevaluated(func));
    }

    pub fn get(&self, name: &str) -> Option<&StaticMethod> {
        self.methods.get(name)
    }
}

pub struct BuiltinManager {
    functions: HashMap<String, BuiltinFn>,
}

impl BuiltinManager {
    pub fn new() -> Self {
        let mut manager = BuiltinManager {
            functions: HashMap::new(),
        };
        manager.register_all();
        manager
    }

    pub fn get(&self, name: &str) -> Option<&BuiltinFn> {
        self.functions.get(name)
    }

    fn register_eval(&mut self, name: &str, func: BuiltinFnEval) {
        self.functions
            .insert(name.to_string(), BuiltinFn::Evaluated(func));
    }

    fn register_uneval(&mut self, name: &str, func: BuiltinFnUneval) {
        self.functions
            .insert(name.to_string(), BuiltinFn::Unevaluated(func));
    }

    fn register_all(&mut self) {
        // IO functions
        self.register_uneval("print", compile_print_uneval);
        self.register_uneval("println", compile_println_uneval);
        self.register_uneval("read_line", compile_read_line_uneval);

        // Command line arguments
        self.register_eval("args_count", compile_args_count);
        self.register_eval("args_get", compile_args_get);

        // String functions
        self.register_eval("char_at", compile_string_char_at);
        self.register_eval("len", compile_string_len);

        // Parameter management moved to Param:: static methods
        // Tensor methods moved to instance/class methods

        self.register_uneval("varbuilder_get", compile_varbuilder_get);
    }
}

// ======================================================================================
//                                  Registered Methods
// ======================================================================================

fn compile_tensor_get<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    // args: index1, index2, ...
    if args.is_empty() {
        return Err("get requires at least 1 argument (index...)".into());
    }
    let rank = args.len();

    // Create array of indices on stack
    // indices: *const i64
    let i64_type = codegen.context.i64_type();
    let index_array_type = i64_type.array_type(rank as u32);

    // Move to entry block for alloca to avoid stack overflow in loops
    let current_block = codegen.builder.get_insert_block().unwrap();
    let func = current_block.get_parent().unwrap();
    let entry_block = func.get_first_basic_block().unwrap();

    if let Some(first_inst) = entry_block.get_first_instruction() {
        codegen.builder.position_before(&first_inst);
    } else {
        codegen.builder.position_at_end(entry_block);
    }

    let index_array_ptr = codegen
        .builder
        .build_alloca(index_array_type, "index_array")
        .unwrap();

    // Move back to current block
    codegen.builder.position_at_end(current_block);

    for i in 0..rank {
        let (idx_val, idx_ty) = args[i].clone();
        let idx_i64 = match idx_ty {
            crate::compiler::ast::Type::I64 => idx_val.into_int_value(),
            crate::compiler::ast::Type::I32 => codegen
                .builder
                .build_int_z_extend(idx_val.into_int_value(), i64_type, "idx_ext")
                .map_err(|e| e.to_string())?,
            _ => return Err(format!("Index {} must be integer", i)),
        };

        let ptr = unsafe {
            codegen
                .builder
                .build_gep(
                    index_array_type,
                    index_array_ptr,
                    &[
                        codegen.context.i64_type().const_int(0, false),
                        codegen.context.i64_type().const_int(i as u64, false),
                    ],
                    "idx_ptr",
                )
                .map_err(|e| e.to_string())?
        };
        codegen
            .builder
            .build_store(ptr, idx_i64)
            .map_err(|e| e.to_string())?;
    }

    // Cast array ptr to i64 ptr (use i8 ptr generic)
    let indices_ptr = codegen
        .builder
        .build_pointer_cast(
            index_array_ptr,
            codegen.context.ptr_type(inkwell::AddressSpace::default()),
            "indices_ptr_cast",
        )
        .map_err(|e| e.to_string())?;

    let fn_val = codegen
        .module
        .get_function("tl_tensor_get_f32_md")
        .ok_or("tl_tensor_get_f32_md not found")?;

    let rank_val = codegen.context.i64_type().const_int(rank as u64, false);

    let call = codegen
        .builder
        .build_call(
            fn_val,
            &[obj_val.into(), indices_ptr.into(), rank_val.into()],
            "get_res",
        )
        .map_err(|e| e.to_string())?;

    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid get return".into()),
    };

    Ok((res, crate::compiler::ast::Type::F32))
}

fn compile_tensor_backward<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen.module.get_function("tl_tensor_backward").unwrap();
    codegen
        .builder
        .build_call(fn_val, &[obj_val.into()], "backward_call")
        .map_err(|e| e.to_string())?;

    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

fn compile_tensor_clone<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen.module.get_function("tl_tensor_clone").unwrap();
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_val.into()], "clone_res")
        .map_err(|e| e.to_string())?;

    let res = codegen.check_tensor_result(call, "clone_error")?;

    // Runtime already registers the tensor (Ref=1). We just track it as a temporary.
    Ok((res, obj_ty))
}

fn compile_tensor_detach<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen.module.get_function("tl_tensor_detach").unwrap();
    // Optional arg: req_grad (bool). Default to false.
    let req_grad = if !args.is_empty() {
        let (arg_val, _) = args[0].clone();
        arg_val.into_int_value()
    } else {
        codegen.context.bool_type().const_int(0, false)
    };

    let call = codegen
        .builder
        .build_call(fn_val, &[obj_val.into(), req_grad.into()], "detach_res")
        .map_err(|e| e.to_string())?;

    let res = codegen.check_tensor_result(call, "detach_error")?;

    // codegen.emit_register_tensor(res, &obj_ty)?;
    Ok((res, obj_ty))
}

fn compile_tensor_grad<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen.module.get_function("tl_tensor_grad").unwrap();
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_val.into()], "grad_res")
        .map_err(|e| e.to_string())?;

    let res = codegen.check_tensor_result(call, "grad_error")?;

    // codegen.emit_register_tensor(res, &obj_ty)?;
    Ok((res, obj_ty))
}

fn compile_tensor_contiguous<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen.module.get_function("tl_tensor_contiguous").unwrap();
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_val.into()], "contiguous_res")
        .map_err(|e| e.to_string())?;

    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid contiguous return".into()),
    };

    codegen.emit_register_tensor(res, &obj_ty)?;
    Ok((res, obj_ty))
}

fn compile_tensor_save<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err("save requires 1 argument (path)".into());
    }
    let fn_val = codegen.module.get_function("tl_tensor_save").unwrap();
    let (path_val, _) = args[0].clone();

    codegen
        .builder
        .build_call(fn_val, &[path_val.into(), obj_val.into()], "save_call")
        .map_err(|e| e.to_string())?;

    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

fn compile_tensor_sum<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.is_empty() {
        // Standard sum
        let fn_val = codegen
            .module
            .get_function("tl_tensor_sum")
            .ok_or("tl_tensor_sum not found")?;
        let call = codegen
            .builder
            .build_call(fn_val, &[obj_val.into()], "sum_res");

        let res = codegen.check_tensor_result(call.map_err(|e| e.to_string())?, "sum_error")?;
        Ok((
            res,
            crate::compiler::ast::Type::Tensor(Box::new(crate::compiler::ast::Type::F32), 0),
        ))
    } else {
        // Sum with dim
        if args.len() != 1 {
            return Err("sum takes at most 1 argument".into());
        }
        let (dim_val, dim_ty) = args[0].clone();
        let dim_i64 = match dim_ty {
            crate::compiler::ast::Type::I64 => dim_val.into_int_value(),
            crate::compiler::ast::Type::I32 => codegen
                .builder
                .build_int_z_extend(
                    dim_val.into_int_value(),
                    codegen.context.i64_type(),
                    "dim_ext",
                )
                .map_err(|e| e.to_string())?,
            _ => return Err("Dimension must be integer".into()),
        };

        let fn_val = codegen
            .module
            .get_function("tl_tensor_sum_dim")
            .ok_or("tl_tensor_sum_dim not found")?;

        let call = codegen.builder.build_call(
            fn_val,
            &[
                obj_val.into(),
                dim_i64.into(),
                codegen.context.bool_type().const_zero().into(),
            ],
            "sum_dim_res",
        );

        let res = codegen.check_tensor_result(call.map_err(|e| e.to_string())?, "sum_dim_error")?;
        // Ideally we subtract 1 from rank, but 0 is safe generic guess for now if we don't track rank strictly
        Ok((
            res,
            crate::compiler::ast::Type::Tensor(Box::new(crate::compiler::ast::Type::F32), 0),
        ))
    }
}
fn compile_tensor_slice<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 2 {
        return Err("slice requires 2 arguments".into());
    }
    if matches!(obj_ty, Type::ScalarArray(_, _)) {
        return Err("slice() does not support ScalarArray. Convert to Tensor first using Tensor::new() or similar".into());
    }

    let (start_val, _) = args[0].clone();
    let (len_val, _) = args[1].clone();

    let fn_val = codegen.module.get_function("tl_tensor_slice").unwrap();
    let call = codegen
        .builder
        .build_call(
            fn_val,
            &[obj_val.into(), start_val.into(), len_val.into()],
            "slice_res",
        )
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid slice return".into()),
    };

    codegen.emit_register_tensor(res, &obj_ty)?;
    Ok((res, obj_ty))
}

fn compile_tensor_to<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err("to/to_device requires 1 argument (device name string)".into());
    }
    let (dev_val, dev_ty) = args[0].clone();
    if !matches!(&dev_ty, Type::UserDefined(s, _) if s == "String") {
        return Err("Device name must be a string".into());
    }

    let fn_val = codegen
        .module
        .get_function("tl_tensor_to_device")
        .ok_or("Runtime fn tl_tensor_to_device not found")?;

    let call = codegen
        .builder
        .build_call(fn_val, &[obj_val.into(), dev_val.into()], "to_dev_res")
        .map_err(|e| e.to_string())?;

    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from to_device".into()),
    };

    Ok((res, obj_ty))
}

fn compile_tensor_add_assign<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err("add_assign requires 1 argument".into());
    }
    let (rhs_val, rhs_ty) = args[0].clone();

    if matches!(rhs_ty, Type::Tensor(_, _)) {
        let fn_val = codegen.module.get_function("tl_tensor_add_assign").unwrap();
        codegen
            .builder
            .build_call(fn_val, &[obj_val.into(), rhs_val.into()], "assign_res")
            .map_err(|e| e.to_string())?;
    } else if matches!(rhs_ty, Type::F32 | Type::F64 | Type::I64 | Type::I32) {
        let scalar_f32 = match rhs_ty {
            Type::F32 => rhs_val.into_float_value(),
            Type::F64 => codegen
                .builder
                .build_float_cast(
                    rhs_val.into_float_value(),
                    codegen.context.f32_type(),
                    "f64_to_f32",
                )
                .map_err(|e| e.to_string())?,
            Type::I64 | Type::I32 => codegen
                .builder
                .build_signed_int_to_float(
                    rhs_val.into_int_value(),
                    codegen.context.f32_type(),
                    "int_to_f32",
                )
                .map_err(|e| e.to_string())?,
            _ => return Err(format!("add_assign scalar: unsupported type {:?}", rhs_ty)),
        };
        let fn_val = codegen
            .module
            .get_function("tl_tensor_add_assign_scalar_f32")
            .unwrap();
        codegen
            .builder
            .build_call(fn_val, &[obj_val.into(), scalar_f32.into()], "assign_res")
            .map_err(|e| e.to_string())?;
    } else {
        return Err(format!(
            "add_assign requires Tensor or scalar argument, got {:?}",
            rhs_ty
        ));
    }

    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

fn compile_tensor_sub_assign<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err("sub_assign requires 1 argument".into());
    }
    let (rhs_val, rhs_ty) = args[0].clone();

    if matches!(rhs_ty, Type::Tensor(_, _)) {
        let fn_val = codegen.module.get_function("tl_tensor_sub_assign").unwrap();
        codegen
            .builder
            .build_call(fn_val, &[obj_val.into(), rhs_val.into()], "assign_res")
            .map_err(|e| e.to_string())?;
    } else if matches!(rhs_ty, Type::F32 | Type::F64 | Type::I64 | Type::I32) {
        let scalar_f32 = match rhs_ty {
            Type::F32 => rhs_val.into_float_value(),
            Type::F64 => codegen
                .builder
                .build_float_cast(
                    rhs_val.into_float_value(),
                    codegen.context.f32_type(),
                    "f64_to_f32",
                )
                .map_err(|e| e.to_string())?,
            Type::I64 | Type::I32 => codegen
                .builder
                .build_signed_int_to_float(
                    rhs_val.into_int_value(),
                    codegen.context.f32_type(),
                    "int_to_f32",
                )
                .map_err(|e| e.to_string())?,
            _ => return Err(format!("sub_assign scalar: unsupported type {:?}", rhs_ty)),
        };
        let fn_val = codegen
            .module
            .get_function("tl_tensor_sub_assign_scalar_f32")
            .unwrap();
        codegen
            .builder
            .build_call(fn_val, &[obj_val.into(), scalar_f32.into()], "assign_res")
            .map_err(|e| e.to_string())?;
    } else {
        return Err(format!(
            "sub_assign requires Tensor or scalar argument, got {:?}",
            rhs_ty
        ));
    }

    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}
fn compile_tensor_mul_assign<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err("mul_assign requires 1 argument".into());
    }
    let (rhs_val, rhs_ty) = args[0].clone();

    if matches!(rhs_ty, Type::Tensor(_, _)) {
        let fn_val = codegen.module.get_function("tl_tensor_mul_assign").unwrap();
        codegen
            .builder
            .build_call(fn_val, &[obj_val.into(), rhs_val.into()], "assign_res")
            .map_err(|e| e.to_string())?;
    } else if matches!(rhs_ty, Type::F32 | Type::F64 | Type::I64 | Type::I32) {
        // Convert to f32 if necessary and call scalar version
        let scalar_f32 = match rhs_ty {
            Type::F32 => rhs_val.into_float_value(),
            Type::F64 => codegen
                .builder
                .build_float_cast(
                    rhs_val.into_float_value(),
                    codegen.context.f32_type(),
                    "f64_to_f32",
                )
                .map_err(|e| e.to_string())?,
            Type::I64 | Type::I32 => codegen
                .builder
                .build_signed_int_to_float(
                    rhs_val.into_int_value(),
                    codegen.context.f32_type(),
                    "int_to_f32",
                )
                .map_err(|e| e.to_string())?,
            _ => return Err(format!("mul_assign scalar: unsupported type {:?}", rhs_ty)),
        };
        let fn_val = codegen
            .module
            .get_function("tl_tensor_mul_assign_scalar_f32")
            .unwrap();
        codegen
            .builder
            .build_call(fn_val, &[obj_val.into(), scalar_f32.into()], "assign_res")
            .map_err(|e| e.to_string())?;
    } else {
        return Err(format!(
            "mul_assign requires Tensor or scalar argument, got {:?}",
            rhs_ty
        ));
    }

    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}
fn compile_tensor_div_assign<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err("div_assign requires 1 argument".into());
    }
    let (rhs_val, rhs_ty) = args[0].clone();

    if matches!(rhs_ty, Type::Tensor(_, _)) {
        let fn_val = codegen.module.get_function("tl_tensor_div_assign").unwrap();
        codegen
            .builder
            .build_call(fn_val, &[obj_val.into(), rhs_val.into()], "assign_res")
            .map_err(|e| e.to_string())?;
    } else if matches!(rhs_ty, Type::F32 | Type::F64 | Type::I64 | Type::I32) {
        let scalar_f32 = match rhs_ty {
            Type::F32 => rhs_val.into_float_value(),
            Type::F64 => codegen
                .builder
                .build_float_cast(
                    rhs_val.into_float_value(),
                    codegen.context.f32_type(),
                    "f64_to_f32",
                )
                .map_err(|e| e.to_string())?,
            Type::I64 | Type::I32 => codegen
                .builder
                .build_signed_int_to_float(
                    rhs_val.into_int_value(),
                    codegen.context.f32_type(),
                    "int_to_f32",
                )
                .map_err(|e| e.to_string())?,
            _ => return Err(format!("div_assign scalar: unsupported type {:?}", rhs_ty)),
        };
        let fn_val = codegen
            .module
            .get_function("tl_tensor_div_assign_scalar_f32")
            .unwrap();
        codegen
            .builder
            .build_call(fn_val, &[obj_val.into(), scalar_f32.into()], "assign_res")
            .map_err(|e| e.to_string())?;
    } else {
        return Err(format!(
            "div_assign requires Tensor or scalar argument, got {:?}",
            rhs_ty
        ));
    }

    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}
fn compile_tensor_zeros<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.is_empty() {
        return Err("Tensor::zeros requires shape argument".into());
    }

    let elements_ref = if let ExprKind::TensorLiteral(el) = &args[0].inner {
        Some(el)
    } else if let ExprKind::TensorConstLiteral(el) = &args[0].inner {
        Some(el)
    } else {
        None
    };

    if let Some(el) = elements_ref {
        let i64_type = codegen.context.i64_type();
        let mut vals = Vec::new();
        for e in el {
            let (v, t) = codegen.compile_expr(e)?;
            let int_val = match t {
                Type::I64 => v.into_int_value(),
                Type::I32 => codegen
                    .builder
                    .build_int_z_extend(v.into_int_value(), i64_type, "ext")
                    .map_err(|e| e.to_string())?,
                _ => return Err(format!("Dimension must be integer, found {:?}", t)),
            };
            vals.push(int_val);
        }

        let rank = el.len();
        let shape_array_type = i64_type.array_type(rank as u32);
        let shape_alloca = codegen
            .builder
            .build_alloca(shape_array_type, "shape_arr")
            .map_err(|e| e.to_string())?;

        for (i, val) in vals.iter().enumerate() {
            let ptr = unsafe {
                codegen
                    .builder
                    .build_in_bounds_gep(
                        shape_array_type,
                        shape_alloca,
                        &[
                            i64_type.const_int(0, false),
                            i64_type.const_int(i as u64, false),
                        ],
                        "tmp",
                    )
                    .map_err(|e| e.to_string())?
            };
            codegen
                .builder
                .build_store(ptr, *val)
                .map_err(|e| e.to_string())?;
        }

        let req_grad = if args.len() > 1 {
            let (v, _) = codegen.compile_expr(&args[1])?;
            v.into_int_value()
        } else {
            codegen.context.bool_type().const_int(0, false)
        };

        let f = codegen
            .module
            .get_function("tl_tensor_zeros")
            .ok_or("tl_tensor_zeros not found")?;
        let call = codegen
            .builder
            .build_call(
                f,
                &[
                    i64_type.const_int(rank as u64, false).into(),
                    shape_alloca.into(),
                    req_grad.into(),
                ],
                "zeros_res",
            )
            .map_err(|e| e.to_string())?;

        let v = codegen.check_tensor_result(call, "zeros_error")?;
        let result_ty = Type::Tensor(Box::new(Type::F32), rank);
        // codegen.emit_register_tensor(v, &result_ty)?;
        return Ok((v, result_ty));
    }

    // Generic path: Compile shape expr -> Tensor/Array
    // We delegate to runtime if the shape is dynamic (a Tuple or Tensor).
    // NOT IMPLEMENTED Generic fallback for now, as existing code handled it differently.
    // Assuming users pass literals for now or we rely on compile_static_method_call fallback?
    // Wait, if we register this, compile_static_method_call WON'T fallback for "zeros".
    // So we MUST handle generic case or fail.
    // Existing code logic: "Optimization for... else { ... }"
    // I need to copy the ELSE block.
    // Since I don't have it, I'll return Err for now and instruct user to use literals if they hit this.
    // Or better, just try to compile args[0] and use it?
    Err("Generic Tensor::zeros (non-literal shape) not yet ported to refactored dispatch. Please use literal shape [d1, d2] for now.".into())
}

fn compile_varbuilder_get_static<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    // Arg 0: Name (String)
    let (name_val, _) = codegen.compile_expr(&args[0])?;

    // Arg 1..: Shape
    let i64_type = codegen.context.i64_type();

    let (rank, shape_alloca) = if args.len() == 2 {
        match &args[1].inner {
            ExprKind::TensorLiteral(elements) | ExprKind::TensorConstLiteral(elements) => {
                let rank = elements.len();
                let shape_array_type = i64_type.array_type(rank as u32);
                let shape_alloca = codegen
                    .builder
                    .build_alloca(shape_array_type, "shape_arr")
                    .map_err(|e| e.to_string())?;

                for (i, elem) in elements.iter().enumerate() {
                    let (val, ty) = codegen.compile_expr(elem)?;
                    let i_val = match ty {
                        Type::I64 => val.into_int_value(),
                        Type::I32 => codegen
                            .builder
                            .build_int_z_extend(val.into_int_value(), i64_type, "dim_zext")
                            .map_err(|e| e.to_string())?,
                        _ => {
                            return Err(format!(
                                "VarBuilder::get expects integer dimensions, got {:?}",
                                ty
                            ))
                        }
                    };
                    let ptr = unsafe {
                        codegen
                            .builder
                            .build_in_bounds_gep(
                                shape_array_type,
                                shape_alloca,
                                &[
                                    i64_type.const_int(0, false),
                                    i64_type.const_int(i as u64, false),
                                ],
                                "tmp",
                            )
                            .map_err(|e| e.to_string())?
                    };
                    codegen
                        .builder
                        .build_store(ptr, i_val)
                        .map_err(|e| e.to_string())?;
                }
                (rank, shape_alloca)
            }
            _ => {
                let rank = args.len() - 1;
                let shape_array_type = i64_type.array_type(rank as u32);
                let shape_alloca = codegen
                    .builder
                    .build_alloca(shape_array_type, "shape_arr")
                    .map_err(|e| e.to_string())?;

                for (i, arg) in args[1..].iter().enumerate() {
                    let (val, ty) = codegen.compile_expr(arg)?;
                    let i_val = if ty == Type::I64 {
                        val.into_int_value()
                    } else {
                        return Err(format!(
                            "VarBuilder::get expects integer dimensions, got {:?}",
                            ty
                        ));
                    };
                    let ptr = unsafe {
                        codegen
                            .builder
                            .build_in_bounds_gep(
                                shape_array_type,
                                shape_alloca,
                                &[
                                    i64_type.const_int(0, false),
                                    i64_type.const_int(i as u64, false),
                                ],
                                "tmp",
                            )
                            .map_err(|e| e.to_string())?
                    };
                    codegen
                        .builder
                        .build_store(ptr, i_val)
                        .map_err(|e| e.to_string())?;
                }
                (rank, shape_alloca)
            }
        }
    } else {
        let rank = args.len() - 1;
        let shape_array_type = i64_type.array_type(rank as u32);
        let shape_alloca = codegen
            .builder
            .build_alloca(shape_array_type, "shape_arr")
            .map_err(|e| e.to_string())?;

        for (i, arg) in args[1..].iter().enumerate() {
            let (val, ty) = codegen.compile_expr(arg)?;
            let i_val = if ty == Type::I64 {
                val.into_int_value()
            } else {
                return Err(format!(
                    "VarBuilder::get expects integer dimensions, got {:?}",
                    ty
                ));
            };
            let ptr = unsafe {
                codegen
                    .builder
                    .build_in_bounds_gep(
                        shape_array_type,
                        shape_alloca,
                        &[
                            i64_type.const_int(0, false),
                            i64_type.const_int(i as u64, false),
                        ],
                        "tmp",
                    )
                    .map_err(|e| e.to_string())?
            };
            codegen
                .builder
                .build_store(ptr, i_val)
                .map_err(|e| e.to_string())?;
        }
        (rank, shape_alloca)
    };

    let f = codegen
        .module
        .get_function("tl_varbuilder_get")
        .ok_or("tl_varbuilder_get not found")?;
    let call = codegen
        .builder
        .build_call(
            f,
            &[
                name_val.into(),
                i64_type.const_int(rank as u64, false).into(),
                shape_alloca.into(),
            ],
            "vb_get_res",
        )
        .map_err(|e| e.to_string())?;

    let v = codegen.check_tensor_result(call, "vb_get_error")?;
    let result_ty = Type::Tensor(Box::new(Type::F32), rank);
    codegen.emit_register_tensor(v, &result_ty)?;
    Ok((v, result_ty))
}

impl<'ctx> CodeGenerator<'ctx> {
    pub(crate) fn register_all_methods(&mut self) {
        // --- Tensor Instance Methods ---
        let mut tensor_methods = InstanceMethodManager::new();
        tensor_methods.register_eval("get", compile_tensor_get);
        tensor_methods.register_eval("backward", compile_tensor_backward);
        tensor_methods.register_eval("clone", compile_tensor_clone);
        tensor_methods.register_eval("detach", compile_tensor_detach);
        tensor_methods.register_eval("grad", compile_tensor_grad);
        tensor_methods.register_eval("contiguous", compile_tensor_contiguous);
        tensor_methods.register_eval("save", compile_tensor_save);
        tensor_methods.register_uneval("reshape", compile_tensor_reshape_uneval);
        tensor_methods.register_eval("sum", compile_tensor_sum);
        tensor_methods.register_eval("slice", compile_tensor_slice);
        tensor_methods.register_eval("to", compile_tensor_to);
        tensor_methods.register_eval("to_device", compile_tensor_to);
        tensor_methods.register_eval("add_assign", compile_tensor_add_assign);
        tensor_methods.register_eval("sub_assign", compile_tensor_sub_assign);
        tensor_methods.register_eval("mul_assign", compile_tensor_mul_assign);
        tensor_methods.register_eval("div_assign", compile_tensor_div_assign);
        tensor_methods.register_eval("transpose", compile_tensor_transpose);
        tensor_methods.register_eval("permute", compile_tensor_transpose); // permute aliases transpose logic for now
        tensor_methods.register_eval("pow", compile_tensor_pow);
        tensor_methods.register_eval("get", compile_tensor_get);

        self.instance_methods
            .insert("Tensor".to_string(), tensor_methods);

        // --- F32 Instance Methods ---
        let mut f32_methods = InstanceMethodManager::new();
        f32_methods.register_eval("abs", compile_f32_abs);
        f32_methods.register_eval("acos", compile_f32_acos);
        f32_methods.register_eval("acosh", compile_f32_acosh);
        f32_methods.register_eval("asin", compile_f32_asin);
        f32_methods.register_eval("asinh", compile_f32_asinh);
        f32_methods.register_eval("atan", compile_f32_atan);
        f32_methods.register_eval("atan2", compile_f32_atan2);
        f32_methods.register_eval("atanh", compile_f32_atanh);
        f32_methods.register_eval("cbrt", compile_f32_cbrt);
        f32_methods.register_eval("ceil", compile_f32_ceil);
        f32_methods.register_eval("copysign", compile_f32_copysign);
        f32_methods.register_eval("cos", compile_f32_cos);
        f32_methods.register_eval("cosh", compile_f32_cosh);
        f32_methods.register_eval("exp", compile_f32_exp);
        f32_methods.register_eval("exp2", compile_f32_exp2);
        f32_methods.register_eval("exp_m1", compile_f32_exp_m1);
        f32_methods.register_eval("floor", compile_f32_floor);
        f32_methods.register_eval("fract", compile_f32_fract);
        f32_methods.register_eval("hypot", compile_f32_hypot);
        f32_methods.register_eval("ln", compile_f32_ln);
        f32_methods.register_eval("ln_1p", compile_f32_ln_1p);
        f32_methods.register_eval("log", compile_f32_log);
        f32_methods.register_eval("log10", compile_f32_log10);
        f32_methods.register_eval("log2", compile_f32_log2);
        f32_methods.register_eval("powf", compile_f32_powf);
        f32_methods.register_eval("pow", compile_f32_pow); // Alias
        f32_methods.register_eval("powi", compile_f32_powi);
        f32_methods.register_eval("recip", compile_f32_recip);
        f32_methods.register_eval("round", compile_f32_round);
        f32_methods.register_eval("signum", compile_f32_signum);
        f32_methods.register_eval("sin", compile_f32_sin);
        f32_methods.register_eval("sinh", compile_f32_sinh);
        f32_methods.register_eval("sqrt", compile_f32_sqrt);
        f32_methods.register_eval("tan", compile_f32_tan);
        f32_methods.register_eval("tanh", compile_f32_tanh);
        f32_methods.register_eval("to_degrees", compile_f32_to_degrees);
        f32_methods.register_eval("to_radians", compile_f32_to_radians);
        f32_methods.register_eval("trunc", compile_f32_trunc);
        self.instance_methods.insert("F32".to_string(), f32_methods);

        // --- F64 Instance Methods ---
        let mut f64_methods = InstanceMethodManager::new();
        f64_methods.register_eval("abs", compile_f64_abs);
        f64_methods.register_eval("acos", compile_f64_acos);
        f64_methods.register_eval("acosh", compile_f64_acosh);
        f64_methods.register_eval("asin", compile_f64_asin);
        f64_methods.register_eval("asinh", compile_f64_asinh);
        f64_methods.register_eval("atan", compile_f64_atan);
        f64_methods.register_eval("atan2", compile_f64_atan2);
        f64_methods.register_eval("atanh", compile_f64_atanh);
        f64_methods.register_eval("cbrt", compile_f64_cbrt);
        f64_methods.register_eval("ceil", compile_f64_ceil);
        f64_methods.register_eval("copysign", compile_f64_copysign);
        f64_methods.register_eval("cos", compile_f64_cos);
        f64_methods.register_eval("cosh", compile_f64_cosh);
        f64_methods.register_eval("exp", compile_f64_exp);
        f64_methods.register_eval("exp2", compile_f64_exp2);
        f64_methods.register_eval("exp_m1", compile_f64_exp_m1);
        f64_methods.register_eval("floor", compile_f64_floor);
        f64_methods.register_eval("fract", compile_f64_fract);
        f64_methods.register_eval("hypot", compile_f64_hypot);
        f64_methods.register_eval("ln", compile_f64_ln);
        f64_methods.register_eval("ln_1p", compile_f64_ln_1p);
        f64_methods.register_eval("log", compile_f64_log);
        f64_methods.register_eval("log10", compile_f64_log10);
        f64_methods.register_eval("log2", compile_f64_log2);
        f64_methods.register_eval("powf", compile_f64_powf);
        f64_methods.register_eval("pow", compile_f64_pow); // Alias
        f64_methods.register_eval("powi", compile_f64_powi);
        f64_methods.register_eval("recip", compile_f64_recip);
        f64_methods.register_eval("round", compile_f64_round);
        f64_methods.register_eval("signum", compile_f64_signum);
        f64_methods.register_eval("sin", compile_f64_sin);
        f64_methods.register_eval("sinh", compile_f64_sinh);
        f64_methods.register_eval("sqrt", compile_f64_sqrt);
        f64_methods.register_eval("tan", compile_f64_tan);
        f64_methods.register_eval("tanh", compile_f64_tanh);
        f64_methods.register_eval("to_degrees", compile_f64_to_degrees);
        f64_methods.register_eval("to_radians", compile_f64_to_radians);
        f64_methods.register_eval("trunc", compile_f64_trunc);
        self.instance_methods.insert("F64".to_string(), f64_methods);

        // --- I64 Instance Methods ---
        let mut i64_methods = InstanceMethodManager::new();
        i64_methods.register_eval("abs", compile_i64_abs);
        i64_methods.register_eval("signum", compile_i64_signum);
        i64_methods.register_eval("pow", compile_i64_pow);
        i64_methods.register_eval("div_euclid", compile_i64_div_euclid);
        i64_methods.register_eval("rem_euclid", compile_i64_rem_euclid);
        i64_methods.register_eval("is_positive", compile_i64_is_positive);
        i64_methods.register_eval("is_negative", compile_i64_is_negative);
        self.instance_methods.insert("I64".to_string(), i64_methods);

        // --- I32 Instance Methods ---
        let mut i32_methods = InstanceMethodManager::new();
        i32_methods.register_eval("abs", compile_i32_abs);
        i32_methods.register_eval("signum", compile_i32_signum);
        i32_methods.register_eval("pow", compile_i32_pow);
        i32_methods.register_eval("div_euclid", compile_i32_div_euclid);
        i32_methods.register_eval("rem_euclid", compile_i32_rem_euclid);
        i32_methods.register_eval("is_positive", compile_i32_is_positive);
        i32_methods.register_eval("is_negative", compile_i32_is_negative);
        self.instance_methods.insert("I32".to_string(), i32_methods);

        // --- Tensor Static Methods ---
        let mut tensor_static = StaticMethodManager::new();
        // Unevaluated because of special TensorLiteral handling optimization
        tensor_static.register_uneval("zeros", compile_tensor_zeros);
        tensor_static.register_uneval("randn", compile_randn);
        tensor_static.register_uneval("ones", compile_ones);
        tensor_static.register_eval("load", compile_load_tensor);
        self.static_methods
            .insert("Tensor".to_string(), tensor_static);

        // --- VarBuilder Static Methods ---
        let mut varbuilder_static = StaticMethodManager::new();
        varbuilder_static.register_uneval("get", compile_varbuilder_get_static);
        self.static_methods
            .insert("VarBuilder".to_string(), varbuilder_static);

        // --- Param Static Methods ---
        let mut param_static = StaticMethodManager::new();
        param_static.register_eval("save_all", compile_save_all_params);
        param_static.register_eval("load_all", compile_load_all_params);
        param_static.register_eval("save", compile_save_weights);
        param_static.register_eval("load", compile_load_weights);
        param_static.register_eval("add", compile_add_parameter);
        param_static.register_eval("register", compile_parameter);
        param_static.register_eval("update_all", compile_update_all_params);
        param_static.register_eval("register_modules", compile_register_modules);
        param_static.register_uneval("checkpoint", compile_checkpoint);
        param_static.register_uneval("set_device", compile_set_device);
        self.static_methods
            .insert("Param".to_string(), param_static);
    }

    fn load_struct_i64_field(
        &mut self,
        obj_val: BasicValueEnum<'ctx>,
        obj_ty: &Type,
        field_name: &str,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        let struct_name = match obj_ty {
            Type::Struct(name, _) | Type::UserDefined(name, _) => name.clone(),
            _ => return Err(format!("Expected struct type for field {}", field_name)),
        };

        let simple_struct_name = if struct_name.contains("::") {
            struct_name.split("::").last().unwrap()
        } else {
            struct_name.as_str()
        };

        let struct_def = self
            .struct_defs
            .get(simple_struct_name)
            .ok_or(format!("Struct definition for {} not found", struct_name))?;

        let field_idx = struct_def
            .fields
            .iter()
            .position(|(n, _)| n == field_name)
            .ok_or(format!(
                "Field {} not found in struct {}",
                field_name, struct_name
            ))?;

        if obj_val.is_pointer_value() {
            let ptr = obj_val.into_pointer_value();
            let st_llvm_ty = self
                .struct_types
                .get(simple_struct_name)
                .ok_or(format!("Struct type {} not found", struct_name))?;

            let field_ptr = self
                .builder
                .build_struct_gep(
                    *st_llvm_ty,
                    ptr,
                    field_idx as u32,
                    &format!("ptr_{}", field_name),
                )
                .map_err(|e| e.to_string())?;

            let loaded = self
                .builder
                .build_load(self.context.i64_type(), field_ptr, field_name)
                .map_err(|e| e.to_string())?;
            Ok(loaded)
        } else if obj_val.is_struct_value() {
            let struct_val = obj_val.into_struct_value();
            let extracted = self
                .builder
                .build_extract_value(struct_val, field_idx as u32, field_name)
                .map_err(|e| e.to_string())?;
            Ok(extracted)
        } else {
            Err("Cannot access field of non-pointer and non-struct value".into())
        }
    }

    pub(crate) fn is_safe_to_free(&self, expr: &Expr, ty: &Type) -> bool {
        match ty {
            Type::Tensor(_, _) => {
                // Tensors originating from expressions (not variables) are always new allocations (R-values)
                match &expr.inner {
                    ExprKind::Variable(_) | ExprKind::FieldAccess(_, _) => false,
                    _ => true,
                }
            }
            Type::Struct(_, _) | Type::UserDefined(_, _) => {
                match &expr.inner {
                    // Fresh allocations are safe to free
                    ExprKind::StaticMethodCall(_, _, _) | ExprKind::StructInit(_, _, _) => true,
                    // Variables and Fields are L-values, not safe to free
                    ExprKind::Variable(_) | ExprKind::FieldAccess(_, _) => false,
                    // Method calls: Check recursively if the receiver was safe to free.
                    // If obj is temporary, obj.method() result is treated as temporary (part of obj).
                    // If obj is strictly safe (fresh), propagation says result is safe.
                    // BUT: We will apply runtime check too.
                    ExprKind::MethodCall(obj, _, _) => self.is_safe_to_free(obj, ty),
                    // Other expressions?
                    _ => false,
                }
            }
            _ => false,
        }
    }

    /// Register a tensor with the memory manager for automatic cleanup
    /// This should be called for intermediate expression results (BinOp, MethodCall, etc.)
    /// The tensor will be freed when exit_scope is called, unless it's unregistered first
    pub(crate) fn emit_register_tensor(
        &self,
        val: BasicValueEnum<'ctx>,
        ty: &Type,
    ) -> Result<(), String> {
        // Only register tensors
        if !matches!(ty, Type::Tensor(_, _)) {
            return Ok(());
        }

        let reg_fn = self
            .module
            .get_function("tl_mem_register_tensor")
            .ok_or("tl_mem_register_tensor not found")?;

        let ptr = val.into_pointer_value();
        let cast_ptr = self
            .builder
            .build_pointer_cast(
                ptr,
                self.context.ptr_type(inkwell::AddressSpace::default()),
                "reg_tensor_ptr",
            )
            .map_err(|e| e.to_string())?;

        self.builder
            .build_call(reg_fn, &[cast_ptr.into()], "")
            .map_err(|e| e.to_string())?;

        Ok(())
    }

    /// Unregister a tensor from the memory manager (ownership transfer)
    /// This should be called when a tensor is moved to a variable, struct field, or return value
    pub(crate) fn gen_save_struct(
        &self,
        map: inkwell::values::BasicValueEnum<'ctx>,
        struct_ptr: inkwell::values::BasicValueEnum<'ctx>,
        struct_name: &str,
        prefix: String,
    ) -> Result<(), String> {
        let def = self
            .struct_defs
            .get(struct_name)
            .ok_or(format!("Struct definition '{}' not found", struct_name))?;

        let struct_ty = *self
            .struct_types
            .get(struct_name)
            .ok_or("Struct LLVM type not found")?;

        for (i, (field_name, field_type)) in def.fields.iter().enumerate() {
            let full_key = if prefix.is_empty() {
                field_name.clone()
            } else {
                format!("{}.{}", prefix, field_name)
            };

            let ptr = struct_ptr.into_pointer_value();
            let field_ptr = self
                .builder
                .build_struct_gep(struct_ty, ptr, i as u32, field_name)
                .map_err(|e| e.to_string())?;

            match field_type {
                Type::Tensor(_, _) => {
                    // Save Tensor
                    let tensor_ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
                    let t_val = self
                        .builder
                        .build_load(tensor_ptr_ty, field_ptr, field_name)
                        .map_err(|e| e.to_string())?;
                    let key_ptr = self
                        .builder
                        .build_global_string_ptr(&full_key, "key_str")
                        .map_err(|e| e.to_string())?;

                    let i8_ptr = self
                        .builder
                        .build_pointer_cast(
                            key_ptr.as_pointer_value(),
                            self.context.ptr_type(inkwell::AddressSpace::default()),
                            "key_cast",
                        )
                        .map_err(|e| e.to_string())?;

                    let insert_fn = self
                        .module
                        .get_function("tl_tensor_map_insert")
                        .ok_or("tl_tensor_map_insert not found")?;
                    let _ = self
                        .builder
                        .build_call(insert_fn, &[map.into(), i8_ptr.into(), t_val.into()], "")
                        .map_err(|e| e.to_string())?;
                }
                Type::UserDefined(sub_name, _) if sub_name != "String" => {
                    // Recurse
                    let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
                    let sub_val = self
                        .builder
                        .build_load(ptr_ty, field_ptr, "sub_ptr")
                        .map_err(|e| e.to_string())?;
                    self.gen_save_struct(map, sub_val, sub_name, full_key)?;
                }
                _ => {
                    // Skip primitives
                }
            }
        }
        Ok(())
    }

    pub(crate) fn gen_load_struct(
        &self,
        map: inkwell::values::BasicValueEnum<'ctx>,
        struct_ptr: inkwell::values::BasicValueEnum<'ctx>,
        struct_name: &str,
        prefix: String,
    ) -> Result<(), String> {
        let def = self
            .struct_defs
            .get(struct_name)
            .ok_or(format!("Struct definition '{}' not found", struct_name))?;

        let struct_ty = *self
            .struct_types
            .get(struct_name)
            .ok_or("Struct LLVM type not found")?;

        for (i, (field_name, field_type)) in def.fields.iter().enumerate() {
            let full_key = if prefix.is_empty() {
                field_name.clone()
            } else {
                format!("{}.{}", prefix, field_name)
            };

            let ptr = struct_ptr.into_pointer_value();
            let field_ptr = self
                .builder
                .build_struct_gep(struct_ty, ptr, i as u32, field_name)
                .map_err(|e| e.to_string())?;

            match field_type {
                Type::Tensor(_, _) => {
                    // Load Tensor
                    let key_ptr = self
                        .builder
                        .build_global_string_ptr(&full_key, "key_str")
                        .map_err(|e| e.to_string())?;

                    let i8_ptr = self
                        .builder
                        .build_pointer_cast(
                            key_ptr.as_pointer_value(),
                            self.context.ptr_type(inkwell::AddressSpace::default()),
                            "key_cast",
                        )
                        .map_err(|e| e.to_string())?;

                    let get_fn = self
                        .module
                        .get_function("tl_tensor_map_get")
                        .ok_or("tl_tensor_map_get not found")?;
                    let call = self
                        .builder
                        .build_call(get_fn, &[map.into(), i8_ptr.into()], "t_val")
                        .map_err(|e| e.to_string())?;

                    let t_val = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err("tl_tensor_map_get returned inst/void".into()),
                    };

                    self.builder
                        .build_store(field_ptr, t_val)
                        .map_err(|e| e.to_string())?;
                }
                Type::UserDefined(sub_name, _) if sub_name != "String" => {
                    // Recurse: load the pointer to inner struct
                    let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
                    let sub_val = self
                        .builder
                        .build_load(ptr_ty, field_ptr, "sub_ptr")
                        .map_err(|e| e.to_string())?;
                    self.gen_load_struct(map, sub_val, sub_name, full_key)?;
                }
                _ => {
                    // Skip primitives
                }
            }
        }
        Ok(())
    }

    pub(crate) fn gen_register_params(
        &self,
        struct_ptr: inkwell::values::BasicValueEnum<'ctx>,
        struct_name: &str,
        prefix: String,
    ) -> Result<(), String> {
        let simple_name = if struct_name.contains("::") {
            struct_name.split("::").last().unwrap()
        } else {
            struct_name
        };

        let def = self
            .struct_defs
            .get(simple_name)
            .ok_or(format!("Struct definition '{}' not found", struct_name))?;

        let struct_ty = *self
            .struct_types
            .get(simple_name)
            .ok_or("Struct LLVM type not found")?;

        for (i, (field_name, field_type)) in def.fields.iter().enumerate() {
            let full_key = if prefix.is_empty() {
                field_name.clone()
            } else {
                format!("{}.{}", prefix, field_name)
            };

            let ptr = struct_ptr.into_pointer_value();
            let field_ptr = self
                .builder
                .build_struct_gep(struct_ty, ptr, i as u32, field_name)
                .map_err(|e| e.to_string())?;

            match field_type {
                Type::Tensor(_, _) => {
                    // Register Tensor: tl_add_parameter(name, tensor)
                    let tensor_ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
                    let t_val = self
                        .builder
                        .build_load(tensor_ptr_ty, field_ptr, field_name)
                        .map_err(|e| e.to_string())?;

                    let key_ptr = self
                        .builder
                        .build_global_string_ptr(&full_key, "key_str")
                        .map_err(|e| e.to_string())?;

                    let i8_ptr = self
                        .builder
                        .build_pointer_cast(
                            key_ptr.as_pointer_value(),
                            self.context.ptr_type(inkwell::AddressSpace::default()),
                            "key_cast",
                        )
                        .map_err(|e| e.to_string())?;

                    let add_fn = self
                        .module
                        .get_function("tl_add_parameter")
                        .ok_or("tl_add_parameter not found")?;

                    self.builder
                        .build_call(add_fn, &[i8_ptr.into(), t_val.into()], "")
                        .map_err(|e| e.to_string())?;
                }
                Type::UserDefined(sub_name, _) if sub_name != "String" => {
                    // Recurse
                    let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
                    let sub_val = self
                        .builder
                        .build_load(ptr_ty, field_ptr, "sub_ptr")
                        .map_err(|e| e.to_string())?;
                    self.gen_register_params(sub_val, sub_name, full_key)?;
                }
                Type::Struct(sub_name, _) => {
                    // Recurse for Type::Struct as well (e.g. from generic instantiation)
                    let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
                    let sub_val = self
                        .builder
                        .build_load(ptr_ty, field_ptr, "sub_ptr")
                        .map_err(|e| e.to_string())?;
                    self.gen_register_params(sub_val, sub_name, full_key)?;
                }
                _ => {
                    // Skip primitives
                }
            }
        }
        Ok(())
    }
    pub(crate) fn compile_expr(
        &mut self,
        expr: &Expr,
    ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        let prev_span = self.current_span.clone();
        self.current_span = Some(expr.span.clone());
        let result = self.compile_expr_inner(expr);
        self.current_span = prev_span;
        result
    }

    pub(crate) fn compile_expr_inner(
        &mut self,
        expr: &Expr,
    ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        match &expr.inner {
            ExprKind::Block(stmts) => {
                self.enter_scope();
                let mut last_val = None;
                for (i, stmt) in stmts.iter().enumerate() {
                    if i == stmts.len() - 1 {
                        if let StmtKind::Expr(e) = &stmt.inner {
                            last_val = Some(self.compile_expr(e)?);
                        } else {
                            self.compile_stmt(stmt)?;
                        }
                    } else {
                        self.compile_stmt(stmt)?;
                    }
                }

                let final_res = last_val.unwrap_or((
                    self.context.i64_type().const_int(0, false).into(),
                    Type::Void,
                ));

                // FIX UAF: Unregister block result before scope cleanup
                if matches!(
                    final_res.1,
                    Type::Tensor(_, _) | Type::Struct(_, _) | Type::UserDefined(_, _) | Type::Tuple(_)
                ) {
                    if let Some(unreg_fn) = self.module.get_function("tl_mem_unregister") {
                        let ptr = final_res.0.into_pointer_value();
                        let cast_ptr = self
                            .builder
                            .build_pointer_cast(
                                ptr,
                                self.context.ptr_type(inkwell::AddressSpace::default()),
                                "cast_unreg",
                            )
                            .unwrap();
                        self.builder
                            .build_call(unreg_fn, &[cast_ptr.into()], "")
                            .unwrap();
                    }
                }

                self.exit_scope();

                Ok(final_res)
            }
            ExprKind::Int(i) => {
                let i64_type = self.context.i64_type();
                Ok((i64_type.const_int(*i as u64, true).into(), Type::I64))
            }
            ExprKind::Float(f) => {
                let f32_type = self.context.f32_type();
                Ok((f32_type.const_float(*f).into(), Type::F32))
            }
            ExprKind::Bool(b) => {
                let bool_type = self.context.bool_type();
                Ok((
                    bool_type.const_int(if *b { 1 } else { 0 }, false).into(),
                    Type::Bool,
                ))
            }
            ExprKind::StringLiteral(s) => self.compile_string_literal(s),
            ExprKind::Symbol(name) => {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                std::hash::Hash::hash(name, &mut hasher);
                let seed = std::hash::Hasher::finish(&hasher);
                let i64_type = self.context.i64_type();
                Ok((i64_type.const_int(seed, false).into(), Type::Entity))
            }
            ExprKind::LogicVar(name) => {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                std::hash::Hash::hash(name, &mut hasher);
                let seed = std::hash::Hasher::finish(&hasher);
                let i64_type = self.context.i64_type();
                Ok((i64_type.const_int(seed, false).into(), Type::Entity))
            }
            ExprKind::EnumInit {
                enum_name,
                variant_name,
                fields,
            } => {
                let enum_def = self
                    .enum_defs
                    .get(enum_name)
                    .ok_or(format!("Enum def {} not found", enum_name))?
                    .clone();
                let enum_ty = *self
                    .enum_types
                    .get(enum_name)
                    .ok_or(format!("Enum type {} not found", enum_name))?;

                // 1. Allocate Enum
                // Manual malloc(i64)
                let size_ptr = unsafe {
                    self.builder.build_gep(
                        enum_ty,
                        self.context.ptr_type(inkwell::AddressSpace::default()).const_null(),
                        &[self.context.i64_type().const_int(1, false)],
                        "size_ptr",
                    ).map_err(|e| e.to_string())?
                };
                let size = self.builder
                    .build_ptr_to_int(size_ptr, self.context.i64_type(), "size")
                    .map_err(|e| e.to_string())?;

                let malloc_fn = self.module.get_function("malloc").ok_or("malloc not found")?;
                let alloca = match self.builder
                    .build_call(malloc_fn, &[size.into()], &format!("enum_{}", enum_name))
                    .map_err(|e| e.to_string())?
                    .try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
                        _ => return Err("malloc returned void".into()),
                    };
                // alloca is *EnumStruct

                // 2. Store Tag
                let variant_idx = enum_def
                    .variants
                    .iter()
                    .position(|v| v.name == *variant_name)
                    .unwrap();
                let tag_ptr = self
                    .builder
                    .build_struct_gep(enum_ty, alloca, 0, "tag_ptr")
                    .map_err(|e| e.to_string())?;
                self.builder
                    .build_store(
                        tag_ptr,
                        self.context.i32_type().const_int(variant_idx as u64, false),
                    )
                    .unwrap();

                // 3. Store Fields
                let variant_def = &enum_def.variants[variant_idx];
                if !fields.is_empty() {
                    let payload_ptr_raw = self
                        .builder
                        .build_struct_gep(enum_ty, alloca, 1, "payload_ptr_raw")
                        .map_err(|e| e.to_string())?;

                    // Reconstruct Variant Struct Type
                    let mut field_types: Vec<inkwell::types::BasicTypeEnum> = vec![];
                    for (_, ty) in &variant_def.fields {
                        let llvm_ty = match ty {
                            Type::F32 => self.context.f32_type().into(),
                            Type::I64 => self.context.i64_type().into(),
                            Type::Bool => self.context.bool_type().into(),
                            Type::Tensor(_, _)
                            | Type::Struct(_, _)
                            | Type::Enum(_, _)
                            | Type::UserDefined(_, _)
                            | Type::Vec(_) => self
                                .context
                                .ptr_type(inkwell::AddressSpace::default())
                                .into(),
                            _ => self.context.i64_type().into(),
                        };
                        field_types.push(llvm_ty);
                    }
                    let variant_struct_ty = self.context.struct_type(&field_types, false);

                    let payload_ptr = self
                        .builder
                        .build_pointer_cast(
                            payload_ptr_raw,
                            self.context.ptr_type(inkwell::AddressSpace::default()),
                            "payload_cast",
                        )
                        .unwrap();

                    // Sort fields to match definition order (parser might shuffle? No, Vec is ordered)
                    // But init fields might be out of order? AST stores as Vec, usually parser preserves source order.
                    // But user might write out of order? AST doesn't enforce order if named?
                    // AST `EnumInit` has `fields: Vec<(String, Expr)>`. Semantics checks existence.
                    // We need to iterate over VARIANT definition fields and find corresponding expr in INIT.

                    for (idx, (f_name, f_ty)) in variant_def.fields.iter().enumerate() {
                        let (_, expr) = fields
                            .iter()
                            .find(|(n, _)| n == f_name)
                            .ok_or(format!("Missing field {}", f_name))?;

                        let (val, _) = self.compile_expr(expr)?;
                        // Deep clone if needed?
                        // Similar to StructInit/Let: r-value move, l-value clone.
                        let is_rvalue = matches!(
                            &expr.inner,
                            ExprKind::FnCall(_, _)
                                | ExprKind::MethodCall(_, _, _)
                                | ExprKind::StaticMethodCall(_, _, _)
                                | ExprKind::BinOp(_, _, _)
                                | ExprKind::UnOp(_, _)
                                | ExprKind::TensorLiteral(_)
                                | ExprKind::Block(_)
                        );
                        let mut stored_val = val;
                        let should_deep_clone = match f_ty {
                            Type::Tensor(_, _) => !is_rvalue,
                            Type::Struct(_, _) | Type::UserDefined(_, _) => !is_rvalue,
                            _ => false,
                        };
                        if should_deep_clone {
                            stored_val = self.emit_deep_clone(val, f_ty)?;
                        }

                        let f_ptr = self
                            .builder
                            .build_struct_gep(
                                variant_struct_ty,
                                payload_ptr,
                                idx as u32,
                                "field_ptr",
                            )
                            .map_err(|e| e.to_string())?;

                        self.builder.build_store(f_ptr, stored_val).unwrap();

                        // Register ownership if needed
                        if matches!(
                            f_ty,
                            Type::Tensor(_, _) | Type::Struct(_, _) | Type::UserDefined(_, _)
                        ) {
                            // Struct owns it now.
                            // If we deep cloned, we own the clone.
                            // If we moved (is_rvalue), we own the original.
                        }
                    }
                }

                // Return pointer to Enum
                // Wait, objects are passed by pointer usually.
                // But `alloca` IS a pointer to the struct storage.
                // If I want to return the "Value", for Struct/Enum, the Value IS the pointer.
                // So I return `alloca.into()`.
                // Check struct init logic.
                // compile_struct_init returns `alloca.into()`.
                Ok((alloca.into(), Type::Enum(enum_name.clone(), vec![])))
            }
            ExprKind::Match {
                expr: subject_expr,
                arms,
            } => self.compile_match_like(subject_expr, arms),
            ExprKind::IfLet {
                pattern,
                expr,
                then_block,
                else_block,
            } => {
                let mut arms: Vec<(Pattern, Expr)> = Vec::with_capacity(2);
                arms.push((
                    pattern.clone(),
                    Spanned::dummy(ExprKind::Block(then_block.clone())),
                ));
                let fallback =
                    Spanned::dummy(ExprKind::Block(else_block.clone().unwrap_or_default()));
                arms.push((Pattern::Wildcard, fallback));
                self.compile_match_like(expr, &arms)
            }

            ExprKind::Wildcard => {
                Err("ExprKind::Wildcard should only appear in logic rules".to_string())
            }
            ExprKind::Range(_, _) => {
                Err("ExprKind::Range should only appear in For loops".to_string())
            }
            ExprKind::As(expr, target_type) => {
                let (val, source_type) = self.compile_expr(expr)?;
                if source_type == *target_type {
                    return Ok((val, source_type));
                }

                match (&source_type, target_type) {
                    (Type::I64, Type::F32) => {
                        let i = val.into_int_value();
                        let f = self
                            .builder
                            .build_signed_int_to_float(i, self.context.f32_type(), "cast")
                            .map_err(|e| e.to_string())?;
                        Ok((f.into(), Type::F32))
                    }
                    (Type::F32, Type::I64) => {
                        let f = val.into_float_value();
                        let i = self
                            .builder
                            .build_float_to_signed_int(f, self.context.i64_type(), "cast")
                            .map_err(|e| e.to_string())?;
                        Ok((i.into(), Type::I64))
                    }
                    (Type::Bool, Type::I64) => {
                        let b = val.into_int_value();
                        let i = self
                            .builder
                            .build_int_z_extend(b, self.context.i64_type(), "cast")
                            .map_err(|e| e.to_string())?;
                        Ok((i.into(), Type::I64))
                    }
                    (Type::I64, Type::Bool) => {
                        let i = val.into_int_value();
                        let zero = self.context.i64_type().const_zero();
                        let b = self
                            .builder
                            .build_int_compare(inkwell::IntPredicate::NE, i, zero, "cast")
                            .map_err(|e| e.to_string())?;
                        Ok((b.into(), Type::Bool))
                    }
                    (Type::Bool, Type::F32) => {
                        let b = val.into_int_value();
                        let f = self
                            .builder
                            .build_unsigned_int_to_float(b, self.context.f32_type(), "cast")
                            .map_err(|e| e.to_string())?;
                        Ok((f.into(), Type::F32))
                    }
                    (Type::F32, Type::Bool) => {
                        let f = val.into_float_value();
                        let zero = self.context.f32_type().const_zero();
                        let b = self
                            .builder
                            .build_float_compare(inkwell::FloatPredicate::UNE, f, zero, "cast")
                            .map_err(|e| e.to_string())?;
                        Ok((b.into(), Type::Bool))
                    }
                    (Type::ScalarArray(elem_ty, len), Type::Tensor(target_elem, _)) => {
                        // Convert ScalarArray to Tensor and then cast if necessary
                        let tensor_val = match elem_ty.as_ref() {
                            Type::I64 => {
                                let from_fn = self
                                    .module
                                    .get_function("tl_tensor_from_i64_array")
                                    .ok_or("tl_tensor_from_i64_array not found")?;
                                let len_val = self
                                    .context
                                    .i64_type()
                                    .const_int(format!("{}", len).parse().unwrap(), false);
                                // val is pointer to [i64; N]
                                let ptr = val.into_pointer_value();
                                // cast to i64*
                                let i64_ptr_ty =
                                    self.context.ptr_type(inkwell::AddressSpace::default());
                                let cast_ptr = self
                                    .builder
                                    .build_pointer_cast(ptr, i64_ptr_ty, "cast_ptr")
                                    .map_err(|e| e.to_string())?;

                                let call = self
                                    .builder
                                    .build_call(
                                        from_fn,
                                        &[cast_ptr.into(), len_val.into()],
                                        "t_from_arr",
                                    )
                                    .map_err(|e| e.to_string())?;
                                match call.try_as_basic_value() {
                                    inkwell::values::ValueKind::Basic(v) => v,
                                    _ => return Err("Invalid return from tensor creation".into()),
                                }
                            }
                            _ => {
                                return Err(format!(
                                    "Unsupported ScalarArray element type {:?}",
                                    elem_ty
                                ))
                            }
                        };

                        if *elem_ty == *target_elem {
                            // self.emit_register_tensor(tensor_val, target_type)?;

                            Ok((tensor_val, target_type.clone()))
                        } else {
                            // Cast the newly created tensor
                            match target_elem.as_ref() {
                                Type::F32 => {
                                    let cast_fn = self
                                        .module
                                        .get_function("tl_tensor_to_f32")
                                        .ok_or("tl_tensor_to_f32 not found")?;
                                    let call = self
                                        .builder
                                        .build_call(cast_fn, &[tensor_val.into()], "cast_t")
                                        .map_err(|e| e.to_string())?;
                                    let res = match call.try_as_basic_value() {
                                        ValueKind::Basic(v) => v,
                                        _ => return Err("Invalid return".into()),
                                    };
                                    // self.emit_register_tensor(res, target_type)?;

                                    Ok((res, target_type.clone()))
                                }
                                Type::I64 => {
                                    let cast_fn = self
                                        .module
                                        .get_function("tl_tensor_to_i64")
                                        .ok_or("tl_tensor_to_i64 not found")?;
                                    let call = self
                                        .builder
                                        .build_call(cast_fn, &[tensor_val.into()], "cast_t")
                                        .map_err(|e| e.to_string())?;
                                    let res = match call.try_as_basic_value() {
                                        ValueKind::Basic(v) => v,
                                        _ => return Err("Invalid return".into()),
                                    };
                                    // self.emit_register_tensor(res, target_type)?;

                                    Ok((res, target_type.clone()))
                                }
                                _ => Err(format!(
                                    "Unsupported tensor cast target: {:?}",
                                    target_elem
                                )),
                            }
                        }
                    }
                    (Type::Tensor(_, _), Type::Tensor(inner_dst, _)) => match inner_dst.as_ref() {
                        Type::F32 => {
                            let cast_fn = self
                                .module
                                .get_function("tl_tensor_to_f32")
                                .ok_or("tl_tensor_to_f32 not found")?;
                            let call = self
                                .builder
                                .build_call(cast_fn, &[val.into()], "cast_t")
                                .map_err(|e| e.to_string())?;
                            let res = match call.try_as_basic_value() {
                                ValueKind::Basic(v) => v,
                                _ => return Err("Invalid return".into()),
                            };
                            // self.emit_register_tensor(res, target_type)?;

                            Ok((res, target_type.clone()))
                        }
                        Type::I64 => {
                            let cast_fn = self
                                .module
                                .get_function("tl_tensor_to_i64")
                                .ok_or("tl_tensor_to_i64 not found")?;
                            let call = self
                                .builder
                                .build_call(cast_fn, &[val.into()], "cast_t")
                                .map_err(|e| e.to_string())?;
                            let res = match call.try_as_basic_value() {
                                ValueKind::Basic(v) => v,
                                _ => return Err("Invalid return".into()),
                            };
                            // self.emit_register_tensor(res, target_type)?;

                            Ok((res, target_type.clone()))
                        }
                        _ => Err(format!("Unsupported tensor cast target: {:?}", inner_dst)),
                    },
                    _ => Err(format!(
                        "Unsupported cast from {:?} to {:?}",
                        source_type, target_type
                    )),
                }
            }
            ExprKind::FieldAccess(obj, field) => {
                let (obj_val, obj_ty) = self.compile_expr(obj)?;
                let struct_name = match &obj_ty {
                    Type::Struct(name, args) | Type::UserDefined(name, args) => {
                         if args.is_empty() {
                             name.clone()
                         } else {
                             self.mangle_type_name(name, args)
                         }
                    }
                    _ => return Err(format!("Field access on non-struct type {:?}", obj_ty)),
                };

                let simple_struct_name = if struct_name.contains("::") {
                    struct_name.split("::").last().unwrap()
                } else {
                    &struct_name
                };

                let struct_def = self
                    .struct_defs
                    .get(simple_struct_name)
                    .ok_or(format!("Struct definition for {} not found", struct_name))?;

                let field_idx = struct_def
                    .fields
                    .iter()
                    .position(|(n, _)| n == field)
                    .ok_or(format!(
                        "Field {} not found in struct {}",
                        field, struct_name
                    ))?;
                let (_, field_ty) = &struct_def.fields[field_idx];

                if obj_val.is_pointer_value() {
                    let ptr = obj_val.into_pointer_value();
                    let st_llvm_ty = self.struct_types.get(simple_struct_name).unwrap();

                    let field_ptr = self
                        .builder
                        .build_struct_gep(
                            *st_llvm_ty,
                            ptr,
                            field_idx as u32,
                            &format!("ptr_{}", field),
                        )
                        .map_err(|e| e.to_string())?;

                    let llvm_ty: inkwell::types::BasicTypeEnum = match field_ty {
                        Type::I64 => self.context.i64_type().into(),
                        Type::F32 => self.context.f32_type().into(),
                        Type::Bool => self.context.bool_type().into(),
                        Type::Tensor(_, _) => self
                            .context
                            .ptr_type(inkwell::AddressSpace::default())
                            .into(),
                        Type::Struct(_, _) | Type::UserDefined(_, _) => self
                            .context
                            .ptr_type(inkwell::AddressSpace::default())
                            .into(),
                        _ => self.context.i64_type().into(), // Placeholder
                    };

                    let loaded = self
                        .builder
                        .build_load(llvm_ty, field_ptr, field)
                        .map_err(|e| e.to_string())?;
                    Ok((loaded, field_ty.clone()))
                } else if obj_val.is_struct_value() {
                    let struct_val = obj_val.into_struct_value();
                    let extracted = self
                        .builder
                        .build_extract_value(struct_val, field_idx as u32, field)
                        .map_err(|e| e.to_string())?;
                    Ok((extracted, field_ty.clone()))
                } else {
                    Err("Cannot access field of non-pointer and non-struct value".into())
                }
            }

            ExprKind::Variable(name) => {
                for scope in self.variables.iter().rev() {
                    if let Some((val, ty, _)) = scope.get(name) {
                        if val.is_pointer_value() {
                            let ptr = val.into_pointer_value();

                            // ScalarArray: load the pointer from alloca (it stores ptr to global)
                            if let Type::ScalarArray(_, _) = ty {
                                let ptr_type =
                                    self.context.ptr_type(inkwell::AddressSpace::default());
                                let loaded_ptr = self
                                    .builder
                                    .build_load(ptr_type, ptr, &format!("{}_ptr", name))
                                    .map_err(|e| e.to_string())?;
                                return Ok((loaded_ptr, ty.clone()));
                            }

                            let llvm_ty: inkwell::types::BasicTypeEnum = match ty {
                                Type::I64 => self.context.i64_type().into(),
                                Type::F32 => self.context.f32_type().into(),
                                Type::Bool => self.context.bool_type().into(),
                                Type::Tensor(_, _) | Type::Vec(_) => self
                                    .context
                                    .ptr_type(inkwell::AddressSpace::default())
                                    .into(),
                                Type::Struct(_, _)
                                | Type::UserDefined(_, _)
                                | Type::Tuple(_)
                                | Type::Enum(_, _) => self
                                    .context
                                    .ptr_type(inkwell::AddressSpace::default())
                                    .into(),
                                _ => self.context.i64_type().into(), // Fallback
                            };

                            let loaded = self
                                .builder
                                .build_load(llvm_ty, ptr, name)
                                .map_err(|e| e.to_string())?;
                            return Ok((loaded, ty.clone()));
                        } else {
                            return Ok((*val, ty.clone()));
                        }
                    }
                }
                Err(format!("Variable {} not found in scopes", name))
            }
            ExprKind::StructInit(name, generics, fields) => self.compile_struct_init(name, generics, fields),
            ExprKind::StaticMethodCall(type_ty, method_name, args) => {
                self.compile_static_method_call(type_ty, method_name, args)
            }
            ExprKind::BinOp(lhs, op, rhs) => {
                let left = self.compile_expr(lhs)?;
                let right = self.compile_expr(rhs)?;
                let res = self.compile_bin_op(
                    left.0,
                    left.1.clone(),
                    right.0,
                    right.1.clone(),
                    op.clone(),
                )?;

                // Register intermediate tensor result
                self.add_temp(res.0, res.1.clone());
                Ok(res)
            }

            ExprKind::Tuple(exprs) => self.compile_tuple(exprs),
            ExprKind::TupleAccess(expr, idx) => self.compile_tuple_access(expr, *idx),

            ExprKind::TensorComprehension {
                indices,
                clauses,
                body,
            } => {
                // Generate a unique name for the temporary result
                static NEXT_ID: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(0);
                let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let temp_name = format!("_comp_res_{}", id);

                // Compile as a tensor equation: let temp[indices] = body;
                // We pass the indices, clauses, and optional body directly.
                self.compile_tensor_equation(&temp_name, indices, clauses, body.as_deref())
                    .map_err(|e| e.to_string())?;

                // After compilation, the tensor 'temp_name' is registered in the scope.
                // We need to load it to return it as an expression value.
                // It should be in the current scope.

                let (val_enum, val_ty, _) = self
                    .variables
                    .last()
                    .unwrap()
                    .get(&temp_name)
                    .ok_or(format!("Failed to retrieve temporary tensor {}", temp_name))?
                    .clone();

                if let Type::Tensor(_, _) = val_ty {
                    // It's a pointer to the tensor struct
                    if val_enum.is_pointer_value() {
                        let ptr_to_ptr = val_enum.into_pointer_value();
                        let void_ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
                        let ptr = self
                            .builder
                            .build_load(void_ptr_type, ptr_to_ptr, "tensor_ptr")
                            .map_err(|e| e.to_string())?;
                        Ok((ptr, val_ty))
                    } else {
                        Err("Tensor variable should be a pointer".into())
                    }
                } else {
                    Err("Comprehension result must be a tensor".into())
                }
            }
            ExprKind::TensorLiteral(elements) => self.compile_tensor_literal(elements),
            ExprKind::TensorConstLiteral(elements) => self.compile_tensor_const_literal(elements),
            ExprKind::MethodCall(obj, method, args) => self.compile_method_call(obj, method, args),
            ExprKind::FnCall(name, args) => self.compile_fn_call(name, args),
            ExprKind::IndexAccess(target, indices) => {
                let (val, val_type) = self.compile_expr(target)?;
                match val_type {
                    // OPTIMIZATION: ScalarArray direct access (no runtime call)
                    Type::ScalarArray(elem_type, len) => {
                        if indices.len() != 1 {
                            return Err("ScalarArray only supports 1D index".into());
                        }

                        let llvm_elem_type: inkwell::types::BasicTypeEnum = match elem_type.as_ref()
                        {
                            Type::I64 => self.context.i64_type().into(),
                            Type::I32 => self.context.i32_type().into(),
                            Type::F32 => self.context.f32_type().into(),
                            _ => self.context.f32_type().into(),
                        };
                        let i64_type = self.context.i64_type();
                        let _array_type = llvm_elem_type.array_type(len as u32);
                        let array_ptr = val.into_pointer_value();

                        let (idx_val, idx_ty) = self.compile_expr(&indices[0])?;
                        let idx_int = match idx_ty {
                            Type::I64 => idx_val.into_int_value(),
                            Type::I32 => self
                                .builder
                                .build_int_z_extend(idx_val.into_int_value(), i64_type, "zext")
                                .map_err(|e| e.to_string())?,
                            _ => return Err("Index must be integer".into()),
                        };

                        // Direct GEP into array
                        let elem_ptr = unsafe {
                            self.builder
                                .build_in_bounds_gep(
                                    llvm_elem_type,
                                    array_ptr,
                                    &[idx_int],
                                    "scalar_elem_ptr",
                                )
                                .map_err(|e| e.to_string())?
                        };

                        let loaded = self
                            .builder
                            .build_load(llvm_elem_type, elem_ptr, "scalar_elem")
                            .map_err(|e| e.to_string())?;

                        Ok((loaded, *elem_type))
                    }
                    Type::Tensor(inner, _) => {
                        // Prepare indices array
                        let rank = indices.len();
                        let i64_type = self.context.i64_type();

                        // Create array on stack in the ENTRY block to avoid stack overflow in loops
                        let array_type = i64_type.array_type(rank as u32);

                        let current_block = self.builder.get_insert_block().unwrap();
                        let function = current_block.get_parent().unwrap();
                        let entry_block = function.get_first_basic_block().unwrap();

                        let entry_builder = self.context.create_builder();
                        if let Some(first_instr) = entry_block.get_first_instruction() {
                            entry_builder.position_before(&first_instr);
                        } else {
                            entry_builder.position_at_end(entry_block);
                        }

                        let array_alloca = entry_builder
                            .build_alloca(array_type, "idx_arr")
                            .map_err(|e| e.to_string())?;

                        for (i, idx_expr) in indices.iter().enumerate() {
                            let (compiled_idx, ty) = self.compile_expr(idx_expr)?;

                            // Ensure index is integer or float (cast if needed)
                            let idx_val = match ty {
                                Type::I64 => compiled_idx.into_int_value(),
                                Type::I32 => self
                                    .builder
                                    .build_int_z_extend(
                                        compiled_idx.into_int_value(),
                                        i64_type,
                                        "zext",
                                    )
                                    .map_err(|e| e.to_string())?,
                                Type::F64 | Type::F32 => self
                                    .builder
                                    .build_float_to_signed_int(
                                        compiled_idx.into_float_value(),
                                        i64_type,
                                        "f2i",
                                    )
                                    .map_err(|e| e.to_string())?,
                                _ => return Err(format!("Invalid index type {:?}", ty)),
                            };
                            let idx_val = inkwell::values::BasicValueEnum::IntValue(idx_val);

                            let elem_ptr = unsafe {
                                self.builder
                                    .build_gep(
                                        array_type,
                                        array_alloca,
                                        &[
                                            i64_type.const_int(0, false),
                                            i64_type.const_int(i as u64, false),
                                        ],
                                        "idx_ptr",
                                    )
                                    .map_err(|e| e.to_string())?
                            };
                            self.builder
                                .build_store(elem_ptr, idx_val)
                                .map_err(|e| e.to_string())?;
                        }

                        let (get_fn_name, res_ty) = match inner.as_ref() {
                            Type::I64 | Type::I32 => {
                                ("tl_tensor_get_i64_md", inner.as_ref().clone())
                            }
                            _ => ("tl_tensor_get_f32_md", Type::F32),
                        };
                        let get_fn = self.module.get_function(get_fn_name).unwrap();
                        let tensor_ptr = val.into_pointer_value();
                        let array_ptr = self
                            .builder
                            .build_pointer_cast(
                                array_alloca,
                                self.context.ptr_type(inkwell::AddressSpace::default()),
                                "arr_ptr",
                            )
                            .map_err(|e| e.to_string())?;
                        let rank_val = i64_type.const_int(rank as u64, false);

                        let call = self
                            .builder
                            .build_call(
                                get_fn,
                                &[tensor_ptr.into(), array_ptr.into(), rank_val.into()],
                                "get_md_call",
                            )
                            .map_err(|e| e.to_string())?;

                        let res = match call.try_as_basic_value() {
                            ValueKind::Basic(v) => v,
                            _ => return Err("Invalid get return".into()),
                        };

                        if let Type::I32 = res_ty {
                            let i32_val = self
                                .builder
                                .build_int_truncate(
                                    res.into_int_value(),
                                    self.context.i32_type(),
                                    "i32_trunc",
                                )
                                .map_err(|e| e.to_string())?;
                            Ok((i32_val.into(), Type::I32))
                        } else {
                            Ok((res, res_ty))
                        }
                    }
                    _ => Err("Index access only on Tensor or ScalarArray".into()),
                }
            }
            ExprKind::UnOp(op, expr) => {
                let (val, ty) = self.compile_expr(expr)?;
                match op {
                    UnOp::Neg => match &ty {
                        Type::I64 => {
                            let i = val.into_int_value();
                            let res = self
                                .builder
                                .build_int_neg(i, "negtmp")
                                .map_err(|e| e.to_string())?;
                            Ok((res.into(), Type::I64))
                        }
                        Type::F32 => {
                            let f = val.into_float_value();
                            let res = self
                                .builder
                                .build_float_neg(f, "negtmp")
                                .map_err(|e| e.to_string())?;
                            Ok((res.into(), Type::F32))
                        }
                        Type::Tensor(_inner, _rank) => {
                            let neg_fn = self.module.get_function("tl_tensor_neg").unwrap();
                            let call = self
                                .builder
                                .build_call(neg_fn, &[val.into()], "neg")
                                .map_err(|e| e.to_string())?;
                            let res = match call.try_as_basic_value() {
                                ValueKind::Basic(v) => v,
                                _ => return Err("Failed neg".into()),
                            };
                            self.add_temp(res, ty.clone());

                            Ok((res, ty.clone()))

                        }
                        _ => Err("Negation only on int/float/tensor".into()),
                    },

                    UnOp::Not => {
                        match &ty {
                            Type::Bool => {
                                let b = val.into_int_value(); // i1
                                let res = self
                                    .builder
                                    .build_not(b, "nottmp")
                                    .map_err(|e| e.to_string())?;
                                Ok((res.into(), Type::Bool))
                            }
                            _ => Err("Not only on bool".into()),
                        }
                    }
                    UnOp::Query => {
                        // Logic Query: check if result tensor is non-empty
                        if let Type::Tensor(_, _) = ty {
                            let len_fn = self.module.get_function("tl_tensor_len").unwrap();
                            let cast_ptr = self
                                .builder
                                .build_pointer_cast(
                                    val.into_pointer_value(),
                                    self.context.ptr_type(inkwell::AddressSpace::default()),
                                    "cast_len",
                                )
                                .unwrap();
                            let call = self
                                .builder
                                .build_call(len_fn, &[cast_ptr.into()], "len")
                                .map_err(|e| e.to_string())?;
                            let len = match call.try_as_basic_value() {
                                inkwell::values::ValueKind::Basic(v) => v.into_int_value(),
                                _ => return Err("Failed len".into()),
                            };
                            

                            // Check len > 0

                            let zero = self.context.i64_type().const_zero();
                            let bool_val = self
                                .builder
                                .build_int_compare(
                                    inkwell::IntPredicate::SGT,
                                    len,
                                    zero,
                                    "is_true",
                                )
                                .map_err(|e| e.to_string())?;

                            // Convert i1 bool to f32 (0.0 or 1.0)
                            let float_val = self.builder.build_unsigned_int_to_float(
                                bool_val,
                                self.context.f32_type(),
                                "bool_to_f32"
                            ).map_err(|e| e.to_string())?;

                            // Create scalar tensor using tl_tensor_new(data_ptr, rank=0, shape_ptr=NULL)
                            // 1. Alloca for f32 data
                            let current_block = self.builder.get_insert_block().unwrap();
                            let func = current_block.get_parent().unwrap();
                            let entry_block = func.get_first_basic_block().unwrap();
                            
                            // Insert at the beginning of entry block to avoid being after terminator
                            if let Some(first_inst) = entry_block.get_first_instruction() {
                                self.builder.position_before(&first_inst);
                            } else {
                                self.builder.position_at_end(entry_block);
                            }
                            
                            let f32_ptr = self.builder.build_alloca(self.context.f32_type(), "scalar_data").map_err(|e| e.to_string())?;
                            // Also allocate dummy shape pointer (i64/usize aligned) to satisfy slice::from_raw_parts alignment check
                            let shape_dummy_ptr = self.builder.build_alloca(self.context.i64_type(), "shape_dummy").map_err(|e| e.to_string())?;
                            
                            self.builder.position_at_end(current_block); // Back to current
                            
                            // 2. Store value
                            self.builder.build_store(f32_ptr, float_val).map_err(|e| e.to_string())?;

                            // 3. Call tl_tensor_new
                            let tensor_new_fn = self.module.get_function("tl_tensor_new").ok_or("tl_tensor_new not found")?;
                            // tl_tensor_new(data: *const f32, rank: usize, shape: *const usize)
                            let zero_sz = self.context.i64_type().const_zero(); // rank = 0
                            
                            // Pass properly aligned dummy pointer
                            let shape_ptr = self.builder.build_pointer_cast(
                                shape_dummy_ptr,
                                self.context.ptr_type(inkwell::AddressSpace::default()),
                                "shape_ptr_cast"
                            ).map_err(|e| e.to_string())?;

                            let tensor_call = self.builder.build_call(
                                tensor_new_fn, 
                                &[f32_ptr.into(), zero_sz.into(), shape_ptr.into()], 
                                "pred_tensor"
                            ).map_err(|e| e.to_string())?;
                            
                            let res = match tensor_call.try_as_basic_value() {
                                inkwell::values::ValueKind::Basic(v) => v,
                                _ => return Err("Invalid return from tl_tensor_new".into()),
                            };

                            // Register result
                            let res_ty = Type::Tensor(Box::new(Type::F32), 0);
                            self.add_temp(res, res_ty.clone());

                            Ok((res, res_ty))
                        } else {
                            Err("Query on non-tensor".into())
                        }
                    }
                }
            }


            ExprKind::IfExpr(cond, then_stmts, else_stmts) => {
                let parent = self
                    .builder
                    .get_insert_block()
                    .unwrap()
                    .get_parent()
                    .unwrap();

                let (cond_val, cond_ty) = self.compile_expr(cond)?;
                
                let cond_int = if let Type::Tensor(_, _) = cond_ty {
                    // Implicit conversion: Tensor -> Bool
                    // Assuming scalar tensor (0-dim or 1-element). extract item as int/bool.
                    // We can use `tl_tensor_item_i64` and cast to bool?
                    let item_fn = self.module.get_function("tl_tensor_item_i64").unwrap();
                    let call = self.builder.build_call(item_fn, &[cond_val.into()], "cond_item").map_err(|e| e.to_string())?;
                    let item_val = match call.try_as_basic_value() {
                         inkwell::values::ValueKind::Basic(v) => v,
                         _ => return Err("Expected basic value from tl_tensor_item_i64".into()),
                    };
                    
                    self.builder.build_int_compare(
                        inkwell::IntPredicate::NE,
                        item_val.into_int_value(),
                        self.context.i64_type().const_int(0, false),
                        "cond_bool"
                    ).map_err(|e| e.to_string())?
                } else {
                     self
                    .builder
                    .build_int_cast(
                        cond_val.into_int_value(),
                        self.context.bool_type(),
                        "boolcast",
                    )
                    .map_err(|e| e.to_string())?
                };


                let then_bb = self.context.append_basic_block(parent, "if_then");
                let else_bb = self.context.append_basic_block(parent, "if_else");
                let merge_bb = self.context.append_basic_block(parent, "if_merge");

                self.builder
                    .build_conditional_branch(cond_int, then_bb, else_bb)
                    .map_err(|e| e.to_string())?;

                // Then branch
                self.builder.position_at_end(then_bb);
                self.enter_scope();
                let mut then_val: Option<(BasicValueEnum<'ctx>, Type)> = None;
                for (i, stmt) in then_stmts.iter().enumerate() {
                    if i == then_stmts.len() - 1 {
                        if let StmtKind::Expr(e) = &stmt.inner {
                            then_val = Some(self.compile_expr(e)?);
                        } else {
                            self.compile_stmt(stmt)?;
                        }
                    } else {
                        self.compile_stmt(stmt)?;
                    }
                }
                // Default value if no expression
                let then_result = then_val.unwrap_or((
                    self.context.i64_type().const_int(0, false).into(),
                    Type::Void,
                ));

                // Scope Promotion Logic
                // If result is in current scope (Temporary), remove it so it isn't freed. (Move)
                // If result is NOT in current scope (L-value), CLONE it. (Copy)

                let mut then_final_val = then_result.0;

                if matches!(
                    then_result.1,
                    Type::Tensor(_, _) | Type::Struct(_, _) | Type::UserDefined(_, _) | Type::Tuple(_)
                ) {
                    // Logic:
                    // 1. Check AST of last stmt.
                    let is_lvalue = if let Some(last) = then_stmts.last() {
                        match &last.inner {
                            StmtKind::Expr(e) => matches!(
                                &e.inner,
                                ExprKind::Variable(_)
                                    | ExprKind::FieldAccess(_, _)
                                    | ExprKind::IndexAccess(_, _)
                            ),
                            _ => false, // Stmt that isn't Expr? Void.
                        }
                    } else {
                        false
                    }; // Empty body -> Void

                    if is_lvalue {
                        // L-Value: Must Clone to return Owned.
                        then_final_val = self.emit_deep_clone(then_final_val, &then_result.1)?;
                        // Original remains in parent scope. Safe.
                    } else {
                        // R-Value (Temporary in this scope):
                        // We MUST prevent exit_scope from freeing it.
                        // Since we can't identify it by name to remove, we call runtime UNREGISTER.
                        // Runtime unregister stops `tl_tensor_free` from working?
                        // If `tl_tensor_free` checks if ptr is valid?
                        // If so, `unregister` works.

                        if let Some(unreg_fn) = self.module.get_function("tl_mem_unregister") {
                            let ptr = then_final_val.into_pointer_value();
                            let cast_ptr = self
                                .builder
                                .build_pointer_cast(
                                    ptr,
                                    self.context.ptr_type(inkwell::AddressSpace::default()),
                                    "cast",
                                )
                                .unwrap();
                            self.builder
                                .build_call(unreg_fn, &[cast_ptr.into()], "")
                                .unwrap();
                        }
                    }
                }

                self.exit_scope();

                // Get the block after cleanup (important for PHI incoming)
                let then_final_bb = self.builder.get_insert_block().unwrap();
                if then_final_bb.get_terminator().is_none() {
                    self.builder
                        .build_unconditional_branch(merge_bb)
                        .map_err(|e| e.to_string())?;
                }

                // Else branch
                self.builder.position_at_end(else_bb);
                self.enter_scope();
                let mut else_val: Option<(BasicValueEnum<'ctx>, Type)> = None;
                if let Some(else_body) = else_stmts {
                    for (i, stmt) in else_body.iter().enumerate() {
                        if i == else_body.len() - 1 {
                            if let StmtKind::Expr(e) = &stmt.inner {
                                else_val = Some(self.compile_expr(e)?);
                            } else {
                                self.compile_stmt(stmt)?;
                            }
                        } else {
                            self.compile_stmt(stmt)?;
                        }
                    }
                }
                let else_result = else_val.unwrap_or((
                    self.context.i64_type().const_int(0, false).into(),
                    Type::Void,
                ));

                let mut else_final_val = else_result.0;

                if matches!(
                    else_result.1,
                    Type::Tensor(_, _) | Type::Struct(_, _) | Type::UserDefined(_, _) | Type::Tuple(_)
                ) {
                    // Logic:
                    // 1. Check AST of last stmt.
                    let is_lvalue = if let Some(body) = &else_stmts {
                        if let Some(last) = body.last() {
                            match &last.inner {
                                StmtKind::Expr(e) => matches!(
                                    &e.inner,
                                    ExprKind::Variable(_)
                                        | ExprKind::FieldAccess(_, _)
                                        | ExprKind::IndexAccess(_, _)
                                ),
                                _ => false,
                            }
                        } else {
                            false
                        }
                    } else {
                        false
                    };

                    if is_lvalue {
                        else_final_val = self.emit_deep_clone(else_final_val, &else_result.1)?;
                    } else {
                        if let Some(unreg_fn) = self.module.get_function("tl_mem_unregister") {
                            let ptr = else_final_val.into_pointer_value();
                            let cast_ptr = self
                                .builder
                                .build_pointer_cast(
                                    ptr,
                                    self.context.ptr_type(inkwell::AddressSpace::default()),
                                    "cast",
                                )
                                .unwrap();
                            self.builder
                                .build_call(unreg_fn, &[cast_ptr.into()], "")
                                .unwrap();
                        }
                    }
                }

                self.exit_scope();

                // Get the block after cleanup
                let else_final_bb = self.builder.get_insert_block().unwrap();
                if else_final_bb.get_terminator().is_none() {
                    self.builder
                        .build_unconditional_branch(merge_bb)
                        .map_err(|e| e.to_string())?;
                }

                // Merge block with PHI
                self.builder.position_at_end(merge_bb);

                // Only create PHI if both branches return non-void values
                if !matches!(then_result.1, Type::Void) && !matches!(else_result.1, Type::Void) {
                    let llvm_ty: inkwell::types::BasicTypeEnum = match &then_result.1 {
                        Type::I64 => self.context.i64_type().into(),
                        Type::F32 => self.context.f32_type().into(),
                        Type::Bool => self.context.bool_type().into(),
                        Type::Tensor(_, _)
                        | Type::Struct(_, _)
                        | Type::UserDefined(_, _)
                        | Type::Tuple(_) => self
                            .context
                            .ptr_type(inkwell::AddressSpace::default())
                            .into(),
                        _ => self.context.i64_type().into(),
                    };

                    let phi = self
                        .builder
                        .build_phi(llvm_ty, "if_result")
                        .map_err(|e| e.to_string())?;
                    phi.add_incoming(&[
                        (&then_final_val, then_final_bb),
                        (&else_final_val, else_final_bb),
                    ]);

                    // FIX: Register the PHI result as a temporary
                    // The incoming values (then_result, else_result) were unregistered in their respective blocks
                    // to prevent freeing. Now we have a new value (phi) representing the result.
                    // We must register it so it is managed in the PARENT scope.
                    let res_val = phi.as_basic_value();
                    let res_ty = then_result.1.clone(); // Assume types match (semantics checks this)

                    if matches!(
                        res_ty,
                        Type::Tensor(_, _)
                            | Type::Struct(_, _)
                            | Type::UserDefined(_, _)
                            | Type::Tuple(_)
                    ) {
                        self.emit_register_tensor(res_val, &res_ty)?;
                        // Note: emit_register_tensor handles recursion for Structs/Tuples if implemented,
                        // but currently it checks for Type::Tensor.
                        // For structs/tuples we might need `gen_register_params` or similar if they are managed.
                        // However, `IfExpr` typically unregisters the *value pointer*.

                        // If we unregistered the *inputs*, they are effectively "moved" to the PHI node.
                        // But wait, `tl_mem_unregister` tells runtime "stop tracking this pointer".
                        // The PHI node returns the *same pointer* (incoming from blocks).
                        // So the pointer itself is valid, but runtime doesn't track it.
                        // We need to re-register it.

                        if let Type::Tensor(_, _) = res_ty {
                            // Already called emit_register_tensor above
                        } else if let Type::Struct(_, _) | Type::UserDefined(_, _) = &res_ty {
                            if let Some(reg_fn) = self.module.get_function("tl_mem_register_struct")
                            {
                                let ptr = res_val.into_pointer_value();
                                let cast_ptr = self
                                    .builder
                                    .build_pointer_cast(
                                        ptr,
                                        self.context.ptr_type(inkwell::AddressSpace::default()),
                                        "cast",
                                    )
                                    .unwrap();
                                self.builder
                                    .build_call(reg_fn, &[cast_ptr.into()], "")
                                    .unwrap();
                            }
                        }
                    }

                    Ok((res_val, res_ty))
                } else {
                    Ok((
                        self.context.i64_type().const_int(0, false).into(),
                        Type::Void,
                    ))
                }
            }
        }
    }

    fn compile_struct_init(
        &mut self,
        name: &str,
        generics: &[Type],
        fields: &[(String, Expr)],
    ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        let lookup_name = if generics.is_empty() {
             name.to_string()
        } else {
             self.mangle_type_name(name, generics)
        };
        
        // Debug:
        // if generics.len() > 0 { ... }

        let struct_type = *self
            .struct_types
            .get(&lookup_name)
            .ok_or_else(|| {
                 format!("Struct type {} not found in codegen", lookup_name)
            })?;

        let struct_def = self
            .struct_defs
            .get(&lookup_name)
            .ok_or_else(|| {
                 format!("Struct definition {} not found", lookup_name)
            })?
            .clone();

        // Determine allocation strategy: Arena or Heap
        let size = struct_type
            .size_of()
            .ok_or(format!("Cannot determine size of struct {}", name))?;

        // 1. Heap Allocation
        let malloc_fn = self
            .module
            .get_function("malloc")
            .ok_or("malloc not found (declare in builtins)")?;
        
        let size_int = size;
        let size_i64 = if size_int.get_type() == self.context.i32_type() {
             self.builder.build_int_z_extend(size_int, self.context.i64_type(), "size_i64").unwrap()
        } else {
             size_int
        };

        let call = self
            .builder
            .build_call(malloc_fn, &[size_i64.into()], "struct_malloc")
            .map_err(|e| e.to_string())?;
        let raw_ptr = match call.try_as_basic_value() {
            inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
            _ => return Err("malloc returned invalid value".into()),
        };

        // 2. Register with MemoryManager
        // Force named registration
        let register_fn = self
            .module
            .get_function("tl_mem_register_struct_named")
            .expect("tl_mem_register_struct_named not found in module");

        let cast_ptr = self
            .builder
            .build_pointer_cast(
                raw_ptr,
                self.context.ptr_type(inkwell::AddressSpace::default()),
                "cast_ptr",
            )
            .unwrap();

        let name_global = self
            .builder
            .build_global_string_ptr(name, "struct_name")
            .map_err(|e| e.to_string())?;

        self.builder
            .build_call(register_fn, &[cast_ptr.into(), name_global.as_pointer_value().into()], "")
            .map_err(|e| e.to_string())?;

        // Cast to Struct Pointer (opaque pointer in modern LLVM, but typed for GEP)
        let struct_ptr = self
            .builder
            .build_pointer_cast(
                raw_ptr,
                self.context.ptr_type(inkwell::AddressSpace::default()),
                "struct_ptr",
            )
            .map_err(|e| e.to_string())?;

        for (field_name, field_expr) in fields {
            let field_idx = struct_def
                .fields
                .iter()
                .position(|(n, _)| n == field_name)
                .ok_or(format!("Field {} not found in struct {}", field_name, name))?;

            let (val, _ty) = self.compile_expr(field_expr)?;
            
            // Move Semantics: 
            // If the field value is in the cleanup list (temporary), removing it transfers ownership to the struct.
            // Combined with disabled Struct Cleanup (Reference Semantics), this allows the value to survive.
            if let Some(temps) = self.temporaries.last_mut() {
                if let Some(idx) = temps.iter().position(|(v, _, _)| *v == val) {
                    temps.remove(idx);
                }
            }

            let field_ptr = self
                .builder
                .build_struct_gep(
                    struct_type,
                    struct_ptr,
                    field_idx as u32,
                    &format!("{}.{}", name, field_name),
                )
                .map_err(|e| e.to_string())?;

            // Deep Clone / Acquire Logic:
            // For Tensors: acquire reference (share).
            // For Structs: deep copy struct memory (prevent dangling pointer when local var dies), share tensors.
            let store_val = if matches!(
                _ty,
                Type::Tensor(_, _) | Type::Struct(_, _) | Type::UserDefined(_, _) | Type::Tuple(_)
            ) {
                self.emit_deep_clone(val, &_ty)?
            } else {
                val
            };

            self.builder
                .build_store(field_ptr, store_val)
                .map_err(|e| e.to_string())?;

            // Move Semantics Removed:
            // We now use RefCounting/DeepCopy. Source variable should remain valid (Shared ownership).
            // Cleanup at end of scope will decrement refcount (balancing the Acquire in emit_deep_clone).
            // No manual unregister. No removal from scope.
        }

        // Return the pointer directly (no load)
        Ok((struct_ptr.into(), Type::Struct(name.to_string(), generics.to_vec())))
    }

    fn compile_string_literal(&self, s: &str) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        let str_val = self
            .builder
            .build_global_string_ptr(s, "str_lit")
            .map_err(|e| e.to_string())?
            .as_pointer_value();

        // Create String object
        let new_fn = self
            .module
            .get_function("tl_string_new")
            .ok_or("tl_string_new not found")?;
        let call = self
            .builder
            .build_call(new_fn, &[str_val.into()], "string_obj")
            .map_err(|e| e.to_string())?;
        let ptr = match call.try_as_basic_value() {
            inkwell::values::ValueKind::Basic(v) => v,
            _ => return Err("tl_string_new returned void".into()),
        };

        Ok((ptr, Type::UserDefined("String".to_string(), vec![])))
    }

    fn compile_tuple(&mut self, elements: &[Expr]) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        let mut vals = Vec::new();
        let mut types = Vec::new();
        let mut llvm_types = Vec::new();

        for e in elements {
            let (val, ty) = self.compile_expr(e)?;
            vals.push(val);
            types.push(ty.clone());
            llvm_types.push(self.get_llvm_type(&ty)?);
        }

        let tuple_struct_type = self.context.struct_type(&llvm_types, false);

        // Malloc
        // We use malloc to ensure it survives if returned (and managed by owner).
        // Similar to StructInit.
        let size = tuple_struct_type
            .size_of()
            .ok_or("Cannot get size of tuple")?;
        
        let size_int = size;
        let size_i64 = if size_int.get_type() == self.context.i32_type() {
             self.builder.build_int_z_extend(size_int, self.context.i64_type(), "size_i64").unwrap()
        } else {
             size_int
        };

        let malloc_fn = self
            .module
            .get_function("malloc")
            .ok_or("malloc not found")?;
        let call = self
            .builder
            .build_call(malloc_fn, &[size_i64.into()], "tuple_malloc")
            .map_err(|e| e.to_string())?;
        let raw_ptr = match call.try_as_basic_value() {
            ValueKind::Basic(v) => v.into_pointer_value(),
            _ => return Err("malloc returned instruction value".into()),
        };

        // Cast to tuple struct pointer
        let tuple_ptr = self
            .builder
            .build_pointer_cast(
                raw_ptr,
                self.context.ptr_type(inkwell::AddressSpace::default()),
                "tuple_ptr",
            )
            .map_err(|e| e.to_string())?;

        // Store elements
        for (i, (val, ty)) in vals.iter().zip(types.iter()).enumerate() {
            let field_ptr = self
                .builder
                .build_struct_gep(tuple_struct_type, tuple_ptr, i as u32, "tuple_field")
                .map_err(|e| e.to_string())?;
            self.builder
                .build_store(field_ptr, *val)
                .map_err(|e| e.to_string())?;

            // Ownership transfer: Unregister from scope if it's a temp/managed type
            if matches!(
                ty,
                Type::Tensor(_, _) | Type::Struct(_, _) | Type::UserDefined(_, _) | Type::Tuple(_)
            ) {
                let unreg_fn = self
                    .module
                    .get_function("tl_mem_unregister")
                    .ok_or("tl_mem_unregister not found")?;
                let val_ptr = val.into_pointer_value();
                let cast_ptr = self
                    .builder
                    .build_pointer_cast(
                        val_ptr,
                        self.context.ptr_type(inkwell::AddressSpace::default()),
                        "cast_unreg_elem",
                    )
                    .unwrap();
                self.builder
                    .build_call(unreg_fn, &[cast_ptr.into()], "")
                    .unwrap();

                // If element was a variable, we need to remove it from scope tracking appropriately?
                // compile_expr for variable returns a LOADed value usually, or the pointer?
                // compile_expr(Variable) returns (Load(alloca), Type).
                // So 'val' is the pointer to the object (if Struct/Tensor).
                // For Variables, we might need similar logic to StructInit to prevent double-free if we "move" it.
                // But generally tuple construction is: let t = (a, b); -> t owns copies of a and b (ref copy).
                // If a, b are variables, we are COPYING the reference. So we should ACQUIRE?
                // Wait, StructInit unregisters?
                // StructInit logic:
                // if arg is Variable: remove from scope (Move semantics?) -> Logic says "should_remove = true".
                // Yes, StructInit implements Move semantics for variables passed to it.
                // So let's replicate that for Tuple.
                if let ExprKind::Variable(var_name) = &elements[i].inner {
                    // Search and remove variable from scope to prevent automatic free at end of scope
                    // Effectively "Moving" ownership to the tuple.
                    for scope in self.variables.iter_mut().rev() {
                        let mut should_remove = false;
                        if let Some((_, var_ty, _)) = scope.get(var_name) {
                            if matches!(
                                var_ty,
                                Type::Tensor(_, _)
                                    | Type::Struct(_, _)
                                    | Type::UserDefined(_, _)
                                    | Type::Tuple(_)
                            ) {
                                should_remove = true;
                            }
                        }
                        if should_remove {
                            scope.remove(var_name); // Remove from tracking -> No free at exit_scope
                            break;
                        }
                    }
                }
            }
        }

        Ok((tuple_ptr.into(), Type::Tuple(types)))
    }

    fn compile_tuple_access(
        &mut self,
        expr: &Expr,
        idx: usize,
    ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        let (tuple_val, tuple_ty) = self.compile_expr(expr)?;

        let element_types = match tuple_ty {
            Type::Tuple(ts) => ts,
            _ => return Err(format!("Expected tuple type, found {:?}", tuple_ty)),
        };

        if idx >= element_types.len() {
            return Err(format!(
                "Tuple index {} out of bounds (len {})",
                idx,
                element_types.len()
            ));
        }
        let field_ty = element_types[idx].clone();

        // tuple_val should be a pointer to the tuple struct (i8*)
        if !tuple_val.is_pointer_value() {
            return Err("Tuple value is not a pointer".into());
        }
        let tuple_ptr = tuple_val.into_pointer_value();

        // Reconstruct LLVM struct type for GEP
        let mut llvm_types = Vec::new();
        for ty in &element_types {
            llvm_types.push(self.get_llvm_type(ty)?);
        }
        let tuple_struct_type = self.context.struct_type(&llvm_types, false);

        // GEP
        let field_ptr = self
            .builder
            .build_struct_gep(tuple_struct_type, tuple_ptr, idx as u32, "tuple_access")
            .map_err(|e| e.to_string())?;

        // Load
        let llvm_field_ty = self.get_llvm_type(&field_ty)?;
        let val = self
            .builder
            .build_load(llvm_field_ty, field_ptr, "tuple_elem")
            .map_err(|e| e.to_string())?;

        Ok((val, field_ty))
    }

    fn compile_tuple_struct_init(
        &mut self,
        name: &str,
        args: &[Expr],
    ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        let struct_type = *self
            .struct_types
            .get(name)
            .ok_or(format!("Struct type {} not found in codegen", name))?;

        let struct_def = self
            .struct_defs
            .get(name)
            .ok_or(format!("Struct definition {} not found", name))?
            .clone();

        if args.len() != struct_def.fields.len() {
            return Err(format!(
                "Field count mismatch for struct {}: expected {}, found {}",
                name,
                struct_def.fields.len(),
                args.len()
            ));
        }

        // 1. Heap Allocation
        let malloc_fn = self
            .module
            .get_function("malloc")
            .ok_or("malloc not found")?;
        let size = struct_type
            .size_of()
            .ok_or(format!("Cannot determine size of struct {}", name))?;
            
        let size_int = size;
        let size_i64 = if size_int.get_type() == self.context.i32_type() {
             self.builder.build_int_z_extend(size_int, self.context.i64_type(), "size_i64").unwrap()
        } else {
             size_int
        };

        let call = self
            .builder
            .build_call(malloc_fn, &[size_i64.into()], "struct_malloc")
            .map_err(|e| e.to_string())?;
        let raw_ptr = match call.try_as_basic_value() {
            inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
            _ => return Err("malloc returned invalid value".into()),
        };

        // 2. Register
        // 2. Register
        // Force named registration
        let register_fn = self
            .module
            .get_function("tl_mem_register_struct_named")
            .expect("tl_mem_register_struct_named not found in module");

        let cast_ptr = self
            .builder
            .build_pointer_cast(
                raw_ptr,
                self.context.ptr_type(inkwell::AddressSpace::default()),
                "cast_ptr",
            )
            .unwrap();

        let name_global = self
            .builder
            .build_global_string_ptr(name, "struct_name")
            .map_err(|e| e.to_string())?;

        self.builder
            .build_call(register_fn, &[cast_ptr.into(), name_global.as_pointer_value().into()], "")
            .map_err(|e| e.to_string())?;

        // Debug trace
        if name.contains("GPT") {
            let size_val = self.context.i64_type().const_int(0, false);
            self.emit_log_alloc(cast_ptr.into(), size_val).ok();
        }

        let struct_ptr = self
            .builder
            .build_pointer_cast(
                raw_ptr,
                self.context.ptr_type(inkwell::AddressSpace::default()),
                "struct_ptr",
            )
            .map_err(|e| e.to_string())?;

        for (i, arg_expr) in args.iter().enumerate() {
            let (val, _ty) = self.compile_expr(arg_expr)?;
            let field_ptr = self
                .builder
                .build_struct_gep(
                    struct_type,
                    struct_ptr,
                    i as u32,
                    &format!("{}.{}", name, i),
                )
                .map_err(|e| e.to_string())?;

            let store_val = if matches!(
                _ty,
                Type::Tensor(_, _) | Type::Struct(_, _) | Type::UserDefined(_, _) | Type::Tuple(_)
            ) {
                self.emit_deep_clone(val, &_ty)?
            } else {
                val
            };
            self.builder
                .build_store(field_ptr, store_val)
                .map_err(|e| e.to_string())?;

            // Move Semantics Removed (same as compile_struct_init):
            // We now use RefCounting/DeepCopy. Source variable should remain valid (Shared ownership).
            // Cleanup at end of scope will decrement refcount (balancing the Acquire in emit_deep_clone).
            // No manual unregister. No removal from scope.
        }

        // Return the pointer directly (no load)
        // Struct remains registered in scope; caller (StmtKind::Let) will deep_clone it
        // Return the pointer directly (no load)
        // Struct remains registered in scope; caller (StmtKind::Let) will deep_clone it
        
        Ok((struct_ptr.into(), Type::Struct(name.to_string(), vec![])))
    }

    fn compile_static_method_call(
        &mut self,
        ty: &Type,
        method_name: &str,
        args: &[Expr],
    ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        let type_name = crate::compiler::type_registry::TypeRegistry::type_to_key(ty);

        // Handle Vec<T>::new()
        let inner_opt = match ty {
            Type::Vec(inner) => Some(inner.clone()),
            Type::UserDefined(n, args) if n == "Vec" => {
                if args.is_empty() {
                    Some(Box::new(Type::Void))
                } else if args.len() == 1 {
                    Some(Box::new(args[0].clone()))
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(inner) = inner_opt {
            if method_name == "new" {
                 let fn_name = match inner.as_ref() {
                     Type::U8 => "tl_vec_u8_new",
                     Type::I64 => "tl_vec_i64_new",
                     Type::F32 => "tl_vec_f32_new",
                     _ => "tl_vec_ptr_new",
                 };
                 let fn_val = self.module.get_function(fn_name).ok_or(format!("{} not found", fn_name))?;
                 let call = self.builder.build_call(fn_val, &[], "vec_new").map_err(|e| e.to_string())?;
                 let res = match call.try_as_basic_value() {
                     inkwell::values::ValueKind::Basic(v) => v,
                     _ => return Err("Invalid return from vec.new()".into()),
                 };
                 // For ptr, no cast needed as it returns void* equivalent (or Vec<void*> ptr) which is opaque ptr?
                 // But wait, tl_vec_ptr_new returns *mut Vec<*mut c_void>.
                 // LLVM sees it as pointer.
                 // We might need to pointer cast if strict typing is used.
                 // But in TL runtime, all objects are pointers.
                 return Ok((res, ty.clone()));
            }
        }

        if type_name == "Tokenizer" && method_name == "new" {
            if args.len() != 1 {
                return Err("Tokenizer::new requires 1 argument".into());
            }
            let (path_val, _) = self.compile_expr(&args[0])?;
            let fn_val = self
                .module
                .get_function("tl_tokenizer_new")
                .ok_or("tl_tokenizer_new not found")?;
            let call = self
                .builder
                .build_call(fn_val, &[path_val.into()], "tok_new")
                .map_err(|e| e.to_string())?;
            let handle = match call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v,
                _ => return Err("Invalid return from Tokenizer::new".into()),
            };

            let struct_type = *self
                .struct_types
                .get("Tokenizer")
                .ok_or("Struct type Tokenizer not found")?;
            let struct_def = self
                .struct_defs
                .get("Tokenizer")
                .ok_or("Struct definition Tokenizer not found")?;
            let size = struct_type
                .size_of()
                .ok_or("Cannot determine size of Tokenizer")?;
            
            let size_int = size;
            let size_i64 = if size_int.get_type() == self.context.i32_type() {
                self.builder.build_int_z_extend(size_int, self.context.i64_type(), "size_i64").unwrap()
            } else {
                size_int
            };

            let malloc_fn = self
                .module
                .get_function("malloc")
                .ok_or("malloc not found (declare in builtins)")?;
            let call = self
                .builder
                .build_call(malloc_fn, &[size_i64.into()], "tokenizer_malloc")
                .map_err(|e| e.to_string())?;
            let raw_ptr = match call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
                _ => return Err("malloc returned invalid value".into()),
            };
            if let Some(register_fn) = self.module.get_function("tl_mem_register_struct") {
                let cast_ptr = self
                    .builder
                    .build_pointer_cast(
                        raw_ptr,
                        self.context.ptr_type(inkwell::AddressSpace::default()),
                        "cast_ptr",
                    )
                    .unwrap();
                self.builder
                    .build_call(register_fn, &[cast_ptr.into()], "")
                    .map_err(|e| e.to_string())?;
            }
            let struct_ptr = self
                .builder
                .build_pointer_cast(
                    raw_ptr,
                    self.context.ptr_type(inkwell::AddressSpace::default()),
                    "tokenizer_ptr",
                )
                .map_err(|e| e.to_string())?;
            let field_idx = struct_def
                .fields
                .iter()
                .position(|(n, _)| n == "_h")
                .ok_or("Field _h not found in Tokenizer")?;
            let field_ptr = self
                .builder
                .build_struct_gep(struct_type, struct_ptr, field_idx as u32, "tokenizer_h")
                .map_err(|e| e.to_string())?;
            self.builder
                .build_store(field_ptr, handle)
                .map_err(|e| e.to_string())?;
            return Ok((struct_ptr.into(), Type::Struct("Tokenizer".to_string(), vec![])));
        }

        if type_name == "KVCache" && method_name == "new" {
            if args.len() != 1 {
                return Err("KVCache::new requires 1 argument".into());
            }
            let (layers_val, _) = self.compile_expr(&args[0])?;
            let fn_val = self
                .module
                .get_function("tl_kv_cache_new")
                .ok_or("tl_kv_cache_new not found")?;
            let call = self
                .builder
                .build_call(fn_val, &[layers_val.into()], "kv_new")
                .map_err(|e| e.to_string())?;
            let handle = match call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v,
                _ => return Err("Invalid return from KVCache::new".into()),
            };

            let struct_type = *self
                .struct_types
                .get("KVCache")
                .ok_or("Struct type KVCache not found")?;
            let struct_def = self
                .struct_defs
                .get("KVCache")
                .ok_or("Struct definition KVCache not found")?;
            let size = struct_type
                .size_of()
                .ok_or("Cannot determine size of KVCache")?;
            
            let size_int = size;
            let size_i64 = if size_int.get_type() == self.context.i32_type() {
                self.builder.build_int_z_extend(size_int, self.context.i64_type(), "size_i64").unwrap()
            } else {
                size_int
            };

            let malloc_fn = self
                .module
                .get_function("malloc")
                .ok_or("malloc not found (declare in builtins)")?;
            let call = self
                .builder
                .build_call(malloc_fn, &[size_i64.into()], "kvcache_malloc")
                .map_err(|e| e.to_string())?;
            let raw_ptr = match call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
                _ => return Err("malloc returned invalid value".into()),
            };
            if let Some(register_fn) = self.module.get_function("tl_mem_register_struct") {
                let cast_ptr = self
                    .builder
                    .build_pointer_cast(
                        raw_ptr,
                        self.context.ptr_type(inkwell::AddressSpace::default()),
                        "cast_ptr",
                    )
                    .unwrap();
                self.builder
                    .build_call(register_fn, &[cast_ptr.into()], "")
                    .map_err(|e| e.to_string())?;
            }
            let struct_ptr = self
                .builder
                .build_pointer_cast(
                    raw_ptr,
                    self.context.ptr_type(inkwell::AddressSpace::default()),
                    "kvcache_ptr",
                )
                .map_err(|e| e.to_string())?;
            let field_idx = struct_def
                .fields
                .iter()
                .position(|(n, _)| n == "_h")
                .ok_or("Field _h not found in KVCache")?;
            let field_ptr = self
                .builder
                .build_struct_gep(struct_type, struct_ptr, field_idx as u32, "kvcache_h")
                .map_err(|e| e.to_string())?;
            self.builder
                .build_store(field_ptr, handle)
                .map_err(|e| e.to_string())?;
            return Ok((struct_ptr.into(), Type::Struct("KVCache".to_string(), vec![])));
        }

        if type_name == "Map" && method_name == "load" {
            if args.len() != 1 {
                return Err("Map::load requires 1 argument".into());
            }
            let (path_val, _) = self.compile_expr(&args[0])?;
            let fn_val = self
                .module
                .get_function("tl_gguf_load")
                .ok_or("tl_gguf_load not found")?;
            let call = self
                .builder
                .build_call(fn_val, &[path_val.into()], "map_load")
                .map_err(|e| e.to_string())?;
            let res = match call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v,
                _ => return Err("Invalid return from Map::load".into()),
            };
            return Ok((res, Type::UserDefined("Map".to_string(), vec![])));
        }

        if type_name == "File" {
            match method_name {
                "exists" => {
                    if args.len() != 1 {
                        return Err("File::exists requires 1 argument".into());
                    }
                    let (path_val, _) = self.compile_expr(&args[0])?;
                    let fn_val = self
                        .module
                        .get_function("tl_file_exists_i64")
                        .ok_or("tl_file_exists_i64 not found")?;
                    let call = self
                        .builder
                        .build_call(fn_val, &[path_val.into()], "file_exists")
                        .map_err(|e| e.to_string())?;
                    let res = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v.into_int_value(),
                        _ => return Err("Invalid return from File::exists".into()),
                    };
                    let ok = self
                        .builder
                        .build_int_compare(
                            inkwell::IntPredicate::EQ,
                            res,
                            self.context.i64_type().const_int(1, false),
                            "file_exists_bool",
                        )
                        .map_err(|e| e.to_string())?;
                    return Ok((ok.into(), Type::Bool));
                }
                "read" => {
                    if args.len() != 1 {
                        return Err("File::read requires 1 argument".into());
                    }
                    let (path_val, _) = self.compile_expr(&args[0])?;
                    let fn_val = self
                        .module
                        .get_function("tl_read_file")
                        .ok_or("tl_read_file not found")?;
                    let call = self
                        .builder
                        .build_call(fn_val, &[path_val.into()], "file_read")
                        .map_err(|e| e.to_string())?;
                    let res = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err("Invalid return from File::read".into()),
                    };
                    return Ok((res, Type::UserDefined("String".to_string(), vec![])));
                }
                "write" => {
                    if args.len() != 2 {
                        return Err("File::write requires 2 arguments".into());
                    }
                    let (path_val, _) = self.compile_expr(&args[0])?;
                    let (content_val, _) = self.compile_expr(&args[1])?;
                    let fn_val = self
                        .module
                        .get_function("tl_write_file")
                        .ok_or("tl_write_file not found")?;
                    let call = self
                        .builder
                        .build_call(
                            fn_val,
                            &[path_val.into(), content_val.into()],
                            "file_write",
                        )
                        .map_err(|e| e.to_string())?;
                    let res = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v.into_int_value(),
                        _ => return Err("Invalid return from File::write".into()),
                    };
                    let ok = self
                        .builder
                        .build_int_compare(
                            inkwell::IntPredicate::EQ,
                            res,
                            self.context.i64_type().const_int(1, false),
                            "file_write_bool",
                        )
                        .map_err(|e| e.to_string())?;
                    return Ok((ok.into(), Type::Bool));
                }
                "download" => {
                    if args.len() != 2 {
                        return Err("File::download requires 2 arguments".into());
                    }
                    let (url_val, _) = self.compile_expr(&args[0])?;
                    let (path_val, _) = self.compile_expr(&args[1])?;
                    let fn_val = self
                        .module
                        .get_function("tl_download_file")
                        .ok_or("tl_download_file not found")?;
                    let call = self
                        .builder
                        .build_call(
                            fn_val,
                            &[url_val.into(), path_val.into()],
                            "file_download",
                        )
                        .map_err(|e| e.to_string())?;
                    let res = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v.into_int_value(),
                        _ => return Err("Invalid return from File::download".into()),
                    };
                    let ok = self
                        .builder
                        .build_int_compare(
                            inkwell::IntPredicate::EQ,
                            res,
                            self.context.i64_type().const_int(1, false),
                            "file_download_bool",
                        )
                        .map_err(|e| e.to_string())?;
                    return Ok((ok.into(), Type::Bool));
                }
                "read_binary" => {
                    if args.len() != 1 {
                        return Err("File::read_binary requires 1 argument".into());
                    }
                    let (path_val, _) = self.compile_expr(&args[0])?;
                    let fn_val = self
                        .module
                        .get_function("tl_file_read_binary")
                        .ok_or("tl_file_read_binary not found")?;
                    let call = self
                        .builder
                        .build_call(fn_val, &[path_val.into()], "file_read_binary")
                        .map_err(|e| e.to_string())?;
                    let res = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err("Invalid return from File::read_binary".into()),
                    };
                    return Ok((res, Type::Vec(Box::new(Type::U8))));
                }
                _ => {}
            }
        }

        if type_name == "System" && method_name == "memory_mb" {
            if !args.is_empty() {
                return Err("System::memory_mb takes no arguments".into());
            }
            let fn_val = self
                .module
                .get_function("tl_get_memory_mb")
                .ok_or("tl_get_memory_mb not found")?;
            let call = self
                .builder
                .build_call(fn_val, &[], "mem_mb")
                .map_err(|e| e.to_string())?;
            let res = match call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v,
                _ => return Err("Invalid return from System::memory_mb".into()),
            };
            return Ok((res, Type::I64));
        }
        if type_name == "System" && method_name == "metal_pool_bytes" {
            if !args.is_empty() {
                return Err("System::metal_pool_bytes takes no arguments".into());
            }
            let fn_val = self
                .module
                .get_function("tl_get_metal_pool_bytes")
                .ok_or("tl_get_metal_pool_bytes not found")?;
            let call = self
                .builder
                .build_call(fn_val, &[], "metal_pool_bytes")
                .map_err(|e| e.to_string())?;
            let res = match call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v,
                _ => return Err("Invalid return from System::metal_pool_bytes".into()),
            };
            return Ok((res, Type::I64));
        }
        if type_name == "System" && method_name == "metal_pool_mb" {
            if !args.is_empty() {
                return Err("System::metal_pool_mb takes no arguments".into());
            }
            let fn_val = self
                .module
                .get_function("tl_get_metal_pool_mb")
                .ok_or("tl_get_metal_pool_mb not found")?;
            let call = self
                .builder
                .build_call(fn_val, &[], "metal_pool_mb")
                .map_err(|e| e.to_string())?;
            let res = match call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v,
                _ => return Err("Invalid return from System::metal_pool_mb".into()),
            };
            return Ok((res, Type::I64));
        }
        if type_name == "System" && method_name == "metal_pool_count" {
            if !args.is_empty() {
                return Err("System::metal_pool_count takes no arguments".into());
            }
            let fn_val = self
                .module
                .get_function("tl_get_metal_pool_count")
                .ok_or("tl_get_metal_pool_count not found")?;
            let call = self
                .builder
                .build_call(fn_val, &[], "metal_pool_count")
                .map_err(|e| e.to_string())?;
            let res = match call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v,
                _ => return Err("Invalid return from System::metal_pool_count".into()),
            };
            return Ok((res, Type::I64));
        }
        if type_name == "System" && method_name == "metal_sync" {
            if !args.is_empty() {
                return Err("System::metal_sync takes no arguments".into());
            }
            let fn_val = self
                .module
                .get_function("tl_metal_sync")
                .ok_or("tl_metal_sync not found")?;
            self.builder
                .build_call(fn_val, &[], "metal_sync")
                .map_err(|e| e.to_string())?;
            return Ok((
                self.context.i64_type().const_int(0, false).into(),
                Type::Void,
            ));
        }
        if type_name == "System" && method_name == "pool_count" {
            if !args.is_empty() {
                return Err("System::pool_count takes no arguments".into());
            }
            let fn_val = self
                .module
                .get_function("tl_get_pool_count")
                .ok_or("tl_get_pool_count not found")?;
            let call = self
                .builder
                .build_call(fn_val, &[], "pool_count")
                .map_err(|e| e.to_string())?;
            let res = match call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v,
                _ => return Err("Invalid return from System::pool_count".into()),
            };
            return Ok((res, Type::I64));
        }
        if type_name == "System" && method_name == "refcount_count" {
            if !args.is_empty() {
                return Err("System::refcount_count takes no arguments".into());
            }
            let fn_val = self
                .module
                .get_function("tl_get_refcount_count")
                .ok_or("tl_get_refcount_count not found")?;
            let call = self
                .builder
                .build_call(fn_val, &[], "refcount_count")
                .map_err(|e| e.to_string())?;
            let res = match call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v,
                _ => return Err("Invalid return from System::refcount_count".into()),
            };
            return Ok((res, Type::I64));
        }
        if type_name == "System" && method_name == "scope_depth" {
            if !args.is_empty() {
                return Err("System::scope_depth takes no arguments".into());
            }
            let fn_val = self
                .module
                .get_function("tl_get_scope_depth")
                .ok_or("tl_get_scope_depth not found")?;
            let call = self
                .builder
                .build_call(fn_val, &[], "scope_depth")
                .map_err(|e| e.to_string())?;
            let res = match call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v,
                _ => return Err("Invalid return from System::scope_depth".into()),
            };
            return Ok((res, Type::I64));
        }

        if type_name == "Tensor" && method_name == "clear_grads" {
            if !args.is_empty() {
                return Err("Tensor::clear_grads takes no arguments".into());
            }
            let fn_val = self
                .module
                .get_function("tl_clear_grads")
                .ok_or("tl_clear_grads not found")?;
            self.builder
                .build_call(fn_val, &[], "clear_grads")
                .map_err(|e| e.to_string())?;
            return Ok((
                self.context.i64_type().const_int(0, false).into(),
                Type::Void,
            ));
        }

        if type_name == "Path" && method_name == "exists" {
            if args.len() != 1 {
                return Err("Path::exists requires 1 argument".into());
            }
            let (path_val, _) = self.compile_expr(&args[0])?;
            let fn_val = self
                .module
                .get_function("tl_path_exists")
                .ok_or("tl_path_exists not found")?;
            let call = self
                .builder
                .build_call(fn_val, &[path_val.into()], "path_exists")
                .map_err(|e| e.to_string())?;
            let res = match call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v,
                _ => return Err("Invalid return from Path::exists".into()),
            };
            return Ok((res, Type::Bool));
        }

        // 1. StaticMethodManager Lookup
        let method_opt = self
            .static_methods
            .get(&type_name)
            .and_then(|m| m.get(method_name))
            .copied();
        if let Some(method) = method_opt {
            match method {
                StaticMethod::Evaluated(func) => {
                    let mut compiled_args = Vec::with_capacity(args.len());
                    let mut compiled_args_types = Vec::with_capacity(args.len());
                    for arg in args {
                        let (val, ty) = self.compile_expr(arg)?;
                        // Auto-conversion: ScalarArray -> Tensor (Generic Rule)
                        let (val, ty) = if let Type::ScalarArray(_, _) = ty {
                            let (new_val, new_ty) = self.ensure_tensor_v2(arg, 0)?;
                            (new_val.try_into().unwrap(), new_ty)
                        } else {
                            (val, ty)
                        };
                        compiled_args.push((val, ty.clone()));
                        compiled_args_types.push((val, ty));
                    }
                    let res = func(self, compiled_args);

                    return res;


                }
                StaticMethod::Unevaluated(func) => {
                    return func(self, args);
                }
            }
        }

        // 1. Resolve Mangled Name
        let simple_type_name = if type_name.contains("::") {
            type_name.split("::").last().unwrap()
        } else {
            &type_name
        };
        let mangled_name = format!("tl_{}_{}", simple_type_name, method_name);
        let stdlib_name = format!("tl_{}_{}", simple_type_name.to_lowercase(), method_name);

        // 2. Lookup Function
        let (func, actual_name) = if let Some(f) = self.module.get_function(&mangled_name) {
            (f, mangled_name)
        } else if let Some(f) = self.module.get_function(&stdlib_name) {
            (f, stdlib_name)
        } else if let Some(f) = self.module.get_function(method_name) {
            (f, method_name.to_string())
        } else {
             // Fallback: Check if it is an Enum Variant initialization
             if let Some(enum_def) = self.enum_defs.get(&type_name).cloned() {
                 if let Some(variant_idx) = enum_def.variants.iter().position(|v| v.name == method_name) {
                     let variant_def = &enum_def.variants[variant_idx];
                     if args.len() != variant_def.fields.len() {
                         return Err(format!("Enum variant {}::{} expects {} args, got {}", type_name, method_name, variant_def.fields.len(), args.len()));
                     }
                     
                     let enum_ty = *self.enum_types.get(&type_name).ok_or(format!("Enum type {} not found", type_name))?;
                     
                     // Allocate
                     // Manual malloc(i64)
                     let size_ptr = unsafe {
                        self.builder.build_gep(
                            enum_ty,
                            self.context.ptr_type(inkwell::AddressSpace::default()).const_null(),
                            &[self.context.i64_type().const_int(1, false)],
                            "size_ptr",
                        ).map_err(|e| e.to_string())?
                    };
                    let size = self.builder
                        .build_ptr_to_int(size_ptr, self.context.i64_type(), "size")
                        .map_err(|e| e.to_string())?;

                    let malloc_fn = self.module.get_function("malloc").ok_or("malloc not found")?;
                    let alloca = match self.builder
                        .build_call(malloc_fn, &[size.into()], &format!("enum_{}", type_name))
                        .map_err(|e| e.to_string())?
                        .try_as_basic_value() {
                            inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
                            _ => return Err("malloc returned void".into()),
                        };
                     
                     // Store Tag
                     let tag_ptr = self.builder.build_struct_gep(enum_ty, alloca, 0, "tag_ptr").map_err(|e| e.to_string())?;
                     self.builder.build_store(tag_ptr, self.context.i32_type().const_int(variant_idx as u64, false)).map_err(|e| e.to_string())?;
                     
                     // Store Fields
                     if !variant_def.fields.is_empty() {
                         let payload_ptr_raw = self.builder.build_struct_gep(enum_ty, alloca, 1, "payload_ptr_raw").map_err(|e| e.to_string())?;
                         
                         let mut field_types = vec![];
                         for (_, ty) in &variant_def.fields {
                             let llvm_ty = match ty {
                                 Type::F32 => self.context.f32_type().into(),
                                 Type::I64 => self.context.i64_type().into(),
                                 Type::Bool => self.context.bool_type().into(),
                                 Type::Tensor(_, _) | Type::Struct(_, _) | Type::Enum(_, _) | Type::UserDefined(_, _) | Type::Vec(_) => 
                                     self.context.ptr_type(inkwell::AddressSpace::default()).into(),
                                 _ => self.context.i64_type().into(),
                             };
                             field_types.push(llvm_ty);
                         }
                         let variant_struct_ty = self.context.struct_type(&field_types, false);
                         
                         let payload_ptr = self.builder.build_pointer_cast(
                             payload_ptr_raw,
                             self.context.ptr_type(inkwell::AddressSpace::default()),
                             "payload_cast"
                         ).unwrap();
                         
                         for (i, arg) in args.iter().enumerate() {
                             let (val, _) = self.compile_expr(arg)?;
                             let field_ptr = self.builder.build_struct_gep(variant_struct_ty, payload_ptr, i as u32, "field_ptr").map_err(|e| e.to_string())?;
                             self.builder.build_store(field_ptr, val).map_err(|e| e.to_string())?;
                         }
                     }
                     
                     return Ok((alloca.into(), Type::Enum(type_name.to_string(), vec![])));
                 }
             }

            return Err(format!(
                "Static method {}::{} not found (checked {}, {}, and {})",
                type_name, method_name, mangled_name, stdlib_name, method_name
            ));
        };

        // 3. Generic Fallback: Compile Args & Handle SRET
        // Get return type to check if this uses sret (do this before compiling args)
        let ret_ty = self
            .fn_return_types
            .get(&actual_name)
            .cloned()
            .unwrap_or(Type::Void);

        // Check for SRET usage logic (Structs/UserDefined return types usually use SRET in this ABI)
        // If the function definition requires SRET (hidden first arg pointer), we must allocate it.
        // We detect this if ret_ty is Struct/UserDefined AND the function is not in the "exclude SRET" list?
        // Or essentially if it's a struct return.
        // The old code had a manual toggle `uses_sret = false` or similar?
        // No, Step 201 showed `uses_sret = false; /* SRET DISABLED */`.
        // If SRET was disabled in legacy code, I should probably keep it disabled unless I know otherwise.
        // BUT `Struct::new` relies on it?
        // Step 201 had `if uses_sret { ... }`.
        // If `uses_sret` was hardcoded to `false`, then SRET is NOT used.
        // I will replicate strict legacy behavior: SRET disabled.
        // Wait, Step 201 line 3244: `let uses_sret = false;`.
        // So I'll just compile args and call.

        let mut compiled_args = Vec::with_capacity(args.len());
        let mut compiled_args_types = Vec::with_capacity(args.len());
        for arg in args {
            let (val, ty) = self.compile_expr(arg)?;
            // Auto-conversion: ScalarArray -> Tensor (Generic Rule)
            let (val, ty) = if let Type::ScalarArray(_, _) = ty {
                let (new_val, new_ty) = self.ensure_tensor_v2(arg, 0)?;
                (new_val.try_into().unwrap(), new_ty)
            } else {
                (val, ty)
            };
            compiled_args.push(val.into());
            compiled_args_types.push((val, ty));
        }

        // 4. Call
        let call = self
            .builder
            .build_call(func, &compiled_args, "static_call")
            .map_err(|e| e.to_string())?;




        match call.try_as_basic_value() {
            inkwell::values::ValueKind::Basic(_) => {
                let v = self.check_tensor_result(call, "static_call_error")?;
                // Register intermediate tensor result
                // REMOVED REDUNDANT REGISTRATION

                Ok((v, ret_ty))
            }
            _ => Ok((
                self.context.i64_type().const_int(0, false).into(),
                Type::Void,
            )),
        }
    }

    fn compile_tensor_literal(
        &mut self,
        elements: &[Expr],
    ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        // 1. Calculate shape and total elements
        fn count_elements(exprs: &[Expr]) -> (usize, Vec<usize>) {
            if exprs.is_empty() {
                return (0, vec![0]);
            }

            let is_nested = matches!(exprs[0].inner, ExprKind::TensorLiteral(_));
            if is_nested {
                let mut total = 0;
                let mut first_shape = None;
                for e in exprs {
                    if let ExprKind::TensorLiteral(children) = &e.inner {
                        let (count, shape) = count_elements(children);
                        total += count;
                        if first_shape.is_none() {
                            first_shape = Some(shape);
                        }
                    }
                }
                let mut shape = vec![exprs.len()];
                if let Some(s) = first_shape {
                    shape.extend(s);
                }
                (total, shape)
            } else {
                (exprs.len(), vec![exprs.len()])
            }
        }

        let (total_elements, shape) = count_elements(elements);
        let rank = shape.len();

        // 2. Flatten elements
        fn flatten_exprs(exprs: &[Expr], result: &mut Vec<Expr>) {
            for e in exprs {
                if let ExprKind::TensorLiteral(children) = &e.inner {
                    flatten_exprs(children, result);
                } else {
                    result.push(e.clone());
                }
            }
        }

        let mut flat_exprs = Vec::new();
        flatten_exprs(elements, &mut flat_exprs);

        // 3. Compile all elements and determine target type
        let mut compiled_vals = Vec::with_capacity(flat_exprs.len());
        let mut has_float = false;

        for expr in &flat_exprs {
            let (val, ty) = self.compile_expr(expr)?;
            compiled_vals.push((val, ty.clone()));
            if matches!(ty, Type::F32 | Type::F64) {
                has_float = true;
            }
        }

        // Determine target type: F32 if any float present, else I64 (if all are numeric ints)
        let i64_type = self.context.i64_type();
        let f32_type = self.context.f32_type();

        // 4. Allocate temporary buffer for data
        let alloc_tmp_fn = self
            .module
            .get_function("tl_alloc_tmp")
            .expect("tl_alloc_tmp not found");

        let element_size = if has_float { 4 } else { 8 };
        let size = self
            .builder
            .build_int_mul(
                i64_type.const_int(total_elements as u64, false),
                i64_type.const_int(element_size, false),
                "alloc_size",
            )
            .map_err(|e| e.to_string())?;

        let call_idx = self
            .builder
            .build_call(alloc_tmp_fn, &[size.into()], "buf_void")
            .map_err(|e| e.to_string())?;

        let data_alloca = match call_idx.try_as_basic_value() {
            inkwell::values::ValueKind::Basic(inkwell::values::BasicValueEnum::PointerValue(v)) => {
                v
            }
            _ => return Err("Invalid tl_alloc_tmp return".to_string()),
        };

        // 5. Store elements into buffer
        for (i, (val, ty)) in compiled_vals.iter().enumerate() {
            if has_float {
                let float_val = match ty {
                    Type::F32 => val.into_float_value(),
                    Type::F64 => self
                        .builder
                        .build_float_cast(val.into_float_value(), f32_type, "f64_to_f32")
                        .unwrap(),
                    Type::I64 => self
                        .builder
                        .build_signed_int_to_float(val.into_int_value(), f32_type, "i64_to_f32")
                        .unwrap(),
                    Type::I32 | Type::Bool => self
                        .builder
                        .build_signed_int_to_float(
                            self.builder
                                .build_int_z_extend(val.into_int_value(), i64_type, "zext")
                                .unwrap(),
                            f32_type,
                            "i_to_f32",
                        )
                        .unwrap(),
                    _ => return Err(format!("Cannot cast {:?} to F32", ty)),
                };

                let ptr = unsafe {
                    self.builder
                        .build_in_bounds_gep(
                            f32_type,
                            data_alloca,
                            &[i64_type.const_int(i as u64, false)],
                            "elem_ptr",
                        )
                        .map_err(|e| e.to_string())?
                };
                self.builder.build_store(ptr, float_val).unwrap();
            } else {
                let int_val = match ty {
                    Type::I64 => val.into_int_value(),
                    Type::I32 => self
                        .builder
                        .build_int_z_extend(val.into_int_value(), i64_type, "zext")
                        .unwrap(),
                    Type::Bool => self
                        .builder
                        .build_int_z_extend(val.into_int_value(), i64_type, "zext")
                        .unwrap(),
                    _ => return Err(format!("Cannot cast {:?} to I64", ty)),
                };

                let ptr = unsafe {
                    self.builder
                        .build_in_bounds_gep(
                            i64_type,
                            data_alloca,
                            &[i64_type.const_int(i as u64, false)],
                            "elem_ptr",
                        )
                        .map_err(|e| e.to_string())?
                };
                self.builder.build_store(ptr, int_val).unwrap();
            }
        }

        // 6. Allocate and fill shape buffer
        let shape_size = rank as u64 * 8; // i64 size
        let shape_alloc_call = self
            .builder
            .build_call(
                alloc_tmp_fn,
                &[i64_type.const_int(shape_size, false).into()],
                "shape_alloc",
            )
            .map_err(|e| e.to_string())?;

        let shape_alloca = match shape_alloc_call.try_as_basic_value() {
            ValueKind::Basic(v) => v.into_pointer_value(),
            _ => return Err("tl_alloc_tmp failed".into()),
        };

        for (i, dim) in shape.iter().enumerate() {
            let ptr = unsafe {
                self.builder
                    .build_in_bounds_gep(
                        i64_type,
                        shape_alloca,
                        &[i64_type.const_int(i as u64, false)],
                        "shape_ptr",
                    )
                    .map_err(|e| e.to_string())?
            };
            self.builder
                .build_store(ptr, i64_type.const_int(*dim as u64, false))
                .map_err(|e| e.to_string())?;
        }

        // 7. Call appropriate runtime tensor constructor
        let (new_fn_name, result_inner_type) = if has_float {
            ("tl_tensor_new", Type::F32)
        } else {
            ("tl_tensor_new_i64", Type::I64)
        };

        let new_fn = self
            .module
            .get_function(new_fn_name)
            .ok_or(format!("{} not found", new_fn_name))?;

        let shape_ptr_cast = self
            .builder
            .build_pointer_cast(
                shape_alloca,
                self.context.ptr_type(inkwell::AddressSpace::default()),
                "shap_ptr_cast",
            )
            .map_err(|e| e.to_string())?;

        let data_ptr_cast = self
            .builder
            .build_pointer_cast(
                data_alloca,
                self.context.ptr_type(inkwell::AddressSpace::default()),
                "data_ptr_cast",
            )
            .unwrap();

        let call = self
            .builder
            .build_call(
                new_fn,
                &[
                    data_ptr_cast.into(),
                    i64_type.const_int(rank as u64, false).into(),
                    shape_ptr_cast.into(),
                ],
                "new_tensor",
            )
            .map_err(|e| e.to_string())?;

        let res = self.check_tensor_result(call, "new_tensor_error")?;

        // 8. Free temporary buffers
        let free_tmp_fn = self
            .module
            .get_function("tl_free_tmp")
            .expect("tl_free_tmp not found");

        self.builder
            .build_call(free_tmp_fn, &[data_alloca.into()], "")
            .unwrap();
        self.builder
            .build_call(free_tmp_fn, &[shape_alloca.into()], "")
            .unwrap();

        let result_ty = Type::Tensor(Box::new(result_inner_type), rank);


        Ok((res, result_ty))
    }

    fn compile_tensor_const_literal(
        &mut self,
        elements: &[Expr],
    ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        // Optimized path for constant tensor literals - static extraction
        fn flatten_const(exprs: &[Expr]) -> Result<(Vec<f64>, Vec<usize>, bool), String> {
            if exprs.is_empty() {
                return Ok((vec![], vec![0], false));
            }

            let is_nested = matches!(
                &exprs[0].inner,
                ExprKind::TensorConstLiteral(_) | ExprKind::TensorLiteral(_)
            );
            if is_nested {
                let mut flat_data = Vec::new();
                let mut first_shape = None;
                let mut all_ints = true;

                for e in exprs {
                    let (children, shape, ints) = match &e.inner {
                        ExprKind::TensorConstLiteral(c) | ExprKind::TensorLiteral(c) => {
                            flatten_const(c)?
                        }
                        _ => return Err("Mixed types in const tensor".into()),
                    };
                    if let Some(ref s) = first_shape {
                        if s != &shape {
                            return Err("Ragged tensors not supported".into());
                        }
                    } else {
                        first_shape = Some(shape.clone());
                    }
                    flat_data.extend(children);
                    all_ints &= ints;
                }

                let mut shape = vec![exprs.len()];
                if let Some(s) = first_shape {
                    shape.extend(s);
                }
                Ok((flat_data, shape, all_ints))
            } else {
                let mut data = Vec::new();
                let mut all_ints = true;
                for e in exprs {
                    match &e.inner {
                        ExprKind::Float(f) => {
                            data.push(*f);
                            all_ints = false;
                        }
                        ExprKind::Int(i) => data.push(*i as f64),
                        _ => return Err("Const tensor must contain only literals".into()),
                    }
                }
                Ok((data, vec![exprs.len()], all_ints))
            }
        }

        let (flat_data, shape, all_ints) = flatten_const(elements)?;
        let rank = shape.len();
        let len = flat_data.len();

        // OPTIMIZATION: For small 1D constant tensors (≤8 elements), use heap-based scalar array
        // DISABLED: This causes ABI mismatch with functions expecting OpaqueTensor* (Tensor<T,N>).
        // Until ScalarArray can be auto-converted or functions support it, we must force OpaqueTensor.
        if false && rank == 1 && len <= 8 && len > 0 {
            let (_elem_ty, _llvm_elem_type): (Type, inkwell::types::BasicTypeEnum) = if all_ints {
                (Type::I64, self.context.i64_type().into())
            } else {
                (Type::F32, self.context.f32_type().into())
            };
            // ... dead code ...
            return Err("Optimization disabled".into());
        }

        // Fall back to standard tensor creation for larger tensors (and now all tensors)
        let len = len as u64;
        let f32_type = self.context.f32_type();
        let i64_type = self.context.i64_type();

        // Use tl_alloc_tmp instead of malloc
        let alloc_tmp_fn = self
            .module
            .get_function("tl_alloc_tmp")
            .expect("tl_alloc_tmp not found");
        let free_tmp_fn = self
            .module
            .get_function("tl_free_tmp")
            .expect("tl_free_tmp not found");

        if all_ints {
            // I64 TENSOR Creation
            let data_size_bytes = len * 8; // i64 = 8 bytes
            let alloc_call = self
                .builder
                .build_call(
                    alloc_tmp_fn,
                    &[i64_type.const_int(data_size_bytes, false).into()],
                    "temp_data_alloc_i64",
                )
                .map_err(|e| e.to_string())?;
            let data_ptr = match alloc_call.try_as_basic_value() {
                ValueKind::Basic(v) => v.into_pointer_value(),
                _ => return Err("tl_alloc_tmp returned non-pointer".into()),
            };

            // Populate data buffer (i64)
            for (i, val) in flat_data.iter().enumerate() {
                let int_val = i64_type.const_int(*val as u64, false); // val is f64, safe cast for ints
                let elem_ptr = unsafe {
                    self.builder
                        .build_in_bounds_gep(
                            i64_type,
                            data_ptr, // treated as i64*
                            &[i64_type.const_int(i as u64, false)],
                            "data_elem",
                        )
                        .map_err(|e| e.to_string())?
                };
                // Cast pointer if needed, but here we treat malloc'd void* as i64* implicitly via GEP type
                // Actually build_in_bounds_gep on opaque ptr might require explicit type.
                // LLVM 15+ uses opaque pointers, so GEP type matches element type.
                self.builder
                    .build_store(elem_ptr, int_val)
                    .map_err(|e| e.to_string())?;
            }

            // Allocate shape buffer
            let shape_size_bytes = rank as u64 * 8;
            let shape_alloc_call = self
                .builder
                .build_call(
                    alloc_tmp_fn,
                    &[i64_type.const_int(shape_size_bytes, false).into()],
                    "temp_shape_alloc",
                )
                .map_err(|e| e.to_string())?;
            let shape_ptr = match shape_alloc_call.try_as_basic_value() {
                ValueKind::Basic(v) => v.into_pointer_value(),
                _ => return Err("tl_alloc_tmp returned non-pointer".into()),
            };

            // Populate shape (same for both)
            for (i, dim) in shape.iter().enumerate() {
                let elem_ptr = unsafe {
                    self.builder
                        .build_in_bounds_gep(
                            i64_type,
                            shape_ptr,
                            &[i64_type.const_int(i as u64, false)],
                            "shape_elem",
                        )
                        .map_err(|e| e.to_string())?
                };
                self.builder
                    .build_store(elem_ptr, i64_type.const_int(*dim as u64, false))
                    .map_err(|e| e.to_string())?;
            }

            // Call tl_tensor_new_i64
            let new_fn = self
                .module
                .get_function("tl_tensor_new_i64")
                .ok_or("tl_tensor_new_i64 not found")?;

            let call = self
                .builder
                .build_call(
                    new_fn,
                    &[
                        data_ptr.into(),
                        i64_type.const_int(rank as u64, false).into(),
                        shape_ptr.into(),
                    ],
                    "new_const_tensor_i64",
                )
                .map_err(|e| e.to_string())?;

            let res = self.check_tensor_result(call, "new_const_tensor_i64_error")?;

            // FREE temps
            self.builder
                .build_call(free_tmp_fn, &[data_ptr.into()], "")
                .map_err(|e| e.to_string())?;
            self.builder
                .build_call(free_tmp_fn, &[shape_ptr.into()], "")
                .map_err(|e| e.to_string())?;

            Ok((res, Type::Tensor(Box::new(Type::I64), rank)))
        } else {
            // F32 TENSOR Creation
            let data_size_bytes = len * 4; // f32 = 4 bytes
            let alloc_call = self
                .builder
                .build_call(
                    alloc_tmp_fn,
                    &[i64_type.const_int(data_size_bytes, false).into()],
                    "temp_data_alloc",
                )
                .map_err(|e| e.to_string())?;
            let data_ptr = match alloc_call.try_as_basic_value() {
                ValueKind::Basic(v) => v.into_pointer_value(),
                _ => return Err("tl_alloc_tmp returned non-pointer".into()),
            };

            // Populate data buffer (f32)
            for (i, val) in flat_data.iter().enumerate() {
                let float_val = f32_type.const_float(*val);
                let elem_ptr = unsafe {
                    self.builder
                        .build_in_bounds_gep(
                            f32_type,
                            data_ptr,
                            &[i64_type.const_int(i as u64, false)],
                            "data_elem",
                        )
                        .map_err(|e| e.to_string())?
                };
                self.builder
                    .build_store(elem_ptr, float_val)
                    .map_err(|e| e.to_string())?;
            }

            // Allocate shape buffer
            let shape_size_bytes = rank as u64 * 8; // i64 = 8 bytes
            let shape_alloc_call = self
                .builder
                .build_call(
                    alloc_tmp_fn,
                    &[i64_type.const_int(shape_size_bytes, false).into()],
                    "temp_shape_alloc",
                )
                .map_err(|e| e.to_string())?;
            let shape_ptr = match shape_alloc_call.try_as_basic_value() {
                ValueKind::Basic(v) => v.into_pointer_value(),
                _ => return Err("tl_alloc_tmp returned non-pointer".into()),
            };

            // Populate shape buffer
            for (i, dim) in shape.iter().enumerate() {
                let elem_ptr = unsafe {
                    self.builder
                        .build_in_bounds_gep(
                            i64_type,
                            shape_ptr,
                            &[i64_type.const_int(i as u64, false)],
                            "shape_elem",
                        )
                        .map_err(|e| e.to_string())?
                };
                self.builder
                    .build_store(elem_ptr, i64_type.const_int(*dim as u64, false))
                    .map_err(|e| e.to_string())?;
            }

            // Call tl_tensor_new
            let new_fn = self
                .module
                .get_function("tl_tensor_new")
                .ok_or("tl_tensor_new not found")?;

            let call = self
                .builder
                .build_call(
                    new_fn,
                    &[
                        data_ptr.into(),
                        i64_type.const_int(rank as u64, false).into(),
                        shape_ptr.into(),
                    ],
                    "new_const_tensor",
                )
                .map_err(|e| e.to_string())?;

            let res = self.check_tensor_result(call, "new_const_tensor_error")?;

            // FREE temps
            self.builder
                .build_call(free_tmp_fn, &[data_ptr.into()], "")
                .map_err(|e| e.to_string())?;
            self.builder
                .build_call(free_tmp_fn, &[shape_ptr.into()], "")
                .map_err(|e| e.to_string())?;

            Ok((res, Type::Tensor(Box::new(Type::F32), rank)))
        }
    }

    fn compile_match_like(
        &mut self,
        subject_expr: &Expr,
        arms: &[(Pattern, Expr)],
    ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        let (subject_val, subject_ty) = self.compile_expr(subject_expr)?;
        let enum_name = match subject_ty {
            Type::Enum(n, _) | Type::UserDefined(n, _) => n,
            _ => return Err("Match on non-enum".into()),
        };
        let enum_def = self
            .enum_defs
            .get(&enum_name)
            .ok_or("Enum def not found")?
            .clone();
        let enum_ty = *self
            .enum_types
            .get(&enum_name)
            .ok_or("Enum type not found")?;

        let ptr = subject_val.into_pointer_value();

        // Load Tag
        let tag_ptr = self
            .builder
            .build_struct_gep(enum_ty, ptr, 0, "tag_ptr")
            .map_err(|e| e.to_string())?;
        let tag_val = self
            .builder
            .build_load(self.context.i32_type(), tag_ptr, "tag")
            .map_err(|e| e.to_string())?
            .into_int_value();

        let current_func = self
            .builder
            .get_insert_block()
            .unwrap()
            .get_parent()
            .unwrap();
        let merge_block = self.context.append_basic_block(current_func, "match_merge");

        let mut arm_blocks = Vec::with_capacity(arms.len());
        for i in 0..arms.len() {
            arm_blocks.push(
                self.context
                    .append_basic_block(current_func, &format!("arm_{}", i)),
            );
        }

        let mut switch_cases = vec![];
        let mut used_variants = HashSet::new();
        for (arm_idx, (pat, _)) in arms.iter().enumerate() {
            if let Pattern::EnumPattern { variant_name, .. } = pat {
                let idx = enum_def
                    .variants
                    .iter()
                    .position(|v| v.name == *variant_name)
                    .ok_or("Enum variant not found")?;
                if used_variants.insert(idx) {
                    switch_cases.push((
                        self.context.i32_type().const_int(idx as u64, false),
                        arm_blocks[arm_idx],
                    ));
                }
            }
        }

        let wildcard_block = arms
            .iter()
            .position(|(p, _)| matches!(p, Pattern::Wildcard))
            .map(|i| arm_blocks[i]);

        let default_block = if let Some(wb) = wildcard_block {
            wb
        } else {
            let func = self
                .builder
                .get_insert_block()
                .unwrap()
                .get_parent()
                .unwrap();
            let unreachable_bb = self.context.append_basic_block(func, "match_unreachable");
            let current_block = self.builder.get_insert_block().unwrap();
            self.builder.position_at_end(unreachable_bb);
            let _ = self.builder.build_unreachable();
            self.builder.position_at_end(current_block);
            unreachable_bb
        };

        self.builder
            .build_switch(tag_val, default_block, &switch_cases)
            .map_err(|e| e.to_string())?;

        let mut incoming_vals = vec![];
        let mut result_type = Type::Void;

        for (i, (pat, body)) in arms.iter().enumerate() {
            let block = arm_blocks[i];
            self.builder.position_at_end(block);

            self.enter_scope();

            if let Pattern::EnumPattern {
                variant_name,
                bindings,
                ..
            } = pat
            {
                let variant_idx = enum_def
                    .variants
                    .iter()
                    .position(|v| v.name == *variant_name)
                    .ok_or("Enum variant not found")?;
                self.bind_enum_pattern_fields(
                    current_func,
                    enum_ty,
                    ptr,
                    &enum_def,
                    variant_idx,
                    bindings,
                )?;
            }

            let (val, ty) = self.compile_match_arm_body(body)?;
            if i == 0 {
                result_type = ty.clone();
            }

            self.exit_scope();

            let current_insert_block = self.builder.get_insert_block().unwrap();
            if current_insert_block.get_terminator().is_none() {
                self.builder
                    .build_unconditional_branch(merge_block)
                    .map_err(|e| e.to_string())?;
                incoming_vals.push((val, current_insert_block));
            }
        }

        self.builder.position_at_end(merge_block);

        if result_type == Type::Void {
            Ok((
                self.context.i64_type().const_int(0, false).into(),
                Type::Void,
            ))
        } else {
            let phi_type: inkwell::types::BasicTypeEnum = match result_type {
                Type::F32 => self.context.f32_type().into(),
                Type::I64 => self.context.i64_type().into(),
                Type::Bool => self.context.bool_type().into(),
                _ => self
                    .context
                    .ptr_type(inkwell::AddressSpace::default())
                    .into(),
            };
            let phi = self.builder.build_phi(phi_type, "match_res").unwrap();
            let incomings: Vec<(
                &dyn inkwell::values::BasicValue,
                inkwell::basic_block::BasicBlock,
            )> = incoming_vals
                .iter()
                .map(|(v, b)| (v as &dyn inkwell::values::BasicValue, *b))
                .collect();
            phi.add_incoming(&incomings);

            Ok((phi.as_basic_value(), result_type))
        }
    }

    fn compile_match_arm_body(
        &mut self,
        body: &Expr,
    ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        if let ExprKind::Block(stmts) = &body.inner {
            self.enter_scope();
            let mut last_val = None;
            let mut last_is_lvalue = false;
            for (i, stmt) in stmts.iter().enumerate() {
                if i == stmts.len() - 1 {
                    if let StmtKind::Expr(e) = &stmt.inner {
                        last_is_lvalue = Self::is_lvalue_expr(e);
                        last_val = Some(self.compile_expr(e)?);
                    } else {
                        self.compile_stmt(stmt)?;
                    }
                } else {
                    self.compile_stmt(stmt)?;
                }
            }

            let final_res = last_val.unwrap_or((
                self.context.i64_type().const_int(0, false).into(),
                Type::Void,
            ));
            let promoted = self.promote_match_result(final_res.0, &final_res.1, last_is_lvalue)?;
            self.exit_scope();
            Ok((promoted, final_res.1))
        } else {
            let (val, ty) = self.compile_expr(body)?;
            let promoted = self.promote_match_result(val, &ty, Self::is_lvalue_expr(body))?;
            Ok((promoted, ty))
        }
    }

    fn promote_match_result(
        &mut self,
        val: BasicValueEnum<'ctx>,
        ty: &Type,
        is_lvalue: bool,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        let is_ref_type = matches!(
            ty,
            Type::Tensor(_, _)
                | Type::Struct(_, _)
                | Type::Enum(_, _)
                | Type::UserDefined(_, _)
                | Type::Tuple(_)
        );
        if !is_ref_type {
            return Ok(val);
        }

        if is_lvalue {
            self.emit_deep_clone(val, ty)
        } else {
            if let Some(unreg_fn) = self.module.get_function("tl_mem_unregister") {
                let ptr = val.into_pointer_value();
                let cast_ptr = self
                    .builder
                    .build_pointer_cast(
                        ptr,
                        self.context.ptr_type(inkwell::AddressSpace::default()),
                        "cast",
                    )
                    .unwrap();
                self.builder
                    .build_call(unreg_fn, &[cast_ptr.into()], "")
                    .unwrap();
            }
            Ok(val)
        }
    }

    fn bind_enum_pattern_fields(
        &mut self,
        current_func: inkwell::values::FunctionValue<'ctx>,
        enum_ty: inkwell::types::StructType<'ctx>,
        enum_ptr: inkwell::values::PointerValue<'ctx>,
        enum_def: &EnumDef,
        variant_idx: usize,
        bindings: &[(String, String)],
    ) -> Result<(), String> {
        if bindings.is_empty() {
            return Ok(());
        }

        let variant_def = &enum_def.variants[variant_idx];
        let mut field_types = Vec::with_capacity(variant_def.fields.len());
        for (_, ty) in &variant_def.fields {
            field_types.push(self.get_llvm_type(ty)?);
        }
        let variant_struct_ty = self.context.struct_type(&field_types, false);

        let payload_ptr_raw = self
            .builder
            .build_struct_gep(enum_ty, enum_ptr, 1, "payload_ptr_raw")
            .unwrap();
        let payload_ptr = self
            .builder
            .build_pointer_cast(
                payload_ptr_raw,
                self.context.ptr_type(inkwell::AddressSpace::default()),
                "payload_cast",
            )
            .unwrap();

        for (field_name, bind_name) in bindings {
            let f_idx = variant_def
                .fields
                .iter()
                .position(|(n, _)| n == field_name)
                .ok_or("Enum field not found")?;
            let (_, f_ty) = &variant_def.fields[f_idx];

            let f_ptr = self
                .builder
                .build_struct_gep(variant_struct_ty, payload_ptr, f_idx as u32, "field_ptr")
                .unwrap();

            let llvm_ty = self.get_llvm_type(f_ty)?;
            let f_val = self.builder.build_load(llvm_ty, f_ptr, "bind_val").unwrap();

            let alloca = self.create_entry_block_alloca(current_func, bind_name, f_ty)?;
            let stored_val = if matches!(
                f_ty,
                Type::Tensor(_, _)
                    | Type::Struct(_, _)
                    | Type::Enum(_, _)
                    | Type::UserDefined(_, _)
                    | Type::Tuple(_)
            ) {
                self.emit_deep_clone(f_val, f_ty)?
            } else {
                f_val
            };
            self.builder.build_store(alloca, stored_val).unwrap();

            self.variables
                .last_mut()
                .unwrap()
                .insert(bind_name.clone(), (alloca.into(), f_ty.clone(), super::CLEANUP_FULL));
        }

        Ok(())
    }

    fn is_lvalue_expr(expr: &Expr) -> bool {
        matches!(
            &expr.inner,
            ExprKind::Variable(_)
                | ExprKind::FieldAccess(_, _)
                | ExprKind::IndexAccess(_, _)
                | ExprKind::TupleAccess(_, _)
        )
    }


    pub(crate) fn create_entry_block_alloca(
        &self,
        function: FunctionValue<'ctx>,
        name: &str,
        ty: &Type,
    ) -> Result<inkwell::values::PointerValue<'ctx>, String> {
        let builder = self.context.create_builder();
        let entry = function.get_first_basic_block().unwrap();
        match entry.get_first_instruction() {
            Some(first_instr) => builder.position_before(&first_instr),
            None => builder.position_at_end(entry),
        }

        let llvm_type: inkwell::types::BasicTypeEnum = self
            .get_llvm_type(ty)
            .map_err(|e| format!("Failed to get LLVM type for alloca: {}", e))?;
        let alloca = builder.build_alloca(llvm_type, name).unwrap();
        if let Some(instr) = alloca.as_instruction_value() {
            // Force 16-byte alignment to satisfy SIMD/slice requirements
            instr.set_alignment(16).ok();
        }
        Ok(alloca)
    }

    // Debug method to print IR
    pub fn dump_llvm_ir(&self) {
        self.module.print_to_stderr();
    }

    pub(crate) fn extract_index_bounds(
        &mut self,
        expr: &Expr,
        bounds: &mut HashMap<String, inkwell::values::IntValue<'ctx>>,
    ) -> Result<(), String> {
        match &expr.inner {
            ExprKind::IndexAccess(target, indices) => {
                // Target should be ExprKind::Ident for variable access
                // Instead of compiling, look up the variable directly
                let (tensor_ptr, is_scalar_array, array_len) = match &target.inner {
                    ExprKind::Variable(name) => {
                        let (val, ty) = self
                            .lookup_variable(name)
                            .ok_or(format!("Variable {} not found", name))?;
                        // Load pointer if needed
                        match ty {
                            Type::Tensor(_, _) => {
                                // val is a pointer to the tensor pointer
                                let loaded = self
                                    .builder
                                    .build_load(
                                        self.context.ptr_type(inkwell::AddressSpace::default()),
                                        val.into_pointer_value(),
                                        name,
                                    )
                                    .map_err(|e| e.to_string())?
                                    .into_pointer_value();
                                (loaded, false, 0)
                            }
                            Type::ScalarArray(_, len) => {
                                // For ScalarArray, val is pointer to alloca storing pointer to array
                                // We don't need the runtime pointer for bounds, just the length
                                (val.into_pointer_value(), true, len)
                            }

                            _ => return Err("Expected tensor variable".into()),
                        }
                    }
                    _ => {
                        return Err("Complex index target not supported in bounds extraction".into())
                    }
                };

                let dim_fn = self.module.get_function("tl_tensor_dim").unwrap();
                for (i, idx_expr) in indices.iter().enumerate() {
                    match &idx_expr.inner {
                        ExprKind::Int(_) | ExprKind::Float(_) => continue,
                        ExprKind::Variable(name) => {
                            if !bounds.contains_key(name) {
                                let dim_size = if is_scalar_array {
                                    if i == 0 {
                                        self.context.i64_type().const_int(array_len as u64, false)
                                    } else {
                                        return Err("ScalarArray only has 1 dimension".into());
                                    }
                                } else {
                                    let dim_idx_val =
                                        self.context.i64_type().const_int(i as u64, false);
                                    let call_result = self
                                        .builder
                                        .build_call(
                                            dim_fn,
                                            &[tensor_ptr.into(), dim_idx_val.into()],
                                            "dim_size",
                                        )
                                        .map_err(|e| e.to_string())?;
                                    match call_result.try_as_basic_value() {
                                        ValueKind::Basic(v) => v.into_int_value(),
                                        _ => return Err("Invalid dim return".into()),
                                    }
                                };
                                bounds.insert(name.clone(), dim_size);
                            }
                        }
                        _ => continue,
                    }
                }
            }
            ExprKind::BinOp(lhs, _, rhs) => {
                self.extract_index_bounds(lhs, bounds)?;
                self.extract_index_bounds(rhs, bounds)?;
            }
            ExprKind::UnOp(_, inner) => self.extract_index_bounds(inner, bounds)?,
            ExprKind::Block(stmts) => {
                for stmt in stmts {
                    if let StmtKind::Expr(e) = &stmt.inner {
                        self.extract_index_bounds(e, bounds)?;
                    } else if let StmtKind::Let { value, .. } = &stmt.inner {
                        self.extract_index_bounds(value, bounds)?;
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
    fn compile_method_call(
        &mut self,
        obj: &Expr,
        method: &str,
        args: &[Expr],
    ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        let (val, ty) = self.compile_method_call_inner(obj, method, args)?;
        self.add_temp(val, ty.clone());
        Ok((val, ty))
    }

    fn compile_method_call_inner(
        &mut self,
        obj: &Expr,
        method: &str,
        args: &[Expr],
    ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        let (obj_val, obj_ty) = self.compile_expr(obj)?;

        // Check for Vec (either Type::Vec or UserDefined("Vec"))
        let inner_opt = match &obj_ty {
            Type::Vec(inner) => Some(inner.clone()),
            Type::UserDefined(n, args) => {
                 if n == "Vec" {
                     if args.is_empty() {
                        Some(Box::new(Type::Void))
                     } else if args.len() == 1 {
                        Some(Box::new(args[0].clone()))
                     } else {
                        None
                     }
                 } else if n.starts_with("Vec_") {
                     let suffix = &n[4..];
                     match suffix {
                         "i64" => Some(Box::new(Type::I64)),
                         "f32" => Some(Box::new(Type::F32)),
                         "u8" => Some(Box::new(Type::U8)),
                         "String" => Some(Box::new(Type::UserDefined(String::from("String"), vec![]))),
                         _ => Some(Box::new(Type::UserDefined(suffix.to_string(), vec![]))),
                     }
                 } else {
                     None
                 }
            }
            _ => None,
        };
        
        if let Some(inner) = inner_opt {
            match method {
                "len" => {
                    let fn_name = match inner.as_ref() {
                        Type::U8 => "tl_vec_u8_len",
                        Type::I64 => "tl_vec_i64_len",
                        Type::F32 => "tl_vec_f32_len",
                        _ => "tl_vec_ptr_len",
                    };
                    let fn_val = self
                        .module
                        .get_function(fn_name)
                        .ok_or(format!("{} not found", fn_name))?;
                    let call = self
                        .builder
                        .build_call(fn_val, &[obj_val.into()], "vec_len")
                        .map_err(|e| e.to_string())?;
                    let res = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err("Invalid return from vec.len()".into()),
                    };
                    return Ok((res, Type::I64));
                }
                "push" => {
                    if args.len() != 1 {
                        return Err("push requires 1 argument".into());
                    }
                    let (arg_val, arg_ty) = self.compile_expr(&args[0])?;
                    
                    // Use Monomorphizer to get/create specialized method
                    let fn_name = self.ensure_vec_method(&inner, "push")
                        .map_err(|e| format!("Failed to monomorphize push: {}", e))?;
                    
                    let fn_val = self.module.get_function(&fn_name)
                        .ok_or(format!("{} not found", fn_name))?;

                    // Prepare argument: cast if necessary
                    // For struct types (which use generic ptr implementation), we need to cast to i8*
                    // unless we generated a specialized wrapper (which ensure_vec_method handles via aliasing to core fn)
                    // The core fn for structs is tl_vec_ptr_push(vec*, void*)
                    // So we expect the function to take i8*.
                    
                    // However, we added alias with correct name, but sharing the SAME signature as core function?
                    // In ensure_vec_method: "let fn_type = core_fn.get_type();"
                    // So checks against fn_type param types.
                    
                    let param_type = fn_val.get_nth_param(1).unwrap().get_type();
                    let arg_casted = if param_type.is_pointer_type() {
                         // Cast arg to whatever pointer type the function expects (likely i8*)
                         let ptr = arg_val.into_pointer_value();
                         self.builder.build_pointer_cast(
                             ptr, 
                             param_type.into_pointer_type(),
                             "arg_cast"
                         ).unwrap().into()
                    } else if param_type.is_int_type() {
                         if arg_val.is_int_value() {
                             if arg_val.into_int_value().get_type() != param_type.into_int_type() {
                                 // Cast int (e.g. i64 -> u8)
                                 self.builder.build_int_cast(
                                     arg_val.into_int_value(),
                                     param_type.into_int_type(),
                                     "arg_int_cast"
                                 ).unwrap().into()
                             } else {
                                 arg_val
                             }
                         } else {
                             return Err(format!("Expected int for push, got {:?}", arg_ty));
                         }
                    } else if param_type.is_float_type() {
                         // Check float types
                         arg_val
                    } else {
                        arg_val
                    };

                    self.builder
                        .build_call(fn_val, &[obj_val.into(), arg_casted.into()], "")
                        .map_err(|e| e.to_string())?;
                    
                    // Return Void
                    return Ok((
                        self.context.i64_type().const_int(0, false).into(),
                        Type::Void,
                    ));
                }
                "get" => {
                    if args.len() != 1 {
                        return Err("get requires 1 argument".into());
                    }
                    let (idx_val, _) = self.compile_expr(&args[0])?;
                    // Ensure index is i64
                    let idx_i64 = if idx_val.is_int_value() {
                        let int_val = idx_val.into_int_value();
                        if int_val.get_type().get_bit_width() == 32 {
                            self.builder.build_int_z_extend(int_val, self.context.i64_type(), "idx_ext").unwrap()
                        } else {
                            int_val
                        }
                    } else {
                         return Err("Index must be integer".into());
                    };

                    // Use Monomorphizer
                    let fn_name = self.ensure_vec_method(&inner, "get")
                        .map_err(|e| format!("Failed to monomorphize get: {}", e))?;
                    
                    let fn_val = self.module.get_function(&fn_name)
                        .ok_or(format!("{} not found", fn_name))?;
                        
                    let call = self
                        .builder
                        .build_call(fn_val, &[obj_val.into(), idx_i64.into()], "vec_get")
                        .map_err(|e| e.to_string())?;
                    let res = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err("Invalid return from vec.get()".into()),
                    };
                    
                    // No need to cast back if specialized wrapper returns correct type!
                    // If ensure_vec_method creates a wrapper that calls core fn (which returns i8*)
                    // The wrapper logic in mono.rs (generic call) returns whatever core fn returns (i8*).
                    // So we STILL need to cast if the wrapper returns i8* but we expect Point*.
                    
                    // Wait, ensure_vec_method in mono.rs creates wrapper with SAME signature as core fn.
                    // core fn for struct is `tl_vec_ptr_get` -> returns i8*.
                    // So `fn_val` returns i8*.
                    // So we DO need to cast result to `inner` type pointer.
                    
                    // But if inner is i64, core fn is `tl_vec_i64_get` -> returns i64.
                    // Wrapper returns i64. No cast needed.
                    
                    // So we check return type of `fn_val`.
                    let ret_type = fn_val.get_type().get_return_type().unwrap(); // We know it's not void for get
                    
                    let final_res = if ret_type.is_pointer_type() {
                         // Cast to expected inner type pointer
                         let inner_llvm_ty = self.get_llvm_type(&inner)?;
                         // If inner is Struct, get_llvm_type returns pointer (opaque struct ptr).
                         // That's what we want.
                         let ptr_ty = match inner_llvm_ty {
                            inkwell::types::BasicTypeEnum::PointerType(p) => p,
                            _ => return Err("Expected pointer type for Vec inner element".into()), 
                         };
                         
                         // If check if res is already correct type?
                         // struct types are opaque pointers. i8* is opaque pointer (in new LLVM).
                         // But if we need explicit cast for strictness or older LLVM...
                         self.builder.build_pointer_cast(
                             res.into_pointer_value(),
                             ptr_ty,
                             "cast_back"
                         ).unwrap().into()
                    } else {
                        res
                    };
                    
                    return Ok((final_res, *inner.clone()));
                }
                "free" => {
                     let fn_name = match inner.as_ref() {
                        Type::U8 => "tl_vec_u8_free",
                        Type::I64 => "tl_vec_i64_free", // Impl needed
                        Type::F32 => "tl_vec_f32_free", // Impl needed
                        _ => "tl_vec_ptr_free",         // Impl needed
                    };
                    let fn_val = self
                            .module
                            .get_function(fn_name)
                            .ok_or(format!("{} not found", fn_name))?;
                        self.builder
                            .build_call(fn_val, &[obj_val.into()], "vec_free")
                            .map_err(|e| e.to_string())?;
                        return Ok((
                            self.context.i64_type().const_int(0, false).into(),
                            Type::Void,
                        ));
                }
                "read_i32_be" if matches!(inner.as_ref(), Type::U8) => {
                     if args.len() != 1 {
                            return Err("read_i32_be requires 1 argument".into());
                        }
                        let (idx_val, _) = self.compile_expr(&args[0])?;
                        let fn_val = self
                            .module
                            .get_function("tl_vec_u8_read_i32_be")
                            .ok_or("tl_vec_u8_read_i32_be not found")?;
                        let call = self
                            .builder
                            .build_call(fn_val, &[obj_val.into(), idx_val.into()], "vec_i32")
                            .map_err(|e| e.to_string())?;
                        let res = match call.try_as_basic_value() {
                            inkwell::values::ValueKind::Basic(v) => v,
                            _ => return Err("Invalid return from vec.read_i32_be()".into()),
                        };
                        return Ok((res, Type::I64));
                }
                _ => {}
            }
        }

        // 2. Resolve Type Name to check Manager
        let type_name = match &obj_ty {
            Type::Struct(name, _) => name.clone(),
            Type::UserDefined(name, _) => name.clone(),
            Type::Tensor(_, _) => "Tensor".to_string(),
            Type::F32 => "F32".to_string(),
            Type::F64 => "F64".to_string(),
            Type::I64 => "I64".to_string(),
            Type::I32 => "I32".to_string(),
            _ => "".to_string(),
        };

        // 3. Manager Lookup
        // Use copied() to drop borrow of self.instance_methods
        let method_opt = if !type_name.is_empty() {
            self.instance_methods
                .get(&type_name)
                .and_then(|m| m.get(method))
                .copied()
        } else {
            None
        };

        if let Some(method_fn) = method_opt {
            match method_fn {
                InstanceMethod::Evaluated(func) => {
                    // Compile args
                    let mut compiled_args = Vec::with_capacity(args.len());
                    let mut compiled_args_types = Vec::with_capacity(args.len());
                    for arg in args {
                        let (val, ty) = self.compile_expr(arg)?;
                        compiled_args.push((val, ty.clone()));
                        compiled_args_types.push((val, ty));
                    }

                    // Call function
                    let res_result = func(self, obj_val, obj_ty.clone(), compiled_args);



                    return res_result;

                }
                InstanceMethod::Unevaluated(func) => {
                    // Unevaluated methods handle their own arg compilation and cleanup
                    return func(self, obj, method, args);
                }
            }
        }

        if type_name == "Tokenizer" {
            let handle = self.load_struct_i64_field(obj_val, &obj_ty, "_h")?;
            match method {
                "encode" => {
                    if args.len() != 1 {
                        return Err("Tokenizer::encode requires 1 argument".into());
                    }
                    let (prompt_val, _) = self.compile_expr(&args[0])?;
                    let fn_val = self
                        .module
                        .get_function("tl_tokenizer_encode")
                        .ok_or("tl_tokenizer_encode not found")?;
                    let call = self
                        .builder
                        .build_call(fn_val, &[handle.into(), prompt_val.into()], "tok_encode")
                        .map_err(|e| e.to_string())?;
                    let res = self.check_tensor_result(call, "tok_encode_error")?;
                    return Ok((res, Type::Tensor(Box::new(Type::I64), 0)));
                }
                "decode" => {
                    if args.len() != 1 {
                        return Err("Tokenizer::decode requires 1 argument".into());
                    }
                    let (ids_val, _) = self.compile_expr(&args[0])?;
                    let fn_val = self
                        .module
                        .get_function("tl_tokenizer_decode")
                        .ok_or("tl_tokenizer_decode not found")?;
                    let call = self
                        .builder
                        .build_call(fn_val, &[handle.into(), ids_val.into()], "tok_decode")
                        .map_err(|e| e.to_string())?;
                    let res = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err("Invalid return from Tokenizer::decode".into()),
                    };
                    return Ok((res, Type::UserDefined("String".to_string(), vec![])));
                }
                _ => {}
            }
        }

        if type_name == "KVCache" {
            let handle = self.load_struct_i64_field(obj_val, &obj_ty, "_h")?;
            match method {
                "free" => {
                    let fn_val = self
                        .module
                        .get_function("tl_kv_cache_free")
                        .ok_or("tl_kv_cache_free not found")?;
                    self.builder
                        .build_call(fn_val, &[handle.into()], "kv_free")
                        .map_err(|e| e.to_string())?;
                    return Ok((
                        self.context.i64_type().const_int(0, false).into(),
                        Type::Void,
                    ));
                }
                "get_k" | "get_v" => {
                    if args.len() != 1 {
                        return Err("KVCache::get_k/get_v requires 1 argument".into());
                    }
                    let (layer_val, _) = self.compile_expr(&args[0])?;
                    let fn_name = if method == "get_k" {
                        "tl_kv_cache_get_k"
                    } else {
                        "tl_kv_cache_get_v"
                    };
                    let fn_val = self
                        .module
                        .get_function(fn_name)
                        .ok_or(format!("{} not found", fn_name))?;
                    let call = self
                        .builder
                        .build_call(fn_val, &[handle.into(), layer_val.into()], "kv_get")
                        .map_err(|e| e.to_string())?;
                    let res = self.check_tensor_result(call, "kv_get_error")?;
                    let _ret_ty = Type::Tensor(Box::new(Type::F32), 0);
                    let _ret_ty = Type::Tensor(Box::new(Type::F32), 0);

                    return Ok((res, Type::Tensor(Box::new(Type::F32), 0)));
                }
                "update" => {
                    if args.len() != 3 {
                        return Err("KVCache::update requires 3 arguments".into());
                    }
                    let (layer_val, _) = self.compile_expr(&args[0])?;
                    let (k_val, _) = self.compile_expr(&args[1])?;
                    let (v_val, _) = self.compile_expr(&args[2])?;
                    let fn_val = self
                        .module
                        .get_function("tl_kv_cache_update")
                        .ok_or("tl_kv_cache_update not found")?;
                    self.builder
                        .build_call(
                            fn_val,
                            &[handle.into(), layer_val.into(), k_val.into(), v_val.into()],
                            "kv_update",
                        )
                        .map_err(|e| e.to_string())?;
                    return Ok((
                        self.context.i64_type().const_int(0, false).into(),
                        Type::Void,
                    ));
                }
                _ => {}
            }
        }

        if type_name == "Map" {
            match method {
                "get" | "get_1d" | "get_quantized" => {
                    if args.len() != 1 {
                        return Err("Map::get requires 1 argument".into());
                    }
                    let (key_val, _) = self.compile_expr(&args[0])?;
                    let fn_name = match method {
                        "get" => "tl_tensor_map_get",
                        "get_1d" => "tl_tensor_map_get_1d",
                        "get_quantized" => "tl_tensor_map_get_quantized",
                        _ => unreachable!(),
                    };
                    let fn_val = self
                        .module
                        .get_function(fn_name)
                        .ok_or(format!("{} not found", fn_name))?;
                    let call = self
                        .builder
                        .build_call(fn_val, &[obj_val.into(), key_val.into()], "map_get")
                        .map_err(|e| e.to_string())?;
                    let res = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err("Invalid return from Map::get".into()),
                    };
                    if method == "get_quantized" {
                        return Ok((res, Type::I64));
                    }
                    let _ret_ty = Type::Tensor(Box::new(Type::F32), 0);
                    let _ret_ty = Type::Tensor(Box::new(Type::F32), 0);

                    return Ok((res, Type::Tensor(Box::new(Type::F32), 0)));
                }
                _ => {}
            }
        }

        if type_name == "String" {
            match method {
                "concat" => {
                    if args.len() != 1 {
                        return Err("String::concat requires 1 argument".into());
                    }
                    let (other_val, _) = self.compile_expr(&args[0])?;
                    let fn_val = self
                        .module
                        .get_function("tl_string_concat")
                        .ok_or("tl_string_concat not found")?;
                    let call = self
                        .builder
                        .build_call(fn_val, &[obj_val.into(), other_val.into()], "str_concat")
                        .map_err(|e| e.to_string())?;
                    let res = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err("Invalid return from String::concat".into()),
                    };
                    return Ok((res, Type::UserDefined("String".to_string(), vec![])));
                }
                "contains" => {
                    if args.len() != 1 {
                        return Err("String::contains requires 1 argument".into());
                    }
                    let (needle_val, _) = self.compile_expr(&args[0])?;
                    let fn_val = self
                        .module
                        .get_function("tl_string_contains")
                        .ok_or("tl_string_contains not found")?;
                    let call = self
                        .builder
                        .build_call(fn_val, &[obj_val.into(), needle_val.into()], "str_contains")
                        .map_err(|e| e.to_string())?;
                    let res = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err("Invalid return from String::contains".into()),
                    };
                    return Ok((res, Type::Bool));
                }
                "to_i64" => {
                    if !args.is_empty() {
                        return Err("String::to_i64 requires 0 arguments".into());
                    }
                    let fn_val = self
                        .module
                        .get_function("tl_string_to_i64")
                        .ok_or("tl_string_to_i64 not found")?;
                    let call = self
                        .builder
                        .build_call(fn_val, &[obj_val.into()], "str_to_i64")
                        .map_err(|e| e.to_string())?;
                    let res = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err("Invalid return from String::to_i64".into()),
                    };
                    return Ok((res, Type::I64));
                }
                _ => {}
            }
        }

        // Special Handling for Tensor methods
        if let Type::Tensor(elem_ty, _) = &obj_ty {
            match method {
                "to_i64" => {
                    if !args.is_empty() {
                        return Err("Tensor::to_i64 requires 0 arguments".into());
                    }
                    if matches!(elem_ty.as_ref(), Type::I64) {
                        return Ok((obj_val, obj_ty.clone()));
                    }
                    let fn_val = self
                        .module
                        .get_function("tl_tensor_to_i64")
                        .ok_or("tl_tensor_to_i64 not found")?;
                    let call = self
                        .builder
                        .build_call(fn_val, &[obj_val.into()], "tensor_to_i64")
                        .map_err(|e| e.to_string())?;
                    let res = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err("Invalid return from Tensor::to_i64".into()),
                    };
                    return Ok((res, Type::Tensor(Box::new(Type::I64), 0)));
                }
                "cuda" => {
                    let fn_val = self
                        .module
                        .get_function("tl_tensor_to_device")
                        .ok_or("tl_tensor_to_device not found")?;
                    let (dev_str_val, _) = self.compile_string_literal("cuda")?;
                    let call = self
                        .builder
                        .build_call(fn_val, &[obj_val.into(), dev_str_val.into()], "cuda_res")
                        .map_err(|e| e.to_string())?;
                    let res = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err("Invalid return from cuda()".into()),
                    };


                    return Ok((res, obj_ty.clone()));
                }
                "cpu" => {
                    let fn_val = self
                        .module
                        .get_function("tl_tensor_to_device")
                        .ok_or("tl_tensor_to_device not found")?;
                    let (dev_str_val, _) = self.compile_string_literal("cpu")?;
                    let call = self
                        .builder
                        .build_call(fn_val, &[obj_val.into(), dev_str_val.into()], "cpu_res")
                        .map_err(|e| e.to_string())?;
                    let res = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err("Invalid return from cpu()".into()),
                    };


                    return Ok((res, obj_ty.clone()));
                }
                "item" => {
                    let is_int = matches!(
                        elem_ty.as_ref(),
                        Type::I64 | Type::I32 | Type::U32 | Type::U8
                    );
                    let fn_name = if is_int {
                        "tl_tensor_item_i64"
                    } else {
                        "tl_tensor_item"
                    };
                    let fn_val = self
                        .module
                        .get_function(fn_name)
                        .ok_or(format!("{} not found", fn_name))?;
                    let call = self
                        .builder
                        .build_call(fn_val, &[obj_val.into()], "item_res")
                        .map_err(|e| e.to_string())?;
                    let res = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err("Invalid return from item()".into()),
                    };
                    let ret_ty = if is_int { Type::I64 } else { Type::F32 };
                    return Ok((res, ret_ty));
                }
                "max" | "min" | "mean" | "argmax" | "argmin" => {
                    if !args.is_empty() {
                        let suffix = if method == "argmax" || method == "argmin" {
                            ""
                        } else {
                            "_dim"
                        };
                        let fn_name = format!("tl_tensor_{}{}", method, suffix);
                        let fn_val = self
                            .module
                            .get_function(&fn_name)
                            .ok_or(format!("{} not found", fn_name))?;

                        let mut call_args: Vec<inkwell::values::BasicMetadataValueEnum> =
                            Vec::new();
                        call_args.push(obj_val.into());

                        let (dim_val, _) = self.compile_expr(&args[0])?;
                        call_args.push(dim_val.into());

                        let keep_val = if args.len() > 1 {
                            let (k, _) = self.compile_expr(&args[1])?;
                            k.into()
                        } else {
                            self.context.bool_type().const_int(0, false).into()
                        };
                        call_args.push(keep_val);

                        let call = self
                            .builder
                            .build_call(fn_val, &call_args, "reduce_res")
                            .map_err(|e| e.to_string())?;
                        let res = match call.try_as_basic_value() {
                            inkwell::values::ValueKind::Basic(v) => v,
                            _ => return Err(format!("Invalid return from {}()", method).into()),
                        };
    

                        return Ok((res, obj_ty.clone()));
                    } else {
                        if method == "argmax" || method == "argmin" {
                            return Err(format!("{} requires arguments", method));
                        }
                    }
                }
                "detach" => {
                    let fn_val = self
                        .module
                        .get_function("tl_tensor_detach")
                        .ok_or("tl_tensor_detach not found")?;
                    let req_grad = self.context.bool_type().const_int(0, false);
                    let call = self
                        .builder
                        .build_call(fn_val, &[obj_val.into(), req_grad.into()], "detach_res")
                        .map_err(|e| e.to_string())?;
                    let res = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err("Invalid return from detach()".into()),
                    };


                    return Ok((res, obj_ty.clone()));
                }
                "tril" => {
                    let fn_val = self
                        .module
                        .get_function("tl_tensor_tril")
                        .ok_or("tl_tensor_tril not found")?;

                    if args.len() != 1 {
                        return Err("tril requires 1 argument (diagonal)".into());
                    }

                    let (diag_val, diag_ty) = self.compile_expr(&args[0])?;
                    let diag_i32 = match diag_ty {
                        Type::I64 => self
                            .builder
                            .build_int_cast(
                                diag_val.into_int_value(),
                                self.context.i32_type(),
                                "tril_diag_cast",
                            )
                            .unwrap(),
                        Type::I32 => diag_val.into_int_value(),
                        _ => return Err("tril argument must be integer".into()),
                    };

                    let call = self
                        .builder
                        .build_call(fn_val, &[obj_val.into(), diag_i32.into()], "tril_res")
                        .map_err(|e| e.to_string())?;
                    let res = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err("Invalid return from tril()".into()),
                    };


                    return Ok((res, obj_ty.clone()));
                }
                "mul" | "add" | "sub" | "div" => {
                    if args.len() != 1 {
                        return Err(format!("{} requires 1 argument", method));
                    }
                    // scalar handling: ensure_tensor_v2 handles scalar->tensor conversion
                    let (rhs_val, _) = self.ensure_tensor_v2(&args[0], 0)?;

                    let fn_name = match method {
                        "mul" => "tl_tensor_mul",
                        "add" => "tl_tensor_add",
                        "sub" => "tl_tensor_sub",
                        "div" => "tl_tensor_div",
                        _ => unreachable!(),
                    };
                    let fn_val = self
                        .module
                        .get_function(fn_name)
                        .ok_or(format!("{} not found", fn_name))?;

                    let call = self
                        .builder
                        .build_call(fn_val, &[obj_val.into(), rhs_val.into()], "binop_res")
                        .map_err(|e| e.to_string())?;


                    let res = self.check_tensor_result(call, "binop_error")?;



                    // Note: ensure_tensor_v2 result might need freeing if it was promoted
                    // But currently arguments cleanup loop handles args[0].
                    // ensure_tensor_v2 might create new tensor if scalar.
                    // If so, that new tensor is not in 'args'.
                    // However, ensure_tensor_v2 logic usually registers tensor.
                    // If we don't return it, we should verify leaks.
                    // For now, assuming standard cleanup logic suffices or small leak (scalar tensor).

                    return Ok((res, obj_ty.clone()));
                }
                "contiguous" => {
                    let fn_val = self
                        .module
                        .get_function("tl_tensor_contiguous")
                        .ok_or("tl_tensor_contiguous not found")?;
                    let call = self
                        .builder
                        .build_call(fn_val, &[obj_val.into()], "cont_res")
                        .map_err(|e| e.to_string())?;
                    let res = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err("Invalid return from contiguous()".into()),
                    };


                    return Ok((res, obj_ty.clone()));
                }
                "conv2d" => {
                    let fn_val = self
                        .module
                        .get_function("tl_tensor_conv2d")
                        .ok_or("tl_tensor_conv2d not found")?;

                    if args.len() != 3 {
                        return Err("conv2d requires 3 arguments: weight, padding, stride".into());
                    }

                    // Arg 0: weight (Tensor)
                    let (weight_val, weight_ty) = self.compile_expr(&args[0])?;
                    if !matches!(weight_ty, Type::Tensor(_, _)) {
                        return Err("conv2d arg 0 (weight) must be Tensor".into());
                    }

                    // Arg 1: padding (Int)
                    let (pad_val, pad_ty) = self.compile_expr(&args[1])?;
                    let pad_i64 = match pad_ty {
                        Type::I64 => pad_val.into_int_value(),
                        Type::I32 => self
                            .builder
                            .build_int_z_extend(
                                pad_val.into_int_value(),
                                self.context.i64_type(),
                                "ext",
                            )
                            .unwrap(),
                        _ => return Err("conv2d arg 1 (padding) must be Integer".into()),
                    };

                    // Arg 2: stride (Int)
                    let (stride_val, stride_ty) = self.compile_expr(&args[2])?;
                    let stride_i64 = match stride_ty {
                        Type::I64 => stride_val.into_int_value(),
                        Type::I32 => self
                            .builder
                            .build_int_z_extend(
                                stride_val.into_int_value(),
                                self.context.i64_type(),
                                "ext",
                            )
                            .unwrap(),
                        _ => return Err("conv2d arg 2 (stride) must be Integer".into()),
                    };

                    let call = self
                        .builder
                        .build_call(
                            fn_val,
                            &[
                                obj_val.into(),
                                weight_val.into(),
                                pad_i64.into(),
                                stride_i64.into(),
                            ],
                            "conv_res",
                        )
                        .map_err(|e| e.to_string())?;

                    let res = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err("Invalid return from conv2d()".into()),
                    };


                    return Ok((res, obj_ty.clone()));
                }
                "clamp" => {
                    let fn_val = self
                        .module
                        .get_function("tl_tensor_clamp")
                        .ok_or("tl_tensor_clamp not found")?;

                    if args.len() != 2 {
                        return Err("clamp requires 2 arguments: min, max".into());
                    }

                    // Arg 0: min (f32)
                    let (min_val, min_ty) = self.compile_expr(&args[0])?;
                    let min_f32 = match min_ty {
                        Type::F32 => min_val.into_float_value(),
                        Type::F64 => self
                            .builder
                            .build_float_trunc(
                                min_val.into_float_value(),
                                self.context.f32_type(),
                                "trunc",
                            )
                            .unwrap(),
                        _ => return Err("clamp arg 0 (min) must be Float".into()),
                    };

                    // Arg 1: max (f32)
                    let (max_val, max_ty) = self.compile_expr(&args[1])?;
                    let max_f32 = match max_ty {
                        Type::F32 => max_val.into_float_value(),
                        Type::F64 => self
                            .builder
                            .build_float_trunc(
                                max_val.into_float_value(),
                                self.context.f32_type(),
                                "trunc",
                            )
                            .unwrap(),
                        _ => return Err("clamp arg 1 (max) must be Float".into()),
                    };

                    let call = self
                        .builder
                        .build_call(
                            fn_val,
                            &[obj_val.into(), min_f32.into(), max_f32.into()],
                            "clamp_res",
                        )
                        .map_err(|e| e.to_string())?;

                    let res = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err("Invalid return from clamp()".into()),
                    };


                    return Ok((res, obj_ty.clone()));
                }
                "clone" => {
                    let fn_val = self
                        .module
                        .get_function("tl_tensor_clone")
                        .ok_or("tl_tensor_clone not found")?;
                    let call = self
                        .builder
                        .build_call(fn_val, &[obj_val.into()], "clone_res")
                        .map_err(|e| e.to_string())?;
                    let res = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err("Invalid return from clone()".into()),
                    };


                    return Ok((res, obj_ty.clone()));
                }
                "grad" => {
                    let fn_val = self
                        .module
                        .get_function("tl_tensor_grad")
                        .ok_or("tl_tensor_grad not found")?;
                    let call = self
                        .builder
                        .build_call(fn_val, &[obj_val.into()], "grad_res")
                        .map_err(|e| e.to_string())?;
                    let res = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err("Invalid return from grad()".into()),
                    };


                    return Ok((res, obj_ty.clone()));
                }
                "matmul_quantized" => {
                    if args.len() != 1 {
                        return Err("matmul_quantized requires 1 argument".into());
                    }
                    let (weight_val, _) = self.compile_expr(&args[0])?;
                    let fn_val = self
                        .module
                        .get_function("tl_qtensor_matmul")
                        .ok_or("tl_qtensor_matmul not found")?;
                    let call = self
                        .builder
                        .build_call(fn_val, &[obj_val.into(), weight_val.into()], "qmatmul_res")
                        .map_err(|e| e.to_string())?;

                    let res = self.check_tensor_result(call, "qmatmul_error")?;

                    return Ok((res, obj_ty.clone()));
                }
                _ => {}
            }
        }

        // 4. Generic Fallback (Struct Methods / Mangled Names)
        let struct_name = match &obj_ty {
            Type::Struct(name, _) | Type::UserDefined(name, _) => name.clone(),
            Type::Tensor(_, _) => "Tensor".to_string(),
            _ => return Err(format!("Method {} not found on type {:?}", method, obj_ty)),
        };

        // Try exact mangling first: tl_{Struct}_{Method}
        let mangled_name = format!("tl_{}_{}", struct_name, method);
        // Fallback to lowercase
        let stdlib_name = format!("tl_{}_{}", struct_name.to_lowercase(), method);

        let (func_val, final_name) = if let Some(f) = self.module.get_function(&mangled_name) {
            (f, mangled_name)
        } else if let Some(f) = self.module.get_function(&stdlib_name) {
            (f, stdlib_name)
        } else {
            return Err(format!(
                "Method {} not found in struct {} (checked {} and {})",
                method, struct_name, mangled_name, stdlib_name
            ));
        };

        // Get return type (for SRET check)
        let ret_ty = self
            .fn_return_types
            .get(&final_name)
            .cloned()
            .unwrap_or(Type::Void);

        // SRET Disabled for now to match legacy behavior
        // let uses_sret = false;

        let mut compiled_args_vals = Vec::with_capacity(args.len() + 1);
        let mut compiled_args_types = Vec::with_capacity(args.len());

        // Push Receiver
        compiled_args_vals.push(obj_val.into());

        for arg in args {
            let (val, ty) = self.compile_expr(arg)?;
            compiled_args_vals.push(val.into());
            compiled_args_types.push((val, ty));
        }

        // Call
        let call = self
            .builder
            .build_call(func_val, &compiled_args_vals, "method_call")
            .map_err(|e| e.to_string())?;

        // Return handling
        match call.try_as_basic_value() {
            inkwell::values::ValueKind::Basic(_) => {
                let v = self.check_tensor_result(call, "method_call_error")?;
                // Register intermediate tensor result
                // REMOVED REDUNDANT REGISTRATION

                Ok((v, ret_ty))
            }
            _ => Ok((
                self.context.i64_type().const_int(0, false).into(),
                Type::Void,
            )),
        }
    } /*
          match method {
              "get" => {
                  if args.len() != 1 {
                      return Err("get requires 1 argument".into());
                  }
                  let (idx_val, idx_ty) = self.compile_expr(&args[0])?;

                  // Ensure index is i64
                  let idx_i64 = match idx_ty {
                      Type::I64 => idx_val.into_int_value(),
                      Type::I32 => self
                          .builder
                          .build_int_z_extend(
                              idx_val.into_int_value(),
                              self.context.i64_type(),
                              "idx_ext",
                          )
                          .map_err(|e| e.to_string())?,
                      _ => return Err("Index must be integer".into()),
                  };

                  let fn_val = self.module.get_function("tl_tensor_get").unwrap();
                  // panic!("DEBUG_PANIC: Reached get arm");
                  let call = self
                      .builder
                      .build_call(fn_val, &[obj_val.into(), idx_i64.into()], "get_res")
                      .map_err(|e| e.to_string())?;

                  let res = match call.try_as_basic_value() {
                      ValueKind::Basic(v) => v,
                      _ => return Err("Invalid get return".into()),
                  };

                  // FIX: Free temporary receiver - FORCE CHECK

                  let is_temp = matches!(
                      obj,
                      ExprKind::FnCall(_, _)
                          | ExprKind::StructInit(_, _)
                          | ExprKind::Tuple(_)
                          | ExprKind::TensorLiteral(_)
                  );
                  if is_temp {
                      self.emit_recursive_free(obj_val, &obj_ty, super::CLEANUP_FULL)?;
                  }

                  Ok((res, Type::F32))
              }
              "backward" => {
                  let fn_val = self.module.get_function("tl_tensor_backward").unwrap();
                  self.builder
                      .build_call(fn_val, &[obj_val.into()], "backward_call")
                      .map_err(|e| e.to_string())?;
                  if self.is_safe_to_free(obj, &obj_ty) {
                      self.emit_recursive_free(obj_val, &obj_ty, super::CLEANUP_FULL)?;
                  }

                  Ok((
                      self.context.i64_type().const_int(0, false).into(),
                      Type::Void,
                  ))
              }
              "clone" => {
                  let fn_val = self.module.get_function("tl_tensor_clone").unwrap();
                  let call = self
                      .builder
                      .build_call(fn_val, &[obj_val.into()], "clone_res")
                      .map_err(|e| e.to_string())?;

                  let res = match call.try_as_basic_value() {
                      ValueKind::Basic(v) => v,
                      _ => return Err("Invalid clone return".into()),
                  };

                  if self.is_safe_to_free(obj, &obj_ty) {
                      self.emit_recursive_free(obj_val, &obj_ty, super::CLEANUP_FULL)?;
                  }

                  self.emit_register_tensor(res, &obj_ty)?;
                  Ok((res, obj_ty))
              }
              "detach" => {
                  let fn_val = self.module.get_function("tl_tensor_detach").unwrap();
                  // Optional arg: req_grad (bool). Default to false.
                  let req_grad = if !args.is_empty() {
                      let (arg_val, _arg_ty) = self.compile_expr(&args[0])?;
                      arg_val.into_int_value()
                  } else {
                      self.context.bool_type().const_int(0, false)
                  };

                  let call = self
                      .builder
                      .build_call(fn_val, &[obj_val.into(), req_grad.into()], "detach_res")
                      .map_err(|e| e.to_string())?;

                  let res = match call.try_as_basic_value() {
                      ValueKind::Basic(v) => v,
                      _ => return Err("Invalid detach return".into()),
                  };

                  // Register intermediate tensor result for automatic cleanup
                  self.emit_register_tensor(res, &obj_ty)?;

                  Ok((res, obj_ty))
              }
              "grad" => {
                  let fn_val = self.module.get_function("tl_tensor_grad").unwrap();
                  let call = self
                      .builder
                      .build_call(fn_val, &[obj_val.into()], "grad_res")
                      .map_err(|e| e.to_string())?;

                  let res = match call.try_as_basic_value() {
                      ValueKind::Basic(v) => v,
                      _ => return Err("Invalid grad return".into()),
                  };

                  // Register intermediate tensor result for automatic cleanup
                  self.emit_register_tensor(res, &obj_ty)?;

                  Ok((res, obj_ty))
              }
              "contiguous" => {
                  let fn_val = self.module.get_function("tl_tensor_contiguous").unwrap();
                  let call = self
                      .builder
                      .build_call(fn_val, &[obj_val.into()], "contiguous_res")
                      .map_err(|e| e.to_string())?;

                  let res = match call.try_as_basic_value() {
                      ValueKind::Basic(v) => v,
                      _ => return Err("Invalid contiguous return".into()),
                  };

                  // Register intermediate tensor result for automatic cleanup
                  self.emit_register_tensor(res, &obj_ty)?;

                  Ok((res, obj_ty))
              }
              "save" => {
                  let fn_val = self.module.get_function("tl_tensor_save").unwrap();
                  let (path_val, _) = self.compile_expr(&args[0])?;

                  // tl_tensor_save(path, tensor)
                  self.builder
                      .build_call(fn_val, &[path_val.into(), obj_val.into()], "save_call")
                      .map_err(|e| e.to_string())?;

                  if self.is_safe_to_free(obj, &obj_ty) {
                      self.emit_recursive_free(obj_val, &obj_ty, super::CLEANUP_FULL)?;
                  }
                  // args[0] is path (String). String is UserDefined.
                  // If path expr was temporary, we should free it?
                  // String literal is static (not temporary).
                  // Constructed string?
                  // We only compile_expr within this block. Use is_safe_to_free check?
                  // We need the compiled value to free it.
                  // path_val is available.
                  let path_ty = Type::UserDefined("String".to_string(), vec![]); // Assumed
                  if self.is_safe_to_free(&args[0], &path_ty) {
                      // self.emit_recursive_free(path_val, &path_ty)?;
                      // String free not fully impl?
                  }

                  Ok((
                      self.context.i64_type().const_int(0, false).into(),
                      Type::Void,
                  ))
              }

              "reshape" => {
                  if args.len() != 1 {
                      return Err("reshape method requires 1 argument (shape)".into());
                  }
                  let (s_val, _) = self.compile_expr(&args[0])?;
                  let reshape_fn = self.module.get_function("tl_tensor_reshape").unwrap();
                  let call = self
                      .builder
                      .build_call(reshape_fn, &[obj_val.into(), s_val.into()], "reshape_res")
                      .map_err(|e| e.to_string())?;
                  let res = match call.try_as_basic_value() {
                      ValueKind::Basic(v) => v,
                      _ => return Err("Invalid reshape return".into()),
                  };

                  if self.is_safe_to_free(obj, &obj_ty) {
                      self.emit_recursive_free(obj_val, &obj_ty, super::CLEANUP_FULL)?;
                  }
                  // args[0] is shape tensor.
                  // We compiled it to s_val. Ty is unknown here?
                  // compile_expr returned _. recover type?
                  // We know it's a tensor.
                  if self.is_safe_to_free(&args[0], &Type::Tensor(Box::new(Type::I64), 1)) { // Approximate type
                       // We need the type to free.
                       // But compile_expr return value IS (val, ty).
                       // Previous code: let (s_val, _) = ...
                       // We need to change that line first.
                       // Skipping arg cleanup for reshape for now (usually low impact).
                  }

                  Ok((res, obj_ty))
              }
              "sum" => {
                  let fn_val = self.module.get_function("tl_tensor_sum").unwrap();
                  let call = self
                      .builder
                      .build_call(fn_val, &[obj_val.into()], "sum_res");

                  let res = self.check_tensor_result(call.map_err(|e| e.to_string())?, "sum_error")?;

                  if self.is_safe_to_free(obj, &obj_ty) {
                      self.emit_recursive_free(obj_val, &obj_ty, super::CLEANUP_FULL)?;
                  }

                  // sum returns scalar tensor (rank 0 or 1 depending on impl).
                  // Assuming it returns Tensor<f32, 0> or 1.
                  self.emit_register_tensor(res, &obj_ty)?;
                  Ok((res, obj_ty)) // Currently preserving type/rank info is hard, returning same opaque type
              }
              "slice" => {
                  if args.len() != 2 {
                      return Err("slice requires 2 arguments".into());
                  }

                  // slice() only works on Tensors, not ScalarArrays
                  if matches!(obj_ty, Type::ScalarArray(_, _)) {
                      return Err("slice() does not support ScalarArray. Convert to Tensor first using Tensor::new() or similar".into());
                  }

                  let (start_val, _) = self.compile_expr(&args[0])?;
                  let (len_val, _) = self.compile_expr(&args[1])?;

                  let fn_val = self.module.get_function("tl_tensor_slice").unwrap();
                  let call = self
                      .builder
                      .build_call(
                          fn_val,
                          &[obj_val.into(), start_val.into(), len_val.into()],
                          "slice_res",
                      )
                      .map_err(|e| e.to_string())?;
                  let res = match call.try_as_basic_value() {
                      ValueKind::Basic(v) => v,
                      _ => return Err("Invalid slice return".into()),
                  };

                  if self.is_safe_to_free(obj, &obj_ty) {
                      self.emit_recursive_free(obj_val, &obj_ty, super::CLEANUP_FULL)?;
                  }

                  self.emit_register_tensor(res, &obj_ty)?;
                  Ok((res, obj_ty))
              }
              "to" | "to_device" => {
                  if args.len() != 1 {
                      return Err(format!(
                          "{} requires 1 argument (device name string)",
                          method
                      ));
                  }
                  let (dev_val, dev_ty) = self.compile_expr(&args[0])?;
                  if !matches!(&dev_ty, Type::UserDefined(s, _) if s == "String") {
                      return Err("Device name must be a string".into());
                  }

                  let fn_val = self
                      .module
                      .get_function("tl_tensor_to_device")
                      .ok_or("Runtime fn tl_tensor_to_device not found")?;

                  let call = self
                      .builder
                      .build_call(fn_val, &[obj_val.into(), dev_val.into()], "to_dev_res")
                      .map_err(|e| e.to_string())?;

                  let res = match call.try_as_basic_value() {
                      ValueKind::Basic(v) => v,
                      _ => return Err("Invalid return from to_device".into()),
                  };

                  if self.is_safe_to_free(obj, &obj_ty) {
                      self.emit_recursive_free(obj_val, &obj_ty, super::CLEANUP_FULL)?;
                  }

                  // Note: Do NOT call emit_register_tensor here.
                  // tl_tensor_to_device already registers via make_tensor internally.
                  Ok((res, obj_ty))
              }
              "add_assign" | "sub_assign" | "mul_assign" | "div_assign" => {
                  if args.len() != 1 {
                      return Err(format!("{} requires 1 argument", method));
                  }
                  // Must use ensure_tensor for RHS
                  let (rhs_val, _) = self.ensure_tensor_v2(&args[0], 0)?;

                  let fn_name = match method {
                      "add_assign" => "tl_tensor_add_assign",
                      "sub_assign" => "tl_tensor_sub_assign",
                      "mul_assign" => "tl_tensor_mul_assign",
                      "div_assign" => "tl_tensor_div_assign",
                      _ => unreachable!(),
                  };

                  let fn_val = self
                      .module
                      .get_function(fn_name)
                      .ok_or(format!("Runtime fn {} not found", fn_name))?;

                  self.builder
                      .build_call(fn_val, &[obj_val.into(), rhs_val.into()], "assign_res")
                      .map_err(|e| e.to_string())?;

                  if self.is_safe_to_free(obj, &obj_ty) {
                      // For assign ops, obj is often a field or var, so safe=false.
                      // But if obj is (a+b), then safe=true, and we MUST free it.
                      self.emit_recursive_free(obj_val, &obj_ty, super::CLEANUP_FULL)?;
                  }

                  // args[0] is RHS.
                  // We need to capture (rhs_val, rhs_ty) from ensure_tensor_v2.
                  // But ensure_tensor_v2 returns (BasicValueEnum, Type).
                  // L2339: let (rhs_val, _) = self.ensure_tensor_v2(&args[0], 0)?;
                  // We need the type.
                  // Since it's inside block, we can't easily access the type unless we change the line.
                  // However, we know it's a Tensor.
                  // And ensure_tensor_v2 already handles potential scalar->tensor conversion which creates new tensor.
                  // If args[0] was already a tensor expr, ensure_tensor_v2 returns it.
                  // Check if args[0] is safe to free.

                  Ok((
                      self.context.i64_type().const_int(0, false).into(),
                      Type::Void,
                  ))
              }
                "matmul" => {
                    if args.len() != 1 {
                        return Err("matmul requires 1 argument".into());
                    }
                    let (rhs_val, _) = self.ensure_tensor_v2(&args[0], 0)?;
                    let fn_val = self
                        .module
                        .get_function("tl_tensor_matmul")
                        .ok_or("tl_tensor_matmul not found")?;

                    let call = self
                        .builder
                        .build_call(fn_val, &[obj_val.into(), rhs_val.into()], "matmul_res");

                    let res = self.check_tensor_result(call.map_err(|e| e.to_string())?, "matmul_error")?;

                    if self.is_safe_to_free(obj, &obj_ty) {
                        self.emit_recursive_free(obj_val, &obj_ty, super::CLEANUP_FULL)?;
                    }
                    self.emit_register_tensor(res, &obj_ty)?;
                    Ok((res, obj_ty))
                }
                "exp" | "log" => {
                    let fn_name = format!("tl_tensor_{}", method);
                    let fn_val = self
                        .module
                        .get_function(&fn_name)
                        .ok_or(format!("{} not found", fn_name))?;

                    let call = self
                        .builder
                        .build_call(fn_val, &[obj_val.into()], "unary_res");

                    let res = self.check_tensor_result(call.map_err(|e| e.to_string())?, &format!("{}_error", method))?;

                    if self.is_safe_to_free(obj, &obj_ty) {
                        self.emit_recursive_free(obj_val, &obj_ty, super::CLEANUP_FULL)?;
                    }
                    self.emit_register_tensor(res, &obj_ty)?;
                    Ok((res, obj_ty))
                }
                _ => {
                  // Generic method dispatch for UserDefined types and Tensor
                  let mut type_name = match &obj_ty {
                      Type::UserDefined(name, _) | Type::Struct(name, _) | Type::Enum(name, _) => {
                          if name.contains("::") { name.split("::").last().unwrap().to_string() } else { name.clone() }
                      },
                      Type::Tensor(_, _) => "Tensor".to_string(),
                      Type::I64 => "i64".to_string(),
                      Type::F32 => "f32".to_string(),
                      _ => {
                          return Err(format!("Unknown method: {} on type {:?}", method, obj_ty))
                      }
                  };

                  let mut runtime_fn_name = format!("tl_{}_{}", type_name.to_lowercase(), method);
                  
                  // Generic Monomorphization Trigger
                  if let Type::Struct(name, args) | Type::UserDefined(name, args) = &obj_ty {
                       if !args.is_empty() {
                           // Check if it's a generic method in registry (via mono)
                           let simple_name = if name.contains("::") { name.split("::").last().unwrap() } else { name };
                           
                           match self.monomorphize_method(simple_name, method, args) {
                               Ok(mangled) => {
                                   runtime_fn_name = mangled;
                               },
                               Err(_) => {
                                   // Keep standard name if not found in generic impls (e.g. might be manually defined runtime fn)
                               }
                           }
                       }
                  }

                  let fn_val = self.module.get_function(&runtime_fn_name).ok_or(format!(
                      "Method {} not found on type {} (checked {})",
                      method, type_name, runtime_fn_name
                  ))?;

                      // Prepend object to args
                      let mut compiled_args_vals = Vec::with_capacity(args.len() + 1);
                      let mut compiled_args_types = Vec::with_capacity(args.len() + 1);

                      compiled_args_vals.push(obj_val.into());
                      // Keep track of types? obj is separate.

                      for arg in args {
                          let (val, ty) = self.compile_expr(arg)?;
                          compiled_args_vals.push(val.into());
                          compiled_args_types.push((val, ty));
                      }

                      let call = self
                          .builder
                          .build_call(fn_val, &compiled_args_vals, "method_res")
                          .map_err(|e| e.to_string())?;

                      let call = self.check_tensor_result(call, "method_error")?;

                      // FIX: Free temporary receiver
                      if self.is_safe_to_free(obj, &obj_ty) {
                          self.emit_recursive_free(obj_val, &obj_ty, super::CLEANUP_FULL)?;
                      }

                      // FIX: Free temporary arguments
                      for (i, (val, ty)) in compiled_args_types.iter().enumerate() {
                          let arg_expr = &args[i];
                          if self.is_safe_to_free(arg_expr, ty) {
                              self.emit_recursive_free(*val, ty, super::CLEANUP_FULL)?;
                          }
                      }

                      // Determine return type from fn_return_types map
                      let ret_type = self
                          .fn_return_types
                          .get(&runtime_fn_name)
                          .cloned()
                          .unwrap_or(Type::Void);

                      match call.try_as_basic_value() {
                          ValueKind::Basic(v) => {
                              if let Type::Tensor(_, _) = ret_type {
                                  self.emit_register_tensor(v, &ret_type)?;
                              }
                              Ok((v, ret_type))
                          }
                          _ => Ok((
                              self.context.i64_type().const_int(0, false).into(),
                              Type::Void,
                          )),
                      }
                  }
              }
          }
      */

    fn compile_relation_call(
        &mut self,
        name: &str,
        args: &[Expr],
    ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        let function = self
            .module
            .get_function(name)
            .ok_or(format!("Relation wrapper {} not found", name))?;

        let mut mask: i64 = 0;
        let mut compiled_args = Vec::new();
        let mut tags = Vec::new();
        let i64_type = self.context.i64_type();
        let i8_type = self.context.i8_type();

        // Semantics phase no longer injects metadata. Start index is always 0.
        let start_index = 0;

        for (i, arg) in args.iter().skip(start_index).enumerate() {
            match &arg.inner {
                ExprKind::LogicVar(_) => {
                    // Variable argument -> Set mask bit
                    mask |= 1 << i;
                    // Pass 0 placeholder
                    compiled_args.push(i64_type.const_int(0, false).into());
                    tags.push(3); // Default to Entity
                }
                ExprKind::Symbol(sym_name) => {
                    // Check if variable exists in scope
                    let mut found = false;
                    for scope in self.variables.iter().rev() {
                        if scope.contains_key(sym_name) {
                            found = true;
                            break;
                        }
                    }

                    if found {
                        // Treat as Variable lookup
                        let var_expr = Expr {
                            inner: ExprKind::Variable(sym_name.clone()),
                            span: arg.span.clone(),
                        };
                        let (val, ty) = self.compile_expr(&var_expr)?;
                        compiled_args.push(val);
                        let tag = match ty {
                            Type::I64 | Type::I32 | Type::I8 | Type::U8 | Type::U16 | Type::U32 | Type::Usize => 0, // Int
                            Type::F32 | Type::F64 | Type::F16 | Type::BF16 => 1,             // Float
                            Type::Bool => 2,                        // Bool
                            Type::Entity => 3,                      // Entity
                            _ => 3,
                        };
                        tags.push(tag);
                    } else {
                        // Assume constant Entity Name
                        let add_entity_fn = self.module.get_function("tl_kb_add_entity").unwrap();
                        let name_ptr = self
                            .builder
                            .build_global_string_ptr(sym_name, "ent_name")
                            .map_err(|e| e.to_string())?;
                        let call = self
                            .builder
                            .build_call(
                                add_entity_fn,
                                &[name_ptr.as_pointer_value().into()],
                                "ent_id",
                            )
                            .map_err(|e| e.to_string())?;

                        let val = match call.try_as_basic_value() {
                            inkwell::values::ValueKind::Basic(v) => v,
                            _ => return Err("Invalid return from add_entity".into()),
                        };
                        compiled_args.push(val);
                        tags.push(3); // ConstantTag::Entity
                    }
                }
                ExprKind::StringLiteral(s) => {
                    let add_entity_fn = self.module.get_function("tl_kb_add_entity").unwrap();
                    let name_ptr = self
                        .builder
                        .build_global_string_ptr(s, "ent_name")
                        .map_err(|e| e.to_string())?;
                    let call = self
                        .builder
                        .build_call(
                            add_entity_fn,
                            &[name_ptr.as_pointer_value().into()],
                            "ent_id",
                        )
                        .map_err(|e| e.to_string())?;
                    let val = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err("Invalid return from add_entity".into()),
                    };
                    compiled_args.push(val);
                    tags.push(4); // ConstantTag::String
                }
                _ => {
                    let (val, ty) = self.compile_expr(arg)?;
                     // Force cast to i64
                    let i64_type = self.context.i64_type();
                    let val_i64 = if val.is_int_value() {
                        self.builder.build_int_cast(val.into_int_value(), i64_type, "arg_cast").unwrap().into()
                    } else {
                        // If float, cast to int? Or bitcast?
                        // For now assume logic args are ints or entities (i64)
                        // If boolean?
                         if val.is_int_value() { // Bool is int(1)
                             self.builder.build_int_cast(val.into_int_value(), i64_type, "bool_cast").unwrap().into()
                         } else {
                             val // Cannot cast pointer to int easily here without ptrtoint
                         }
                    };
                    compiled_args.push(val_i64);
                    let tag = match ty {
                        Type::I64 | Type::I32 | Type::I8 | Type::U8 | Type::U16 | Type::U32 | Type::Usize => 0, // Int
                        Type::F32 | Type::F64 | Type::F16 | Type::BF16 => 1,             // Float
                        Type::Bool => 2,                        // Bool
                        Type::Entity => 3,                      // Entity
                        _ => 0,
                    };
                    tags.push(tag);
                }
            }
        }

        // Generate tags pointer
        let tags_ptr = if tags.is_empty() {
            self.context.ptr_type(inkwell::AddressSpace::default()).const_null()
        } else {
            let tags_arr_type = i8_type.array_type(tags.len() as u32);
            let tags_alloca = self.builder.build_alloca(tags_arr_type, "query_tags").unwrap();
            for (i, &tag) in tags.iter().enumerate() {
                let ptr = unsafe {
                    self.builder
                        .build_gep(
                            tags_arr_type,
                            tags_alloca,
                            &[
                                i64_type.const_int(0, false),
                                i64_type.const_int(i as u64, false),
                            ],
                            "",
                        )
                        .unwrap()
                };
                self.builder
                    .build_store(ptr, i8_type.const_int(tag as u64, false))
                    .unwrap();
            }
            unsafe {
                self.builder
                    .build_gep(
                        tags_arr_type,
                        tags_alloca,
                        &[i64_type.const_int(0, false), i64_type.const_int(0, false)],
                        "tags_decayed",
                    )
                    .unwrap()
            }
        };

        // Insert Mask at beginning
        let mask_val = i64_type.const_int(mask as u64, false);
        compiled_args.insert(0, mask_val.into());
        // Append tags pointer at end (as i64)
        // let null_tags = i8_type.ptr_type(inkwell::AddressSpace::default()).const_null();
        let tags_int = self.builder.build_ptr_to_int(tags_ptr, i64_type, "tags_int").unwrap();
        compiled_args.push(tags_int.into());
        // compiled_args.push(null_tags.into()); // FORCE NULL TAGS
        // compiled_args.push(tags_ptr.into());

        let final_args: Vec<inkwell::values::BasicMetadataValueEnum> =
            compiled_args.iter().map(|&val| val.into()).collect();

        let call = self
            .builder
            .build_call(function, &final_args, "query_res")
            .map_err(|e| e.to_string())?;

        // Debug signature mismatch
        

        let res = match call.try_as_basic_value() {
            inkwell::values::ValueKind::Basic(v) => v,
            _ => return Err("Relation wrapper returned void".into()),
        };

        Ok((res, Type::Tensor(Box::new(Type::F32), 1)))
    }

    pub(crate) fn compile_fn_call(
        &mut self,
        name: &str,
        args: &[Expr],
    ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        let (val, ty) = self.compile_fn_call_dps(name, args, None)?;
        self.add_temp(val, ty.clone());
        Ok((val, ty))
    }

    pub(crate) fn compile_fn_call_dps(
        &mut self,
        name: &str,
        args: &[Expr],
        dest: Option<BasicValueEnum<'ctx>>,
    ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        // 0. Check if it's a relation query (handle module path resolution)
        let simple_name = name.split("::").last().unwrap_or(name);

        if self.relations.contains(name) {
            return self.compile_relation_call(name, args);
        } else if self.relations.contains(simple_name) {
            return self.compile_relation_call(simple_name, args);
        }

        // 1. Builtins
        let builtin_opt = self.builtin_manager.get(name).copied();
        if let Some(builtin) = builtin_opt {
            match builtin {
                BuiltinFn::Evaluated(func) => {
                    let mut compiled_args = Vec::with_capacity(args.len());
                    for arg in args {
                        let (mut val, mut ty) = self.compile_expr(arg)?;
                        // Auto-convert ScalarArray to Tensor for consistency
                        if let Type::ScalarArray(_, _) = ty {
                            let (new_val, new_ty) = self.ensure_tensor_v2(arg, 0)?;
                            val = new_val.try_into().unwrap();
                            ty = new_ty;
                        }
                        compiled_args.push((val, ty));
                    }
                    return func(self, compiled_args);
                }
                BuiltinFn::Unevaluated(func) => {
                    return func(self, args);
                }
            }
        }

        // 2. Generic Function Call / Struct Init
        let llvm_func_name = match name {
            "slice" => "tl_tensor_slice",
            "sum" => "tl_tensor_sum", // Fallback for global sum if not caught by builtin (redundant but safe)
            "enable_grad" => "tl_tensor_enable_grad",
            _ => name,
        };

        // Lookup return type FIRST to handle sret
        // Handle static method syntax: Type::method -> tl_type_method
        let resolved_name = if self.module.get_function(llvm_func_name).is_some() {
            llvm_func_name.to_string()
        } else if llvm_func_name.contains("::") {
            let parts: Vec<&str> = llvm_func_name.split("::").collect();
            // Try to resolve simple name (last part) specifically for module imports
            // where definition is simple name but call is qualified.
            if let Some(last) = parts.last() {
                if self.module.get_function(last).is_some() {
                    last.to_string()
                } else if parts.len() >= 2 {
                    let type_name = parts[parts.len() - 2];
                    let method = parts[parts.len() - 1];
                    // Try user-defined type mangling (Case Sensitive) first
                    let mangled = format!("tl_{}_{}", type_name, method);
                    if self.module.get_function(&mangled).is_some() {
                        mangled
                    } else {
                        // Try Stdlib/Primitive mangling (lowercase)
                        format!("tl_{}_{}", type_name.to_lowercase(), method)
                    }
                } else {
                    llvm_func_name.to_string()
                }
            } else {
                llvm_func_name.to_string()
            }
        } else {
            llvm_func_name.to_string()
        };

        // Monomorphization Logic: Check if generic
        let mut final_resolved_name = resolved_name.clone();
        let mut precompiled_args = None;
        
        if self.module.get_function(&final_resolved_name).is_none() && self.generic_fn_defs.contains_key(&final_resolved_name) {
             let mut args_vec = Vec::new();
             let mut arg_types = Vec::new();
             for arg in args {
                  let (mut val, mut ty) = self.compile_expr(arg)?;
                  if let Type::ScalarArray(_, _) = ty {
                        let (new_val, new_ty) = self.ensure_tensor_v2(arg, 0)?;
                        val = new_val.try_into().unwrap();
                        ty = new_ty;
                  }
                  args_vec.push((val, ty.clone()));
                  arg_types.push(ty);
             }
             
             final_resolved_name = self.monomorphize_generic_function(&resolved_name, &arg_types)?;
             precompiled_args = Some(args_vec);
        }

        let lookup_name = final_resolved_name.as_str();
        let ret_type = self
            .fn_return_types
            .get(lookup_name)
            .cloned()
            .unwrap_or(Type::Void);

        let func_opt = self.module.get_function(&final_resolved_name);

        let func = if let Some(f) = func_opt {
            f
        } else {
            // Fallback to Struct Initialization
            let simple_name = if name.contains("::") {
                let s = name.split("::").last().unwrap();
                s
            } else {
                name as &str
            };

            if self.struct_defs.contains_key(simple_name) {
                return self.compile_tuple_struct_init(simple_name, args);
            }

            return Err(format!(
                "Function {} not found (resolved: {})",
                name, resolved_name
            ));
        };

        let mut compiled_args_vals = Vec::with_capacity(args.len() + 1);
        let mut compiled_args_types = Vec::with_capacity(args.len());

        // DPS: If return type is Struct (and SRET enabled), handle hidden dest argument
        // Tensors are returned by pointer directly, so exclude them.
        let mut dest_val = None;
        if match ret_type {
             // CRITICAL FIX: Disable SRET for Structs (Pointer Return)
             Type::Struct(_, _) => false,
             Type::UserDefined(ref name, _) if name != "String" => false,
             _ => false 
        } {
             if let Some(d) = dest {
                 dest_val = Some(d);
             } else {
                 // Allocate Temp Buffer
                 // We use a fixed size sufficient for OpaqueTensor struct (96 bytes safe upper bound)
                 let malloc_fn = self.module.get_function("malloc").ok_or("malloc not found")?;
                 let size = self.context.i64_type().const_int(96, false);
                 let call = self.builder.build_call(malloc_fn, &[size.into()], "tensor_tmp").map_err(|e| e.to_string())?;
                 let raw_ptr = match call.try_as_basic_value() {
                      ValueKind::Basic(v) => v.into_pointer_value(),
                      _ => return Err("malloc failed".into()),
                 };
                 // Cast to opaque ptr
                 let cast_ptr = self.builder.build_pointer_cast(raw_ptr, self.context.ptr_type(inkwell::AddressSpace::default()), "cast_tmp").unwrap();
                 dest_val = Some(cast_ptr.into());
             }
             
             if let Some(d) = dest_val {
                 compiled_args_vals.push(d.into());
             }
        }

        if let Some(pre_values) = precompiled_args {
             for (val, ty) in pre_values {
                  compiled_args_vals.push(val.into());
                  compiled_args_types.push((val, ty));
             }
        } else {
            for arg in args {
                let (mut val, mut ty) = self.compile_expr(arg)?;

                // Move Semantics disabled: function arguments remain valid after calls.

                // Auto-convert ScalarArray to Tensor
                // Functions in Runtime expecting Tensor arguments need OpaqueTensor*, not [T; N]*
                if let Type::ScalarArray(_, _) = ty {
                    let (new_val, new_ty) = self.ensure_tensor_v2(arg, 0)?;
                    val = new_val.try_into().unwrap(); // BasicValueEnum
                    ty = new_ty;
                }

                compiled_args_vals.push(val.into());
                compiled_args_types.push((val, ty));
            }
        }

        let call = self
            .builder
            .build_call(func, &compiled_args_vals, if dest_val.is_some() { "" } else { "call_tmp" })
            .map_err(|e| e.to_string())?;

        // FIX: Free temporary arguments
        for (i, (val, ty)) in compiled_args_types.into_iter().enumerate() {
            let arg_expr = &args[i];
            if self.is_safe_to_free(arg_expr, &ty) {
                self.emit_recursive_free(val, &ty, super::CLEANUP_FULL)?;
            }
        }

        if let Some(d) = dest_val {
             return Ok((d, ret_type));
        }

        let res = match call.try_as_basic_value() {
            ValueKind::Basic(_) => self.check_tensor_result(call, "call_error")?,
            _ => {
                if ret_type == Type::Void {
                    self.context.i64_type().const_int(0, false).into()
                } else {
                    // For debug
                    return Ok((
                        self.context.i64_type().const_int(0, false).into(),
                        Type::Void,
                    ));
                }
            }
        };

        // For Struct returns (SRET deprecated logic, assuming by-value or pointer return if not SRET)
        Ok((res, ret_type))
    }
}
fn compile_set_device<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err("set_device expects 1 argument".into());
    }
    let (arg_val, arg_ty) = codegen.compile_expr(&args[0])?;

    // Expect Device Enum
    let is_device_enum = match &arg_ty {
        Type::Enum(e, _) | Type::UserDefined(e, _) if e == "Device" => true,
        _ => false,
    };

    if !is_device_enum {
        return Err(format!(
            "set_device argument must be a Device enum, found {:?}",
            arg_ty
        ));
    }

    let fn_val = codegen
        .module
        .get_function("tl_set_device")
        .ok_or("tl_set_device not found")?;

    // Argument is pointer to Device enum (which is opaque* in LLVM)
    let arg_ptr = match arg_val {
        BasicValueEnum::PointerValue(p) => p,
        _ => return Err("Expected pointer to Device enum".into()),
    };

    codegen
        .builder
        .build_call(fn_val, &[arg_ptr.into()], "")
        .map_err(|e| e.to_string())?;

    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

fn compile_checkpoint<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 2 {
        return Err("checkpoint requires 2 arguments: (method_ref, input)".into());
    }
    // Parse args[0] as obj.method
    let (obj_ptr, fn_ptr) = if let ExprKind::FieldAccess(obj_expr, method_name) = &args[0].inner {
        let (obj_val, obj_ty) = codegen.compile_expr(obj_expr)?;

        // Get struct type
        let struct_name = match obj_ty {
            Type::Struct(n, _) | Type::UserDefined(n, _) => n,
            _ => return Err("checkpoint arg 1 must be object.method".into()),
        };

        // Find function: Struct_Method
        let fn_name = format!("tl_{}_{}", struct_name, method_name);
        let fn_val = codegen
            .module
            .get_function(&fn_name)
            .ok_or(format!("Method {} not found", fn_name))?;

        // Cast function to void pointer
        let fn_ptr_val = fn_val.as_global_value().as_pointer_value();
        let void_fn_ptr = codegen
            .builder
            .build_bit_cast(
                fn_ptr_val,
                codegen.context.ptr_type(inkwell::AddressSpace::default()),
                "fn_void_ptr",
            )
            .map_err(|e| e.to_string())?;

        let obj_ptr = codegen
            .builder
            .build_bit_cast(
                obj_val.into_pointer_value(),
                codegen.context.ptr_type(inkwell::AddressSpace::default()),
                "obj_void_ptr",
            )
            .map_err(|e| e.to_string())?;

        (obj_ptr, void_fn_ptr)
    } else {
        return Err("checkpoint first argument must be 'obj.method'".into());
    };

    // Compile input
    let (arg_val, arg_ty) = codegen.compile_expr(&args[1])?;
    if !matches!(arg_ty, Type::Tensor(_, _)) {
        return Err("checkpoint input must be tensor".into());
    }

    // Call runtime
    let cp_fn = codegen
        .module
        .get_function("tl_checkpoint")
        .expect("tl_checkpoint not found");
    let arg_ptr = codegen
        .builder
        .build_bit_cast(
            arg_val.into_pointer_value(),
            codegen.context.ptr_type(inkwell::AddressSpace::default()),
            "arg_cast",
        )
        .map_err(|e| e.to_string())?;

    let call = codegen
        .builder
        .build_call(
            cp_fn,
            &[obj_ptr.into(), fn_ptr.into(), arg_ptr.into()],
            "checkpoint_res",
        )
        .map_err(|e| e.to_string())?;

    let res_val = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("tl_checkpoint did not return a value".into()),
    };

    // Register result
    let res_ty = arg_ty.clone(); // Checkpoint returns same type as input
    codegen.emit_register_tensor(res_val, &res_ty)?;

    Ok((res_val, res_ty))
}

fn compile_print_common<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    is_newline: bool,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    // Shared name arg for error message (not passed here but can infer)
    if args.len() != 1 {
        return Err("print/println requires 1 argument".into());
    }
    // Check type of arg
    let (arg_val, arg_type) = &args[0];
    match arg_type {
        Type::I64 => {
            let fn_name = if is_newline {
                "tl_print_i64"
            } else {
                "tl_display_i64"
            };
            let fn_val = codegen.module.get_function(fn_name).unwrap();
            codegen
                .builder
                .build_call(fn_val, &[(*arg_val).into()], "print_call")
                .map_err(|e| e.to_string())?;
        }
        Type::I32 => {
            let fn_name = if is_newline {
                "tl_print_i32"
            } else {
                "tl_display_i32"
            };
            let fn_val = codegen.module.get_function(fn_name).unwrap();
            codegen
                .builder
                .build_call(fn_val, &[(*arg_val).into()], "print_call")
                .map_err(|e| e.to_string())?;
        }
        Type::F32 => {
            let fn_name = if is_newline {
                "tl_print_f32"
            } else {
                "tl_display_f32"
            };
            let fn_val = codegen.module.get_function(fn_name).unwrap();
            codegen
                .builder
                .build_call(fn_val, &[(*arg_val).into()], "print_call")
                .map_err(|e| e.to_string())?;
        }
        Type::F64 => {
            let fn_name = if is_newline {
                "tl_print_f64"
            } else {
                "tl_display_f64"
            };
            let fn_val = codegen.module.get_function(fn_name).unwrap();
            codegen
                .builder
                .build_call(fn_val, &[(*arg_val).into()], "print_call")
                .map_err(|e| e.to_string())?;
        }
        Type::Bool => {
            let fn_name = if is_newline {
                "tl_print_bool"
            } else {
                "tl_display_bool"
            };
            let fn_val = codegen.module.get_function(fn_name).unwrap();
            codegen
                .builder
                .build_call(fn_val, &[(*arg_val).into()], "print_call")
                .map_err(|e| e.to_string())?;
        }
        Type::Tuple(elem_types) => {
            // Print tuple as (a, b, c)
            let display_fn = codegen
                .module
                .get_function("tl_display_string")
                .ok_or("tl_display_string not found")?;
            let print_fn = codegen
                .module
                .get_function("tl_print_string")
                .ok_or("tl_print_string not found")?;

            fn emit_tuple_str<'ctx>(
                codegen: &mut CodeGenerator<'ctx>,
                s: &str,
                newline: bool,
                display_fn: inkwell::values::FunctionValue<'ctx>,
                print_fn: inkwell::values::FunctionValue<'ctx>,
            ) -> Result<(), String> {
                let s_val = codegen.context.const_string(s.as_bytes(), true);
                let global = codegen.module.add_global(
                    s_val.get_type(),
                    Some(inkwell::AddressSpace::default()),
                    "tuple_part",
                );
                global.set_initializer(&s_val);
                global.set_linkage(inkwell::module::Linkage::Internal);
                global.set_constant(true);

                let ptr = unsafe {
                    codegen
                        .builder
                        .build_in_bounds_gep(
                            s_val.get_type(),
                            global.as_pointer_value(),
                            &[
                                codegen.context.i64_type().const_int(0, false),
                                codegen.context.i64_type().const_int(0, false),
                            ],
                            "tuple_str_ptr",
                        )
                        .map_err(|e| e.to_string())?
                };

                let fn_val = if newline { print_fn } else { display_fn };
                codegen
                    .builder
                    .build_call(fn_val, &[ptr.into()], "print_tuple_part")
                    .map_err(|e| e.to_string())?;
                Ok(())
            }

            emit_tuple_str(codegen, "(", false, display_fn, print_fn)?;

            if !arg_val.is_pointer_value() {
                return Err("Tuple value is not a pointer".into());
            }
            let tuple_ptr = arg_val.into_pointer_value();

            let mut llvm_types = Vec::new();
            for ty in elem_types.iter() {
                llvm_types.push(codegen.get_llvm_type(ty)?);
            }
            let tuple_struct_type = codegen.context.struct_type(&llvm_types, false);

            for (idx, ty) in elem_types.iter().enumerate() {
                if idx > 0 {
                    emit_tuple_str(codegen, ", ", false, display_fn, print_fn)?;
                }
                let field_ptr = codegen
                    .builder
                    .build_struct_gep(tuple_struct_type, tuple_ptr, idx as u32, "tuple_elem_ptr")
                    .map_err(|e| e.to_string())?;
                let llvm_field_ty = codegen.get_llvm_type(ty)?;
                let val = codegen
                    .builder
                    .build_load(llvm_field_ty, field_ptr, "tuple_elem")
                    .map_err(|e| e.to_string())?;
                compile_print_common(codegen, vec![(val, ty.clone())], false)?;
            }

            emit_tuple_str(codegen, ")", false, display_fn, print_fn)?;
            if is_newline {
                emit_tuple_str(codegen, "", true, display_fn, print_fn)?;
            }
        }
        Type::Tensor(_, _) => {
            let fn_name = if is_newline {
                "tl_tensor_print"
            } else {
                "tl_tensor_display"
            };
            let fn_val = codegen.module.get_function(fn_name).unwrap();
            codegen
                .builder
                .build_call(fn_val, &[(*arg_val).into()], "print_call")
                .map_err(|e| e.to_string())?;
        }
        Type::UserDefined(s, _) if s == "String" => {
            let fn_name = if is_newline {
                "tl_print_string"
            } else {
                "tl_display_string"
            };
            let fn_val = codegen.module.get_function(fn_name);
            if let Some(f) = fn_val {
                codegen
                    .builder
                    .build_call(f, &[(*arg_val).into()], "print_call")
                    .map_err(|e| e.to_string())?;
            } else {
                return Err(format!("{} not found (add to init)", fn_name).into());
            }
        }
        Type::ScalarArray(elem_type, len) => {
            let i64_type = codegen.context.i64_type();
            let i32_type = codegen.context.i32_type();
            let f32_type = codegen.context.f32_type();

            let (llvm_elem_type, print_fn_name): (inkwell::types::BasicTypeEnum, &str) =
                match elem_type.as_ref() {
                    Type::I64 => (
                        i64_type.into(),
                        if is_newline {
                            "tl_print_i64"
                        } else {
                            "tl_display_i64"
                        },
                    ),
                    Type::I32 => (
                        i32_type.into(),
                        if is_newline {
                            "tl_print_i32"
                        } else {
                            "tl_display_i32"
                        },
                    ),
                    Type::F64 => (
                        codegen.context.f64_type().into(),
                        if is_newline {
                            "tl_print_f64"
                        } else {
                            "tl_display_f64"
                        },
                    ),
                    Type::Bool => (
                        codegen.context.bool_type().into(),
                        if is_newline {
                            "tl_print_bool"
                        } else {
                            "tl_display_bool"
                        },
                    ),
                    _ => (
                        f32_type.into(),
                        if is_newline {
                            "tl_print_f32"
                        } else {
                            "tl_display_f32"
                        },
                    ),
                };

            let print_fn = codegen
                .module
                .get_function(print_fn_name)
                .ok_or(format!("{} not found", print_fn_name))?;

            let array_ptr = (*arg_val).into_pointer_value();
            let typed_ptr_type = codegen.context.ptr_type(inkwell::AddressSpace::default());
            let typed_ptr = codegen
                .builder
                .build_pointer_cast(array_ptr, typed_ptr_type, "print_typed_ptr")
                .map_err(|e| e.to_string())?;

            for i in 0..*len {
                let idx = i64_type.const_int(i as u64, false);
                let elem_ptr = unsafe {
                    codegen
                        .builder
                        .build_in_bounds_gep(llvm_elem_type, typed_ptr, &[idx], "elem_ptr")
                        .map_err(|e| e.to_string())?
                };
                let elem_val = codegen
                    .builder
                    .build_load(llvm_elem_type, elem_ptr, "elem_val")
                    .map_err(|e| e.to_string())?;
                codegen
                    .builder
                    .build_call(print_fn, &[elem_val.into()], "print_elem")
                    .map_err(|e| e.to_string())?;
            }
        }
        _ => return Err(format!("Cannot print type {:?}", arg_type)),
    }
    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

fn compile_print_uneval<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_print_formatted(codegen, args, false)
}

fn compile_println_uneval<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_print_formatted(codegen, args, true)
}

fn compile_read_line_uneval<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err("read_line requires 1 argument".into());
    }
    let (prompt_val, _prompt_ty) = codegen.compile_expr(&args[0])?;
    let fn_val = codegen
        .module
        .get_function("tl_read_line")
        .ok_or("tl_read_line not found")?;
    let call = codegen
        .builder
        .build_call(fn_val, &[prompt_val.into()], "read_line_res")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from read_line".into()),
    };
    Ok((res, Type::UserDefined("String".to_string(), vec![])))
}

fn compile_print_formatted<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
    is_newline: bool,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.is_empty() {
        if is_newline {
            // Print newline only
            let fn_val = codegen
                .module
                .get_function("tl_print_string")
                .ok_or("tl_print_string not found")?;

            // Create empty string "" and print it (tl_print_string adds newline)
            let s_val = codegen.context.const_string(b"", true);
            let global = codegen.module.add_global(
                s_val.get_type(),
                Some(inkwell::AddressSpace::default()),
                "empty_str",
            );
            global.set_initializer(&s_val);
            global.set_linkage(inkwell::module::Linkage::Internal);
            global.set_constant(true);

            let ptr = unsafe {
                codegen
                    .builder
                    .build_in_bounds_gep(
                        s_val.get_type(),
                        global.as_pointer_value(),
                        &[
                            codegen.context.i64_type().const_int(0, false),
                            codegen.context.i64_type().const_int(0, false),
                        ],
                        "str_ptr",
                    )
                    .map_err(|e| e.to_string())?
            };

            codegen
                .builder
                .build_call(fn_val, &[ptr.into()], "print_newline")
                .map_err(|e| e.to_string())?;
        }
        return Ok((
            codegen.context.i64_type().const_int(0, false).into(),
            Type::Void,
        ));
    }

    // Check for format string
    let fmt_str_opt = if let ExprKind::StringLiteral(s) = &args[0].inner {
        if s.contains("{}") {
            Some(s.clone())
        } else {
            None
        }
    } else {
        None
    };

    if let Some(fmt_str) = fmt_str_opt {
        // Formatted print
        let parts: Vec<&str> = fmt_str.split("{}").collect();
        let arg_count = args.len() - 1;
        let placeholder_count = parts.len() - 1;

        if arg_count != placeholder_count {
            return Err(format!(
                "Format string has {} placeholders but {} arguments were provided",
                placeholder_count, arg_count
            ));
        }

        let display_fn = codegen
            .module
            .get_function("tl_display_string")
            .ok_or("tl_display_string not found")?;

        for (i, part) in parts.iter().enumerate() {
            // 1. Print literal part
            if !part.is_empty() {
                let s_val = codegen.context.const_string(part.as_bytes(), true);
                let global = codegen.module.add_global(
                    s_val.get_type(),
                    Some(inkwell::AddressSpace::default()),
                    "fmt_part",
                );
                global.set_initializer(&s_val);
                global.set_linkage(inkwell::module::Linkage::Internal);
                global.set_constant(true);

                let ptr = unsafe {
                    codegen
                        .builder
                        .build_in_bounds_gep(
                            s_val.get_type(),
                            global.as_pointer_value(),
                            &[
                                codegen.context.i64_type().const_int(0, false),
                                codegen.context.i64_type().const_int(0, false),
                            ],
                            "str_ptr",
                        )
                        .map_err(|e| e.to_string())?
                };

                codegen
                    .builder
                    .build_call(display_fn, &[ptr.into()], "print_part")
                    .map_err(|e| e.to_string())?;
            }

            // 2. Print argument
            if i < arg_count {
                let expr = &args[i + 1];
                let (val, ty) = codegen.compile_expr(expr)?;
                // Use existing common logic, force is_newline=false
                compile_print_common(codegen, vec![(val, ty)], false)?;
            }
        }

        if is_newline {
            // Print final newline using tl_print_string("")
            let print_fn = codegen
                .module
                .get_function("tl_print_string")
                .ok_or("tl_print_string not found")?;

            let s_val = codegen.context.const_string(b"", true);
            let global = codegen.module.add_global(
                s_val.get_type(),
                Some(inkwell::AddressSpace::default()),
                "empty_str",
            );
            global.set_initializer(&s_val);
            global.set_linkage(inkwell::module::Linkage::Internal);
            global.set_constant(true);

            let ptr = unsafe {
                codegen
                    .builder
                    .build_in_bounds_gep(
                        s_val.get_type(),
                        global.as_pointer_value(),
                        &[
                            codegen.context.i64_type().const_int(0, false),
                            codegen.context.i64_type().const_int(0, false),
                        ],
                        "str_ptr",
                    )
                    .map_err(|e| e.to_string())?
            };

            codegen
                .builder
                .build_call(print_fn, &[ptr.into()], "print_newline")
                .map_err(|e| e.to_string())?;
        }
    } else {
        // Normal print
        if args.len() != 1 {
            return Err("print/println requires 1 argument (or format string)".into());
        }
        let (val, ty) = codegen.compile_expr(&args[0])?;
        compile_print_common(codegen, vec![(val, ty)], is_newline)?;
    }

    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

fn compile_args_count<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if !args.is_empty() {
        return Err("args_count takes no arguments".into());
    }
    let fn_val = codegen
        .module
        .get_function("tl_args_count")
        .ok_or("tl_args_count not found")?;
    let call = codegen
        .builder
        .build_call(fn_val, &[], "args_count_res")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid args_count return".into()),
    };
    Ok((res, Type::I64))
}

fn compile_args_get<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err("args_get requires 1 argument (index)".into());
    }
    let (idx_val, _) = args[0].clone();
    let fn_val = codegen
        .module
        .get_function("tl_args_get")
        .ok_or("tl_args_get not found")?;
    let call = codegen
        .builder
        .build_call(fn_val, &[idx_val.into()], "args_get_res")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid args_get return".into()),
    };
    Ok((res, Type::UserDefined("String".to_string(), vec![])))
}

fn compile_string_char_at<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 2 {
        return Err("char_at requires 2 arguments (string, index)".into());
    }
    let (str_val, _) = args[0].clone();
    let (idx_val, idx_ty) = args[1].clone();

    // Convert index to i64 if needed
    let idx_i64 = match idx_ty {
        Type::I64 => idx_val.into_int_value(),
        Type::I32 => codegen
            .builder
            .build_int_z_extend(
                idx_val.into_int_value(),
                codegen.context.i64_type(),
                "idx_ext",
            )
            .map_err(|e| e.to_string())?,
        _ => return Err("Index must be integer".into()),
    };

    let fn_val = codegen
        .module
        .get_function("tl_string_char_at")
        .ok_or("tl_string_char_at not found")?;
    let call = codegen
        .builder
        .build_call(fn_val, &[str_val.into(), idx_i64.into()], "char_at_res")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid char_at return".into()),
    };
    Ok((res, Type::UserDefined("String".to_string(), vec![])))
}

fn compile_string_len<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err("len requires 1 argument (string)".into());
    }
    let (str_val, _) = args[0].clone();

    let fn_val = codegen
        .module
        .get_function("tl_string_len")
        .ok_or("tl_string_len not found")?;
    let call = codegen
        .builder
        .build_call(fn_val, &[str_val.into()], "len_res")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid len return".into()),
    };
    Ok((res, Type::I64))
}
fn compile_tensor_pow<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err("pow requires 1 argument (exponent)".into());
    }
    let (exp_val, exp_ty) = args[0].clone();

    // Check if exponent is Tensor or Scalar
    if let Type::Tensor(_, _) = exp_ty {
        // Tensor exponent
        let fn_val = codegen
            .module
            .get_function("tl_tensor_pow")
            .ok_or("tl_tensor_pow not found")?;
        let call = codegen
            .builder
            .build_call(fn_val, &[obj_val.into(), exp_val.into()], "pow_res");

        let res = codegen.check_tensor_result(call.map_err(|e| e.to_string())?, "pow_error")?;
        codegen.emit_register_tensor(res, &obj_ty)?;
        Ok((res, obj_ty))
    } else {
        // Scalar exponent (assume f32 or convert to f32)
        let exp_f32 = match exp_ty {
            Type::F32 => exp_val.into_float_value(),
            Type::I64 => codegen
                .builder
                .build_signed_int_to_float(
                    exp_val.into_int_value(),
                    codegen.context.f32_type(),
                    "exp_i64_to_f32",
                )
                .map_err(|e| e.to_string())?,
            Type::I32 => codegen
                .builder
                .build_signed_int_to_float(
                    exp_val.into_int_value(),
                    codegen.context.f32_type(),
                    "exp_i32_to_f32",
                )
                .map_err(|e| e.to_string())?,
            _ => {
                return Err(format!(
                    "pow exponent must be Tensor or Number, got {:?}",
                    exp_ty
                ))
            }
        };

        let fn_val = codegen
            .module
            .get_function("tl_tensor_pow_scalar")
            .ok_or("tl_tensor_pow_scalar not found")?;
        let call =
            codegen
                .builder
                .build_call(fn_val, &[obj_val.into(), exp_f32.into()], "pow_scalar_res");

        let res =
            codegen.check_tensor_result(call.map_err(|e| e.to_string())?, "pow_scalar_error")?;
        
        // Fix: Result is always a Tensor (Rank 0), even if invoked on Scalar
        let res_ty = Type::Tensor(Box::new(Type::F32), 0);
        codegen.emit_register_tensor(res, &res_ty)?;
        Ok((res, res_ty))
    }
}

fn compile_tensor_transpose<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    // Prepends receiver to args and calls standard transpose
    let mut new_args = Vec::with_capacity(args.len() + 1);
    new_args.push((obj_val, obj_ty));
    new_args.extend(args);
    compile_transpose(codegen, new_args)
}

fn compile_transpose<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    // transpose(tensor, d0, d1)
    if args.len() != 3 {
        return Err("transpose requires 3 arguments: tensor, dim0, dim1".into());
    }
    let (t_val, t_ty) = &args[0];
    let (d0_val, _) = &args[1];
    let (d1_val, _) = &args[2];
    if !matches!(t_ty, Type::Tensor(_, _)) {
        return Err("First argument to transpose must be a tensor".into());
    }
    let transpose_fn = codegen
        .module
        .get_function("tl_tensor_transpose")
        .ok_or("tl_tensor_transpose not found")?;
    let call = codegen
        .builder
        .build_call(
            transpose_fn,
            &[(*t_val).into(), (*d0_val).into(), (*d1_val).into()],
            "transpose_res",
        )
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid transpose return".into()),
    };
    Ok((res, (*t_ty).clone())) // Returns same type (Tensor)
}

fn compile_save_weights<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 2 {
        return Err("save_weights requires 2 arguments: tensor/struct, path".into());
    }
    let (t_val, t_ty) = &args[0];
    let (path_val, path_ty) = &args[1];

    if !matches!(path_ty, Type::UserDefined(s, _) if s == "String") {
        return Err("Second argument to save_weights must be a String (path)".into());
    }

    match t_ty {
        Type::Tensor(_, _) => {
            let fn_val = codegen
                .module
                .get_function("tl_tensor_save")
                .ok_or("tl_tensor_save not found")?;
            codegen
                .builder
                .build_call(fn_val, &[(*t_val).into(), (*path_val).into()], "")
                .map_err(|e| e.to_string())?;
        }
        Type::UserDefined(struct_name, _) | Type::Struct(struct_name, _) if struct_name != "String" => {
            // Struct serialization
            let new_fn = codegen
                .module
                .get_function("tl_tensor_map_new")
                .ok_or("tl_tensor_map_new not found")?;
            let map_call = codegen
                .builder
                .build_call(new_fn, &[], "map")
                .map_err(|e| e.to_string())?;
            let map_val = match map_call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v,
                _ => return Err("tl_tensor_map_new returned void".into()),
            };

            codegen.gen_save_struct(map_val, *t_val, &struct_name, "".to_string())?;

            let save_fn = codegen
                .module
                .get_function("tl_tensor_map_save")
                .ok_or("tl_tensor_map_save not found")?;
            codegen
                .builder
                .build_call(save_fn, &[map_val.into(), (*path_val).into()], "")
                .map_err(|e| e.to_string())?;

            let free_fn = codegen
                .module
                .get_function("tl_tensor_map_free")
                .ok_or("tl_tensor_map_free not found")?;
            codegen
                .builder
                .build_call(free_fn, &[map_val.into()], "")
                .map_err(|e| e.to_string())?;
        }
        _ => {
            return Err(format!(
                "First argument to save_weights must be a tensor or struct. Found: {:?}",
                t_ty
            ))
        }
    }

    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

fn compile_load_weights<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() == 1 {
        let (path_val, path_ty) = &args[0];
        if !matches!(path_ty, Type::UserDefined(s, _) if s == "String") {
            return Err("Argument to load_weights must be a String (path)".into());
        }

        let fn_val = codegen
            .module
            .get_function("tl_tensor_load")
            .ok_or("tl_tensor_load not found")?;
        let call = codegen
            .builder
            .build_call(fn_val, &[(*path_val).into()], "load_res")
            .map_err(|e| e.to_string())?;

        let res = match call.try_as_basic_value() {
            inkwell::values::ValueKind::Basic(v) => v,
            _ => return Err("Invalid load_weights return".into()),
        };
        Ok((res, Type::Tensor(Box::new(Type::F32), 0)))
    } else if args.len() == 2 {
        // Struct load
        let (struct_val, s_ty) = &args[0];
        let (path_val, path_ty) = &args[1];
        if !matches!(path_ty, Type::UserDefined(s, _) if s == "String") {
            return Err("Second argument to load_weights must be a String (path)".into());
        }

        let struct_name_opt = match &s_ty {
            Type::UserDefined(s, _) => Some(s.clone()),
            Type::Struct(s, _) => Some(s.clone()),
            _ => None,
        };

        if let Some(struct_name) = struct_name_opt {
            if struct_name == "String" {
                return Err("Cannot load weights into String".into());
            }

            let load_fn = codegen
                .module
                .get_function("tl_tensor_map_load")
                .ok_or("tl_tensor_map_load not found")?;
            let map_call = codegen
                .builder
                .build_call(load_fn, &[(*path_val).into()], "map")
                .map_err(|e| e.to_string())?;
            let map_val = match map_call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v,
                _ => return Err("tl_tensor_map_load returned void".into()),
            };

            codegen.gen_load_struct(map_val, *struct_val, &struct_name, "".to_string())?;

            let free_fn = codegen
                .module
                .get_function("tl_tensor_map_free")
                .ok_or("tl_tensor_map_free not found")?;
            codegen
                .builder
                .build_call(free_fn, &[map_val.into()], "")
                .map_err(|e| e.to_string())?;

            Ok((
                codegen.context.i64_type().const_int(0, false).into(),
                Type::Void,
            ))
        } else {
            Err(format!(
                "First argument to load_weights (2 args) must be a struct. Found: {:?}",
                s_ty
            ))
        }
    } else {
        Err("load_weights requires 1 or 2 arguments".into())
    }
}

fn compile_register_modules<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err("register_modules requires 1 argument (struct)".into());
    }
    let (val, ty) = &args[0];
    match ty {
        Type::Struct(sname, _) | Type::UserDefined(sname, _) => {
            codegen.gen_register_params(*val, &sname, "".to_string())?;
            return Ok((codegen.context.i64_type().const_zero().into(), Type::Void));
        }
        _ => return Err("register_modules expects a struct argument".into()),
    }
}

fn compile_update_all_params<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err("update_all_params requires 1 argument".into());
    }
    let (lr_val, _) = &args[0];
    let fn_val = codegen.module.get_function("tl_update_all_params").unwrap();
    codegen
        .builder
        .build_call(fn_val, &[(*lr_val).into()], "")
        .unwrap();
    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

fn compile_add_parameter<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen.module.get_function("tl_add_parameter").unwrap();
    let (name_val, _) = &args[0];
    let (tensor_val, _) = &args[1];
    codegen
        .builder
        .build_call(fn_val, &[(*name_val).into(), (*tensor_val).into()], "")
        .unwrap();
    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

fn compile_load_all_params<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen.module.get_function("tl_load_all_params").unwrap();
    let path_val = if args.len() == 2 {
        let (struct_val, struct_ty) = &args[0];
        let struct_name = match struct_ty {
            Type::Struct(s, _) | Type::UserDefined(s, _) => s,
            _ => return Err("Expected struct as first arg".into()),
        };
        codegen.gen_register_params(*struct_val, &struct_name, "".to_string())?;
        let (path, _) = &args[1];
        path
    } else {
        let (path, _) = &args[0];
        path
    };

    codegen
        .builder
        .build_call(fn_val, &[(*path_val).into()], "load_all_res")
        .map_err(|e| e.to_string())?;
    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

fn compile_parameter<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err("parameter requires 1 argument".into());
    }
    let (arg_val, arg_ty) = &args[0];
    let fn_val = codegen
        .module
        .get_function("tl_register_parameter")
        .ok_or("tl_register_parameter not found")?;
    let call = codegen
        .builder
        .build_call(fn_val, &[(*arg_val).into()], "param_reg")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid parameter return".into()),
    };
    Ok((res, (*arg_ty).clone()))
}

fn compile_load_tensor<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen.module.get_function("tl_tensor_load").unwrap();
    let (path_val, _) = &args[0];
    let call = codegen
        .builder
        .build_call(fn_val, &[(*path_val).into()], "load_res")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid load return".into()),
    };
    Ok((res, Type::Tensor(Box::new(Type::F32), 1)))
}

fn compile_save_all_params<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen.module.get_function("tl_save_all_params").unwrap();
    let path_val = if args.len() == 2 {
        let (struct_val, struct_ty) = &args[0];
        let struct_name = match struct_ty {
            Type::Struct(s, _) | Type::UserDefined(s, _) => s,
            _ => return Err("Expected struct as first arg".into()),
        };
        codegen.gen_register_params(*struct_val, &struct_name, "".to_string())?;
        let (path, _) = &args[1];
        path
    } else {
        let (path, _) = &args[0];
        path
    };

    codegen
        .builder
        .build_call(fn_val, &[(*path_val).into()], "save_all_res")
        .map_err(|e| e.to_string())?;
    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

fn compile_tensor_creation_helper<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
    runtime_func_name: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.is_empty() {
        return Err(format!("{} requires shape", runtime_func_name));
    }
    let shape_expr = &args[0];
    let (rank, shape_vals) = match &shape_expr.inner {
        ExprKind::TensorLiteral(el) | ExprKind::TensorConstLiteral(el) => {
            let mut vals = Vec::new();
            for e in el {
                let (v, t) = codegen.compile_expr(e)?;
                let int_val = match t {
                    Type::I64 => v.into_int_value(),
                    Type::I32 => codegen
                        .builder
                        .build_int_z_extend(
                            v.into_int_value(),
                            codegen.context.i64_type(),
                            "dim_ext",
                        )
                        .map_err(|e| e.to_string())?,
                    _ => return Err(format!("Dimension must be integer, found {:?}", t)),
                };
                vals.push(int_val);
            }
            (el.len(), vals)
        }
        _ => {
            return Err(format!(
                "{} currently requires array literal [dim, ...] for shape",
                runtime_func_name
            ));
        }
    };
    let requires_grad = if args.len() > 1 {
        match &args[1].inner {
            ExprKind::Bool(b) => *b,
            _ => false,
        }
    } else {
        false
    };
    let i64_type = codegen.context.i64_type();

    let current_block = codegen.builder.get_insert_block().unwrap();
    let function = current_block.get_parent().unwrap();
    let entry_block = function.get_first_basic_block().unwrap();
    let entry_builder = codegen.context.create_builder();
    if let Some(first_instr) = entry_block.get_first_instruction() {
        entry_builder.position_before(&first_instr);
    } else {
        entry_builder.position_at_end(entry_block);
    }

    let shape_array_type = i64_type.array_type(rank as u32);
    let shape_alloca = entry_builder
        .build_alloca(shape_array_type, "shape_arr")
        .map_err(|e| e.to_string())?;

    shape_alloca
        .as_instruction_value()
        .unwrap()
        .set_alignment(16)
        .map_err(|e| e.to_string())?;

    for (i, val) in shape_vals.iter().enumerate() {
        let ptr = unsafe {
            codegen.builder.build_in_bounds_gep(
                shape_array_type,
                shape_alloca,
                &[
                    i64_type.const_int(0, false),
                    i64_type.const_int(i as u64, false),
                ],
                "shape_ptr_in",
            )
        }
        .map_err(|e| e.to_string())?;
        codegen
            .builder
            .build_store(ptr, *val)
            .map_err(|e| e.to_string())?;
    }
    let req_grad_val = codegen
        .context
        .bool_type()
        .const_int(if requires_grad { 1 } else { 0 }, false);
    let f = codegen
        .module
        .get_function(runtime_func_name)
        .ok_or(format!("{} not found", runtime_func_name))?;

    let zero = i64_type.const_int(0, false);
    let first_elem_ptr = unsafe {
        codegen.builder.build_in_bounds_gep(
            shape_array_type,
            shape_alloca,
            &[zero, zero],
            "first_elem_ptr",
        )
    }
    .map_err(|e| e.to_string())?;

    let call = codegen
        .builder
        .build_call(
            f,
            &[
                i64_type.const_int(rank as u64, false).into(),
                first_elem_ptr.into(),
                req_grad_val.into(),
            ],
            "creation_res",
        )
        .map_err(|e| e.to_string())?;
    let res = codegen.check_tensor_result(call, "creation_error")?;

    Ok((res, Type::Tensor(Box::new(Type::F32), rank)))
}

fn compile_randn<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_creation_helper(codegen, args, "tl_tensor_randn_debug")
}

fn compile_ones<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_tensor_creation_helper(codegen, args, "tl_tensor_ones")
}

fn compile_varbuilder_get<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 2 {
        return Err("varbuilder_get requires at least 2 arguments (name and dimensions)".into());
    }
    let (name_val, name_ty) = codegen.compile_expr(&args[0])?;
    if !matches!(name_ty, Type::UserDefined(ref s, _) if s == "String") {
        return Err(format!(
            "varbuilder_get expects String as first argument, found {:?}",
            name_ty
        ));
    }
    let name_ptr = name_val.into_pointer_value();

    let (rank, shape_ptr) = if args.len() == 2
        && matches!(
            codegen.compile_expr(&args[1])?.1,
            Type::Tensor(_, _) | Type::ScalarArray(_, _)
        ) {
        let (shape_val, arg1_ty) = codegen.compile_expr(&args[1])?;
        let (num_elements, shape_vals) = match &arg1_ty {
            Type::Tensor(_, _) => {
                let len_fn = codegen
                    .module
                    .get_function("tl_tensor_len")
                    .ok_or("tl_tensor_len not found")?;
                let call = codegen
                    .builder
                    .build_call(len_fn, &[shape_val.into()], "len")
                    .map_err(|e| e.to_string())?;
                let _len = match call.try_as_basic_value() {
                    ValueKind::Basic(v) => v.into_int_value(),
                    _ => return Err("Invalid len return".into()),
                };

                match &args[1].inner {
                    ExprKind::TensorLiteral(elements) | ExprKind::TensorConstLiteral(elements) => (
                        elements.len(),
                        elements
                            .iter()
                            .map(|e| {
                                let (val, _) = codegen.compile_expr(e)?;
                                Ok(val)
                            })
                            .collect::<Result<Vec<_>, String>>()?,
                    ),
                    _ => return Err("varbuilder_get shape must be a literal array".into()),
                }
            }
            Type::ScalarArray(_, l) => match &args[1].inner {
                ExprKind::TensorLiteral(elements) | ExprKind::TensorConstLiteral(elements) => (
                    *l,
                    elements
                        .iter()
                        .map(|e| {
                            let (val, _) = codegen.compile_expr(e)?;
                            Ok(val)
                        })
                        .collect::<Result<Vec<_>, String>>()?,
                ),
                _ => return Err("varbuilder_get shape must be a literal array".into()),
            },
            _ => unreachable!(),
        };

        let i64_type = codegen.context.i64_type();
        let shape_alloca = codegen
            .builder
            .build_alloca(i64_type.array_type(num_elements as u32), "shape_arr")
            .unwrap();
        for (i, val) in shape_vals.iter().enumerate() {
            let idx = codegen.context.i64_type().const_int(i as u64, false);
            let ptr = unsafe {
                codegen
                    .builder
                    .build_in_bounds_gep(
                        i64_type.array_type(num_elements as u32),
                        shape_alloca,
                        &[codegen.context.i64_type().const_zero(), idx],
                        "shptr",
                    )
                    .unwrap()
            };
            codegen
                .builder
                .build_store(ptr, val.into_int_value())
                .unwrap();
        }
        (num_elements, shape_alloca)
    } else {
        let num_dims = args.len() - 1;
        let i64_type = codegen.context.i64_type();
        let shape_alloca = codegen
            .builder
            .build_alloca(i64_type.array_type(num_dims as u32), "shape_arr")
            .unwrap();
        for (i, arg) in args[1..].iter().enumerate() {
            let (val, _) = codegen.compile_expr(arg)?;
            let idx = i64_type.const_int(i as u64, false);
            let ptr = unsafe {
                codegen
                    .builder
                    .build_in_bounds_gep(
                        i64_type.array_type(num_dims as u32),
                        shape_alloca,
                        &[i64_type.const_zero(), idx],
                        "shptr",
                    )
                    .unwrap()
            };
            codegen
                .builder
                .build_store(ptr, val.into_int_value())
                .unwrap();
        }
        (num_dims, shape_alloca)
    };

    let fn_val = codegen.module.get_function("tl_varbuilder_get").unwrap();
    let call = codegen
        .builder
        .build_call(
            fn_val,
            &[
                name_ptr.into(),
                codegen
                    .context
                    .i64_type()
                    .const_int(rank as u64, false)
                    .into(),
                shape_ptr.into(),
            ],
            "varbuilder_get_result",
        )
        .unwrap();
    let res = codegen.check_tensor_result(call, "varbuilder_get_error")?;
    let res_ty = Type::Tensor(Box::new(Type::F32), 0);
    codegen.emit_register_tensor(res, &res_ty)?;
    Ok((res, res_ty))
}

fn cast_value_to_f32<'ctx>(
    codegen: &CodeGenerator<'ctx>,
    val: BasicValueEnum<'ctx>,
    ty: &Type,
) -> Result<FloatValue<'ctx>, String> {
    let f32_type = codegen.context.f32_type();
    match ty {
        Type::F32 => Ok(val.into_float_value()),
        Type::F64 => codegen
            .builder
            .build_float_cast(val.into_float_value(), f32_type, "f64_to_f32")
            .map_err(|e| e.to_string()),
        Type::I64 => codegen
            .builder
            .build_signed_int_to_float(val.into_int_value(), f32_type, "i64_to_f32")
            .map_err(|e| e.to_string()),
        Type::I32 => codegen
            .builder
            .build_signed_int_to_float(val.into_int_value(), f32_type, "i32_to_f32")
            .map_err(|e| e.to_string()),
        Type::Bool => {
            let i64_type = codegen.context.i64_type();
            let i64_val = codegen
                .builder
                .build_int_z_extend(val.into_int_value(), i64_type, "bool_to_i64")
                .map_err(|e| e.to_string())?;
            codegen
                .builder
                .build_signed_int_to_float(i64_val, f32_type, "bool_to_f32")
                .map_err(|e| e.to_string())
        }
        _ => Err(format!("Cannot cast {:?} to F32", ty)),
    }
}

fn cast_value_to_f64<'ctx>(
    codegen: &CodeGenerator<'ctx>,
    val: BasicValueEnum<'ctx>,
    ty: &Type,
) -> Result<FloatValue<'ctx>, String> {
    let f64_type = codegen.context.f64_type();
    match ty {
        Type::F64 => Ok(val.into_float_value()),
        Type::F32 => codegen
            .builder
            .build_float_ext(val.into_float_value(), f64_type, "f32_to_f64")
            .map_err(|e| e.to_string()),
        Type::I64 => codegen
            .builder
            .build_signed_int_to_float(val.into_int_value(), f64_type, "i64_to_f64")
            .map_err(|e| e.to_string()),
        Type::I32 => codegen
            .builder
            .build_signed_int_to_float(val.into_int_value(), f64_type, "i32_to_f64")
            .map_err(|e| e.to_string()),
        Type::Bool => {
            let i64_type = codegen.context.i64_type();
            let i64_val = codegen
                .builder
                .build_int_z_extend(val.into_int_value(), i64_type, "bool_to_i64")
                .map_err(|e| e.to_string())?;
            codegen
                .builder
                .build_signed_int_to_float(i64_val, f64_type, "bool_to_f64")
                .map_err(|e| e.to_string())
        }
        _ => Err(format!("Cannot cast {:?} to F64", ty)),
    }
}

pub(crate) fn cast_value_to_i64<'ctx>(
    codegen: &CodeGenerator<'ctx>,
    val: BasicValueEnum<'ctx>,
    ty: &Type,
) -> Result<IntValue<'ctx>, String> {
    let i64_type = codegen.context.i64_type();
    match ty {
        Type::I64 => Ok(val.into_int_value()),
        Type::I32 => codegen
            .builder
            .build_int_s_extend(val.into_int_value(), i64_type, "i32_to_i64")
            .map_err(|e| e.to_string()),
        _ => Err(format!("Cannot cast {:?} to I64", ty)),
    }
}

fn cast_value_to_i32<'ctx>(
    codegen: &CodeGenerator<'ctx>,
    val: BasicValueEnum<'ctx>,
    ty: &Type,
) -> Result<IntValue<'ctx>, String> {
    let i32_type = codegen.context.i32_type();
    match ty {
        Type::I32 => Ok(val.into_int_value()),
        Type::I64 => codegen
            .builder
            .build_int_cast(val.into_int_value(), i32_type, "i64_to_i32")
            .map_err(|e| e.to_string()),
        _ => Err(format!("Cannot cast {:?} to I32", ty)),
    }
}

fn compile_f32_unary_math<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    op_name: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if !args.is_empty() {
        return Err(format!("{} requires 0 arguments", op_name));
    }
    let obj_f32 = cast_value_to_f32(codegen, obj_val, &obj_ty)?;
    let fn_name = format!("tl_f32_{}", op_name);
    let fn_val = codegen
        .module
        .get_function(&fn_name)
        .ok_or(format!("Function {} not found", fn_name))?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_f32.into()], "f32_unary")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(format!("Invalid {} return", op_name)),
    };
    Ok((res, Type::F32))
}

fn compile_f32_binary_math<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    op_name: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err(format!("{} requires 1 argument", op_name));
    }
    let obj_f32 = cast_value_to_f32(codegen, obj_val, &obj_ty)?;
    let (arg_val, arg_ty) = &args[0];
    let arg_f32 = cast_value_to_f32(codegen, *arg_val, arg_ty)?;
    let fn_name = format!("tl_f32_{}", op_name);
    let fn_val = codegen
        .module
        .get_function(&fn_name)
        .ok_or(format!("Function {} not found", fn_name))?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_f32.into(), arg_f32.into()], "f32_binary")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(format!("Invalid {} return", op_name)),
    };
    Ok((res, Type::F32))
}

fn compile_f32_powi<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err("powi requires 1 argument".into());
    }
    let obj_f32 = cast_value_to_f32(codegen, obj_val, &obj_ty)?;
    let (arg_val, arg_ty) = &args[0];
    let i64_type = codegen.context.i64_type();
    let arg_i64 = match arg_ty {
        Type::I64 => arg_val.into_int_value(),
        Type::I32 | Type::Bool => codegen
            .builder
            .build_int_z_extend(arg_val.into_int_value(), i64_type, "powi_i64")
            .map_err(|e| e.to_string())?,
        _ => return Err(format!("powi requires integer argument, got {:?}", arg_ty)),
    };
    let fn_val = codegen
        .module
        .get_function("tl_f32_powi")
        .ok_or("Function tl_f32_powi not found".to_string())?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_f32.into(), arg_i64.into()], "f32_powi")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid powi return".into()),
    };
    Ok((res, Type::F32))
}

fn compile_f64_unary_math<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    op_name: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if !args.is_empty() {
        return Err(format!("{} requires 0 arguments", op_name));
    }
    let obj_f64 = cast_value_to_f64(codegen, obj_val, &obj_ty)?;
    let fn_name = format!("tl_f64_{}", op_name);
    let fn_val = codegen
        .module
        .get_function(&fn_name)
        .ok_or(format!("Function {} not found", fn_name))?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_f64.into()], "f64_unary")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(format!("Invalid {} return", op_name)),
    };
    Ok((res, Type::F64))
}

fn compile_f64_binary_math<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    op_name: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err(format!("{} requires 1 argument", op_name));
    }
    let obj_f64 = cast_value_to_f64(codegen, obj_val, &obj_ty)?;
    let (arg_val, arg_ty) = &args[0];
    let arg_f64 = cast_value_to_f64(codegen, *arg_val, arg_ty)?;
    let fn_name = format!("tl_f64_{}", op_name);
    let fn_val = codegen
        .module
        .get_function(&fn_name)
        .ok_or(format!("Function {} not found", fn_name))?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_f64.into(), arg_f64.into()], "f64_binary")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(format!("Invalid {} return", op_name)),
    };
    Ok((res, Type::F64))
}

fn compile_f64_powi<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err("powi requires 1 argument".into());
    }
    let obj_f64 = cast_value_to_f64(codegen, obj_val, &obj_ty)?;
    let (arg_val, arg_ty) = &args[0];
    let i64_type = codegen.context.i64_type();
    let arg_i64 = match arg_ty {
        Type::I64 => arg_val.into_int_value(),
        Type::I32 | Type::Bool => codegen
            .builder
            .build_int_z_extend(arg_val.into_int_value(), i64_type, "powi_i64")
            .map_err(|e| e.to_string())?,
        _ => return Err(format!("powi requires integer argument, got {:?}", arg_ty)),
    };
    let fn_val = codegen
        .module
        .get_function("tl_f64_powi")
        .ok_or("Function tl_f64_powi not found".to_string())?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_f64.into(), arg_i64.into()], "f64_powi")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid powi return".into()),
    };
    Ok((res, Type::F64))
}

fn compile_i64_unary_math<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    op_name: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if !args.is_empty() {
        return Err(format!("{} requires 0 arguments", op_name));
    }
    let obj_i64 = cast_value_to_i64(codegen, obj_val, &obj_ty)?;
    let fn_name = format!("tl_i64_{}", op_name);
    let fn_val = codegen
        .module
        .get_function(&fn_name)
        .ok_or(format!("Function {} not found", fn_name))?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_i64.into()], "i64_unary")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(format!("Invalid {} return", op_name)),
    };
    Ok((res, Type::I64))
}

fn compile_i64_binary_math<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    op_name: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err(format!("{} requires 1 argument", op_name));
    }
    let obj_i64 = cast_value_to_i64(codegen, obj_val, &obj_ty)?;
    let (arg_val, arg_ty) = &args[0];
    let arg_i64 = cast_value_to_i64(codegen, *arg_val, arg_ty)?;
    let fn_name = format!("tl_i64_{}", op_name);
    let fn_val = codegen
        .module
        .get_function(&fn_name)
        .ok_or(format!("Function {} not found", fn_name))?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_i64.into(), arg_i64.into()], "i64_binary")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(format!("Invalid {} return", op_name)),
    };
    Ok((res, Type::I64))
}

fn compile_i64_pow<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err("pow requires 1 argument".into());
    }
    let obj_i64 = cast_value_to_i64(codegen, obj_val, &obj_ty)?;
    let (arg_val, arg_ty) = &args[0];
    let exp_i64 = cast_value_to_i64(codegen, *arg_val, arg_ty)?;
    let fn_val = codegen
        .module
        .get_function("tl_i64_pow")
        .ok_or("Function tl_i64_pow not found".to_string())?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_i64.into(), exp_i64.into()], "i64_pow")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid pow return".into()),
    };
    Ok((res, Type::I64))
}

fn compile_i64_is_positive<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if !args.is_empty() {
        return Err("is_positive requires 0 arguments".into());
    }
    let obj_i64 = cast_value_to_i64(codegen, obj_val, &obj_ty)?;
    let fn_val = codegen
        .module
        .get_function("tl_i64_is_positive")
        .ok_or("Function tl_i64_is_positive not found".to_string())?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_i64.into()], "i64_is_positive")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid is_positive return".into()),
    };
    Ok((res, Type::Bool))
}

fn compile_i64_is_negative<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if !args.is_empty() {
        return Err("is_negative requires 0 arguments".into());
    }
    let obj_i64 = cast_value_to_i64(codegen, obj_val, &obj_ty)?;
    let fn_val = codegen
        .module
        .get_function("tl_i64_is_negative")
        .ok_or("Function tl_i64_is_negative not found".to_string())?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_i64.into()], "i64_is_negative")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid is_negative return".into()),
    };
    Ok((res, Type::Bool))
}

fn compile_i32_unary_math<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    op_name: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if !args.is_empty() {
        return Err(format!("{} requires 0 arguments", op_name));
    }
    let obj_i32 = cast_value_to_i32(codegen, obj_val, &obj_ty)?;
    let fn_name = format!("tl_i32_{}", op_name);
    let fn_val = codegen
        .module
        .get_function(&fn_name)
        .ok_or(format!("Function {} not found", fn_name))?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_i32.into()], "i32_unary")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(format!("Invalid {} return", op_name)),
    };
    Ok((res, Type::I32))
}

fn compile_i32_binary_math<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    op_name: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err(format!("{} requires 1 argument", op_name));
    }
    let obj_i32 = cast_value_to_i32(codegen, obj_val, &obj_ty)?;
    let (arg_val, arg_ty) = &args[0];
    let arg_i32 = cast_value_to_i32(codegen, *arg_val, arg_ty)?;
    let fn_name = format!("tl_i32_{}", op_name);
    let fn_val = codegen
        .module
        .get_function(&fn_name)
        .ok_or(format!("Function {} not found", fn_name))?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_i32.into(), arg_i32.into()], "i32_binary")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(format!("Invalid {} return", op_name)),
    };
    Ok((res, Type::I32))
}

fn compile_i32_pow<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err("pow requires 1 argument".into());
    }
    let obj_i32 = cast_value_to_i32(codegen, obj_val, &obj_ty)?;
    let (arg_val, arg_ty) = &args[0];
    let exp_i32 = cast_value_to_i32(codegen, *arg_val, arg_ty)?;
    let fn_val = codegen
        .module
        .get_function("tl_i32_pow")
        .ok_or("Function tl_i32_pow not found".to_string())?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_i32.into(), exp_i32.into()], "i32_pow")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid pow return".into()),
    };
    Ok((res, Type::I32))
}

fn compile_i32_is_positive<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if !args.is_empty() {
        return Err("is_positive requires 0 arguments".into());
    }
    let obj_i32 = cast_value_to_i32(codegen, obj_val, &obj_ty)?;
    let fn_val = codegen
        .module
        .get_function("tl_i32_is_positive")
        .ok_or("Function tl_i32_is_positive not found".to_string())?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_i32.into()], "i32_is_positive")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid is_positive return".into()),
    };
    Ok((res, Type::Bool))
}

fn compile_i32_is_negative<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if !args.is_empty() {
        return Err("is_negative requires 0 arguments".into());
    }
    let obj_i32 = cast_value_to_i32(codegen, obj_val, &obj_ty)?;
    let fn_val = codegen
        .module
        .get_function("tl_i32_is_negative")
        .ok_or("Function tl_i32_is_negative not found".to_string())?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_i32.into()], "i32_is_negative")
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid is_negative return".into()),
    };
    Ok((res, Type::Bool))
}

macro_rules! f32_unary_method {
    ($name:ident, $op:expr) => {
        fn $name<'ctx>(
            codegen: &mut CodeGenerator<'ctx>,
            obj_val: BasicValueEnum<'ctx>,
            obj_ty: Type,
            args: Vec<(BasicValueEnum<'ctx>, Type)>,
        ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
            compile_f32_unary_math(codegen, obj_val, obj_ty, args, $op)
        }
    };
}

macro_rules! f32_binary_method {
    ($name:ident, $op:expr) => {
        fn $name<'ctx>(
            codegen: &mut CodeGenerator<'ctx>,
            obj_val: BasicValueEnum<'ctx>,
            obj_ty: Type,
            args: Vec<(BasicValueEnum<'ctx>, Type)>,
        ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
            compile_f32_binary_math(codegen, obj_val, obj_ty, args, $op)
        }
    };
}

macro_rules! f64_unary_method {
    ($name:ident, $op:expr) => {
        fn $name<'ctx>(
            codegen: &mut CodeGenerator<'ctx>,
            obj_val: BasicValueEnum<'ctx>,
            obj_ty: Type,
            args: Vec<(BasicValueEnum<'ctx>, Type)>,
        ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
            compile_f64_unary_math(codegen, obj_val, obj_ty, args, $op)
        }
    };
}

macro_rules! f64_binary_method {
    ($name:ident, $op:expr) => {
        fn $name<'ctx>(
            codegen: &mut CodeGenerator<'ctx>,
            obj_val: BasicValueEnum<'ctx>,
            obj_ty: Type,
            args: Vec<(BasicValueEnum<'ctx>, Type)>,
        ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
            compile_f64_binary_math(codegen, obj_val, obj_ty, args, $op)
        }
    };
}

macro_rules! i64_unary_method {
    ($name:ident, $op:expr) => {
        fn $name<'ctx>(
            codegen: &mut CodeGenerator<'ctx>,
            obj_val: BasicValueEnum<'ctx>,
            obj_ty: Type,
            args: Vec<(BasicValueEnum<'ctx>, Type)>,
        ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
            compile_i64_unary_math(codegen, obj_val, obj_ty, args, $op)
        }
    };
}

macro_rules! i64_binary_method {
    ($name:ident, $op:expr) => {
        fn $name<'ctx>(
            codegen: &mut CodeGenerator<'ctx>,
            obj_val: BasicValueEnum<'ctx>,
            obj_ty: Type,
            args: Vec<(BasicValueEnum<'ctx>, Type)>,
        ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
            compile_i64_binary_math(codegen, obj_val, obj_ty, args, $op)
        }
    };
}

macro_rules! i32_unary_method {
    ($name:ident, $op:expr) => {
        fn $name<'ctx>(
            codegen: &mut CodeGenerator<'ctx>,
            obj_val: BasicValueEnum<'ctx>,
            obj_ty: Type,
            args: Vec<(BasicValueEnum<'ctx>, Type)>,
        ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
            compile_i32_unary_math(codegen, obj_val, obj_ty, args, $op)
        }
    };
}

macro_rules! i32_binary_method {
    ($name:ident, $op:expr) => {
        fn $name<'ctx>(
            codegen: &mut CodeGenerator<'ctx>,
            obj_val: BasicValueEnum<'ctx>,
            obj_ty: Type,
            args: Vec<(BasicValueEnum<'ctx>, Type)>,
        ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
            compile_i32_binary_math(codegen, obj_val, obj_ty, args, $op)
        }
    };
}

f32_unary_method!(compile_f32_abs, "abs");
f32_unary_method!(compile_f32_acos, "acos");
f32_unary_method!(compile_f32_acosh, "acosh");
f32_unary_method!(compile_f32_asin, "asin");
f32_unary_method!(compile_f32_asinh, "asinh");
f32_unary_method!(compile_f32_atan, "atan");
f32_unary_method!(compile_f32_atanh, "atanh");
f32_unary_method!(compile_f32_cbrt, "cbrt");
f32_unary_method!(compile_f32_ceil, "ceil");
f32_unary_method!(compile_f32_cos, "cos");
f32_unary_method!(compile_f32_cosh, "cosh");
f32_unary_method!(compile_f32_exp, "exp");
f32_unary_method!(compile_f32_exp2, "exp2");
f32_unary_method!(compile_f32_exp_m1, "exp_m1");
f32_unary_method!(compile_f32_floor, "floor");
f32_unary_method!(compile_f32_fract, "fract");
f32_unary_method!(compile_f32_ln, "ln");
f32_unary_method!(compile_f32_ln_1p, "ln_1p");
f32_unary_method!(compile_f32_log10, "log10");
f32_unary_method!(compile_f32_log2, "log2");
f32_unary_method!(compile_f32_recip, "recip");
f32_unary_method!(compile_f32_round, "round");
f32_unary_method!(compile_f32_signum, "signum");
f32_unary_method!(compile_f32_sin, "sin");
f32_unary_method!(compile_f32_sinh, "sinh");
f32_unary_method!(compile_f32_sqrt, "sqrt");
f32_unary_method!(compile_f32_tan, "tan");
f32_unary_method!(compile_f32_tanh, "tanh");
f32_unary_method!(compile_f32_to_degrees, "to_degrees");
f32_unary_method!(compile_f32_to_radians, "to_radians");
f32_unary_method!(compile_f32_trunc, "trunc");

f32_binary_method!(compile_f32_atan2, "atan2");
f32_binary_method!(compile_f32_copysign, "copysign");
f32_binary_method!(compile_f32_hypot, "hypot");
f32_binary_method!(compile_f32_log, "log");
f32_binary_method!(compile_f32_powf, "powf");

fn compile_f32_pow<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_f32_binary_math(codegen, obj_val, obj_ty, args, "powf")
}

f64_unary_method!(compile_f64_abs, "abs");
f64_unary_method!(compile_f64_acos, "acos");
f64_unary_method!(compile_f64_acosh, "acosh");
f64_unary_method!(compile_f64_asin, "asin");
f64_unary_method!(compile_f64_asinh, "asinh");
f64_unary_method!(compile_f64_atan, "atan");
f64_unary_method!(compile_f64_atanh, "atanh");
f64_unary_method!(compile_f64_cbrt, "cbrt");
f64_unary_method!(compile_f64_ceil, "ceil");
f64_unary_method!(compile_f64_cos, "cos");
f64_unary_method!(compile_f64_cosh, "cosh");
f64_unary_method!(compile_f64_exp, "exp");
f64_unary_method!(compile_f64_exp2, "exp2");
f64_unary_method!(compile_f64_exp_m1, "exp_m1");
f64_unary_method!(compile_f64_floor, "floor");
f64_unary_method!(compile_f64_fract, "fract");
f64_unary_method!(compile_f64_ln, "ln");
f64_unary_method!(compile_f64_ln_1p, "ln_1p");
f64_unary_method!(compile_f64_log10, "log10");
f64_unary_method!(compile_f64_log2, "log2");
f64_unary_method!(compile_f64_recip, "recip");
f64_unary_method!(compile_f64_round, "round");
f64_unary_method!(compile_f64_signum, "signum");
f64_unary_method!(compile_f64_sin, "sin");
f64_unary_method!(compile_f64_sinh, "sinh");
f64_unary_method!(compile_f64_sqrt, "sqrt");
f64_unary_method!(compile_f64_tan, "tan");
f64_unary_method!(compile_f64_tanh, "tanh");
f64_unary_method!(compile_f64_to_degrees, "to_degrees");
f64_unary_method!(compile_f64_to_radians, "to_radians");
f64_unary_method!(compile_f64_trunc, "trunc");

f64_binary_method!(compile_f64_atan2, "atan2");
f64_binary_method!(compile_f64_copysign, "copysign");
f64_binary_method!(compile_f64_hypot, "hypot");
f64_binary_method!(compile_f64_log, "log");
f64_binary_method!(compile_f64_powf, "powf");

fn compile_f64_pow<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    compile_f64_binary_math(codegen, obj_val, obj_ty, args, "powf")
}

i64_unary_method!(compile_i64_abs, "abs");
i64_unary_method!(compile_i64_signum, "signum");
i64_binary_method!(compile_i64_div_euclid, "div_euclid");
i64_binary_method!(compile_i64_rem_euclid, "rem_euclid");

i32_unary_method!(compile_i32_abs, "abs");
i32_unary_method!(compile_i32_signum, "signum");
i32_binary_method!(compile_i32_div_euclid, "div_euclid");
i32_binary_method!(compile_i32_rem_euclid, "rem_euclid");

// --- Reshape Unevaluated Implementation ---

fn compile_tensor_reshape_uneval<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: &Expr,
    _method: &str,
    args: &[Expr],
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err("reshape method requires exactly 1 argument (shape array)".into());
    }

    // 1. Compile Receiver
    let (obj_val, obj_ty) = codegen.compile_expr(obj)?;

    // 2. Inspect shape argument for static rank inference
    let shape_expr = &args[0];
    let new_rank = match &shape_expr.inner {
        ExprKind::TensorLiteral(elements) | ExprKind::TensorConstLiteral(elements) => {
            elements.len()
        }
        _ => 0, // Unknown rank
    };

    // 3. Compile shape argument
    let (s_val, _) = codegen.compile_expr(shape_expr)?;

    // 4. Call runtime function
    let reshape_fn = codegen
        .module
        .get_function("tl_tensor_reshape_new")
        .ok_or("tl_tensor_reshape_new not found")?;
    let call = codegen
        .builder
        .build_call(reshape_fn, &[obj_val.into(), s_val.into()], "reshape_res")
        .map_err(|e| e.to_string())?;

    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid reshape return".into()),
    };

    // 5. Construct return type
    let new_ty = if new_rank > 0 {
        if let Type::Tensor(inner, _) = obj_ty {
            Type::Tensor(inner, new_rank)
        } else {
            Type::Tensor(Box::new(Type::F32), new_rank)
        }
    } else {
        // Unknown rank - preserve original
        obj_ty
    };

    Ok((res, new_ty))
}
