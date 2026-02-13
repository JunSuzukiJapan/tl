use super::CodeGenerator;
use crate::compiler::ast::*;

use inkwell::values::*;
use inkwell::types::BasicType;
use std::collections::{HashMap, HashSet};










// System Static Methods
fn compile_system_method<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    fn_name: &str,
    ret_type: Type,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen.module.get_function(fn_name).ok_or(format!("{} not found", fn_name))?;
    let call = codegen.builder.build_call(fn_val, &[], "sys_call").map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err(format!("Invalid return from {}", fn_name)),
    };
    Ok((res, ret_type))
}

pub fn compile_system_memory_mb<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if !args.is_empty() { return Err("System::memory_mb takes no arguments".into()); }
    compile_system_method(codegen, "tl_get_memory_mb", Type::I64)
}

pub fn compile_system_metal_pool_bytes<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if !args.is_empty() { return Err("System::metal_pool_bytes takes no arguments".into()); }
    compile_system_method(codegen, "tl_get_metal_pool_bytes", Type::I64)
}

pub fn compile_system_metal_pool_mb<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if !args.is_empty() { return Err("System::metal_pool_mb takes no arguments".into()); }
    compile_system_method(codegen, "tl_get_metal_pool_mb", Type::I64)
}

pub fn compile_system_metal_pool_count<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if !args.is_empty() { return Err("System::metal_pool_count takes no arguments".into()); }
    compile_system_method(codegen, "tl_get_metal_pool_count", Type::I64)
}

pub fn compile_system_metal_sync<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if !args.is_empty() { return Err("System::metal_sync takes no arguments".into()); }
    compile_system_method(codegen, "tl_metal_synchronize", Type::Void)
}

pub fn compile_system_pool_count<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if !args.is_empty() { return Err("System::pool_count takes no arguments".into()); }
    compile_system_method(codegen, "tl_get_pool_count", Type::I64)
}

pub fn compile_system_refcount_count<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if !args.is_empty() { return Err("System::refcount_count takes no arguments".into()); }
    compile_system_method(codegen, "tl_get_refcount_count", Type::I64)
}

pub fn compile_system_scope_depth<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if !args.is_empty() { return Err("System::scope_depth takes no arguments".into()); }
    compile_system_method(codegen, "tl_get_scope_depth", Type::I64)
}



pub fn compile_path_exists<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 { return Err("Path::exists requires 1 argument".into()); }
    let fn_val = codegen.module.get_function("tl_path_exists").ok_or("tl_path_exists not found")?;
    let call = codegen.builder.build_call(fn_val, &[args[0].0.into()], "path_exists").map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v, // Bool return type?
        _ => return Err("Invalid return from Path::exists".into()),
    };
    // Original code checks return loop handling for Bool?
    // Line 4886 says: return Ok((res, Type::Bool));
    // So tl_path_exists returns i1 (bool) directly? 
    // Wait, checked the earlier view - yes, it returned res (v) directly as Type::Bool.
    Ok((res, Type::Bool))
}






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
    /// Signature only, no implementation. Used for semantics analysis.
    SignatureOnly,
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
    Option<&Type>, // target_type hint
) -> Result<(BasicValueEnum<'ctx>, Type), String>;

pub type StaticMethodUneval = for<'a, 'ctx> fn(
    &'a mut CodeGenerator<'ctx>,
    &[Expr], // args
    Option<&Type>, // target_type hint
) -> Result<(BasicValueEnum<'ctx>, Type), String>;

#[derive(Clone, Copy)]
pub enum StaticMethod {
    Evaluated(StaticMethodEval),
    Unevaluated(StaticMethodUneval),
    /// Signature only, no implementation. Used for semantics analysis.
    SignatureOnly,
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
        
        // Panic function - diverging, never returns
        self.register_uneval("panic", compile_panic_uneval);

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

    // codegen.emit_register_tensor(res, &obj_ty)?;
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

    // codegen.emit_register_tensor(res, &obj_ty)?;
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
    if !matches!(&dev_ty, Type::String(_)) {
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

#[allow(deprecated)]
fn compile_varbuilder_get_static<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    // Arg 0: Name (String)
    let (name_val, name_ty) = codegen.compile_expr(&args[0])?;
    let name_ptr = if let Type::String(_) = name_ty {
        // String is { i64 ptr } or similar. extraction needed.
        // Assuming String struct layout: Field 0 is ptr (i64).
        let ptr_to_struct = name_val.into_pointer_value();
        let i64_ptr_ty = codegen.context.i64_type().ptr_type(inkwell::AddressSpace::default());
        let ptr_to_first_field = codegen
            .builder
            .build_pointer_cast(ptr_to_struct, i64_ptr_ty, "str_ptr_cast")
            .unwrap();
        let str_addr_i64 = codegen
            .builder
            .build_load(codegen.context.i64_type(), ptr_to_first_field, "str_addr")
            .unwrap()
            .into_int_value();
        let i8_ptr_ty = codegen.context.i8_type().ptr_type(inkwell::AddressSpace::default());
        codegen
            .builder
            .build_int_to_ptr(str_addr_i64, i8_ptr_ty, "cstr_ptr")
            .unwrap()
    } else {
         return Err("VarBuilder::get name must be String".into());
    };

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
                name_ptr.into(),
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
    fn substitute_type_generic(&self, ty: &Type, generics: &[String], args: &[Type]) -> Type {
        match ty {
            Type::Struct(name, inner_args) => {
                 if let Some(idx) = generics.iter().position(|g| g == name) {
                     return args[idx].clone();
                 }
                // If the struct is Generic, we must substitute
                 Type::Struct(name.clone(), inner_args.iter().map(|t| self.substitute_type_generic(t, generics, args)).collect())
            }
            Type::Enum(name, inner_args) => {
                 if let Some(idx) = generics.iter().position(|g| g == name) {
                     return args[idx].clone();
                 }
                 Type::Enum(name.clone(), inner_args.iter().map(|t| self.substitute_type_generic(t, generics, args)).collect())
            }
            Type::Path(segments, inner_args) => {
                 // Check if single-segment path is a type parameter
                 if segments.len() == 1 {
                     if let Some(idx) = generics.iter().position(|g| g == &segments[0]) {
                         return args[idx].clone();
                     }
                 }
                 Type::Path(segments.clone(), inner_args.iter().map(|t| self.substitute_type_generic(t, generics, args)).collect())
            }
            Type::Tensor(inner, rank) => Type::Tensor(Box::new(self.substitute_type_generic(inner, generics, args)), *rank),
            Type::Tuple(types) => Type::Tuple(types.iter().map(|t| self.substitute_type_generic(t, generics, args)).collect()),
            Type::Ptr(inner) => Type::Ptr(Box::new(self.substitute_type_generic(inner, generics, args))),

             _ => ty.clone(),
        }
    }

    /// Parse type arguments from underscore-separated parts (e.g., ["Entry", "i64", "i64"])
    /// Handles nested generics by checking if a part is a known generic type
    fn parse_mangled_type_args(&self, parts: &[&str]) -> Vec<Type> {
        let mut result = Vec::new();
        let mut i = 0;
        while i < parts.len() {
            let s = parts[i];
            // Try primitive types first
            let ty = match s.to_lowercase().as_str() {
                "i64" => Type::I64,
                "i32" => Type::I32,
                "f32" => Type::F32,
                "f64" => Type::F64,
                "bool" => Type::Bool,
                "u8" => Type::U8,
                "u16" => Type::U16,
                "u32" => Type::U32,
                "usize" => Type::Usize,
                "string" => Type::String("String".to_string()),
                "" => { i += 1; continue; }
                _ => {
                    // Check if this is a known generic enum (like Entry)
                    if let Some(enum_def) = self.enum_defs.get(s) {
                        let arity = enum_def.generics.len();
                        if arity > 0 && i + arity < parts.len() {
                            // Consume next 'arity' parts as type arguments
                            let nested_args = self.parse_mangled_type_args(&parts[i + 1..i + 1 + arity]);
                            i += arity;
                            Type::Enum(s.to_string(), nested_args)
                        } else {
                            Type::Enum(s.to_string(), vec![])
                        }
                    // Check if this is a known generic struct
                    } else if let Some(struct_def) = self.struct_defs.get(s) {
                        let arity = struct_def.generics.len();
                        if arity > 0 && i + arity < parts.len() {
                            let nested_args = self.parse_mangled_type_args(&parts[i + 1..i + 1 + arity]);
                            i += arity;
                            Type::Struct(s.to_string(), nested_args)
                        } else {
                            Type::Struct(s.to_string(), vec![])
                        }
                    } else {
                        // Unknown type - return as struct with no args
                        Type::Struct(s.to_string(), vec![])
                    }
                }
            };
            result.push(ty);
            i += 1;
        }
        result
    }

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

    #[allow(deprecated)]
    pub fn load_struct_i64_field(
        &mut self,
        obj_val: BasicValueEnum<'ctx>,
        obj_ty: &Type,
        field_name: &str,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        let struct_name = match obj_ty {
            Type::Struct(name, _) => name.clone(),

            Type::String(_) => "String".to_string(),
            _ => return Err(format!("Expected struct type for field {} (got {:?})", field_name, obj_ty)),
        };

        let simple_struct_name = struct_name.as_str();

        if simple_struct_name == "String" {
            if field_name == "ptr" {
                 // StringStruct is { ptr: *mut c_char, len: i64 }
                 // We need to load the ptr field (index 0) as a pointer
                 let ptr = obj_val.into_pointer_value();
                 
                 // Define StringStruct layout
                 let str_struct_ty = self.context.struct_type(&[
                     self.context.ptr_type(inkwell::AddressSpace::default()).into(), // ptr
                     self.context.i64_type().into(), // len
                 ], false);
                 
                 // GEP to get pointer to ptr field (index 0)
                 let ptr_field_ptr = self.builder
                     .build_struct_gep(str_struct_ty, ptr, 0, "str_ptr_field")
                     .map_err(|_| "Failed to GEP String.ptr")?;
                 
                 // Load the ptr value as pointer type
                 let ptr_val = self.builder
                     .build_load(self.context.ptr_type(inkwell::AddressSpace::default()), ptr_field_ptr, "str_ptr_val")
                     .map_err(|e| e.to_string())?;
                 
                 // Convert pointer to i64 for callers that expect i64
                 let ptr_as_i64 = self.builder
                     .build_ptr_to_int(ptr_val.into_pointer_value(), self.context.i64_type(), "str_ptr_i64")
                     .map_err(|e| e.to_string())?;
                 
                 return Ok(ptr_as_i64.into());
            } else if field_name == "len" || field_name == "cap" {
                 // Basic String struct might have len/cap? 
                 // Current implementation seems to rely on runtime functions for len.
                 // io.rs only asks for "ptr".
                 return Err("String len/cap not directly accessible via field".into());
            }
        }

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
            Type::Struct(_, _) => {
                match &expr.inner {
                    // Fresh allocations are safe to free
                    ExprKind::StaticMethodCall(_, _, _) | ExprKind::StructInit(_, _) => true,
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

        // V4.5: Use tl_tensor_register (mapped to backend-specific implementation)
        let reg_fn = self
            .module
            .get_function("tl_tensor_register")
            .ok_or("tl_tensor_register not found")?;

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
        &mut self,
        map: inkwell::values::BasicValueEnum<'ctx>,
        struct_ptr: inkwell::values::BasicValueEnum<'ctx>,
        struct_name: &str,
        prefix: String,
    ) -> Result<(), String> {
        // Extract simple name from module path (e.g., "mnist_common::Linear" -> "Linear")
        let simple_name = struct_name;

        let def = self
            .struct_defs
            .get(simple_name)
            .ok_or(format!("Struct definition '{}' not found", struct_name))?;

        let struct_ty = *self
            .struct_types
            .get(simple_name)
            .ok_or("Struct LLVM type not found")?;

        let fields = def.fields.clone();
        for (i, (field_name, field_type)) in fields.iter().enumerate() {
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
                    let (key_val, _) = self.compile_string_literal(&full_key)
                        .map_err(|e| format!("Failed to compile key literal: {}", e))?;

                    let insert_fn = self
                        .module
                        .get_function("tl_tensor_map_insert")
                        .ok_or("tl_tensor_map_insert not found")?;
                    let _ = self
                        .builder
                        .build_call(insert_fn, &[map.into(), key_val.into(), t_val.into()], "")
                        .map_err(|e| e.to_string())?;
                }
                Type::Struct(sub_name, _) => {
                    if sub_name == "String" { panic!("Struct(String) found in codegen save"); }
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
        &mut self,
        map: inkwell::values::BasicValueEnum<'ctx>,
        struct_ptr: inkwell::values::BasicValueEnum<'ctx>,
        struct_name: &str,
        prefix: String,
    ) -> Result<(), String> {
        // Extract simple name from module path
        let simple_name = struct_name;

        let def = self
            .struct_defs
            .get(simple_name)
            .ok_or(format!("Struct definition '{}' not found", struct_name))?;

        let struct_ty = *self
            .struct_types
            .get(simple_name)
            .ok_or("Struct LLVM type not found")?;

        let fields = def.fields.clone();
        for (i, (field_name, field_type)) in fields.iter().enumerate() {
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
                    let (key_val, _) = self.compile_string_literal(&full_key)
                        .map_err(|e| format!("Failed to compile key literal: {}", e))?;

                    let get_fn = self
                        .module
                        .get_function("tl_tensor_map_get")
                        .ok_or("tl_tensor_map_get not found")?;
                    let call = self
                        .builder
                        .build_call(get_fn, &[map.into(), key_val.into()], "t_val")
                        .map_err(|e| e.to_string())?;

                    let t_val = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err("tl_tensor_map_get returned inst/void".into()),
                    };

                    self.builder
                        .build_store(field_ptr, t_val)
                        .map_err(|e| e.to_string())?;
                }
                Type::Struct(sub_name, _) => {
                    if sub_name == "String" { panic!("Struct(String) found in codegen load"); }
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
        &mut self,
        struct_ptr: inkwell::values::BasicValueEnum<'ctx>,
        struct_name: &str,
        prefix: String,
    ) -> Result<(), String> {
        let simple_name = struct_name;

        let def = self
            .struct_defs
            .get(simple_name)
            .ok_or(format!("Struct definition '{}' not found", struct_name))?;

        let struct_ty = *self
            .struct_types
            .get(simple_name)
            .ok_or("Struct LLVM type not found")?;

        let fields = def.fields.clone();
        for (i, (field_name, field_type)) in fields.iter().enumerate() {
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

                    let (key_val, _) = self.compile_string_literal(&full_key)
                        .map_err(|e| format!("Failed to compile key literal: {}", e))?;

                    let add_fn = self
                        .module
                        .get_function("tl_add_parameter")
                        .ok_or("tl_add_parameter not found")?;
                    self.builder
                        .build_call(add_fn, &[key_val.into(), t_val.into()], "")
                        .map_err(|e| e.to_string())?;
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
    pub(crate) fn is_last_use(&self, name: &str) -> bool {
        for scope in self.variable_liveness.iter().rev() {
            if let Some(&last_use) = scope.get(name) {
                // If current_time has reached the last_use time, it's a move.
                // Note: last_use is 0 if it was never used or not found in analysis.
                if last_use == 0 { return false; }
                return self.current_time >= last_use;
            }
        }
        false
    }


    fn emit_retain(&self, val: BasicValueEnum<'ctx>, ty: &Type) -> Result<(), String> {
        match ty {
            Type::Tensor(_, _) | Type::TensorShaped(_, _) => {
                // V4.5: Promote returned tensor (remove from scope)
                let promote_fn = self.module.get_function("tl_tensor_promote")
                    .ok_or("tl_tensor_promote not found")?;
                let ptr = val.into_pointer_value();
                let void_ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
                let cast_ptr = self.builder.build_pointer_cast(ptr, void_ptr_type, "cast_prom").map_err(|e| e.to_string())?;
                self.builder.build_call(promote_fn, &[cast_ptr.into()], "").map_err(|e| e.to_string())?;
            }

            Type::Struct(_, _) | Type::String(_) | Type::Enum(_, _) | Type::Path(_, _) => {
                let inc_fn = self.module.get_function("tl_ptr_inc_ref")
                    .or_else(|| {
                         let void_ty = self.context.void_type();
                         let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
                         let ft = void_ty.fn_type(&[ptr_ty.into()], false);
                         Some(self.module.add_function("tl_ptr_inc_ref", ft, None))
                    })
                    .ok_or("tl_ptr_inc_ref decl failed")?;
                let ptr = val.into_pointer_value();
                let void_ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
                let cast_ptr = self.builder.build_pointer_cast(ptr, void_ptr_type, "cast_inc").map_err(|e| e.to_string())?;
                self.builder.build_call(inc_fn, &[cast_ptr.into()], "").map_err(|e| e.to_string())?;
            }
            _ => {}
        }
        Ok(())
    }

    pub(crate) fn compile_expr(
        &mut self,
        expr: &Expr,
    ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        self.current_time += 1;
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
                    Type::Tensor(_, _) | Type::Struct(_, _) | Type::Tuple(_)
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
            ExprKind::CharLiteral(c) => {
                let i32_type = self.context.i32_type();
                Ok((i32_type.const_int(*c as u32 as u64, false).into(), Type::Char("Char".to_string())))
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
                generics,
                payload,
            } => {
                // 0. specialized name handling
                // Extract base name from mangled name (e.g., "Option_i64" -> "Option")
                // and parse generics from mangled suffix if generics is empty
                let (base_name, inferred_generics) = if generics.is_empty() && enum_name.contains('_') {
                    let parts: Vec<&str> = enum_name.splitn(2, '_').collect();
                    if parts.len() == 2 {
                        let base = parts[0].to_string();
                        // Parse ALL type suffixes (e.g., "i64_i64" -> [I64, I64])
                        let type_parts: Vec<&str> = parts[1].split('_').collect();
                        let inferred_types: Vec<Type> = type_parts.iter()
                            .filter_map(|type_part| {
                                match type_part.to_lowercase().as_str() {
                                    "i64" => Some(Type::I64),
                                    "f32" => Some(Type::F32),
                                    "f64" => Some(Type::F64),
                                    "bool" => Some(Type::Bool),
                                    "string" => Some(Type::String("String".to_string())),
                                    "" => None,  // Skip empty parts
                                    other => Some(Type::Struct(other.to_string(), vec![])),
                                }
                            })
                            .collect();
                        (base, inferred_types)
                    } else {
                        (enum_name.clone(), generics.clone())
                    }
                } else {
                    (enum_name.clone(), generics.clone())
                };
                
                let mangled_name = if inferred_generics.is_empty() {
                    enum_name.clone()
                } else {
                    self.mangle_type_name(&base_name, &inferred_generics)
                };

                // 1. On-demand Monomorphization
                // First, try to find already-monomorphized enum by mangled_name
                let mut enum_def = if let Some(def) = self.enum_defs.get(&mangled_name) {
                    def.clone()
                } else if let Some(def) = self.enum_defs.get(enum_name) {
                    // Found exact enum_name (might be mangled already)
                    def.clone()
                } else if let Some(def) = self.enum_defs.get(&base_name) {
                    def.clone()
                } else {
                    return Err(format!("Enum def {} not found (tried: {}, {}, {})", 
                        enum_name, mangled_name, base_name, enum_name));
                };
                
                // If the found enum_def is still generic, monomorphize with inferred or default types
                if !enum_def.generics.is_empty() {
                    let actual_generics = if !inferred_generics.is_empty() {
                        inferred_generics.clone()
                    } else {
                        // Default to I64 for single-param generics like Option<T>, Result<T>
                        vec![Type::I64; enum_def.generics.len()]
                    };
                    let actual_mangled = self.mangle_type_name(&base_name, &actual_generics);
                    
                    // Try to find already-monomorphized version first
                    if let Some(specialized_def) = self.enum_defs.get(&actual_mangled) {
                        enum_def = specialized_def.clone();
                    } else {
                        // Monomorphize on-demand
                        self.monomorphize_enum(&base_name, &actual_generics).map_err(|e| e.to_string())?;
                        enum_def = self.enum_defs.get(&actual_mangled)
                            .ok_or(format!("Monomorphization failed for {} -> {}", base_name, actual_mangled))?
                            .clone();
                    }
                };

                // 2. Ensure enum_type exists
                // Note: After default type monomorphization, enum_def.name may be "Option<i64>" style
                let enum_ty = if let Some(ty) = self.enum_types.get(&enum_def.name) {
                    // Use the name from enum_def (might be monomorphized name)
                    *ty
                } else if let Some(ty) = self.enum_types.get(&mangled_name) {
                    *ty
                } else if let Some(ty) = self.enum_types.get(enum_name) {
                    *ty
                } else {
                    // Need to compile this enum type on-demand
                    // This should work because enum_def has generics=[] after monomorphization
                    self.compile_enum_defs(&[enum_def.clone()])?;
                    *self.enum_types.get(&enum_def.name)
                        .ok_or(format!("Failed to compile enum type {} (from {}), generics={:?}", enum_def.name, enum_name, enum_def.generics))?
                };

                // 2. Allocate Enum
                // Manual malloc(size)
                // Use mangled type for size calculation
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
                let malloc_ptr = match self.builder
                    .build_call(malloc_fn, &[size.into()], &format!("enum_{}", mangled_name))
                    .map_err(|e| e.to_string())?
                    .try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
                        _ => return Err("malloc returned void".into()),
                    };
                // malloc_ptr is *EnumStruct
                
                // Cast to EnumType* (which is WaitStruct*)
                let alloca = self.builder.build_pointer_cast(
                    malloc_ptr,
                    self.context.ptr_type(inkwell::AddressSpace::default()),
                    "enum_cast"
                ).unwrap();

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
                
                // Helper to compile field storage
                let _compile_fields = |fields_def: &Vec<Type>, exprs: &Vec<Expr>, field_names: Option<&Vec<String>>| -> Result<(), String> {
                    let payload_ptr_raw = self
                        .builder
                        .build_struct_gep(enum_ty, alloca, 1, "payload_ptr_raw")
                        .map_err(|e| e.to_string())?;

                    // Reconstruct Variant Struct Type
                    let mut field_types: Vec<inkwell::types::BasicTypeEnum> = vec![];
                    for ty in fields_def {
                        let llvm_ty = match ty {
                            Type::F32 => self.context.f32_type().into(),
                            Type::I64 => self.context.i64_type().into(),
                            Type::Bool => self.context.bool_type().into(),
                            Type::Tensor(_, _)
                            | Type::Struct(_, _)
                            | Type::Enum(_, _)
                            | Type::String(_)
                            | Type::Tuple(_) => self
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

                    // Iterate over DEFINITION fields
                    for (idx, f_ty) in fields_def.iter().enumerate() {
                        // Find expression
                        let expr = if let Some(_names) = field_names {
                             // Struct variant: find by name
                             // exprs is actually not Vec<Expr>, but we pass it as such? 
                             // Wait, payload closure arg needs adjustment.
                             // Actually, let's handle struct/tuple separately in the match below to avoid closure complexity.
                             // Placeholder to compile
                             return Err("Struct variant not supported in closure helper".to_string());
                        } else {
                             // Tuple variant: by index
                             &exprs[idx]
                        };

                        let (val, _) = self.compile_expr(expr)?;
                        
                        // Deep clone logic
                        let is_rvalue = matches!(
                            &expr.inner,
                            ExprKind::FnCall(_, _)
                                | ExprKind::MethodCall(_, _, _)
                                | ExprKind::StaticMethodCall(_, _, _)
                                | ExprKind::BinOp(_, _, _)
                                | ExprKind::UnOp(_, _)
                                | ExprKind::TensorLiteral(_)
                                | ExprKind::Block(_)
                                | ExprKind::Int(_)
                                | ExprKind::Float(_)
                                | ExprKind::Bool(_)
                                | ExprKind::StringLiteral(_)
                        );
                        let mut stored_val = val;
                        let should_deep_clone = match f_ty {
                            Type::Tensor(_, _) => !is_rvalue,
                            Type::Struct(_, _) => !is_rvalue,
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
                    }
                    Ok(())
                };

                match (&variant_def.kind, payload) {
                     (crate::compiler::ast::VariantKind::Unit, crate::compiler::ast::EnumVariantInit::Unit) => {
                         // No fields to store
                     },
                     (crate::compiler::ast::VariantKind::Tuple(types), crate::compiler::ast::EnumVariantInit::Tuple(exprs)) => {
                        let payload_ptr_raw = self
                            .builder
                            .build_struct_gep(enum_ty, alloca, 1, "payload_ptr_raw")
                            .map_err(|e| e.to_string())?;

                        // Reconstruct Variant Struct Type
                        let mut field_types_llvm: Vec<inkwell::types::BasicTypeEnum> = vec![];
                        for ty in types {
                             let llvm_ty = match ty {
                                Type::F32 => self.context.f32_type().into(),
                                Type::I64 => self.context.i64_type().into(),
                                Type::Bool => self.context.bool_type().into(),
                                Type::Struct(_, _)
                                | Type::Enum(_, _)
                                | Type::String(_)
                                | Type::Tensor(_, _)
                                | Type::Tuple(_) => self
                                    .context
                                    .ptr_type(inkwell::AddressSpace::default())
                                    .into(),
                                _ => self.context.i64_type().into(),
                            };
                            field_types_llvm.push(llvm_ty);
                        }

                        let variant_struct_ty = self.context.struct_type(&field_types_llvm, false);

                        let payload_ptr = self
                            .builder
                            .build_pointer_cast(
                                payload_ptr_raw,
                                self.context.ptr_type(inkwell::AddressSpace::default()),
                                "payload_cast",
                            )
                            .unwrap();
                        
                        for (idx, (f_ty, expr)) in types.iter().zip(exprs.iter()).enumerate() {
                             let (val, _) = self.compile_expr(expr)?;
                             
                            // Move semantics for pointer types:
                            // When a variable is used as enum payload, we MOVE ownership.
                            let is_moveable_type = match f_ty {
                                Type::Tensor(_, _) => true,
                                Type::Struct(n, args) => {
                                    // Primitive types don't need move semantics
                                    !(args.is_empty() && 
                                      (n == "I64" || n == "F64" || n == "I32" || n == "F32" || n == "Bool" || 
                                       n == "i64" || n == "f64" || n == "i32" || n == "f32" || n == "bool"))
                                }
                                Type::Enum(_, _) | Type::Tuple(_) => true,
                                _ => false,
                            };

                            
                            // If field_expr is a Variable, mark it as moved (disable cleanup)
                            if is_moveable_type {
                                if let ExprKind::Variable(var_name) = &expr.inner {
                                    for scope in self.variables.iter_mut().rev() {
                                        if let Some((_, _, cleanup_mode)) = scope.get_mut(var_name) {
                                            *cleanup_mode = crate::compiler::codegen::CLEANUP_NONE;
                                            break;
                                        }
                                    }
                                }
                            }
                            
                            let f_ptr = self.builder.build_struct_gep(variant_struct_ty, payload_ptr, idx as u32, "").map_err(|e| e.to_string())?;
                            self.builder.build_store(f_ptr, val).unwrap();
                        }

                     },
                     (crate::compiler::ast::VariantKind::Struct(fields_def), crate::compiler::ast::EnumVariantInit::Struct(exprs)) => {
                        let payload_ptr_raw = self
                            .builder
                            .build_struct_gep(enum_ty, alloca, 1, "payload_ptr_raw")
                            .map_err(|e| e.to_string())?;
                        
                        let mut field_types_llvm: Vec<inkwell::types::BasicTypeEnum> = vec![];
                        for (_, ty) in fields_def {
                             let llvm_ty = match ty {
                                Type::F32 => self.context.f32_type().into(),
                                Type::I64 => self.context.i64_type().into(),
                                Type::Bool => self.context.bool_type().into(),
                                Type::Tensor(_, _)
                                | Type::Struct(_, _)
                                | Type::Enum(_, _) => self
                                    .context
                                    .ptr_type(inkwell::AddressSpace::default())
                                    .into(),
                                Type::String(_) => self
                                    .context
                                    .ptr_type(inkwell::AddressSpace::default())
                                    .into(),
                                Type::Char(_) => self.context.i32_type().into(),
                                _ => self.context.i64_type().into(),
                            };
                            field_types_llvm.push(llvm_ty);
                        }
                        let variant_struct_ty = self.context.struct_type(&field_types_llvm, false);

                        let payload_ptr = self
                            .builder
                            .build_pointer_cast(
                                payload_ptr_raw,
                                self.context.ptr_type(inkwell::AddressSpace::default()),
                                "payload_cast",
                            )
                            .unwrap();
                            
                        for (idx, (f_name, f_ty)) in fields_def.iter().enumerate() {
                             let (_, expr) = exprs.iter().find(|(n, _)| n == f_name).ok_or(format!("Missing field {}", f_name))?;
                             
                             let (val, _) = self.compile_expr(expr)?;
                             
                            let is_rvalue = matches!(
                                &expr.inner,
                                ExprKind::FnCall(_, _)
                                    | ExprKind::MethodCall(_, _, _)
                                    | ExprKind::StaticMethodCall(_, _, _)
                                    | ExprKind::BinOp(_, _, _)
                                    | ExprKind::UnOp(_, _)
                                    | ExprKind::TensorLiteral(_)
                                    | ExprKind::Block(_)
                                    | ExprKind::Int(_)
                                    | ExprKind::Float(_)
                                    | ExprKind::Bool(_)
                                    | ExprKind::StringLiteral(_)
                            );
                            let mut stored_val = val;
                            let should_deep_clone = match f_ty {
                                Type::Tensor(_, _) => !is_rvalue,
                                Type::Struct(_, _) => !is_rvalue,
                                _ => false,
                            };
                            if should_deep_clone {
                                 stored_val = self.emit_deep_clone(val, f_ty)?;
                            }
                            
                            let f_ptr = self.builder.build_struct_gep(variant_struct_ty, payload_ptr, idx as u32, "").map_err(|e| e.to_string())?;
                            self.builder.build_store(f_ptr, stored_val).unwrap();
                        }
                     },
                     _ => return Err(format!("Mismatch between variant definition {:?} and init payload {:?}", variant_def.kind, payload)),
                }

                // Return the monomorphized type so downstream processing (like MethodCall) 
                // gets the correct generics. After monomorphization, enum_def.generics is empty
                // and enum_def.name contains the full name like "Option<i64>".
                // For proper type matching, we return (enum_def.name, []) as generics are baked into the name.
                Ok((alloca.into(), Type::Enum(enum_def.name.clone(), vec![])))
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
                    (Type::I64, Type::F64) => {
                        let i = val.into_int_value();
                        let f = self
                            .builder
                            .build_signed_int_to_float(i, self.context.f64_type(), "cast")
                            .map_err(|e| e.to_string())?;
                        Ok((f.into(), Type::F64))
                    }
                    (Type::F64, Type::I64) => {
                        let f = val.into_float_value();
                        let i = self
                            .builder
                            .build_float_to_signed_int(f, self.context.i64_type(), "cast")
                            .map_err(|e| e.to_string())?;
                        Ok((i.into(), Type::I64))
                    }
                    (Type::F32, Type::F64) => {
                        let f = val.into_float_value();
                        let f64_val = self
                            .builder
                            .build_float_ext(f, self.context.f64_type(), "cast")
                            .map_err(|e| e.to_string())?;
                        Ok((f64_val.into(), Type::F64))
                    }
                    (Type::F64, Type::F32) => {
                        let f = val.into_float_value();
                        let f32_val = self
                            .builder
                            .build_float_trunc(f, self.context.f32_type(), "cast")
                            .map_err(|e| e.to_string())?;
                        Ok((f32_val.into(), Type::F32))
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
                    // Int to Ptr
                    (Type::I64, Type::Ptr(_)) => {
                        let i = val.into_int_value();
                        let ptr = self.builder.build_int_to_ptr(i, self.context.ptr_type(inkwell::AddressSpace::default()), "cast_int_to_ptr").map_err(|e| e.to_string())?;
                        Ok((ptr.into(), target_type.clone()))
                    }
                    // Ptr to Ptr (Bitcast)
                    (Type::Ptr(_), Type::Ptr(_)) => {
                        if val.is_pointer_value() {
                             let ptr = val.into_pointer_value();
                             let new_ptr = self.builder.build_pointer_cast(ptr, self.context.ptr_type(inkwell::AddressSpace::default()), "cast_ptr_ptr").map_err(|e| e.to_string())?;
                             Ok((new_ptr.into(), target_type.clone()))
                        } else if val.is_int_value() {
                             // Handle case where Ptr is represented as I64 (e.g. malloc return if signature mismatch?)
                             let i = val.into_int_value();
                             let new_ptr = self.builder.build_int_to_ptr(i, self.context.ptr_type(inkwell::AddressSpace::default()), "cast_int_ptr").map_err(|e| e.to_string())?;
                             Ok((new_ptr.into(), target_type.clone()))
                        } else {
                             return Err(format!("Invalid value kind for Ptr cast: {:?}", val));
                        }
                    }
                    // Integer Casts
                    (Type::I64, Type::I32) => {
                         let i = val.into_int_value();
                         let t = self.builder.build_int_cast(i, self.context.i32_type(), "cast_i64_i32").map_err(|e| e.to_string())?;
                         Ok((t.into(), Type::I32))
                    }
                    (Type::I32, Type::I64) => {
                         let i = val.into_int_value();
                         let t = self.builder.build_int_z_extend(i, self.context.i64_type(), "cast_i32_i64").map_err(|e| e.to_string())?;
                         Ok((t.into(), Type::I64))
                    }
                    // U8 Casts
                    (Type::I64, Type::Struct(name, _)) | (Type::I32, Type::Struct(name, _)) if name == "u8" => {
                         let i = val.into_int_value();
                         let truncated = self.builder.build_int_cast(i, self.context.i8_type(), "cast_u8").map_err(|e| e.to_string())?;
                         Ok((truncated.into(), target_type.clone()))
                    }
                    (Type::F32, Type::Struct(name, _)) if name == "u8" => {
                         let f = val.into_float_value();
                         let i = self.builder.build_float_to_unsigned_int(f, self.context.i8_type(), "cast_u8").map_err(|e| e.to_string())?;
                         Ok((i.into(), target_type.clone()))
                    }
                    // u8 -> Int Casts
                    (Type::Struct(name, _), Type::I64) if name == "u8" => {
                         let i = val.into_int_value(); // i8
                         let extended = self.builder.build_int_z_extend(i, self.context.i64_type(), "cast_u8_i64").map_err(|e| e.to_string())?;
                         Ok((extended.into(), Type::I64))
                    }
                     (Type::Struct(name, _), Type::I32) if name == "u8" => {
                         let i = val.into_int_value(); // i8
                         let extended = self.builder.build_int_z_extend(i, self.context.i32_type(), "cast_u8_i32").map_err(|e| e.to_string())?;
                         Ok((extended.into(), Type::I32))
                    }
                     (Type::I64, Type::U8) | (Type::I32, Type::U8) => {
                         let i = val.into_int_value();
                         let truncated = self.builder.build_int_cast(i, self.context.i8_type(), "cast_u8").map_err(|e| e.to_string())?;
                         Ok((truncated.into(), Type::U8))
                    }
                    (Type::F32, Type::U8) => {
                         let f = val.into_float_value();
                         let i = self.builder.build_float_to_unsigned_int(f, self.context.i8_type(), "cast_u8").map_err(|e| e.to_string())?;
                         Ok((i.into(), Type::U8))
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

                // Auto-dereference Ref types - REMOVED (Ref not in spec)
                // while let Type::Ref(inner) = obj_ty.clone() {
                //     let ptr = obj_val.into_pointer_value();
                //     let loaded = self.builder.build_load(
                //         self.get_llvm_type(&inner)?,
                //         ptr,
                //         "deref"
                //     ).map_err(|e| e.to_string())?;
                //     obj_val = loaded.into();
                //     obj_ty = *inner;
                // }
                
                // Determine struct name and generic args
                // Note: Vec_u8 etc. may come as Enum due to monomorphize.rs conversion
                // Treat such types as Struct for field access purposes
                // Normalize obj_ty to resolve Path types in generic args
                let normalized_obj_ty = self.normalize_type(&obj_ty);

                let (base_name, generic_args) = match &normalized_obj_ty {
                    Type::Struct(name, args) => (name.clone(), args.clone()),
                    Type::Enum(name, args) => {
                        // Workaround: Some types like Vec_u8 are incorrectly classified as Enum
                        // If it has struct_defs entry, treat it as struct
                        (name.clone(), args.clone())
                    }
                    _ => return Err(format!("Field access on non-struct type {:?}", obj_ty)),
                };


                // Determine mangled name for lookup
                let mangled_name = if generic_args.is_empty() {
                    base_name.clone()
                } else {
                    self.mangle_type_name(&base_name, &generic_args)
                };

                let simple_struct_name = mangled_name.clone();

                /*
                Logic:
                1. Try looking up exact specialized name (e.g. "MyStruct_i64").
                2. If not found, try base name (e.g. "MyStruct") effectively falling back to generic definition.
                   If base name is found, we must substitutions generic params in the field type.
                */
                
                let (struct_def, is_generic_base) = if let Some(def) = self.struct_defs.get(&simple_struct_name) {
                    (def, false)
                } else if let Some(def) = self.struct_defs.get(&base_name) {
                    (def, true)
                } else {
                    // Try extracting base name from underscore-mangled name (e.g., "Vec_u8" -> "Vec")
                    let underscore_base = base_name.split('_').next().unwrap_or(&base_name).to_string();
                    if let Some(def) = self.struct_defs.get(&underscore_base) {
                        (def, true)
                    } else {
                        return Err(format!("Struct definition for {} not found (checked {}, {}, {})", 
                            base_name, simple_struct_name, base_name, underscore_base));
                    }
                };

                let field_idx = struct_def
                    .fields
                    .iter()
                    .position(|(n, _)| n == field)
                    .ok_or_else(|| {

                        format!(
                        "Field {} not found in struct {}. Available: {:?}",
                        field, base_name, struct_def.fields.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>()
                    )})?;
                
                // Retrieve field type and substitute if necessary
                let (_, field_ty_raw) = &struct_def.fields[field_idx];
                
                // If generic_args is empty but we have a mangled name, extract generics from it
                let effective_generic_args = if generic_args.is_empty() && base_name.contains('_') {
                    let parts: Vec<&str> = base_name.split('_').collect();
                    if parts.len() > 1 {
                        self.parse_mangled_type_args(&parts[1..])
                    } else {
                        generic_args.clone()
                    }
                } else {
                    generic_args.clone()
                };
                
                let field_ty = if is_generic_base && !effective_generic_args.is_empty() {
                     self.substitute_type_generic(field_ty_raw, &struct_def.generics, &effective_generic_args)
                } else {
                     field_ty_raw.clone()
                };
                
                // Continue with loading... use simple_struct_name or base_name for struct type lookup?
                // self.struct_types stores llvm type.
                // If we are using generic base, "Vec_i64" might NOT be in struct_types if it wasn't compiled.
                // But monomorphize_method usually generates it?
                // If it isn't in struct_defs, maybe it isn't in struct_types either?
                // If so, we need to register it?
                // Or maybe Vec is Opaque Pointer in struct_types?
                
                // For Vec, we know it translates to i8*.
                // But for general structs, we need correct offset.
                // If we use Generic definition, we assume layout matches logic.
                // For Vec, fields are fixed.
                
                // If struct_types doesn't have specialized name, try base name?
                // BUT LLVM types for specialized structs might differ if fields differ (size).
                // For Vec, size is fixed (ptr, i64, i64).
                // So base struct name in struct_types should be sufficient if registered?
                // CodeGenerator registers structs in init.
                
                let st_llvm_ty = if let Some(t) = self.struct_types.get(&simple_struct_name) {
                     *t
                } else if let Some(t) = self.struct_types.get(&base_name) {
                     *t 
                } else {
                    // Try underscore-based base name (e.g., "Vec_u8" -> "Vec")
                    // and monomorphize on-demand if generics can be inferred
                    let underscore_base = base_name.split('_').next().unwrap_or(&base_name).to_string();
                    if let Some(t) = self.struct_types.get(&underscore_base) {
                        *t
                    } else if !generic_args.is_empty() {
                        // Try to monomorphize on-demand

                        match self.monomorphize_struct(&underscore_base, &generic_args) {
                            Ok(t) => {

                                t
                            }
                            Err(_) => {
                                // Last resort: try to infer generics from underscore suffix
                                let parts: Vec<&str> = base_name.split('_').collect();
                                if parts.len() >= 2 {
                                    let inferred_ty = match parts[1].to_lowercase().as_str() {
                                        "u8" => Type::U8,
                                        "i64" => Type::I64,
                                        "f32" => Type::F32,
                                        "f64" => Type::F64,
                                        "string" => Type::String("String".to_string()),
                                        _ => return Err(format!("LLVM struct type for {} not found", base_name)),
                                    };
                                    self.monomorphize_struct(&underscore_base, &[inferred_ty])
                                        .map_err(|e| format!("Failed to monomorphize {}: {}", base_name, e))?
                                } else {
                                    return Err(format!("LLVM struct type for {} not found", base_name));
                                }
                            }
                        }
                    } else {
                        // generic_args is empty but name is mangled (e.g., Vec_Entry_i64_i64)
                        // For Vec, HashMap, etc., the LLVM layout is fixed regardless of generic args
                        // So we can use the base type's LLVM layout
                        if underscore_base == "Vec" || underscore_base == "HashMap" || underscore_base == "Option" || underscore_base == "Result" {
                            // Try to get base type, then monomorphize with i64 as fallback
                            if let Some(t) = self.struct_types.get(&underscore_base) {
                                *t
                            } else {
                                self.monomorphize_struct(&underscore_base, &[Type::I64])
                                    .map_err(|e| format!("Failed to monomorphize {} for FieldAccess: {}", underscore_base, e))?
                            }
                        } else {
                            // Try to infer generics from underscore suffix (e.g., "Vec_u8" -> Vec<u8>)
                            let parts: Vec<&str> = base_name.split('_').collect();
                            if parts.len() >= 2 {
                                let inferred_ty = match parts[1].to_lowercase().as_str() {
                                    "u8" => Type::U8,
                                    "i64" => Type::I64,
                                    "f32" => Type::F32,
                                    "f64" => Type::F64,
                                    "string" => Type::String("String".to_string()),
                                    _ => return Err(format!("LLVM struct type for {} not found (tried {}, {}, {})", 
                                        base_name, simple_struct_name, base_name, underscore_base)),
                                };
                                self.monomorphize_struct(&underscore_base, &[inferred_ty])
                                    .map_err(|e| format!("Failed to monomorphize {}: {}", base_name, e))?
                            } else {
                                return Err(format!("LLVM struct type for {} not found (tried {}, {}, {})", 
                                    base_name, simple_struct_name, base_name, underscore_base));
                            }
                        }
                    }
                };


                if obj_val.is_pointer_value() {
                    let ptr = obj_val.into_pointer_value();
                    // let st_llvm_ty = self.struct_types.get(simple_struct_name).unwrap(); // Handled above

                    let field_ptr = self
                        .builder
                        .build_struct_gep(
                            st_llvm_ty,
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
                        Type::String(_) => self
                            .context
                            .ptr_type(inkwell::AddressSpace::default())
                            .into(),
                        Type::Char(_) | Type::I32 => self.context.i32_type().into(),
                        Type::Struct(_, _) 
                        | Type::Enum(_, _)
                        | Type::Ptr(_) // FIX: Handle Ptr fields
                        | Type::Tuple(_) => self
                            .context
                            .ptr_type(inkwell::AddressSpace::default())
                            .into(),
                        _ => self.context.i64_type().into(), // Placeholder
                    };

                    let loaded = self
                        .builder
                        .build_load(llvm_ty, field_ptr, field)
                        .map_err(|e| e.to_string())?;
                    self.emit_retain(loaded, &field_ty)?; // FIX: Acquire ownership
                    Ok((loaded, field_ty.clone()))
                } else if obj_val.is_struct_value() {
                    let struct_val = obj_val.into_struct_value();
                    let extracted = self
                        .builder
                        .build_extract_value(struct_val, field_idx as u32, field)
                        .map_err(|e| e.to_string())?;
                    self.emit_retain(extracted, &field_ty)?; // FIX: Acquire ownership
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

                            let llvm_ty = self.get_llvm_type(ty).map_err(|e| e.to_string())?;

                            let loaded = self
                                .builder
                                .build_load(llvm_ty, ptr, name)
                                .map_err(|e| e.to_string())?;

                            // FIX: Must retain ownership for the new variable reference
                            // Otherwise `let v2 = v` creates a second owner without IncRef -> Double Free.
                            self.emit_retain(loaded, ty)?;

                            return Ok((loaded, ty.clone()));
                        } else {
                            // Regular Variable - Retain ownership for caller
                            self.emit_retain(*val, ty)?; 
                            return Ok((*val, ty.clone()));
                        }
                    }
                }
                Err(format!("Variable {} not found in scopes", name))
            }
            ExprKind::StructInit(ty, fields) => {
                 // Normalize Path types to Struct/Enum
                 let normalized_ty = self.normalize_type(ty);
                 let (name, generics) = match &normalized_ty {
                      Type::Struct(name, generics) => (name.clone(), generics.clone()),
                      Type::Enum(name, generics) => (name.clone(), generics.clone()), // Enums might use struct-init syntax?
                      _ => panic!("StructInit type must be Struct or Enum (after normalization), found {:?} (original: {:?})", normalized_ty, ty),
                 };
                 self.compile_struct_init(&name, &generics, fields)
            },
            ExprKind::StaticMethodCall(type_ty, method_name, args) => {
                if method_name == "sizeof" {
                     // For Enum types, we need to get the actual data struct size, not pointer size
                     if let Type::Enum(enum_name, generics) = type_ty {
                         // Try direct lookup first, then mangled name if generics present
                         let lookup_name = if generics.is_empty() {
                             enum_name.clone()
                         } else if self.enum_types.contains_key(enum_name) {
                             enum_name.clone()
                         } else {
                             self.mangle_type_name(enum_name, generics)
                         };
                         
                         // Look up the actual LLVM struct type from enum_types
                         if let Some(enum_struct_type) = self.enum_types.get(&lookup_name) {
                             let size_val = enum_struct_type.size_of().ok_or(format!("Enum type {} has no size", lookup_name))?;
                             return Ok((size_val.into(), Type::I64));
                         } else {
                             return Err(format!("Enum type {} not found in enum_types for sizeof", lookup_name));
                         }
                     }
                     
                     // For Struct types, also check if it's actually an enum (mangled name)
                     if let Type::Struct(name, _) = type_ty {
                         if let Some(enum_struct_type) = self.enum_types.get(name) {
                             // It's actually an enum with a Struct type wrapper
                             let size_val = enum_struct_type.size_of().ok_or(format!("Enum type {} has no size", name))?;
                             return Ok((size_val.into(), Type::I64));
                         }
                     }
                     
                     // Generic T already substituted by Monomorphizer
                     let llvm_ty = self.get_llvm_type(type_ty).map_err(|e| e.to_string())?;
                     let size_val = llvm_ty.size_of().ok_or(format!("Type {:?} has no size (ZST not supported)", type_ty))?;
                     // Cast to i64 if needed? IntValue is generic, but usually i64 for size_t on 64bit
                     // LLVM size_of returns integer type matching target's pointer width.
                     // Our Type::I64 expects LLVM i64.
                     return Ok((size_val.into(), Type::I64));
                 }


                let struct_name = match type_ty {
                    Type::Struct(name, _) => name,
                    Type::Enum(name, _) => name,
                    Type::F32 => "F32",
                    Type::I64 => "I64",
                    Type::Bool => "Bool",
                    Type::String(_) => "String",
                    // Add other types as needed or implement a helper
                    Type::Tensor(_, _) => "Tensor",
                    Type::Path(segments, _) => segments.last().map(|s| s.as_str()).unwrap_or("Unknown"),
                    _ => return Err(format!("Cannot call static method on type {:?}", type_ty)),
                };
                self.compile_static_method_call(struct_name, method_name, args, &type_ty)
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

                        // FIX: テンソル方程式の中間変数 _comp_res_N の所有権を呼び出し元に移転。
                        // tl_tensor_acquire は no-op のため、_comp_res_N と代入先変数が同じポインタを共有する。
                        // 両方が CLEANUP_FULL のままだと exit_scope で二重解放が発生する。
                        if let Some(scope) = self.variables.last_mut() {
                            if let Some(entry) = scope.get_mut(&temp_name) {
                                entry.2 = super::CLEANUP_NONE;
                            }
                        }

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
                match val_type.clone() {
                    Type::Struct(name, _) if name == "Tensor" => {
                        let rank = indices.len();
                        let i64_type = self.context.i64_type();
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

                        let get_fn_name = "tl_tensor_get_f32_md";
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
                        Ok((res, Type::F32))
                    }
                    Type::Struct(_, _) => {
                         // Generic Struct Indexing -> Desugar to .get() method call
                         self.emit_method_call(target, val, val_type, "get", indices)
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


                    Type::Ptr(inner) => {
                         let ptr = val.into_pointer_value();
                         if indices.len() != 1 {
                             return Err("Ptr indexing must be 1D".into());
                         }
                         let (idx_val, _) = self.compile_expr(&indices[0])?;
                         
                         unsafe {
                             let elem_ptr = self.builder.build_gep(
                                 self.context.ptr_type(inkwell::AddressSpace::default()),
                                 ptr,
                                 &[idx_val.into_int_value()],
                                 "ptr_idx"
                             ).map_err(|e| e.to_string())?;
                             
                             let load_ty = self.get_or_monomorphize_type(&inner)?;
                             let val = self.builder.build_load(
                                 load_ty,
                                 elem_ptr,
                                 "ptr_val"
                             ).map_err(|e| e.to_string())?;
                             
                             // If it's a struct/string, should we retain/clone?
                             // Usually reading from array is borrowing or copying.
                             // Rust: Copy type -> Copy. non-Copy -> Move.
                             // Here we implement Move (like *ptr).
                             // We do NOT emit retain/clone automatically on access?
                             // If the result is assigned to variable, assignment logic calls retain if needed?
                             // No, assignment logic calls deep clone if safe_to_free.
                             
                             Ok((val, *inner.clone()))
                         }
                    }
                    _ => Err("Index access only on Tensor or Ptr".into()),
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
                    UnOp::Ref => {
                        // Ref operator: &expr
                        // Always get address. Do not return value directly.
                        // Reference types removed from spec - return Ptr instead
                         // For scalars (L-Value vs R-Value logic)
                        if let ExprKind::Variable(name) = &expr.inner {
                            // Variable: Get address from variables map
                             for scope in self.variables.iter().rev() {
                                if let Some((var_val, _, _)) = scope.get(name) {
                                     // var_val is the Alloca (Pointer) - return as Ptr type
                                     return Ok((var_val.as_basic_value_enum(), Type::Ptr(Box::new(ty))));
                                }
                             }
                             return Err(format!("Variable {} not found for Ref", name));
                        } else {
                            // R-Value: Store to temp, return address as Ptr
                            let current_block = self.builder.get_insert_block().unwrap();
                            let func = current_block.get_parent().unwrap();
                            let alloca = self.create_entry_block_alloca(func, "ref_tmp", &ty)?;
                            
                            self.builder.build_store(alloca, val).map_err(|e| e.to_string())?;
                            return Ok((alloca.into(), Type::Ptr(Box::new(ty))));
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
                } else if cond_val.is_pointer_value() {
                     // Condition is a Tensor (PointerValue). Extract boolean value.
                     // It is likely a Scalar Tensor from a comparison (e.g. gt, lt).
                     // use tl_tensor_item_i64 to extract value (0 or 1)
                     let item_fn = self.module.get_function("tl_tensor_item_i64").or_else(|| self.module.get_function("tl_tensor_item_f32").map(|f| {
                         // Fallback? f32 compare != 0.0?
                         // Ideally comparison returns i64-backed scalar tensor (0 or 1).
                         // Let's assume tl_tensor_item_i64 works for now or if missing try to cast.
                         // But wait, explicit check for tensor type was done via item_fn earlier?
                         // The panic says: cond_val is PointerValue.
                         // The unwrapped item_fn above was for "if cond_val is tensor but identified elsewhere?"
                         // Wait, let's look at lines 3046-3049 which I didn't see yet.
                         f
                     })).ok_or("Runtime function tl_tensor_item_i64 not found to resolve Tensor boolean condition")?;
                     
                     let call = self.builder.build_call(item_fn, &[cond_val.into()], "cond_item").map_err(|e| e.to_string())?;
                     let item_val = match call.try_as_basic_value() {
                          inkwell::values::ValueKind::Basic(v) => v,
                          _ => return Err("Expected basic value from tl_tensor_item_i64".into()),
                     };
                     
                     // item_i64 returns i64. Comparison 0 or 1.
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
                    Type::Tensor(_, _) | Type::Struct(_, _) | Type::Tuple(_) | Type::String(_)
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
                        // FIX: Mark the temporary as CLEANUP_NONE in the compiler's temporaries list.
                        // This prevents emit_cleanup_vars_in_scope from calling emit_recursive_free.
                        self.mark_temp_no_cleanup(then_final_val);
                        
                        // Also call runtime UNREGISTER to remove from memory manager.
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
                    Type::Tensor(_, _) | Type::Struct(_, _) | Type::Tuple(_) | Type::String(_)
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
                        // FIX: Mark the temporary as CLEANUP_NONE in the compiler's temporaries list.
                        self.mark_temp_no_cleanup(else_final_val);
                        
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
                        | Type::String(_)
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
                            | Type::String(_)
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
                        } else if let Type::Struct(_, _) | Type::String(_) = &res_ty {
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
        // Early detection: If `name` is already a mangled name that exists in struct_types,
        // use it directly and ignore generics (avoids double-mangling)
        if !generics.is_empty() && name.contains('_') {
            if let Some(&existing_type) = self.struct_types.get(name) {
                let struct_def = self.struct_defs.get(name)
                    .ok_or_else(|| format!("Struct definition {} not found", name))?
                    .clone();
                return self.compile_struct_alloc(name, &[], &existing_type, &struct_def, fields);
            }
        }
        
        if !generics.is_empty() {
             // Generate mangled name first
             let mangled_name = self.mangle_type_name(name, generics);
             
             // Try to get existing or monomorphize on-demand
             let struct_type = if let Some(t) = self.struct_types.get(&mangled_name) {
                 *t
             } else {
                 // Attempt monomorphization
                 if self.monomorphize_struct(name, generics).is_ok() {
                     // Try again after monomorphization
                     if let Some(t) = self.struct_types.get(&mangled_name) {
                         *t
                     } else {
                         return Err(format!("Struct type {} not found after monomorphization", mangled_name));
                     }
                 } else {
                     // Recovery for double-mangled names (e.g. HashMap_i64_i64 -> HashMap)
                     let def_names: Vec<String> = self.struct_defs.keys().cloned().collect();
                      
                     let mut recovered = false;
                     for def_name in def_names {
                         if name.starts_with(&def_name) && name != def_name {
                             if self.monomorphize_struct(&def_name, generics).is_ok() {
                                 recovered = true;
                                 break;
                             }
                         }
                     }
                     
                     if recovered {
                         if let Some(t) = self.struct_types.get(&mangled_name) {
                             *t
                         } else {
                             // Try with base name mangled
                             let base = name.split('_').next().unwrap_or(name);
                             let base_mangled = self.mangle_type_name(base, generics);
                             *self.struct_types.get(&base_mangled)
                                 .ok_or(format!("Struct type {} not found (tried {} and {})", name, mangled_name, base_mangled))?
                         }
                     } else {
                         return Err(format!("Monomorphization failed for {} with generics {:?}", name, generics));
                     }
                 }
             };
             
             let struct_def = self.struct_defs.get(&mangled_name)
                 .or_else(|| {
                     // Try base name mangled
                     let base = name.split('_').next().unwrap_or(name);
                     let base_mangled = self.mangle_type_name(base, generics);
                     self.struct_defs.get(&base_mangled)
                 })
                 .ok_or(format!("Struct definition {} not found", mangled_name))?
                 .clone();
             
             return self.compile_struct_alloc(name, generics, &struct_type, &struct_def, fields);
        }

        // Non-generic case
        let lookup_name = name.to_string();
        
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
            
        self.compile_struct_alloc(name, generics, &struct_type, &struct_def, fields)
    }

    // Refactored helper to allow calling from recovery path
    fn compile_struct_alloc(
        &mut self,
        _original_name: &str,
        generics: &[Type],
        struct_type: &inkwell::types::StructType<'ctx>,
        struct_def: &crate::compiler::ast::StructDef,
        fields: &[(String, Expr)],
    ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        let name = &struct_def.name; // Use resolved name


        // Determine allocation strategy: Arena or Heap
        let size = struct_type
            .size_of()
            .ok_or(format!("Cannot determine size of struct {}", name))?;

        // ZST Optimization (PhantomData etc): Return NULL, not an aggregate value.
        // The Runtime handles NULL pointers gracefully (ignores them).
        if struct_def.fields.is_empty() {
            let null_ptr = self.context.ptr_type(inkwell::AddressSpace::default()).const_null();
            return Ok((null_ptr.into(), Type::Struct(name.to_string(), generics.to_vec())));
        }

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
            
            // Move Semantics for pointer types:
            // When a variable is used as a struct field initializer, we MOVE ownership.
            // This means:
            // 1. If it's a temporary: remove from cleanup list (already done by try_consume_temp)
            // 2. If it's a variable: set its cleanup_mode to CLEANUP_NONE (ownership transferred)
            
            let moved = self.try_consume_temp(val);
            
            // Check if field_expr is a direct variable access - if so, mark it as moved
            let is_moveable_type = matches!(
                _ty,
                Type::Tensor(_, _) | Type::Struct(_, _) | Type::Tuple(_) | Type::Enum(_, _)
            );
            
            if !moved && is_moveable_type {
                // If field_expr is a Variable, we should transfer ownership (move semantics)
                // by disabling cleanup for the source variable
                if let ExprKind::Variable(var_name) = &field_expr.inner {
                    // Find the variable in scope and set cleanup_mode to CLEANUP_NONE
                    for scope in self.variables.iter_mut().rev() {
                        if let Some((_, _, cleanup_mode)) = scope.get_mut(var_name) {
                            *cleanup_mode = crate::compiler::codegen::CLEANUP_NONE;
                            break;
                        }
                    }
                }
            }

            let field_ptr = self
                .builder
                .build_struct_gep(
                    *struct_type,
                    struct_ptr,
                    field_idx as u32,
                    &format!("{}.{}", name, field_name),
                )
                .map_err(|e| e.to_string())?;

            // Store the value directly (move semantics - no deep clone needed since we're transferring ownership)
            // For scalar types, just store the value.
            // For pointer types (Tensor, Struct, etc.), store the pointer - ownership is transferred.
            self.builder
                .build_store(field_ptr, val)
                .map_err(|e| e.to_string())?;
        }


        // Return the pointer directly (no load)
        Ok((struct_ptr.into(), Type::Struct(name.to_string(), generics.to_vec())))
    }

    pub fn compile_string_literal(&self, s: &str) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        let s_null_term = format!("{}\0", s);
        let str_val = self
            .builder
            .build_global_string_ptr(&s_null_term, "str_lit")
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

        Ok((ptr, Type::String("String".to_string())))
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
                Type::Tensor(_, _) | Type::Struct(_, _) | Type::Tuple(_) | Type::String(_)
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
                                    | Type::Tuple(_)
                                    | Type::String(_)
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
        // Extract simple name from module path
        let simple_name = name;

        let struct_type = *self
            .struct_types
            .get(simple_name)
            .ok_or(format!("Struct type {} not found in codegen", name))?;

        let struct_def = self
            .struct_defs
            .get(simple_name)
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
                Type::Tensor(_, _) | Type::Struct(_, _) | Type::Tuple(_) | Type::String(_)
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

    #[allow(deprecated)]
    pub(crate) fn compile_static_method_call(
        &mut self,
        struct_name: &str,
        method: &str,
        args: &[Expr],
        target_type: &Type,
    ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        // Compatibility aliases for existing logic
        let type_name = struct_name;

        // 0. Check if this is an Enum Variant initialization (priority check)
        //    This handles cases like Entry_i64_i64::Empty where the type_name is
        //    the mangled enum name and method is a variant name.
        if let Some(mut enum_def) = self.enum_defs.get(struct_name).cloned() {
            if let Some(variant_idx) = enum_def.variants.iter().position(|v| v.name == method) {
                // If enum_def is still generic, monomorphize with default type
                if !enum_def.generics.is_empty() {
                    let default_generics = vec![Type::I64; enum_def.generics.len()];
                    let mangled = self.mangle_type_name(struct_name, &default_generics);
                    if let Some(specialized) = self.enum_defs.get(&mangled) {
                        enum_def = specialized.clone();
                    } else {
                        self.monomorphize_enum(struct_name, &default_generics).map_err(|e| e.to_string())?;
                        enum_def = self.enum_defs.get(&mangled)
                            .ok_or(format!("Monomorphization failed for {} -> {}", struct_name, mangled))?
                            .clone();
                    }
                }
                // This is an enum variant constructor, compile it as EnumInit
                return self.compile_enum_variant_as_static_method_call(
                    &enum_def.name, method, args, variant_idx, &enum_def
                );
            }
        }

        // 1. Try TypeManager (AST-defined methods) - find matching overload
        if let Some(type_info) = self.type_manager.get_type(type_name) {
            if type_info.has_static_method(method) {
                // Get overloads and try to find a match
                if let Some(overloads) = type_info.get_static_overloads(method) {
                    // For Unevaluated methods, we need to try each overload
                    // For now, try the first unevaluated if any, or match by arg count
                    for overload in overloads {
                        if overload.arg_types.len() == args.len() {
                            match &overload.impl_fn {
                                StaticMethod::Evaluated(func) => {
                                    let func = *func;
                                    let mut compiled_args = Vec::new();
                                    for arg in args {
                                        compiled_args.push(self.compile_expr(arg)?);
                                    }
                                    return func(self, compiled_args, Some(target_type));
                                }
                                StaticMethod::Unevaluated(func) => {
                                    let func = *func;
                                    return func(self, args, Some(target_type));
                                }
                                StaticMethod::SignatureOnly => {
                                    // This method has no implementation, skip this overload
                                    continue;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // 2. Check self.static_methods (registered via register_all_methods)
        // Copy the method to avoid borrow conflict with self.compile_expr
        let method_opt = self.static_methods.get(type_name).and_then(|m| m.get(method).copied());
        if let Some(method_fn) = method_opt {
            match method_fn {
                StaticMethod::Evaluated(func) => {
                    let mut compiled_args = Vec::new();
                    for arg in args {
                        compiled_args.push(self.compile_expr(arg)?);
                    }
                    return func(self, compiled_args, Some(target_type));
                }
                StaticMethod::Unevaluated(func) => {
                    return func(self, args, Some(target_type));
                }
                StaticMethod::SignatureOnly => {
                    // Should not happen for registered methods, fall through
                }
            }
        }

        // 2.5. Built-in Map::load using tl_gguf_load
        if type_name == "Map" && method == "load" {
            if args.len() != 1 {
                return Err("Map::load requires 1 argument".into());
            }
            let (path_val, path_ty) = self.compile_expr(&args[0])?;
            
            // TL String is StringStruct { ptr: *c_char, len: i64 }
            // tl_gguf_load expects *mut StringStruct
            let string_struct_ptr = if matches!(path_ty, Type::String(_)) {
                let struct_ptr = path_val.into_pointer_value();
                struct_ptr
            } else {
                return Err(format!("Map::load expects String argument, got {:?}", path_ty));
            };
            
            let fn_val = self
                .module
                .get_function("tl_gguf_load")
                .ok_or("tl_gguf_load not found")?;
            
            // Cast to generic pointer if needed (though struct ptr usually works)
            let cast_ptr = self.builder.build_pointer_cast(
                string_struct_ptr,
                self.context.ptr_type(inkwell::AddressSpace::default()), 
                "string_struct_ptr"
            ).map_err(|e| e.to_string())?;

            let call = self
                .builder
                .build_call(fn_val, &[cast_ptr.into()], "map_load")
                .map_err(|e| e.to_string())?;
                
            let res = match call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v,
                _ => return Err("Invalid return from Map::load".into()),
            };
            return Ok((res, Type::Struct("Map".to_string(), vec![])));
        }

        // 3. User Generic Fallback (Monomorphize)
        // Normalize target_type to convert Path to Struct/Enum
        let normalized_target_type = self.normalize_type(target_type);
        let generics = match &normalized_target_type {
            Type::Struct(n, args) if n == type_name => args.clone(),
            Type::Enum(n, args) if n == type_name => args.clone(),
            _ => vec![],
        };
        
        let generic_result = if self.generic_impls.contains_key(type_name) {
             if let Ok(name) = self.monomorphize_method(type_name, method, &generics) {
                 self.module.get_function(&name).map(|f| (f, name))
             } else {
                 None
             }
        } else {
            None
        };
        
        let simple_type_name = &struct_name;
        let mangled_name = format!("tl_{}_{}", simple_type_name, method);
        let stdlib_name = format!("tl_{}_{}", simple_type_name.to_lowercase(), method);

        let (func, actual_name) = if let Some((f, name)) = generic_result {
             (f, name)
        } else if let Some(f) = self.module.get_function(&mangled_name) {
            (f, mangled_name)
        } else if let Some(f) = self.module.get_function(&stdlib_name) {
            (f, stdlib_name)
        } else if let Some(f) = self.module.get_function(method) {
            (f, method.to_string())
        } else {
            // Method not found - enum variant initialization is handled at function start
            return Err(format!(
                "Static method {}::{} not found (checked {}, {}, and {})",
                struct_name, method, mangled_name, stdlib_name, method
            ));
        };

        // 3. Generic Fallback: Compile Args & Handle SRET
        // Get return type - first check registered method_return_types, then fall back to signature inference
        // 3. Generic Fallback: Compile Args & Handle SRET
        // Get return type - first check registered method_return_types, then fall back to signature inference
        let ret_ty = if let Some(ret) = self.method_return_types.get(&actual_name) {
            ret.clone()
        } else {
            self.get_return_type_from_signature(func)
        };

        // SRET Logic
        let uses_sret = matches!(&ret_ty, Type::Struct(name, _) if name != "Tensor" && name != "String");
        let mut sret_ptr = None;

        if uses_sret {
             // OLD: Stack Allocation (alloca) -> Causes Free of Stack Pointer crash
             // NEW: Heap Allocation (malloc + register) -> Correct for RefCounted Structs
             
             // 1. Get Struct Type and Size from CodeGen struct_types map
             let (struct_name, generics) = match &ret_ty {
                 Type::Struct(n, g) => (n, g),
                 _ => return Err("SRET used on non-struct type".into()),
             };
             
             let mangled_name = if generics.is_empty() {
                 struct_name.to_string()
             } else {
                 self.mangle_type_name(struct_name, generics)
             };
             
             // Simple name lookup (as done in compile_struct_init)
             let simple_lookup_name = mangled_name.clone();

             // Try to get existing type, or monomorphize on-demand
             let struct_type = if let Some(st) = self.struct_types.get(&simple_lookup_name)
                 .or_else(|| self.enum_types.get(&simple_lookup_name)) {
                 *st
             } else {
                 // Try on-demand monomorphization
                 if !generics.is_empty() {
                     self.monomorphize_struct(struct_name, generics)?;
                 }
                 *self.struct_types.get(&simple_lookup_name)
                     .or_else(|| self.enum_types.get(&simple_lookup_name))
                     .ok_or_else(|| format!("Struct type {} not found for SRET allocation", simple_lookup_name))?
             };
             
             let size = struct_type.size_of().ok_or("Cannot determine size for SRET struct")?;
             
             // 2. Malloc
             let malloc_fn = self.module.get_function("malloc").ok_or("malloc not found")?;
             let size_i64 = self.builder.build_int_z_extend(size, self.context.i64_type(), "size_i64").unwrap();
             let call_malloc = self.builder.build_call(malloc_fn, &[size_i64.into()], "sret_malloc").map_err(|e| e.to_string())?;
             
             let raw_ptr = match call_malloc.try_as_basic_value() {
                 inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
                 _ => return Err("malloc returned void".into()),
             };
             
             // 3. Register (Initialize RefCount=1, Header)
             // Use generic name or strict name?
             let struct_name_str = match &ret_ty {
                 Type::Struct(n, _) => n.as_str(),
                 _ => "AnonymousStruct",
             };
             let name_global = self.builder.build_global_string_ptr(struct_name_str, "struct_name").unwrap();
             let register_fn = self.module.get_function("tl_mem_register_struct_named").ok_or("tl_mem_register_struct_named not found")?;
             
             let cast_ptr = self.builder.build_pointer_cast(raw_ptr, self.context.ptr_type(inkwell::AddressSpace::default()), "cast_ptr").unwrap();
             let _ = self.builder.build_call(register_fn, &[cast_ptr.into(), name_global.as_pointer_value().into()], "");

             sret_ptr = Some(cast_ptr);
        }

        let mut compiled_args = Vec::with_capacity(args.len());
        let mut compiled_args_types = Vec::with_capacity(args.len());
        for arg in args {
            let (val, ty) = self.compile_expr(arg)?;

            let (val, ty) = if type_name == "VarBuilder" {
                 if let Type::String(_) = ty {
                     // VarBuilder methods expect char* (i8*) not String struct
                     let ptr_to_struct = val.into_pointer_value();
                     let i64_ptr_ty = self.context.i64_type().ptr_type(inkwell::AddressSpace::default());
                     let ptr_to_first_field = self.builder.build_pointer_cast(ptr_to_struct, i64_ptr_ty, "str_ptr_cast").unwrap();
                     let str_addr_i64 = self.builder.build_load(self.context.i64_type(), ptr_to_first_field, "str_addr").unwrap().into_int_value();
                     let i8_ptr_ty = self.context.i8_type().ptr_type(inkwell::AddressSpace::default());
                     let char_ptr = self.builder.build_int_to_ptr(str_addr_i64, i8_ptr_ty, "cstr_ptr").unwrap();
                     (char_ptr.into(), ty)
                 } else {
                     (val, ty)
                 }
            } else {
                (val, ty)
            };
            compiled_args.push(val.into());
            compiled_args_types.push((val, ty));
        }

        // 4. Call
        if let Some(ptr) = sret_ptr {
             compiled_args.insert(0, ptr.into());
        }

        let call = self
            .builder
            .build_call(func, &compiled_args, "static_call")
            .map_err(|e| e.to_string())?;

        if let Some(ptr) = sret_ptr {
             // Return SRET pointer. The "Value" of a struct token is the pointer to its memory.
             // Loading it would unbox the content (e.g. the handle) which is wrong.
             Ok((ptr.into(), ret_ty))
        } else {
             match call.try_as_basic_value() {
                 inkwell::values::ValueKind::Basic(_) => {
                     let v = self.check_tensor_result(call, "static_call_error")?;
                     Ok((v, ret_ty))
                 }
                 _ => Ok((
                     self.context.i64_type().const_int(0, false).into(),
                     Type::Void,
                 )),
             }
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
        let (enum_name, raw_generic_args) = match &subject_ty {
            Type::Enum(n, args) | Type::Struct(n, args) => (n, args.clone()),
            Type::Path(segments, args) => {
                if let Some(n) = segments.last() {
                    (n, args.clone())
                } else {
                    return Err("Match on empty path".into());
                }
            }
            _ => return Err(format!("Match on non-enum: {:?}", subject_ty)),
        };
        
        // If generic_args is empty but enum_name is mangled, lookup the original generics
        let generic_args: Vec<Type> = if raw_generic_args.is_empty() && enum_name.contains('_') {
            // Try to lookup from registered mangled types (accurate method)
            if let Some((_, args)) = self.lookup_mangled_type(enum_name) {
                args
            } else {
                // Fallback to parsing mangled name (less accurate)
                let parts: Vec<&str> = enum_name.split('_').collect();
                if parts.len() > 1 {
                    self.parse_mangled_type_args(&parts[1..])
                } else {
                    raw_generic_args
                }
            }
        } else {
            raw_generic_args
        };


        // Ensure enum layout is generated
        // Check if enum is already specialized (or concrete) to avoid double-monomorphization
        // Check if enum is already specialized (or concrete) to avoid double-monomorphization
        let actual_generic_args = if let Some(def) = self.enum_defs.get(enum_name) {
             if def.generics.is_empty() {
                 &[]
             } else {
                 generic_args.as_slice()
             }
        } else {
             // Def not found yet? Try blindly.
             generic_args.as_slice()
        };
        let mangled_name = self.monomorphize_enum(enum_name, actual_generic_args)
            .map_err(|e| format!("Failed to monomorphize enum {}: {}", enum_name, e))?;
        
        let enum_ty = self.enum_types.get(&mangled_name).cloned()
            .ok_or(format!("Enum LLVM type {} not found", mangled_name))?;

        let enum_def = self.enum_defs.get(&mangled_name)
            .or_else(|| self.enum_defs.get(enum_name))
            .cloned()
            .ok_or(format!("Enum def not found (tried {} and {})", mangled_name, enum_name))?;


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
                // Fetch def - try mangled_name first, then enum_name, then base name extraction
                let enum_def = if let Some(def) = self.enum_defs.get(&mangled_name) {
                    def
                } else if let Some(def) = self.enum_defs.get(enum_name) {
                    def
                } else {
                    // Try to convert underscore format to angle-bracket format
                    let base_name = if enum_name.contains('_') && !enum_name.contains('<') {
                        enum_name.split('_').next().unwrap_or(enum_name)
                    } else {
                        enum_name
                    };
                    // Try angle-bracket version from monomorphize result
                    self.enum_defs.get(&mangled_name)
                        .or_else(|| self.enum_defs.get(base_name))
                        .ok_or_else(|| format!("Enum def '{}' not found in pattern. Tried: {}, {}. Available: {:?}", 
                            enum_name, mangled_name, base_name, self.enum_defs.keys().collect::<Vec<_>>()))?
                };
                
                let simple_variant_name = variant_name.as_str();

                let idx = enum_def
                    .variants
                    .iter()
                    .position(|v| v.name == simple_variant_name)
                    .ok_or_else(|| format!("Enum variant '{}' not found in {}. Available: {:?}", simple_variant_name, enum_name, enum_def.variants.iter().map(|v| &v.name).collect::<Vec<_>>()))?;
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
                let simple_variant_name = variant_name.as_str();

                let variant_idx = enum_def
                    .variants
                    .iter()
                    .position(|v| v.name == simple_variant_name)
                    .ok_or("Enum variant not found in bindings")?;
                self.bind_enum_pattern_fields(
                    current_func,
                    enum_ty,
                    ptr,
                    &enum_def,
                    variant_idx,
                    bindings,
                    &generic_args,
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
                Type::I32 | Type::Char(_) => self.context.i32_type().into(),
                Type::Bool => self.context.bool_type().into(),
                Type::String(_) => self.context.ptr_type(inkwell::AddressSpace::default()).into(),
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
                | Type::String(_)
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
        bindings: &crate::compiler::ast::EnumPatternBindings,
        generic_args: &[Type],
    ) -> Result<(), String> {
        let variant_def = &enum_def.variants[variant_idx];

        // Build substitution map for concrete types
        // If enum_def.generics is empty (monomorphized enum like Entry_i64_i64),
        // try to find the original generic enum definition (e.g., Entry) to get generic param names
        let mut subst: HashMap<String, Type> = HashMap::new();
        
        if enum_def.generics.is_empty() && !generic_args.is_empty() {
            // Try to find the original generic enum definition
            let base_name = enum_def.name.split('_').next().unwrap_or(&enum_def.name);
            if let Some(generic_def) = self.enum_defs.get(base_name) {
                for (i, param_name) in generic_def.generics.iter().enumerate() {
                    if let Some(arg) = generic_args.get(i) {
                        subst.insert(param_name.clone(), arg.clone());
                    }
                }
            }
        } else {
            for (i, param_name) in enum_def.generics.iter().enumerate() {
                if let Some(arg) = generic_args.get(i) {
                    subst.insert(param_name.clone(), arg.clone());
                }
            }
        }

        // Get Payload Pointer (Index 1)
        // Opaque Layout: { i32, [u8...]}
        // We cast this pointer to the Variant Struct Pointer
        let payload_ptr_raw = self.builder
            .build_struct_gep(enum_ty, enum_ptr, 1, "payload_ptr_raw")
            .map_err(|e| e.to_string())?;

        // internal helper to get concrete types for this variant
        let mut field_types = Vec::new();
        match &variant_def.kind {
            crate::compiler::ast::VariantKind::Unit => {},
            crate::compiler::ast::VariantKind::Tuple(types) => {
                for ty in types {
                    let concrete_ty = self.substitute_type_simple_bind(ty, &subst);
                    if concrete_ty != Type::Void {
                        field_types.push(concrete_ty);
                    }
                }
            },
            crate::compiler::ast::VariantKind::Struct(fields) => {
                for (_, ty) in fields {
                    let concrete_ty = self.substitute_type_simple_bind(ty, &subst);
                    if concrete_ty != Type::Void {
                        field_types.push(concrete_ty);
                    }
                }
            }
        }

        // Construct Variant LLVM Struct Type
        let mut llvm_field_types = Vec::new();
        for f_ty in &field_types {
            llvm_field_types.push(self.get_llvm_type(f_ty)?);
        }
        let variant_struct_ty = self.context.struct_type(&llvm_field_types, false);

        // Cast payload ptr to variant struct ptr
        // Use pointer_cast (BitCast)
        // Since payload_ptr_raw is [u8]* (inside struct), and we want { Fields... }*
        // We cast to PointerType(variant_struct_ty)
        let variant_ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
        // Wait, build_pointer_cast works on values.
        // We handle opaque pointers in recent LLVM, so just casting the value is enough?
        // But `build_struct_gep` needs the Type info if we didn't pass it.
        // `builder.build_struct_gep(variant_struct_ty, payload_ptr, idx, ...)`

        let payload_ptr = self.builder.build_pointer_cast(
            payload_ptr_raw,
            variant_ptr_type, // Actually just mapped to ptr
            "payload_cast"
        ).unwrap();


        // Bind fields
        let mut current_field_idx = 0; // Relative to Variant Struct

        match (&variant_def.kind, bindings) {
            (crate::compiler::ast::VariantKind::Unit, crate::compiler::ast::EnumPatternBindings::Unit) => {
                Ok(())
            },
            (crate::compiler::ast::VariantKind::Tuple(types), crate::compiler::ast::EnumPatternBindings::Tuple(vars)) => {
                if types.len() != vars.len() {
                    return Err(format!("Tuple pattern length mismatch: expected {}, found {}", types.len(), vars.len()));
                }
                
                for (i, bind_name) in vars.iter().enumerate() {
                    let field_ty = &types[i];
                    let concrete_ty = self.substitute_type_simple_bind(field_ty, &subst);

                    if concrete_ty == Type::Void {
                         continue;
                    }
                    
                    if bind_name == "_" {
                         current_field_idx += 1;
                         continue;
                    }
                    
                    // Access field from cast payload
                    let f_ptr = self
                        .builder
                        .build_struct_gep(variant_struct_ty, payload_ptr, current_field_idx, "field_ptr")
                        .map_err(|e| e.to_string())?;
                    current_field_idx += 1;

                    let llvm_ty = self.get_llvm_type(&concrete_ty)?;
                    let f_val = self.builder.build_load(llvm_ty, f_ptr, "bind_val").unwrap();

                    let alloca = self.create_entry_block_alloca(current_func, bind_name, &concrete_ty)?;
                    let stored_val = match &concrete_ty {
                        Type::Tuple(ts) if ts.is_empty() => f_val,
                        Type::Tensor(_, _)
                        | Type::Struct(_, _)
                        | Type::Enum(_, _)
                        | Type::Tuple(_)
                        | Type::String(_) => {
                             self.emit_deep_clone(f_val, &concrete_ty)?
                        }
                        _ => f_val,
                    };
                    self.builder.build_store(alloca, stored_val).unwrap();

                    self.variables
                        .last_mut()
                        .unwrap()
                        .insert(bind_name.clone(), (alloca.into(), concrete_ty.clone(), super::CLEANUP_FULL));
                }
                Ok(())
            },
            (crate::compiler::ast::VariantKind::Struct(fields), crate::compiler::ast::EnumPatternBindings::Struct(bindings_list)) => {
                 for (def_field_name, def_ty) in fields {
                     let concrete_ty = self.substitute_type_simple_bind(def_ty, &subst);
                     if concrete_ty == Type::Void { continue; }
                     
                     let binding = bindings_list.iter().find(|(n, _)| n == def_field_name);
                     
                     if let Some((_, bind_var_name)) = binding {
                         if bind_var_name != "_" {
                             // Access field
                             let f_ptr = self
                                 .builder
                                 .build_struct_gep(variant_struct_ty, payload_ptr, current_field_idx, "field_ptr")
                                 .map_err(|e| e.to_string())?;
                             
                             let llvm_ty = self.get_llvm_type(&concrete_ty)?;
                             let f_val = self.builder.build_load(llvm_ty, f_ptr, "bind_val").unwrap();
                             
                             let alloca = self.create_entry_block_alloca(current_func, bind_var_name, &concrete_ty)?;
                             let stored_val = match &concrete_ty {
                                 Type::Tuple(ts) if ts.is_empty() => f_val,
                                 Type::Tensor(_, _)
                                 | Type::Struct(_, _)
                                 | Type::Enum(_, _)
                                 | Type::Tuple(_)
                                 | Type::String(_) => {
                                      self.emit_deep_clone(f_val, &concrete_ty)?
                                 }
                                 _ => f_val,
                             }; 
                             self.builder.build_store(alloca, stored_val).unwrap();
                             
                             self.variables.last_mut().unwrap().insert(bind_var_name.clone(), (alloca.into(), concrete_ty.clone(), super::CLEANUP_FULL));
                         }
                     }
                     // Always increment standard layout index
                     current_field_idx += 1;
                 }
                 Ok(())
            }
             _ => Err("Invalid pattern binding for variant".into())
        }
    }
    
    fn substitute_type_simple_bind(&self, ty: &Type, subst: &HashMap<String, Type>) -> Type {
        let substitutor = crate::compiler::ast_subst::TypeSubstitutor::new(subst.clone());
        substitutor.substitute_type(ty)
    }

/*
             crate::compiler::ast::VariantKind::Unit => Box::new(std::iter::empty()),
             crate::compiler::ast::VariantKind::Tuple(types) => Box::new(types.iter()),
             crate::compiler::ast::VariantKind::Struct(fields) => Box::new(fields.iter().map(|(_, t)| t)),
        };
        
        for ty in fields_iter {
            field_types_llvm.push(self.get_llvm_type(ty)?);
        }
        let variant_struct_ty = self.context.struct_type(&field_types_llvm, false);

        // Get Payload Pointer
        // Always GEP to index 1 (payload) if not unit?
        // Wait, if Unit, layout is just Tag?
        // Enum layout: { i32 tag, Union payload }? Or { i32 tag, [MaxPayloadSize x i8] }?
        // `builtin_ast.rs` defines Enum layout.
        // Assuming Enum is { tag, payload }.
        let payload_ptr_raw = self
            .builder
            .build_struct_gep(enum_ty, enum_ptr, 1, "payload_ptr_raw")
            .map_err(|e| e.to_string())?;
        let payload_ptr = self
            .builder
            .build_pointer_cast(
                payload_ptr_raw,
                self.context.ptr_type(inkwell::AddressSpace::default()),
                "payload_cast",
            )
            .unwrap();

        match (&variant_def.kind, bindings) {
            (crate::compiler::ast::VariantKind::Unit, crate::compiler::ast::EnumPatternBindings::Unit) => {
                Ok(())
            },
            (crate::compiler::ast::VariantKind::Tuple(types), crate::compiler::ast::EnumPatternBindings::Tuple(vars)) => {
                if types.len() != vars.len() {
                    return Err(format!("Tuple pattern length mismatch: expected {}, found {}", types.len(), vars.len()));
                }
                
                for (i, bind_name) in vars.iter().enumerate() {
                    let f_ty = &types[i];
                    
                    let f_ptr = self
                        .builder
                        .build_struct_gep(variant_struct_ty, payload_ptr, i as u32, "field_ptr")
                        .map_err(|e| e.to_string())?;

                    let llvm_ty = self.get_llvm_type(f_ty)?;
                    let f_val = self.builder.build_load(llvm_ty, f_ptr, "bind_val").unwrap();

                    let alloca = self.create_entry_block_alloca(current_func, bind_name, f_ty)?;
                    let stored_val = if matches!(
                        f_ty,
                        Type::Tensor(_, _)
                            | Type::Struct(_, _)
                            | Type::Enum(_, _)
                            
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
            },
            (crate::compiler::ast::VariantKind::Struct(fields), crate::compiler::ast::EnumPatternBindings::Struct(bindings_list)) => {
                for (field_name, bind_name) in bindings_list {
                    let f_idx = fields
                        .iter()
                        .position(|(n, _)| n == field_name)
                        .ok_or(format!("Field {} not found in variant {}", field_name, variant_def.name))?;
                    
                    let (_, f_ty) = &fields[f_idx];

                    let f_ptr = self
                        .builder
                        .build_struct_gep(variant_struct_ty, payload_ptr, f_idx as u32, "field_ptr")
                        .map_err(|e| e.to_string())?;

                    let llvm_ty = self.get_llvm_type(f_ty)?;
                    let f_val = self.builder.build_load(llvm_ty, f_ptr, "bind_val").unwrap();

                    let alloca = self.create_entry_block_alloca(current_func, bind_name, f_ty)?;
                    let stored_val = if matches!(
                        f_ty,
                        Type::Tensor(_, _)
                            | Type::Struct(_, _)
                            | Type::Enum(_, _)
                            
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
            },
             _ => Ok(()), 
        }
*/

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
        if ty == &Type::Void {
        }
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
                let tensor_ptr = match &target.inner {
                    ExprKind::Variable(name) => {
                        let (val, ty) = self
                            .lookup_variable(name)
                            .ok_or(format!("Variable {} not found", name))?;
                        // Load pointer if needed
                        match ty {
                            Type::Tensor(_, _) => {
                                // val is a pointer to the tensor pointer
                                self.builder
                                    .build_load(
                                        self.context.ptr_type(inkwell::AddressSpace::default()),
                                        val.into_pointer_value(),
                                        name,
                                    )
                                    .map_err(|e| e.to_string())?
                                    .into_pointer_value()
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
                                let dim_size = match call_result.try_as_basic_value() {
                                    ValueKind::Basic(v) => v.into_int_value(),
                                    _ => return Err("Invalid dim return".into()),
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
        self.emit_method_call(obj, obj_val, obj_ty, method, args)
    }

    fn emit_method_call(
        &mut self,
        obj_expr_context: &Expr, // Passed for fallback Unevaluated methods which need AST access
        obj_val: BasicValueEnum<'ctx>,
        obj_ty: Type,
        method: &str,
        args: &[Expr],
    ) -> Result<(BasicValueEnum<'ctx>, Type), String> {


        // Try TypeManager first
        let type_name_opt = match &obj_ty {

            Type::Struct(n, _) => Some(n.clone()),
            Type::Enum(n, _) => Some(n.clone()),
            Type::Tensor(_, _) => Some("Tensor".to_string()),
            Type::String(_) => Some("String".to_string()),
            _ => None,
        };

        if let Some(name) = type_name_opt {
             // Handle special logical types if we want to support UserDefined alias canonicalization? 
             // Logic above handles it if UserDefined("Custom") returns "Custom".
             if let Some(type_info) = self.type_manager.get_type(&name) {
                 if type_info.has_instance_method(method) {
                     if let Some(overloads) = type_info.get_instance_overloads(method) {
                         // Try to find matching overload by arg count
                         for overload in overloads {
                             if overload.arg_types.len() == args.len() {
                                 match &overload.impl_fn {
                                     InstanceMethod::Evaluated(func) => {
                                         let func = *func;
                                         let mut compiled_args = Vec::new();
                                         for arg in args {
                                             compiled_args.push(self.compile_expr(arg)?);
                                         }
                                         return func(self, obj_val, obj_ty, compiled_args);
                                     }
                                     InstanceMethod::Unevaluated(func) => {
                                         let func = *func;
                                         return func(self, obj_expr_context, method, args);
                                     }
                                     InstanceMethod::SignatureOnly => {
                                         // No implementation, skip this overload
                                         continue;
                                     }
                                 }
                             }
                         }
                     }
                 }
             }
        }





        // 2. Resolve Type Name to check Manager
        let type_name = match &obj_ty {
            Type::Struct(name, _) => name.clone(),
            Type::Enum(name, _) => name.clone(),
            Type::Tensor(_, _) => "Tensor".to_string(),
            Type::F32 => "F32".to_string(),
            Type::F64 => "F64".to_string(),
            Type::I64 => "I64".to_string(),
            Type::I32 => "I32".to_string(),
            Type::String(_) => "String".to_string(),
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
                    return func(self, obj_expr_context, method, args);
                }
                InstanceMethod::SignatureOnly => {
                    // Should not happen, fall through to lookup elsewhere
                }
            }
        }




        if type_name == "Map" {
            match method {
                "get" | "get_1d" | "get_quantized" => {
                    if args.len() != 1 {
                        return Err("Map::get requires 1 argument".into());
                    }
                    let (key_val, key_ty) = self.compile_expr(&args[0])?;
                    
                    // Runtime expects *mut StringStruct, so pass the String struct pointer directly
                    // Do NOT extract the ptr field - runtime will access it
                    let name_arg = if matches!(key_ty, Type::String(_)) {
                        // key_val is already a pointer to StringStruct
                        key_val
                    } else {
                        return Err(format!("Map::get expects String argument, got {:?}", key_ty));
                    };
                    
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
                        .build_call(fn_val, &[obj_val.into(), name_arg.into()], "map_get")
                        .map_err(|e| e.to_string())?;
                    let res = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err("Invalid return from Map::get".into()),
                    };
                    if method == "get_quantized" {
                        // Return Tensor type so that ownership is tracked correctly
                        return Ok((res, Type::Tensor(Box::new(Type::I8), 2)));
                    }
                    let _ret_ty = Type::Tensor(Box::new(Type::F32), 0);

                    return Ok((res, Type::Tensor(Box::new(Type::F32), 0)));
                }
                _ => {}
            }
        }

        if type_name == "String" {
            let str_struct_ty = self.context.struct_type(&[
                self.context.ptr_type(inkwell::AddressSpace::default()).into(), // ptr
                self.context.i64_type().into(), // len
            ], false);

            match method {

                "print" | "display" => {
                    let fn_val = self
                        .module
                        .get_function("tl_print_string")
                        .ok_or("tl_print_string not found")?;
                    
                    // Pass struct pointer directly
                    self.builder.build_call(fn_val, &[obj_val.into()], "").map_err(|e| e.to_string())?;
                    return Ok((self.context.i64_type().const_zero().into(), Type::Void));
                }
                "len" => {
                    let ptr = obj_val.into_pointer_value();
                    let len_ptr = self.builder.build_struct_gep(str_struct_ty, ptr, 1, "len_ptr").map_err(|_| "Failed to GEP String len")?;
                    let len_val = self.builder.build_load(self.context.i64_type(), len_ptr, "len").map_err(|e| e.to_string())?;
                    return Ok((len_val, Type::I64));
                }

                _ => {}
            }
        }

        // Special Handling for Tensor methods was removed in favor of TypeManager registration.
        // See builtin_types/non_generic/tensor.rs for method implementations.

        // 4. Generic Fallback (Struct Methods / Mangled Names)
        let struct_name = match &obj_ty {

            Type::Struct(name, _) | Type::Enum(name, _) => name.clone(),
            Type::Path(segments, _) => if let Some(n) = segments.last() { n.clone() } else { return Err("Empty path".into()) },
            Type::Tensor(_, _) => "Tensor".to_string(),
            Type::String(_) => "String".to_string(),
            _ => return Err(format!("Method {} not found on type {:?}", method, obj_ty)),
        };

        let type_name = struct_name.clone();



        // Extract simple name from module path for mangling
        let simple_struct_name = struct_name.clone();

        // Try exact mangling first: tl_{Struct}_{Method}



        // Check for generic impls first
        // Extract base name from mangled type name (e.g., "Option<i64>" -> "Option", "Result_i32" -> "Result")
        let (base_type_name, inferred_generics) = if type_name.contains('<') {
            let base = type_name.split('<').next().unwrap_or(&type_name).to_string();
            // Extract generics from e.g., "Option<i64>" -> [I64]
            let generics_str = type_name.trim_start_matches(&base).trim_start_matches('<').trim_end_matches('>');
            let parsed_generics: Vec<Type> = generics_str.split(',')
                .filter_map(|s| {
                    let s = s.trim();
                    match s.to_lowercase().as_str() {
                        "i64" => Some(Type::I64),
                        "i32" => Some(Type::I32),
                        "f32" => Some(Type::F32),
                        "f64" => Some(Type::F64),
                        "bool" => Some(Type::Bool),
                        "u8" => Some(Type::U8),
                        "string" => Some(Type::String("String".to_string())),
                        "" => None,
                        _ => Some(Type::Struct(s.to_string(), vec![])),
                    }
                })
                .collect();
            (base, parsed_generics)
        } else if type_name.contains('_') {
            // Try underscore format: "Result_i32_i32" -> ("Result", [I32, I32])
            // Also handle nested generics like "Vec_Entry_i64_i64" -> ("Vec", [Entry<i64, i64>])
            let parts: Vec<&str> = type_name.split('_').collect();
            if !parts.is_empty() {
                let base = parts[0].to_string();
                let parsed_generics = self.parse_mangled_type_args(&parts[1..]);
                (base, parsed_generics)
            } else {
                (type_name.clone(), vec![])
            }
        } else {
            (type_name.clone(), vec![])
        };

        let generic_func = if self.generic_impls.contains_key(&base_type_name) {
             // Prioritize obj_ty for type args - it has complete type information
             // String parser (inferred_generics) may lose nested generic info (e.g. Entry<K,V> -> Entry)
             let generics = match &obj_ty {
                 Type::Struct(_, args) | Type::Enum(_, args) | Type::Path(_, args) if !args.is_empty() => args.clone(),
                 _ => if !inferred_generics.is_empty() { inferred_generics.clone() } else { vec![] },
             };

             // Try monomorphize
             match self.monomorphize_method(&base_type_name, method, &generics) {
                 Ok(name) => {
                     let f = self.module.get_function(&name).ok_or(format!("{} not found", name))?;
                     Some((f, name))
                 },
                 Err(e) => return Err(e),
             }
        } else {
             None
        };

        let mangled_name = format!("tl_{}_{}", simple_struct_name, method);
        // Fallback to lowercase
        let stdlib_name = format!("tl_{}_{}", simple_struct_name.to_lowercase(), method);

        let (func_val, final_name) = if let Some(res) = generic_func {
            res
        } else if let Some(f) = self.module.get_function(&mangled_name) {
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
        let mut ret_ty = if let Some(ret) = self.method_return_types.get(&final_name) {
             ret.clone()
        } else {
             self.get_return_type_from_signature(func_val)
        };
        
        // FIX: For generic runtime functions (like tl_vec_ptr_get), the return type may have 
        // generic placeholders (e.g., Option<T>) that need to be substituted with actual generics.
        // This is necessary because runtime functions like tl_vec_ptr_get are shared across all
        // Vec<T> types, but the actual T in Option<T> depends on the call site.
        if !inferred_generics.is_empty() {
            match &ret_ty {
                Type::Enum(name, args) if (name == "Option" || name == "Result") && args.len() == 1 => {
                    // For Vec.get -> Option<T>, replace T with the actual element type
                    if method == "get" && base_type_name == "Vec" {
                        ret_ty = Type::Enum(name.clone(), inferred_generics.clone());
                    }
                }
                Type::Enum(name, args) if name == "Result" && args.len() == 2 => {
                    // Handle Result<T, E> if needed
                    if method == "get" && base_type_name == "Vec" {
                        let new_ok = inferred_generics.get(0).cloned().unwrap_or(args[0].clone());
                        let new_err = args[1].clone();
                        ret_ty = Type::Enum(name.clone(), vec![new_ok, new_err]);
                    }
                }
                _ => {}
            }
        }
        
        // For Option.unwrap/unwrap_or, the return type T should be replaced with generics[0]
        // Note: Get generics from obj_ty which has the correct type from the call site
        if base_type_name == "Option" && (method == "unwrap" || method == "unwrap_or") {
            if let Type::Enum(_, args) = &obj_ty {
                if let Some(inner_ty) = args.get(0) {
                    ret_ty = inner_ty.clone();
                }
            }
        }
        
        // For Result.unwrap, the return type T should be replaced with generics[0]
        if base_type_name == "Result" && method == "unwrap" {
            if let Type::Enum(_, args) = &obj_ty {
                if let Some(ok_ty) = args.get(0) {
                    ret_ty = ok_ty.clone();
                }
            }
        }

        // SRET Check
        // String is a pointer (RefCounted), handled as value return, not SRET.
        let uses_sret = match &ret_ty {
            Type::Struct(n, _) => n != "String" && n != "Tensor",
            _ => false,
        };
        let mut sret_ptr = None;

        if uses_sret {
             // OLD: Stack Allocation (alloca) -> Causes Stack Corruption
             // NEW: Heap Allocation (malloc + register) -> Correct for RefCounted Structs/SRET
             
             // 1. Get Struct Type and Size from CodeGen struct_types map
             let (struct_name, generics) = match &ret_ty {
                 Type::Struct(n, g) => (n, g),
                 _ => return Err("SRET used on non-struct type".into()),
             };
             
             let mangled_name = if generics.is_empty() {
                 struct_name.to_string()
             } else {
                 self.mangle_type_name(struct_name, generics)
             };
             
             // Simple name lookup (as done in compile_struct_init)
             let simple_lookup_name = mangled_name.clone();

             let struct_type = self.struct_types.get(&simple_lookup_name)
                 .or_else(|| self.enum_types.get(&simple_lookup_name))
                 .ok_or_else(|| format!("Struct type {} not found for SRET allocation", simple_lookup_name))?;
             
             let size = struct_type.size_of().ok_or("Cannot determine size for SRET struct")?;
             
             // 2. Malloc
             let malloc_fn = self.module.get_function("malloc").ok_or("malloc not found")?;
             let size_i64 = self.builder.build_int_z_extend(size, self.context.i64_type(), "size_i64").unwrap();
             let call_malloc = self.builder.build_call(malloc_fn, &[size_i64.into()], "sret_malloc").map_err(|e| e.to_string())?;
             
             let raw_ptr = match call_malloc.try_as_basic_value() {
                 inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
                 _ => return Err("malloc returned void".into()),
             };
             
             // 3. Register (Initialize RefCount=1, Header)
             // Use generic name or strict name?
             let struct_name_str = match &ret_ty {
                 Type::Struct(n, _) => n.as_str(),
                 _ => "AnonymousStruct",
             };
             let name_global = self.builder.build_global_string_ptr(struct_name_str, "struct_name").unwrap();
             let register_fn = self.module.get_function("tl_mem_register_struct_named").ok_or("tl_mem_register_struct_named not found")?;
             
             let cast_ptr = self.builder.build_pointer_cast(raw_ptr, self.context.ptr_type(inkwell::AddressSpace::default()), "cast_ptr").unwrap();
             let _ = self.builder.build_call(register_fn, &[cast_ptr.into(), name_global.as_pointer_value().into()], "");

             sret_ptr = Some(cast_ptr);
        }

        let mut compiled_args_vals = Vec::with_capacity(args.len() + 1);
        let mut compiled_args_types = Vec::with_capacity(args.len());

        // Push SRET Ptr if needed
        if let Some(ptr) = sret_ptr {
            compiled_args_vals.push(ptr.into());
        }

        // Push Receiver
        compiled_args_vals.push(obj_val.into());

        for arg in args {
            let (val, ty) = self.compile_expr(arg)?;
            
            // ARGUMENT PASSING FIX: Retain ownership because Callee takes ownership (and releases at end)
            // If we don't retain, the Callee's release will free the memory while Caller still holds it (if var).
            // We skip String because it uses manual management not compatible with refcounts map yet.
            let should_retain = match &ty {
                Type::Struct(n, _) if n != "String" => true,
                Type::Enum(_, _) | Type::Tensor(_, _) | Type::Tuple(_) => true,
                _ => false, 
            };
            
            if should_retain {
                 self.emit_retain(val, &ty)?;
            }

            compiled_args_vals.push(val.into());
            compiled_args_types.push((val, ty));
        }

        // Call
        let call = self
            .builder
            .build_call(func_val, &compiled_args_vals, "method_call")
            .map_err(|e| e.to_string())?;

        // Return handling
        // Return handling
        if let Some(ptr) = sret_ptr {
            // Return SRET pointer as value
             Ok((ptr.into(), ret_ty))
        } else {
            match call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(_) => {
                    let v = self.check_tensor_result(call, "method_call_error")?;
                    Ok((v, ret_ty))
                }
                _ => Ok((
                    self.context.i64_type().const_int(0, false).into(),
                    Type::Void,
                )),
            }
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

                  // self.emit_register_tensor(res, &obj_ty)?;
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
                  // self.emit_register_tensor(res, &obj_ty)?;

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
                  // self.emit_register_tensor(res, &obj_ty)?;

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
                  // self.emit_register_tensor(res, &obj_ty)?;

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
                  let path_ty = Type::String("String".to_string()); // Assumed
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

              "slice" => {
                  if args.len() != 2 {
                      return Err("slice requires 2 arguments".into());
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

                  // self.emit_register_tensor(res, &obj_ty)?;
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
                  if !matches!(dev_ty, Type::String(_)) {
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
                    // self.emit_register_tensor(res, &obj_ty)?;
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
                    // self.emit_register_tensor(res, &obj_ty)?;
                    Ok((res, obj_ty))
                }
                _ => {
                  // Generic method dispatch for UserDefined types and Tensor
                  let mut type_name = match &obj_ty {
                      Type::Struct(name, _) | Type::Struct(name, _) | Type::Enum(name, _) => {
                          name.clone()
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
                  if let Type::Struct(name, args) | Type::Struct(name, args) = &obj_ty {
                       if !args.is_empty() {
                           // Check if it's a generic method in registry (via mono)
                           let simple_name = name;
                           
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

                      // Determine return type dynamically from signature
                      let ret_type = self.get_return_type_from_signature(fn_val);

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

    // Helper to derive return type from LLVM Function Signature
    pub(crate) fn get_return_type_from_signature(&self, func: inkwell::values::FunctionValue<'ctx>) -> Type {
        let name = func.get_name().to_str().unwrap_or("");

        // 1. Check if we have an explicit mapping for this function
        if let Some(ty) = self.method_return_types.get(name) {
             return ty.clone();
        }

        let ret = func.get_type().get_return_type();
        match ret {
            None => Type::Void,
            Some(inkwell::types::BasicTypeEnum::IntType(i)) => {
                let width = i.get_bit_width();
                if width == 1 {
                    Type::Bool
                } else if width == 64 {
                    Type::I64
                } else {
                    Type::I32 // Fallback
                }
            }
            Some(inkwell::types::BasicTypeEnum::FloatType(_f)) => {
                 Type::F32 
            }
            Some(inkwell::types::BasicTypeEnum::PointerType(_)) => {
                if name.starts_with("tl_string_") {
                    Type::String("String".to_string())
                } else if name.contains("alloc") || name.contains("init") {
                    // Allocators return generic pointers usually
                    // Use I64 as opaque pointer type placeholder
                    Type::I64
                } else {
                     // Log warning?
                     // Verify if it's a known generic method
                     // For now, default to I64 (opaque handle) to avoid Tensor mismatch logic
                     // UNLESS we are sure it's a tensor method.
                     if name.contains("tensor") {
                         Type::Tensor(Box::new(Type::F32), 0)
                     } else {
                         Type::I64
                     }
                }
            }
            _ => Type::Void, 
        }
    }

    pub(crate) fn compile_fn_call(
        &mut self,
        name: &str,
        args: &[Expr],
    ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        let (val, ty) = self.compile_fn_call_dps(name, args, None)?;
        
        let mode = match &ty {
             Type::Struct(_, _) | Type::String(_) | Type::Tensor(_, _) => super::CLEANUP_FULL,
             _ => super::CLEANUP_STACK,
        };
        
        self.add_temp_with_mode(val, ty.clone(), mode);
        Ok((val, ty))
    }

    pub(crate) fn compile_fn_call_dps(
        &mut self,
        name: &str,
        args: &[Expr],
        dest: Option<BasicValueEnum<'ctx>>,
    ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        // 0. Check if it's a relation query
        let simple_name = name;

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
                        let (val, ty) = self.compile_expr(arg)?;
                        compiled_args.push((val, ty));
                    }
                    return func(self, compiled_args);
                }
                BuiltinFn::Unevaluated(func) => {
                    return func(self, args);
                }
            }
        }

        // Intrinsic: tl_core_hash<T>(val: T) -> i64
        if name.starts_with("tl_core_hash") {
            if args.len() != 1 {
                return Err("tl_core_hash expects exactly 1 argument".to_string());
            }
            let (val, ty) = self.compile_expr(&args[0])?;
            let i64_type = self.context.i64_type();

            let res: inkwell::values::IntValue = match &ty {
                Type::I64 => val.into_int_value(),
                Type::I32 | Type::Char(_) => self.builder.build_int_z_extend(val.into_int_value(), i64_type, "zext").unwrap(),
                Type::Bool => self.builder.build_int_z_extend(val.into_int_value(), i64_type, "zext").unwrap(),
                Type::F32 => {
                    let i32_val = self.builder.build_bit_cast(val.into_float_value(), self.context.i32_type(), "f32_cast").unwrap().into_int_value();
                    self.builder.build_int_z_extend(i32_val, i64_type, "zext").unwrap()
                },
                Type::F64 => {
                     self.builder.build_bit_cast(val.into_float_value(), i64_type, "f64_cast").unwrap().into_int_value()
                },
                Type::String(_) => {
                     let fn_val = self.module.get_function("tl_hash_string")
                         .ok_or("tl_hash_string runtime function not found")?;
                     let call = self.builder.build_call(fn_val, &[val.into()], "hash_call").unwrap();
                     match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v.into_int_value(),
                        _ => return Err("tl_hash_string did not return a value".to_string()),
                     }
                },
                Type::Struct(_, _) | Type::Enum(_, _) | Type::Tensor(_, _) | Type::Tuple(_) => {
                    if val.is_pointer_value() {
                         self.builder.build_ptr_to_int(val.into_pointer_value(), i64_type, "ptr_int").unwrap()
                    } else {
                         // Fallback for immediate structs (very small ones potentially? not standard in TL currently)
                         // Return 0 to be safe/lazy, or error?
                         return Err(format!("Hashing immediate struct/value type not supported: {:?}", ty));
                    }
                },
                _ => return Err(format!("Unsupported type for hash: {:?}", ty)),
            };
            
            return Ok((res.into(), Type::I64));
        }

        // Intrinsic: __builtin_unsafe_to_i64
        if name.starts_with("__builtin_unsafe_to_i64") {
            if args.len() != 1 {
                return Err("__builtin_unsafe_to_i64 expects exactly 1 argument".to_string());
            }
            let (val, ty) = self.compile_expr(&args[0])?;
            
            let i64_type = self.context.i64_type();
            let res = match val {
                inkwell::values::BasicValueEnum::IntValue(i) => {
                    if i.get_type().get_bit_width() == 64 {
                        i.into()
                    } else {
                        // Extend (zext)
                        self.builder.build_int_z_extend(i, i64_type, "zext").unwrap().into()
                    }
                }
                inkwell::values::BasicValueEnum::PointerValue(p) => {
                    self.builder.build_ptr_to_int(p, i64_type, "ptr2int").unwrap().into()
                }
                inkwell::values::BasicValueEnum::FloatValue(f) => {
                    // Bitcast to i32 then zext to i64
                    let i32_type = self.context.i32_type();
                    let as_i32 = self.builder.build_bit_cast(f, i32_type, "f32cast").unwrap().into_int_value();
                    self.builder.build_int_z_extend(as_i32, i64_type, "zext").unwrap().into()
                }
                _ => return Err(format!("Unsupported type for unsafe_to_i64: {:?}", ty)),
            };
            return Ok((res, Type::I64));
        }

        // Intrinsic: __builtin_unsafe_from_i64(val: i64, marker: PhantomData<T>) -> T
        if name.starts_with("__builtin_unsafe_from_i64") {
            if args.len() != 2 {
                return Err("__builtin_unsafe_from_i64 expects 2 arguments (val, marker)".to_string());
            }
            let (val, _val_ty) = self.compile_expr(&args[0])?;
            let (_, marker_ty) = self.compile_expr(&args[1])?;
            
            // Extract T from PhantomData<T>
            let target_type = if let Type::Struct(name, generics) = &marker_ty {
                if name.contains("PhantomData") {
                    if !generics.is_empty() {
                        generics[0].clone()
                    } else if let Some(suffix) = name.strip_prefix("PhantomData_") {
                        // Specialized
                        match suffix {
                            "i64" => Type::I64,
                            "i32" => Type::I32,
                            "f64" => Type::F64,
                            "f32" => Type::F32,
                            "bool" => Type::Bool,
                            "String" => Type::String("String".to_string()),
                            "Char" => Type::Char("Char".to_string()),
                            _ => Type::Struct(suffix.to_string(), vec![]),
                        }
                    } else {
                         return Err(format!("Arg 2 must be PhantomData<T> (specialized or generic), got {:?}", marker_ty));
                    }
                } else {
                    return Err(format!("Arg 2 must be PhantomData<T>, got {:?}", marker_ty));
                }
            } else {
                return Err(format!("Arg 2 must be Struct PhantomData, got {:?}", marker_ty));
            };

            // Cast i64 to T
            let res: inkwell::values::BasicValueEnum = match &target_type {
                Type::I64 => val, // Identity
                Type::Struct(_, _) | Type::Tensor(_, _) => {
                    // i64 -> ptr
                    let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
                    if val.is_int_value() {
                        self.builder.build_int_to_ptr(val.into_int_value(), ptr_type, "int2ptr").unwrap().into()
                    } else if val.is_pointer_value() {
                        val // Identity if already ptr? But arg 0 should be i64.
                    } else {
                         return Err("Input must be int or ptr".to_string());
                    }
                },
                Type::Bool => {
                     // i64 -> bool (trunc)
                     let i1_type = self.context.bool_type();
                     self.builder.build_int_truncate(val.into_int_value(), i1_type, "trunc_bool").unwrap().into()
                },
                Type::String(_) => {
                     // i64 -> ptr
                     let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
                     self.builder.build_int_to_ptr(val.into_int_value(), ptr_type, "int2ptr_str").unwrap().into()
                },
                // Add more types if needed (u8, f32)
                Type::F32 => {
                     // i64 -> i32 -> float
                     let i32_val = self.builder.build_int_truncate(val.into_int_value(), self.context.i32_type(), "trunc_f32").unwrap();
                     self.builder.build_bit_cast(i32_val, self.context.f32_type(), "bitcast").unwrap().into()
                },
                _ => return Err(format!("Unsupported target type for from_i64: {:?}", target_type)),
            };
            
            return Ok((res, target_type));
        }

        // Intrinsic: __builtin_is_ref(marker: PhantomData<T>) -> bool
        if name.starts_with("__builtin_is_ref") {
            if args.len() != 1 {
                return Err("__builtin_is_ref expects 1 argument (marker)".to_string());
            }
            let (_, marker_ty) = self.compile_expr(&args[0])?;
            
            // Extract T
            let target_type = if let Type::Struct(name, generics) = &marker_ty {
                if name.contains("PhantomData") {
                    if !generics.is_empty() {
                        generics[0].clone()
                    } else if let Some(suffix) = name.strip_prefix("PhantomData_") {
                        // Specialized: PhantomData_i64, PhantomData_String, etc.
                        match suffix {
                            "i64" => Type::I64,
                            "i32" => Type::I32,
                            "f64" => Type::F64,
                            "f32" => Type::F32,
                            "bool" => Type::Bool,
                            "String" => Type::String("String".to_string()),
                            "Char" => Type::Char("Char".to_string()),
                            _ => Type::Struct(suffix.to_string(), vec![]), // Assume struct/ref
                        }
                    } else {
                         // PhantomData generic unspecialized with no args? Should not happen in monomorphized code
                         return Err(format!("Arg 1 must be PhantomData<T> (specialized or generic), got {:?}", marker_ty));
                    }
                } else {
                    return Err(format!("Arg 1 must be PhantomData<T>, got {:?}", marker_ty));
                }
            } else {
                return Err(format!("Arg 1 must be Struct PhantomData, got {:?}", marker_ty));
            };

            let is_ref = match target_type {
                Type::Struct(name, _) if name == "String" => false, // String literals crash if treated as RefCounted
                Type::String(_) => false,
                Type::Struct(_, _) | Type::Tensor(_, _) | Type::Enum(_, _) | Type::Tuple(_) => true,
                _ => false,
            };
            
            let bool_val = self.context.bool_type().const_int(if is_ref { 1 } else { 0 }, false);
            return Ok((bool_val.into(), Type::Bool));
        }

        // 2. Generic Function Call / Struct Init
        let llvm_func_name = match name {
            "slice" => "tl_tensor_slice",
            "sum" => "tl_tensor_sum", // Fallback for global sum if not caught by builtin (redundant but safe)
            "enable_grad" => "tl_tensor_enable_grad",
            _ => name,
        };

        // Lookup return type FIRST to handle sret
        // Simplified: FnCall handles simple names. StaticMethodCall handles :: names.
        // We only keep simple name resolution for legacy builtins or simple function lookups.
        let resolved_name = llvm_func_name.to_string();

        // Monomorphization Logic: Check if generic
        let mut final_resolved_name = resolved_name.clone();
        let mut precompiled_args = None;
        
        if self.module.get_function(&final_resolved_name).is_none() && self.generic_fn_defs.contains_key(&final_resolved_name) {
             let mut args_vec = Vec::new();
             let mut arg_types = Vec::new();
             for arg in args {
                  let (val, ty) = self.compile_expr(arg)?;

                  args_vec.push((val, ty.clone()));
                  arg_types.push(ty);
             }
             
             final_resolved_name = self.monomorphize_generic_function(&resolved_name, &arg_types)?;
             precompiled_args = Some(args_vec);
        }

        let func_opt = self.module.get_function(&final_resolved_name);

        let func = if let Some(f) = func_opt {
            f
        } else {
            // Fallback to Struct Initialization
            let simple_name = name;

            if self.struct_defs.contains_key(simple_name) {
                return self.compile_tuple_struct_init(simple_name, args);
            }

            return Err(format!(
                "Function {} not found (resolved: {})",
                name, resolved_name
            ));
        };

        let ret_type = if let Some(ret) = self.method_return_types.get(&resolved_name) {
            ret.clone()
        } else {
            self.get_return_type_from_signature(func)
        };

        let mut compiled_args_vals = Vec::with_capacity(args.len() + 1);
        let mut compiled_args_types = Vec::with_capacity(args.len());

        // DPS: If return type is Struct (and SRET enabled), handle hidden dest argument
        // Tensors are returned by pointer directly, so exclude them.
        let mut dest_val = None;
        let uses_sret = match ret_type {
             Type::Struct(ref name, _) => name != "Tensor" && name != "String",
             _ => false 
        };
        if uses_sret {
             if let Some(d) = dest {
                 dest_val = Some(d);
             } else {
                 // Allocate Temp Buffer on Stack (DPS)
                 let current_block = self.builder.get_insert_block().unwrap();
                 let current_func = current_block.get_parent().unwrap();
                 let alloca = self.create_entry_block_alloca(current_func, "sret_temp", &ret_type)?;
                 dest_val = Some(alloca.into());
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
                let (val, ty) = self.compile_expr(arg)?;

                // ARGUMENT PASSING FIX: Retain ownership because Callee takes ownership (and releases at end)
                let should_retain = match &ty {
                    Type::Struct(n, _) if n != "String" => true,
                    Type::Enum(_, _) | Type::Tensor(_, _) | Type::Tuple(_) => true,
                    _ => false, 
                };
                
                if should_retain {
                        self.emit_retain(val, &ty)?;
                }

                compiled_args_vals.push(val.into());
                compiled_args_types.push((val, ty));
            }
        }
        if let Some(_block) = self.builder.get_insert_block() {
        } else {
        }
        if self.builder.get_insert_block().is_none() {
            return Err(format!("INTERNAL ERROR: Builder has no insert block when calling {}", final_resolved_name));
        }

        for (_i, _arg) in compiled_args_vals.iter().enumerate() {
        }
        let call_name = if ret_type == Type::Void { "" } else { "call_tmp" };

        let call = self
            .builder
            .build_call(func, &compiled_args_vals, call_name)
            .map_err(|e| e.to_string())?;

        // FIX: Free temporary arguments
        for (i, (_, ty)) in compiled_args_types.into_iter().enumerate() {
            let arg_expr = &args[i];
            if self.is_safe_to_free(arg_expr, &ty) {
                // self.emit_recursive_free(val, &ty, super::CLEANUP_FULL)?;
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

        // V4.5: Caller-Side Registration
        // When a function returns a tensor, it is "promoted" (floating).
        // The caller must register it to its own scope immediately.
        // We only do this for USER functions (which have basic blocks). 
        // Builtins/FFI functions (blocks=0) register internally to the current scope.
        if matches!(ret_type, Type::Tensor(_, _)) {
            if func.count_basic_blocks() > 0 {
                self.emit_register_tensor(res, &ret_type)?;
            }
        }

        // Normalize: If return type is Struct("Tensor"), treat it as Type::Tensor
        // (Parser may resolve `-> Tensor` as Struct("Tensor"))
        let ret_type = if let Type::Struct(ref name, _) = ret_type {
            if name == "Tensor" {
                Type::Tensor(Box::new(Type::F32), 0)
            } else {
                ret_type
            }
        } else {
            ret_type
        };

        // For Struct returns (SRET deprecated logic, assuming by-value or pointer return if not SRET)
        Ok((res, ret_type))
    }
}
fn compile_set_device<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err("set_device expects 1 argument".into());
    }
    let (arg_val, arg_ty) = codegen.compile_expr(&args[0])?;

    // Expect Device Enum
    let is_device_enum = match &arg_ty {
        Type::Enum(e, _) | Type::Struct(e, _) if e == "Device" => true,
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
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 2 {
        return Err("checkpoint requires 2 arguments: (method_ref, input)".into());
    }
    // Parse args[0] as obj.method
    let (obj_ptr, fn_ptr) = if let ExprKind::FieldAccess(obj_expr, method_name) = &args[0].inner {
        let (obj_val, obj_ty) = codegen.compile_expr(obj_expr)?;

        // Get struct type
        let struct_name = match obj_ty {
            Type::Struct(n, _) => n,
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
        Type::Char(_) => {
             let fn_name = if is_newline {
                 "tl_print_char"
             } else {
                 "tl_display_char"
             }; // Char is u8 internally (from char_at)
             // Check if arg is Int(8/32/64) or Ptr? generic is usually Any/U64
             // char_at returns Char which usually maps to i8/u8. 
             // If arg_val is i8, we might need cast if function expects i32/u8?
             // Runtime signatures in lib.rs define what expectations are.
             let fn_val = codegen.module.get_function(fn_name).unwrap();
             
             // If we need a cast:
             let arg_casted = if arg_val.is_int_value() {
                 let int_val = arg_val.into_int_value();
                 let i32_type = codegen.context.i32_type();
                 if int_val.get_type() != i32_type {
                      codegen.builder.build_int_cast(int_val, i32_type, "char_cast").unwrap().into()
                 } else {
                      (*arg_val).into()
                 }
             } else {
                 (*arg_val).into()
             };

             codegen
                 .builder
                 .build_call(fn_val, &[arg_casted], "print_call")
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
                let (str_struct_ptr, _) = codegen.compile_string_literal(s)?;
                let ptr = str_struct_ptr.into_pointer_value();

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
        Type::Struct(s, _) | Type::Struct(s, _) if s == "Tensor" => {
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
        Type::String(_) => {
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
        },
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
    Ok((res, Type::String("String".to_string())))
}

/// Compile panic! function - prints error message, calls abort, returns Never type
fn compile_panic_uneval<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err("panic requires 1 argument (error message)".into());
    }
    
    let (msg_val, _msg_ty) = codegen.compile_expr(&args[0])?;
    
    // Print the panic message using tl_display_string
    let display_fn = codegen
        .module
        .get_function("tl_display_string")
        .ok_or("tl_display_string not found")?;
    
    // Print "[PANIC] " prefix using compile_string_literal for proper TL string format
    let (prefix_val, _) = codegen.compile_string_literal("[PANIC] ")?;
    codegen.builder.build_call(display_fn, &[prefix_val.into()], "").unwrap();
    
    // Print the actual message
    codegen.builder.build_call(display_fn, &[msg_val.into()], "").unwrap();
    
    // Print newline
    let (newline_val, _) = codegen.compile_string_literal("\n")?;
    codegen.builder.build_call(display_fn, &[newline_val.into()], "").unwrap();
    
    // Call abort() to terminate the program
    let abort_fn = codegen.module.get_function("abort").ok_or("abort function not found")?;
    codegen.builder.build_call(abort_fn, &[], "").unwrap();
    
    // Insert LLVM unreachable instruction to indicate control doesn't reach here
    codegen.builder.build_unreachable().unwrap();
    
    // Return a dummy value with Never type (code won't actually reach here)
    let dummy = codegen.context.i64_type().const_zero();
    Ok((dummy.into(), Type::Never))
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
                let (str_val, _) = codegen.compile_string_literal(part)?;
                codegen
                    .builder
                    .build_call(display_fn, &[str_val.into()], "print_part")
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

            // Use compile_string_literal to create StringStruct
            let (str_struct_ptr, _) = codegen.compile_string_literal("")?;

            codegen
                .builder
                .build_call(print_fn, &[str_struct_ptr.into()], "print_newline")
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
    Ok((res, Type::String("String".to_string())))
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
    Ok((res, Type::Char("Char".to_string())))
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
    let is_tensor = matches!(t_ty, Type::Tensor(_, _)) 
        || matches!(t_ty, Type::TensorShaped(_, _))
        || matches!(t_ty, Type::Struct(name, _) if name == "Tensor");
        
    if !is_tensor {
        return Err(format!("First argument to transpose must be a tensor. Found: {:?}", t_ty));
    }
    let transpose_fn = codegen
        .module
        .get_function("tl_tensor_transpose")
        .ok_or("tl_tensor_transpose not found")?;

    let t_arg = if let Type::Struct(name, _) = t_ty {
        if name == "Tensor" {
            // Extract handle (i64) from struct
            let handle_i64 = if t_val.is_pointer_value() {
                let ptr = t_val.into_pointer_value();
                let i64_type = codegen.context.i64_type();
                let cast_ptr = codegen.builder.build_pointer_cast(ptr, codegen.context.ptr_type(inkwell::AddressSpace::default()), "cast_tensor_handle").unwrap();
                codegen.builder.build_load(i64_type, cast_ptr, "tensor_handle").unwrap().into_int_value()
            } else if t_val.is_struct_value() {
                codegen.builder.build_extract_value(t_val.into_struct_value(), 0, "tensor_handle").unwrap().into_int_value()
            } else {
                return Err(format!("Unexpected value kind for Struct Tensor: {:?}", t_val));
            };
            
            // Cast i64 handle to Pointer
            codegen.builder.build_int_to_ptr(handle_i64, codegen.context.ptr_type(inkwell::AddressSpace::default()), "handle_ptr").unwrap().into()
        } else {
            *t_val
        }
    } else {
        *t_val
    };

    let call = codegen
        .builder
        .build_call(
            transpose_fn,
            &[t_arg.into(), (*d0_val).into(), (*d1_val).into()],
            "transpose_res",
        )
        .map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid transpose return".into()),
    };

    if let Type::Struct(name, _) = t_ty {
        if name == "Tensor" {
            // Wrap primitive result (ptr) into Struct { handle: i64 }
            let current_block = codegen.builder.get_insert_block().unwrap();
            let parent_fn = current_block.get_parent().unwrap();
            
            // Alloca struct
            let i64_type = codegen.context.i64_type();
            let struct_type = codegen.context.struct_type(&[i64_type.into()], false);
            
            // Manual entry block alloca
            let entry = parent_fn.get_first_basic_block().unwrap();
            let builder = codegen.context.create_builder();
            if let Some(first_instr) = entry.get_first_instruction() {
                builder.position_before(&first_instr);
            } else {
                builder.position_at_end(entry);
            }
            let struct_alloca = builder.build_alloca(struct_type, "tensor_struct_res").unwrap();
            
            // Convert ptr -> i64
            let handle_i64 = codegen.builder.build_ptr_to_int(res.into_pointer_value(), i64_type, "handle_i64").unwrap();
            
            // Store handle (field 0)
            let handle_ptr = codegen.builder.build_struct_gep(struct_type, struct_alloca, 0, "handle_ptr").unwrap();
            codegen.builder.build_store(handle_ptr, handle_i64).unwrap();
            
            // Return pointer to struct
            return Ok((struct_alloca.into(), t_ty.clone()));
        }
    }

    Ok((res, t_ty.clone())) // Returns same type (Tensor)
}

fn compile_save_weights<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 2 {
        return Err("save_weights requires 2 arguments: tensor/struct, path".into());
    }
    let (t_val, t_ty) = &args[0];
    let (path_val, path_ty) = &args[1];

    if !matches!(path_ty, Type::String(_)) {
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
        Type::Struct(struct_name, _) | Type::Struct(struct_name, _) if struct_name != "String" => {
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
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() == 1 {
        let (path_val, path_ty) = &args[0];
        if !matches!(path_ty, Type::String(_)) {
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
        if !matches!(path_ty, Type::String(_)) {
            return Err("Second argument to load_weights must be a String (path)".into());
        }

        let struct_name_opt = match &s_ty {
            Type::Struct(s, _) => Some(s.clone()),
            _ => None,
        };

        if let Some(struct_name) = struct_name_opt {
            if struct_name == "String" {
                panic!("Struct(String) found in load_weights");
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
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err("register_modules requires 1 argument (struct)".into());
    }
    let (val, ty) = &args[0];
    match ty {
        Type::Struct(sname, _) => {
            codegen.gen_register_params(*val, &sname, "".to_string())?;
            return Ok((codegen.context.i64_type().const_zero().into(), Type::Void));
        }
        _ => return Err("register_modules expects a struct argument".into()),
    }
}

fn compile_update_all_params<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
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
    _target: Option<&Type>,
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
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen.module.get_function("tl_load_all_params").unwrap();
    let path_val = if args.len() == 2 {
        let (struct_val, struct_ty) = &args[0];
        let struct_name = match struct_ty {
            Type::Struct(s, _) => s,
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
    _target: Option<&Type>,
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

fn compile_save_all_params<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen.module.get_function("tl_save_all_params").unwrap();
    let path_val = if args.len() == 2 {
        let (struct_val, struct_ty) = &args[0];
        let struct_name = match struct_ty {
            Type::Struct(s, _) => s,
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



fn compile_varbuilder_get<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() < 2 {
        return Err("varbuilder_get requires at least 2 arguments (name and dimensions)".into());
    }
    let (name_val, name_ty) = codegen.compile_expr(&args[0])?;
    if !matches!(name_ty, Type::String(_)) {
        return Err(format!(
            "varbuilder_get expects String as first argument, found {:?}",
            name_ty
        ));
    }
    let name_ptr = name_val.into_pointer_value();

    let (rank, shape_ptr) = if args.len() == 2
        && matches!(
            codegen.compile_expr(&args[1])?.1,
            Type::Tensor(_, _)
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

impl<'ctx> CodeGenerator<'ctx> {
    /// Compile an enum variant constructor that was parsed as StaticMethodCall.
    /// This handles cases like Entry_i64_i64::Empty where the mangled enum name
    /// and variant name come through as a static method call.
    fn compile_enum_variant_as_static_method_call(
        &mut self,
        enum_name: &str,
        variant_name: &str,
        args: &[crate::compiler::ast::Expr],
        variant_idx: usize,
        enum_def: &crate::compiler::ast::EnumDef,
    ) -> Result<(inkwell::values::BasicValueEnum<'ctx>, crate::compiler::ast::Type), String> {
        use crate::compiler::ast::{Type, VariantKind};
        
        let variant_def = &enum_def.variants[variant_idx];
        let field_count = match &variant_def.kind {
            VariantKind::Unit => 0,
            VariantKind::Tuple(t) => t.len(),
            VariantKind::Struct(f) => f.len(),
        };
        if args.len() != field_count {
            return Err(format!("Enum variant {}::{} expects {} args, got {}", enum_name, variant_name, field_count, args.len()));
        }
        
        // Use enum_def.name (monomorphized name like "Option<i64>") instead of enum_name (might be base name "Option")
        let actual_enum_name = &enum_def.name;
        let enum_ty = if let Some(ty) = self.enum_types.get(actual_enum_name) {
            *ty
        } else if let Some(ty) = self.enum_types.get(enum_name) {
            *ty
        } else {
            // Try to compile on-demand
            self.compile_enum_defs(&[enum_def.clone()])?;
            *self.enum_types.get(actual_enum_name)
                .ok_or(format!("Enum type {} not found (tried {} and {})", enum_name, actual_enum_name, enum_name))?
        };
        
        // Allocate memory for enum
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
        
        // Store Tag
        let tag_ptr = self.builder.build_struct_gep(enum_ty, alloca, 0, "tag_ptr").map_err(|e| e.to_string())?;
        let tag_val = variant_idx as u64;
        self.builder.build_store(tag_ptr, self.context.i32_type().const_int(tag_val, false)).unwrap();
        
        // Store Payload if any
        if !args.is_empty() {
            match &variant_def.kind {
                VariantKind::Tuple(types) => {
                    let payload_ptr_raw = self.builder
                        .build_struct_gep(enum_ty, alloca, 1, "payload_ptr_raw")
                        .map_err(|e| e.to_string())?;
                    
                    let mut field_types_llvm: Vec<inkwell::types::BasicTypeEnum> = vec![];
                    for ty in types {
                        let llvm_ty = self.get_llvm_type(ty)?;
                        field_types_llvm.push(llvm_ty);
                    }
                    let variant_struct_ty = self.context.struct_type(&field_types_llvm, false);
                    
                    let payload_ptr = self.builder.build_pointer_cast(
                        payload_ptr_raw,
                        self.context.ptr_type(inkwell::AddressSpace::default()),
                        "payload_cast"
                    ).unwrap();
                    
                    for (idx, (arg, _f_ty)) in args.iter().zip(types.iter()).enumerate() {
                        let (val, _) = self.compile_expr(arg)?;
                        let f_ptr = self.builder.build_struct_gep(variant_struct_ty, payload_ptr, idx as u32, "field_ptr")
                            .map_err(|e| e.to_string())?;
                        self.builder.build_store(f_ptr, val).unwrap();
                    }
                }
                VariantKind::Struct(fields) => {
                    let payload_ptr_raw = self.builder
                        .build_struct_gep(enum_ty, alloca, 1, "payload_ptr_raw")
                        .map_err(|e| e.to_string())?;
                    
                    let mut field_types_llvm: Vec<inkwell::types::BasicTypeEnum> = vec![];
                    for (_, ty) in fields {
                        let llvm_ty = self.get_llvm_type(ty)?;
                        field_types_llvm.push(llvm_ty);
                    }
                    let variant_struct_ty = self.context.struct_type(&field_types_llvm, false);
                    
                    let payload_ptr = self.builder.build_pointer_cast(
                        payload_ptr_raw,
                        self.context.ptr_type(inkwell::AddressSpace::default()),
                        "payload_cast"
                    ).unwrap();
                    
                    for (idx, arg) in args.iter().enumerate() {
                        let (val, _) = self.compile_expr(arg)?;
                        let f_ptr = self.builder.build_struct_gep(variant_struct_ty, payload_ptr, idx as u32, "field_ptr")
                            .map_err(|e| e.to_string())?;
                        self.builder.build_store(f_ptr, val).unwrap();
                    }
                }
                _ => {}
            }
        }
        
        Ok((alloca.into(), Type::Enum(enum_name.to_string(), vec![])))
    }
}
