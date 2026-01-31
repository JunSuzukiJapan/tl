use crate::compiler::codegen::type_manager::{CodeGenType, TypeManager};
use crate::compiler::codegen::expr;
use crate::compiler::codegen::CodeGenerator;
use crate::compiler::ast::Type;
use inkwell::values::BasicValueEnum;

pub fn register_system_types(manager: &mut TypeManager) {
    let mut system = CodeGenType::new("System");
    
    // Time
    system.register_static_method("time", expr::StaticMethod::Evaluated(compile_system_time));
    system.register_static_method("sleep", expr::StaticMethod::Evaluated(compile_system_sleep));
    
    // Memory / Stats
    system.register_static_method("memory_mb", expr::StaticMethod::Evaluated(compile_memory_mb));
    system.register_static_method("pool_count", expr::StaticMethod::Evaluated(compile_pool_count));
    system.register_static_method("refcount_count", expr::StaticMethod::Evaluated(compile_refcount_count));
    system.register_static_method("scope_depth", expr::StaticMethod::Evaluated(compile_scope_depth));
    
    // Metal
    system.register_static_method("metal_pool_bytes", expr::StaticMethod::Evaluated(compile_metal_pool_bytes));
    system.register_static_method("metal_pool_mb", expr::StaticMethod::Evaluated(compile_metal_pool_mb));
    system.register_static_method("metal_pool_count", expr::StaticMethod::Evaluated(compile_metal_pool_count));
    system.register_static_method("metal_sync", expr::StaticMethod::Evaluated(compile_metal_sync));

    manager.register_type(system);
}

fn compile_system_time<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if !args.is_empty() { return Err("System::time takes no arguments".into()); }
    let fn_val = codegen.module.get_function("tl_system_time").ok_or("tl_system_time not found")?;
    let call = codegen.builder.build_call(fn_val, &[], "sys_time").map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from System::time".into()),
    };
    Ok((res, Type::F32))
}

fn compile_system_sleep<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 { return Err("System::sleep requires 1 argument (seconds)".into()); }
    let fn_val = codegen.module.get_function("tl_system_sleep").ok_or("tl_system_sleep not found")?;
    // Arg should be f32
    let arg_val = args[0].0;
     // If arg is int, cast to float? Semantic analyzer handles this? 
     // Usually semantic analyzer casts, but let's assume valid input for now.
    codegen.builder.build_call(fn_val, &[arg_val.into()], "").map_err(|e| e.to_string())?;
    
    // Return void (0 i64)
    let void_val = codegen.context.i64_type().const_int(0, false).into();
    Ok((void_val, Type::Void))
}

// Helper for 0-arg I64 return functions
fn compile_simple_i64_call<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    fn_name: &str,
    debug_name: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen.module.get_function(fn_name).ok_or(format!("{} not found", fn_name))?;
    let call = codegen.builder.build_call(fn_val, &[], debug_name).map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err(format!("Invalid return from {}", debug_name)),
    };
    Ok((res, Type::I64))
}

fn compile_memory_mb<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if !args.is_empty() { return Err("System::memory_mb takes no arguments".into()); }
    compile_simple_i64_call(codegen, "tl_get_memory_mb", "mem_mb")
}

fn compile_pool_count<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if !args.is_empty() { return Err("System::pool_count takes no arguments".into()); }
    compile_simple_i64_call(codegen, "tl_get_pool_count", "pool_count")
}

fn compile_refcount_count<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if !args.is_empty() { return Err("System::refcount_count takes no arguments".into()); }
    compile_simple_i64_call(codegen, "tl_get_refcount_count", "refcount_count")
}

fn compile_scope_depth<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if !args.is_empty() { return Err("System::scope_depth takes no arguments".into()); }
    compile_simple_i64_call(codegen, "tl_get_scope_depth", "scope_depth")
}

fn compile_metal_pool_bytes<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if !args.is_empty() { return Err("System::metal_pool_bytes takes no arguments".into()); }
    compile_simple_i64_call(codegen, "tl_get_metal_pool_bytes", "metal_pool_bytes")
}

fn compile_metal_pool_mb<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if !args.is_empty() { return Err("System::metal_pool_mb takes no arguments".into()); }
    compile_simple_i64_call(codegen, "tl_get_metal_pool_mb", "metal_pool_mb")
}

fn compile_metal_pool_count<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if !args.is_empty() { return Err("System::metal_pool_count takes no arguments".into()); }
    compile_simple_i64_call(codegen, "tl_get_metal_pool_count", "metal_pool_count")
}

fn compile_metal_sync<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if !args.is_empty() { return Err("System::metal_sync takes no arguments".into()); }
    let fn_val = codegen.module.get_function("tl_metal_sync").ok_or("tl_metal_sync not found")?;
    codegen.builder.build_call(fn_val, &[], "metal_sync").map_err(|e| e.to_string())?;
    
    let void_val = codegen.context.i64_type().const_int(0, false).into();
    Ok((void_val, Type::Void))
}
