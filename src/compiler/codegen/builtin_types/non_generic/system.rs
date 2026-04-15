use crate::compiler::codegen::type_manager::{CodeGenType, TypeManager};
use crate::compiler::codegen::CodeGenerator;
use crate::compiler::ast::Type;
use inkwell::values::BasicValueEnum;

pub fn register_system_types(manager: &mut TypeManager) {
    let mut system = CodeGenType::new("System");

    // 引数を持つメソッドは個別に登録
    system.register_evaluated_static_method("time",    compile_system_time,    vec![],            Type::F32);
    system.register_evaluated_static_method("sleep",   compile_system_sleep,   vec![Type::F32],   Type::Void);
    system.register_evaluated_static_method("exit",    compile_system_exit,    vec![Type::I64],   Type::Void);
    system.register_evaluated_static_method("platform",compile_system_platform,vec![],            Type::String("String".to_string()));
    system.register_evaluated_static_method(
        "command", compile_system_command,
        vec![Type::String("String".to_string())],
        Type::String("String".to_string()),
    );

    // 0引数・I64戻り値メソッド（fnポインタで登録）
    system.register_evaluated_static_method("memory_bytes",    compile_memory_bytes,    vec![], Type::I64);
    system.register_evaluated_static_method("pool_count",      compile_pool_count,      vec![], Type::I64);
    system.register_evaluated_static_method("refcount_count",  compile_refcount_count,  vec![], Type::I64);
    system.register_evaluated_static_method("scope_depth",     compile_scope_depth,     vec![], Type::I64);
    system.register_evaluated_static_method("metal_pool_bytes",compile_metal_pool_bytes,vec![], Type::I64);
    system.register_evaluated_static_method("metal_pool_count",compile_metal_pool_count,vec![], Type::I64);

    // 0引数・F64戻り値メソッド
    system.register_evaluated_static_method("memory_mb",    compile_memory_mb,    vec![], Type::F64);
    system.register_evaluated_static_method("metal_pool_mb",compile_metal_pool_mb,vec![], Type::F64);

    // Void戻り値メソッド
    system.register_evaluated_static_method("metal_sync", compile_metal_sync, vec![], Type::Void);
    system.register_evaluated_static_method("mem_report", compile_mem_report, vec![], Type::Void);

    // Internal
    system.register_evaluated_static_method("free_hashmap", compile_free_hashmap, vec![Type::I64], Type::Void);
    system.register_evaluated_static_method("free_memory",  compile_free_memory,  vec![Type::I64], Type::Void);

    manager.register_type(system);
}

fn compile_free_memory<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 { return Err("System::free_memory requires 1 argument (ptr)".into()); }
    
    let fn_val = codegen.module.get_function("tl_mem_free").ok_or("tl_mem_free not found")?;
    
    // Arg is i64, need to cast to void*
    let ptr_int = args[0].0.into_int_value();
    let ptr = codegen.builder.build_int_to_ptr(
        ptr_int, 
        codegen.context.ptr_type(inkwell::AddressSpace::default()), 
        "mem_ptr"
    ).map_err(|e| e.to_string())?;

    codegen.builder.build_call(fn_val, &[ptr.into()], "").map_err(|e| e.to_string())?;
    
    Ok((codegen.context.i64_type().const_int(0, false).into(), Type::Void))
}

fn compile_free_hashmap<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 { return Err("System::free_hashmap requires 1 argument (ptr)".into()); }
    
    let fn_val = codegen.module.get_function("tl_hashmap_free").ok_or("tl_hashmap_free not found")?;
    
    // Arg is i64, need to cast to void* / hashmap*
    let ptr_int = args[0].0.into_int_value();
    let ptr = codegen.builder.build_int_to_ptr(
        ptr_int, 
        codegen.context.ptr_type(inkwell::AddressSpace::default()), 
        "hashmap_ptr"
    ).map_err(|e| e.to_string())?;

    codegen.builder.build_call(fn_val, &[ptr.into()], "").map_err(|e| e.to_string())?;
    
    Ok((codegen.context.i64_type().const_int(0, false).into(), Type::Void))
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
    
    let void_val = codegen.context.i64_type().const_int(0, false).into();
    Ok((void_val, Type::Void))
}

fn compile_system_exit<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 { return Err("System::exit requires 1 argument (code)".into()); }
    let fn_val = codegen.module.get_function("tl_system_exit").unwrap_or_else(|| {
         let void_ty = codegen.context.void_type();
         let i64_ty = codegen.context.i64_type();
         let ft = void_ty.fn_type(&[i64_ty.into()], false);
         codegen.module.add_function("tl_system_exit", ft, None)
    });
    
    // fn_val is FunctionValue directly

    let arg_val = args[0].0;
    let i64_val = if arg_val.is_int_value() {
         arg_val.into_int_value()
    } else {
         return Err("System::exit argument must be int".into());
    };

    codegen.builder.build_call(fn_val, &[i64_val.into()], "").map_err(|e| e.to_string())?;
    
    // Return void (unreachable ideally, but void for checks)
    let void_val = codegen.context.i64_type().const_int(0, false).into();
    Ok((void_val, Type::Void))
}

// Helper for System::platform
fn compile_system_platform<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if !args.is_empty() { return Err("System::platform takes no arguments".into()); }
    let fn_val = codegen.module.get_function("tl_system_platform").ok_or("tl_system_platform not found")?;
    let call = codegen.builder.build_call(fn_val, &[], "sys_platform").map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from System::platform".into()),
    };
    Ok((res, Type::String("String".to_string())))
}

// Helper for System::command
fn compile_system_command<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 { return Err("System::command requires 1 argument (cmd)".into()); }
    let fn_val = codegen.module.get_function("tl_system_command").ok_or("tl_system_command not found")?;
    
    let (cmd_val, cmd_ty) = &args[0];
    let cmd_ptr_val = if matches!(cmd_ty, Type::String(_)) {
        let struct_ty = Type::String("String".to_string());
        codegen.load_struct_i64_field(*cmd_val, &struct_ty, "ptr")?.into_int_value()
    } else {
        return Err(format!("Expected String argument, got {:?}", cmd_ty));
    };
    let cmd_ptr = codegen.builder.build_int_to_ptr(
        cmd_ptr_val,
        codegen.context.ptr_type(inkwell::AddressSpace::default()),
        "cmd_ptr"
    ).map_err(|e| e.to_string())?;

    let call = codegen.builder.build_call(fn_val, &[cmd_ptr.into()], "sys_command").map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from System::command".into()),
    };
    Ok((res, Type::String("String".to_string())))
}

/// 0引数でI64を返すFFI関数を呼ぶ共通ヘルパー。
fn compile_simple_i64_call<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    fn_name: &str,
    debug_name: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen.module.get_function(fn_name).ok_or_else(|| format!("{} not found", fn_name))?;
    let call = codegen.builder.build_call(fn_val, &[], debug_name).map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err(format!("Invalid return from {}", debug_name)),
    };
    Ok((res, Type::I64))
}

/// 0引数でF64を返すFFI関数を呼ぶ共通ヘルパー。
fn compile_simple_f64_call<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    fn_name: &str,
    debug_name: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let fn_val = codegen.module.get_function(fn_name).ok_or_else(|| format!("{} not found", fn_name))?;
    let call = codegen.builder.build_call(fn_val, &[], debug_name).map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err(format!("Invalid return from {}", debug_name)),
    };
    Ok((res, Type::F64))
}


/// I64/F64 返り値の 0 引数 System メソッド群（fnポインタとして登録するための個別ラッパー）。
/// 実装は compile_simple_i64_call / compile_simple_f64_call に委譲。
macro_rules! simple_sys_i64 {
    ($fn_name:ident, $ffi:expr, $method:expr) => {
        fn $fn_name<'ctx>(
            codegen: &mut CodeGenerator<'ctx>,
            args: Vec<(BasicValueEnum<'ctx>, Type)>,
            _target: Option<&Type>,
        ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
            if !args.is_empty() { return Err(concat!("System::", $method, " takes no arguments").into()); }
            compile_simple_i64_call(codegen, $ffi, $method)
        }
    };
}
macro_rules! simple_sys_f64 {
    ($fn_name:ident, $ffi:expr, $method:expr) => {
        fn $fn_name<'ctx>(
            codegen: &mut CodeGenerator<'ctx>,
            args: Vec<(BasicValueEnum<'ctx>, Type)>,
            _target: Option<&Type>,
        ) -> Result<(BasicValueEnum<'ctx>, Type), String> {
            if !args.is_empty() { return Err(concat!("System::", $method, " takes no arguments").into()); }
            compile_simple_f64_call(codegen, $ffi, $method)
        }
    };
}

simple_sys_i64!(compile_memory_bytes,    "tl_get_memory_bytes",    "memory_bytes");
simple_sys_i64!(compile_pool_count,       "tl_get_pool_count",      "pool_count");
simple_sys_i64!(compile_refcount_count,   "tl_get_refcount_count",  "refcount_count");
simple_sys_i64!(compile_scope_depth,      "tl_get_scope_depth",     "scope_depth");
simple_sys_i64!(compile_metal_pool_bytes, "tl_get_metal_pool_bytes","metal_pool_bytes");
simple_sys_i64!(compile_metal_pool_count, "tl_get_metal_pool_count","metal_pool_count");
simple_sys_f64!(compile_memory_mb,        "tl_get_memory_mb",       "memory_mb");
simple_sys_f64!(compile_metal_pool_mb,    "tl_get_metal_pool_mb",   "metal_pool_mb");


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

fn compile_mem_report<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if !args.is_empty() { return Err("System::mem_report takes no arguments".into()); }
    let fn_val = codegen.module.get_function("tl_system_mem_report").ok_or("tl_system_mem_report not found")?;
    codegen.builder.build_call(fn_val, &[], "mem_report").map_err(|e| e.to_string())?;
    
    let void_val = codegen.context.i64_type().const_int(0, false).into();
    Ok((void_val, Type::Void))
}
