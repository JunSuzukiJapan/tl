use crate::compiler::codegen::type_manager::{CodeGenType, TypeManager};
use crate::compiler::codegen::expr;
use crate::compiler::codegen::CodeGenerator;
use crate::compiler::ast::Type;
use inkwell::values::BasicValueEnum;
use inkwell::IntPredicate;

pub fn register_io_types(manager: &mut TypeManager) {
    // File
    let mut file = CodeGenType::new("File");
    file.register_static_method("open", expr::StaticMethod::Evaluated(compile_file_open));
    file.register_static_method("exists", expr::StaticMethod::Evaluated(compile_file_exists));
    file.register_static_method("read", expr::StaticMethod::Evaluated(compile_file_read_static));
    file.register_static_method("write", expr::StaticMethod::Evaluated(compile_file_write));
    file.register_static_method("download", expr::StaticMethod::Evaluated(compile_file_download));
    file.register_static_method("read_binary", expr::StaticMethod::Evaluated(compile_file_read_binary));
    file.register_instance_method("read_string", expr::InstanceMethod::Evaluated(compile_file_read_string));
    file.register_instance_method("write_string", expr::InstanceMethod::Evaluated(compile_file_write_string));
    file.register_instance_method("close", expr::InstanceMethod::Evaluated(compile_file_close));
    manager.register_type(file);

    // Path
    let mut path = CodeGenType::new("Path");
    path.register_static_method("exists", expr::StaticMethod::Evaluated(compile_path_exists));
    manager.register_type(path);

    // Env
    let mut env = CodeGenType::new("Env");
    env.register_static_method("get", expr::StaticMethod::Evaluated(compile_env_get));
    manager.register_type(env);

    // Http
    let mut http = CodeGenType::new("Http");
    http.register_static_method("get", expr::StaticMethod::Evaluated(compile_http_get));
    manager.register_type(http);
}

fn compile_file_write<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 2 {
        return Err("File::write requires 2 arguments".into());
    }
    // args[0] is path, args[1] is content
    let path_val = args[0].0;
    let content_val = args[1].0;

    let fn_val = codegen
        .module
        .get_function("tl_write_file")
        .ok_or("tl_write_file not found")?;
    let call = codegen
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
    let ok = codegen
        .builder
        .build_int_compare(
            IntPredicate::EQ,
            res,
            codegen.context.i64_type().const_int(1, false),
            "file_write_bool",
        )
        .map_err(|e| e.to_string())?;
    Ok((ok.into(), Type::Bool))
}

fn compile_env_get<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 { return Err("Env::get requires 1 argument".into()); }
    let fn_val = codegen.module.get_function("tl_env_get").ok_or("tl_env_get not found")?;
    let call = codegen.builder.build_call(fn_val, &[args[0].0.into()], "env_get").map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from Env::get".into()),
    };
    Ok((res, Type::UserDefined("String".to_string(), vec![])))
}

fn compile_http_get<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
     if args.len() != 1 { return Err("Http::get requires 1 argument".into()); }
    let fn_val = codegen.module.get_function("tl_http_get").ok_or("tl_http_get not found")?;
    let call = codegen.builder.build_call(fn_val, &[args[0].0.into()], "http_get").map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from Http::get".into()),
    };
    Ok((res, Type::UserDefined("String".to_string(), vec![])))
}

fn compile_path_exists<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 { return Err("Path::exists requires 1 argument".into()); }
    
    // 1. Convert String (i8*) to PathBuf
    let tl_path_new = codegen.module.get_function("tl_path_new").ok_or("tl_path_new not found")?;
    let path_val = args[0].0; // String (i8*)
    let path_buf_call = codegen.builder.build_call(tl_path_new, &[path_val.into()], "path_new").map_err(|e| e.to_string())?;
    
    let path_buf_ptr = match path_buf_call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
        _ => return Err("Invalid return from tl_path_new".into()),
    };
    
    // 2. Call tl_path_exists
    let tl_path_exists = codegen.module.get_function("tl_path_exists").ok_or("tl_path_exists not found")?;
    let exists_call = codegen.builder.build_call(tl_path_exists, &[path_buf_ptr.into()], "path_exists").map_err(|e| e.to_string())?;
    
    let exists_val = match exists_call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from tl_path_exists".into()),
    };
    
    // 3. Free PathBuf
    let tl_path_free = codegen.module.get_function("tl_path_free").ok_or("tl_path_free not found")?;
    codegen.builder.build_call(tl_path_free, &[path_buf_ptr.into()], "").map_err(|e| e.to_string())?;

    // Return bool
    // tl_path_exists usually returns i1 or i8 (bool). 
    // We can just return it directly if it matches Type::Bool.
    Ok((exists_val, Type::Bool))
}

pub fn compile_file_open<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 2 { return Err("File::open requires 2 arguments".into()); }
    let fn_val = codegen.module.get_function("tl_file_open").ok_or("tl_file_open not found")?;
    let call = codegen.builder.build_call(fn_val, &[args[0].0.into(), args[1].0.into()], "file_open").map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from File::open".into()),
    };
    Ok((res, Type::UserDefined("File".to_string(), vec![])))
}

pub fn compile_file_exists<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 { return Err("File::exists requires 1 argument".into()); }
    let fn_val = codegen.module.get_function("tl_file_exists_i64").ok_or("tl_file_exists_i64 not found")?;
    let call = codegen.builder.build_call(fn_val, &[args[0].0.into()], "file_exists").map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v.into_int_value(),
        _ => return Err("Invalid return from File::exists".into()),
    };
    let ok = codegen.builder.build_int_compare(inkwell::IntPredicate::EQ, res, codegen.context.i64_type().const_int(1, false), "exists_bool").unwrap();
    Ok((ok.into(), Type::Bool))
}

pub fn compile_file_read_static<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 { return Err("File::read requires 1 argument".into()); }
    let fn_val = codegen.module.get_function("tl_read_file").ok_or("tl_read_file not found")?;
    let call = codegen.builder.build_call(fn_val, &[args[0].0.into()], "file_read").map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from File::read".into()),
    };
    Ok((res, Type::UserDefined("String".to_string(), vec![])))
}

pub fn compile_file_download<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 2 { return Err("File::download requires 2 arguments".into()); }
    let fn_val = codegen.module.get_function("tl_download_file").ok_or("tl_download_file not found")?;
    let call = codegen.builder.build_call(fn_val, &[args[0].0.into(), args[1].0.into()], "file_download").map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v.into_int_value(),
        _ => return Err("Invalid return from File::download".into()),
    };
    let ok = codegen.builder.build_int_compare(inkwell::IntPredicate::EQ, res, codegen.context.i64_type().const_int(1, false), "download_bool").unwrap();
    Ok((ok.into(), Type::Bool))
}

pub fn compile_file_read_binary<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 { return Err("File::read_binary requires 1 argument".into()); }
    let fn_val = codegen.module.get_function("tl_file_read_binary").ok_or("tl_file_read_binary not found")?;
    let call = codegen.builder.build_call(fn_val, &[args[0].0.into()], "file_read_binary").map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from File::read_binary".into()),
    };
    Ok((res, Type::Vec(Box::new(Type::U8))))
}

pub fn compile_file_read_string<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    instance_val: BasicValueEnum<'ctx>,
    _instance_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if !args.is_empty() { return Err("File::read_string takes no arguments".into()); }
    let fn_val = codegen.module.get_function("tl_file_read_string").ok_or("tl_file_read_string not found")?;
    let call = codegen.builder.build_call(fn_val, &[instance_val.into()], "file_read_str").map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from File::read_string".into()),
    };
    Ok((res, Type::UserDefined("String".to_string(), vec![])))
}

pub fn compile_file_write_string<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    instance_val: BasicValueEnum<'ctx>,
    _instance_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err("File::write_string requires 1 argument".into());
    }
    let (content_val, _) = args[0];
    let fn_val = codegen.module.get_function("tl_file_write_string").ok_or("tl_file_write_string not found")?;
    codegen.builder.build_call(
        fn_val,
        &[instance_val.into(), content_val.into()],
        "file_write_str",
    ).map_err(|e| e.to_string())?;
    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

pub fn compile_file_close<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    instance_val: BasicValueEnum<'ctx>,
    _instance_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if !args.is_empty() { return Err("File::close takes no arguments".into()); }
    let fn_val = codegen.module.get_function("tl_file_close").ok_or("tl_file_close not found")?;
    codegen.builder.build_call(fn_val, &[instance_val.into()], "file_close").map_err(|e| e.to_string())?;
    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}
