use crate::compiler::codegen::type_manager::{CodeGenType, TypeManager};
use crate::compiler::codegen::CodeGenerator;
use crate::compiler::ast::Type;
use inkwell::values::BasicValueEnum;
use inkwell::IntPredicate;



pub fn register_io_types(manager: &mut TypeManager) {
    let string_type = Type::String("String".to_string());
    
    // File
    let mut file = CodeGenType::new("File");
    
    // File::open(path: String, mode: String) -> File
    file.register_evaluated_static_method(
        "open", 
        compile_file_open, 
        vec![string_type.clone(), string_type.clone()],
        Type::Struct("File".to_string(), vec![])
    );
    
    // File::exists(path: String) -> Bool
    file.register_evaluated_static_method(
        "exists", 
        compile_file_exists, 
        vec![string_type.clone()],
        Type::Bool
    );
    
    // File::read(path: String) -> String
    file.register_evaluated_static_method(
        "read", 
        compile_file_read_static, 
        vec![string_type.clone()],
        string_type.clone()
    );
    
    // File::write(path: String, content: String) -> Bool
    file.register_evaluated_static_method(
        "write", 
        compile_file_write, 
        vec![string_type.clone(), string_type.clone()],
        Type::Bool
    );
    
    // File::download(url: String, path: String) -> Bool
    file.register_evaluated_static_method(
        "download", 
        compile_file_download, 
        vec![string_type.clone(), string_type.clone()],
        Type::Bool
    );

    // File::read_binary(path: String) -> Vec<u8>
    file.register_evaluated_static_method(
        "read_binary",
        compile_file_read_binary,
        vec![string_type.clone()],
        Type::Struct("Vec".to_string(), vec![Type::U8])
    );

    // File.read_string() -> String
    file.register_evaluated_instance_method(
        "read_string", 
        compile_file_read_string,
        vec![],
        string_type.clone()
    );
    
    // File.write_string(content: String) -> Void
    file.register_evaluated_instance_method(
        "write_string", 
        compile_file_write_string,
        vec![string_type.clone()],
        Type::Void
    );
    
    // File.close() -> Void
    file.register_evaluated_instance_method(
        "close", 
        compile_file_close,
        vec![],
        Type::Void
    );
    
    manager.register_type(file);

    // Path
    let mut path = CodeGenType::new("Path");
    // Path::exists(path: String) -> Bool
    path.register_evaluated_static_method(
        "exists", 
        compile_path_exists, 
        vec![string_type.clone()],
        Type::Bool
    );
    // Path::new(path: String) -> Path
    path.register_evaluated_static_method(
        "new",
        compile_path_new,
        vec![string_type.clone()],
        Type::Struct("Path".to_string(), vec![])
    );
    manager.register_type(path);

    // Env
    let mut env = CodeGenType::new("Env");
    // Env::get(key: String) -> String
    env.register_evaluated_static_method(
        "get", 
        compile_env_get, 
        vec![string_type.clone()],
        string_type.clone()
    );
    // Env::set(key: String, val: String) -> Void
    env.register_evaluated_static_method(
        "set",
        compile_env_set,
        vec![string_type.clone(), string_type.clone()],
        Type::Void
    );
    manager.register_type(env);

    // Http
    let mut http = CodeGenType::new("Http");
    // Http::get(url: String) -> String
    http.register_evaluated_static_method(
        "get", 
        compile_http_get, 
        vec![string_type.clone()],
        string_type.clone()
    );
    // Http::download(url: String, path: String) -> Bool
    http.register_evaluated_static_method(
        "download",
        compile_http_download,
        vec![string_type.clone(), string_type.clone()],
        Type::Bool
    );
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
    let (path_val, path_ty) = &args[0];
    let (content_val, content_ty) = &args[1];

    let extract_ptr = |codegen: &mut CodeGenerator<'ctx>, val: BasicValueEnum<'ctx>, ty: &Type| -> Result<inkwell::values::BasicMetadataValueEnum<'ctx>, String> {
        if matches!(ty, Type::String(_)) {
             let struct_ty = Type::String("String".to_string());
             let ptr_int = codegen.load_struct_i64_field(val, &struct_ty, "ptr")?;
             let ptr = codegen.builder.build_int_to_ptr(
                 ptr_int.into_int_value(),
                 codegen.context.ptr_type(inkwell::AddressSpace::default()),
                 "str_ptr"
             ).map_err(|e| e.to_string())?;
             Ok(ptr.into())
        } else {
             return Err(format!("Expected String argument, got {:?}", ty));
        }
    };

    let path_ptr = extract_ptr(codegen, *path_val, path_ty)?;
    let content_ptr = extract_ptr(codegen, *content_val, content_ty)?;

    let fn_val = codegen
        .module
        .get_function("tl_write_file")
        .ok_or("tl_write_file not found")?;
    let call = codegen
        .builder
        .build_call(
            fn_val,
            &[path_ptr, content_ptr],
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
    Ok((res, Type::String("String".to_string())))
}


    // ... existing compile functions ...

pub fn compile_path_new<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 { return Err("Path::new requires 1 argument".into()); }
    // 1. Convert String (i8*) to PathBuf
    let tl_path_new = codegen.module.get_function("tl_path_new").ok_or("tl_path_new not found")?;
    let (path_val, path_ty) = &args[0]; // String (i8*)
    
    let path_ptr_val = if matches!(path_ty, Type::String(_)) {
        if !path_val.is_pointer_value() {
            return Err(format!("Path::new path must be a pointer, got {:?}", path_val));
        }
        let ptr = path_val.into_pointer_value();
        // Just cast pointer to int to pass through
        codegen.builder.build_ptr_to_int(ptr, codegen.context.i64_type(), "ptr_int").map_err(|e| e.to_string())?
    } else {
         return Err("Path::new argument must be String".into());
    };
    let path_ptr = codegen.builder.build_int_to_ptr(
        path_ptr_val,
        codegen.context.ptr_type(inkwell::AddressSpace::default()),
        "path_ptr_pb"
    ).map_err(|e| e.to_string())?;

    let path_buf_call = codegen.builder.build_call(tl_path_new, &[path_ptr.into()], "path_new").map_err(|e| e.to_string())?;
    
    let path_buf_ptr = match path_buf_call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from tl_path_new".into()),
    };
    
    Ok((path_buf_ptr, Type::Struct("Path".to_string(), vec![])))
}

pub fn compile_env_set<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 2 { return Err("Env::set requires 2 arguments".into()); }
    let fn_val = codegen.module.get_function("tl_env_set").ok_or("tl_env_set not found (runtime)")?;

    // helper to extract string ptr
    let mut extract_ptr = |v: BasicValueEnum<'ctx>, t: &Type| -> Result<inkwell::values::BasicMetadataValueEnum<'ctx>, String> {
         if matches!(t, Type::String(_)) {
              let v_struct = codegen.load_struct_i64_field(v, t, "ptr")?;
              let ptr = codegen.builder.build_int_to_ptr(v_struct.into_int_value(), codegen.context.ptr_type(inkwell::AddressSpace::default()), "str_ptr").map_err(|e| e.to_string())?;
              Ok(ptr.into())
         } else {
              Err("Expected String".into())
         }
    };

    let key_ptr = extract_ptr(args[0].0, &args[0].1)?;
    let val_ptr = extract_ptr(args[1].0, &args[1].1)?;

    codegen.builder.build_call(fn_val, &[key_ptr, val_ptr], "").map_err(|e| e.to_string())?;
    Ok((codegen.context.i64_type().const_int(0, false).into(), Type::Void))
}

pub fn compile_http_download<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    // Reuse compile_file_download or same logic
    compile_file_download(codegen, args, _target)
}

pub fn compile_http_get<'ctx>(
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
    Ok((res, Type::String("String".to_string())))
}

pub fn compile_path_exists<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 { return Err("Path::exists requires 1 argument".into()); }
    
    // 1. Convert String (i8*) to PathBuf
    // 1. Convert String (i8*) to PathBuf
    let tl_path_new = codegen.module.get_function("tl_path_new").ok_or("tl_path_new not found")?;
    let (path_val, path_ty) = &args[0]; // String (i8*)
    
    let path_ptr_val = if matches!(path_ty, Type::String(_)) {
        if !path_val.is_pointer_value() {
            return Err(format!("File::open path must be a pointer, got {:?}", path_val));
        }
        if !path_val.is_pointer_value() {
            return Err(format!("File::open path must be a pointer, got {:?}", path_val));
        }
        let ptr = path_val.into_pointer_value();
        // Just cast pointer to int to pass through
        codegen.builder.build_ptr_to_int(ptr, codegen.context.i64_type(), "ptr_int").map_err(|e| e.to_string())?
    } else {
         return Err("Path::exists argument must be String".into());
    };
    let path_ptr = codegen.builder.build_int_to_ptr(
        path_ptr_val,
        codegen.context.ptr_type(inkwell::AddressSpace::default()),
        "path_ptr_pb"
    ).map_err(|e| e.to_string())?;

    let path_buf_call = codegen.builder.build_call(tl_path_new, &[path_ptr.into()], "path_new").map_err(|e| e.to_string())?;
    
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
    
    // Helper to extract char* from String
    let extract_ptr = |codegen: &mut CodeGenerator<'ctx>, val: BasicValueEnum<'ctx>, ty: &Type| -> Result<inkwell::values::BasicMetadataValueEnum<'ctx>, String> {
        let ptr_int_val = if matches!(ty, Type::String(_)) {
             let struct_ty = Type::String("String".to_string());
             let v = codegen.load_struct_i64_field(val, &struct_ty, "ptr")?;
             v.into_int_value()
        } else {
             return Err(format!("Expected String argument, got {:?}", ty));
        };
        let ptr = codegen.builder.build_int_to_ptr(
            ptr_int_val,
            codegen.context.ptr_type(inkwell::AddressSpace::default()),
            "str_ptr"
        ).map_err(|e| e.to_string())?;
        Ok(ptr.into())
    };

    // Arg 0: Path
    let (path_val, path_ty) = &args[0];
    let path_ptr = extract_ptr(codegen, *path_val, path_ty)?;

    // Arg 1: Mode
    let (mode_val, mode_ty) = &args[1];
    let mode_ptr = extract_ptr(codegen, *mode_val, mode_ty)?;

    let call = codegen.builder.build_call(fn_val, &[path_ptr, mode_ptr], "file_open").map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from File::open".into()),
    };
    Ok((res, Type::Struct("File".to_string(), vec![])))
}

pub fn compile_file_exists<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 { return Err("File::exists requires 1 argument".into()); }
    let fn_val = codegen.module.get_function("tl_file_exists_i64").ok_or("tl_file_exists_i64 not found")?;
    
    let (path_val, path_ty) = &args[0];
    
    let ptr_int_val = if matches!(path_ty, Type::String(_)) {
         let struct_ty = Type::String("String".to_string());
         let v = codegen.load_struct_i64_field(*path_val, &struct_ty, "ptr")?;
         v.into_int_value()
    } else {
         return Err(format!("Expected String argument, got {:?}", path_ty));
    };
    let path_ptr = codegen.builder.build_int_to_ptr(
        ptr_int_val,
        codegen.context.ptr_type(inkwell::AddressSpace::default()),
        "str_ptr"
    ).map_err(|e| e.to_string())?;

    let call = codegen.builder.build_call(fn_val, &[path_ptr.into()], "file_exists").map_err(|e| e.to_string())?;
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
    
    let (path_val, path_ty) = &args[0];
    let path_ptr_val = if matches!(path_ty, Type::String(_)) {
         let struct_ty = Type::String("String".to_string());
         let v = codegen.load_struct_i64_field(*path_val, &struct_ty, "ptr")?;
         v.into_int_value()
    } else {
         return Err(format!("Expected String argument, got {:?}", path_ty));
    };

    let path_ptr = codegen.builder.build_int_to_ptr(
        path_ptr_val,
        codegen.context.ptr_type(inkwell::AddressSpace::default()),
        "path_ptr"
    ).map_err(|e| e.to_string())?;

    let fn_val = codegen.module.get_function("tl_read_file").ok_or("tl_read_file not found")?;
    let call = codegen.builder.build_call(fn_val, &[path_ptr.into()], "file_read").map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from File::read".into()),
    };
    Ok((res, Type::String("String".to_string())))
}

pub fn compile_file_download<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 2 { return Err("File::download requires 2 arguments".into()); }
    
    let extract_ptr = |codegen: &mut CodeGenerator<'ctx>, val: BasicValueEnum<'ctx>, ty: &Type| -> Result<inkwell::values::BasicMetadataValueEnum<'ctx>, String> {
        let ptr_int_val = if matches!(ty, Type::String(_)) {
             let struct_ty = Type::String("String".to_string());
             let v = codegen.load_struct_i64_field(val, &struct_ty, "ptr")?;
             v.into_int_value()
        } else {
             return Err(format!("Expected String argument, got {:?}", ty));
        };
        let ptr = codegen.builder.build_int_to_ptr(
            ptr_int_val,
            codegen.context.ptr_type(inkwell::AddressSpace::default()),
            "str_ptr"
        ).map_err(|e| e.to_string())?;
        Ok(ptr.into())
    };

    let (url_val, url_ty) = &args[0];
    let (path_val, path_ty) = &args[1];

    let url_ptr = extract_ptr(codegen, *url_val, url_ty)?;
    let path_ptr = extract_ptr(codegen, *path_val, path_ty)?;

    let fn_val = codegen.module.get_function("tl_download_file").ok_or("tl_download_file not found")?;
    let call = codegen.builder.build_call(fn_val, &[url_ptr, path_ptr], "file_download").map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v.into_int_value(),
        _ => return Err("Invalid return from File::download".into()),
    };
    let ok = codegen.builder.build_int_compare(inkwell::IntPredicate::EQ, res, codegen.context.i64_type().const_int(1, false), "download_bool").unwrap();
    Ok((ok.into(), Type::Bool))
}

#[allow(deprecated)]
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
    
    // Cast void* to Vec*
    let vec_struct_ty = codegen.struct_types.get("Vec").ok_or("Vec type not found")?;
    let vec_ptr_ty = vec_struct_ty.ptr_type(inkwell::AddressSpace::default());
    let vec_ptr = codegen.builder.build_pointer_cast(
        res.into_pointer_value(),
        vec_ptr_ty,
        "vec_cast"
    ).map_err(|e| e.to_string())?;

    Ok((vec_ptr.into(), Type::Struct("Vec".to_string(), vec![Type::U8])))
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
    Ok((res, Type::String("String".to_string())))
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
