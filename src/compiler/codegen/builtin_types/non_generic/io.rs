use crate::compiler::error::{TlError, CodegenErrorKind};
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

    // File::read_binary(path: String) -> Result<Vec<u8>, TlError>
    file.register_evaluated_static_method(
        "read_binary",
        compile_file_read_binary,
        vec![string_type.clone()],
        Type::Enum("Result".to_string(), vec![Type::Struct("Vec".to_string(), vec![Type::U8]), Type::String("String".to_string())])
    );
    // File::append(path: String, content: String) -> Bool
    file.register_evaluated_static_method(
        "append", 
        compile_file_append, 
        vec![string_type.clone(), string_type.clone()],
        Type::Bool
    );
    // File::delete(path: String) -> Bool
    file.register_evaluated_static_method(
        "delete", 
        compile_file_delete, 
        vec![string_type.clone()],
        Type::Bool
    );
    // File::create_dir(path: String) -> Bool
    file.register_evaluated_static_method(
        "create_dir", 
        compile_file_create_dir, 
        vec![string_type.clone()],
        Type::Bool
    );
    // File::list_dir(path: String) -> Vec<String>
    file.register_evaluated_static_method(
        "list_dir", 
        compile_file_list_dir, 
        vec![string_type.clone()],
        Type::Struct("Vec".to_string(), vec![string_type.clone()])
    );

    // File::write_binary(path: String, data: Vec<u8>) -> Bool
    file.register_evaluated_static_method(
        "write_binary",
        compile_file_write_binary,
        vec![string_type.clone(), Type::Struct("Vec".to_string(), vec![Type::U8])],
        Type::Bool
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
    // Instance methods for Path (signature only for semantics check)
    path.register_instance_signature("is_dir", vec![], Type::Bool);
    path.register_instance_signature("is_file", vec![], Type::Bool);
    path.register_instance_signature("exists", vec![], Type::Bool);
    path.register_instance_signature("to_string", vec![], string_type.clone());

    // Evaluated instance methods for Path
    path.register_evaluated_instance_method("parent", compile_path_parent, vec![], string_type.clone());
    path.register_evaluated_instance_method("file_name", compile_path_file_name, vec![], string_type.clone());
    path.register_evaluated_instance_method("extension", compile_path_extension, vec![], string_type.clone());
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

    // Map (for GGUF tensor loading)
    let any_tensor = Type::Tensor(Box::new(Type::F32), 0);
    let mut map = CodeGenType::new("Map");
    // Map::load(path: String) -> Map  (handled in expr.rs via tl_gguf_load)
    map.register_static_signature("load", vec![string_type.clone()], Type::Struct("Map".to_string(), vec![]));
    // Instance methods (implemented in map_methods.rs)
    use super::map_methods;
    map.register_evaluated_instance_method("get", map_methods::compile_get, vec![string_type.clone()], any_tensor.clone());
    map.register_evaluated_instance_method("get_1d", map_methods::compile_get_1d, vec![string_type.clone()], Type::Tensor(Box::new(Type::F32), 1));
    map.register_evaluated_instance_method("get_quantized", map_methods::compile_get_quantized, vec![string_type.clone()], Type::Tensor(Box::new(Type::I8), 2));
    map.register_evaluated_instance_method("set", map_methods::compile_set, vec![string_type.clone(), any_tensor.clone()], Type::Void);
    map.register_evaluated_instance_method("metadata", map_methods::compile_metadata, vec![], Type::Struct("Map".to_string(), vec![]));
    map.register_evaluated_instance_method("get_i64", map_methods::compile_get_i64, vec![string_type.clone()], Type::I64);
    map.register_evaluated_instance_method("get_string", map_methods::compile_get_string, vec![string_type.clone()], string_type.clone());
    map.register_evaluated_instance_method("get_vec_string", map_methods::compile_get_vec_string, vec![string_type.clone()], Type::Struct("Vec".to_string(), vec![string_type.clone()]));
    manager.register_type(map);
}

fn compile_file_write<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 2 {
        return Err(TlError::from(CodegenErrorKind::Internal("File::write requires 2 arguments".to_string())));
    }
    // args[0] is path, args[1] is content
    let (path_val, path_ty) = &args[0];
    let (content_val, content_ty) = &args[1];

    let extract_ptr = |codegen: &mut CodeGenerator<'ctx>, val: BasicValueEnum<'ctx>, ty: &Type| -> Result<inkwell::values::BasicMetadataValueEnum<'ctx>, TlError> {
        if matches!(ty, Type::String(_)) {
             let struct_ty = Type::String("String".to_string());
             let ptr_int = codegen.load_struct_i64_field(val, &struct_ty, "ptr")?;
             let ptr = codegen.builder.build_int_to_ptr(
                 ptr_int.into_int_value(),
                 codegen.context.ptr_type(inkwell::AddressSpace::default()),
                 "str_ptr"
             ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
             Ok(ptr.into())
        } else {
             return Err(TlError::from(CodegenErrorKind::Internal(format!("Expected String argument, got {:?}", ty))));
        }
    };

    let path_ptr = extract_ptr(codegen, *path_val, path_ty)?;
    let content_ptr = extract_ptr(codegen, *content_val, content_ty)?;

    let fn_val = codegen
        .module
        .get_function("tl_write_file")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_write_file not found".to_string())))?;
    let call = codegen
        .builder
        .build_call(
            fn_val,
            &[path_ptr, content_ptr],
            "file_write",
        )
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v.into_int_value(),
        _ => return Err(CodegenErrorKind::Internal("Invalid return from File::write".to_string()).into()),
    };
    let ok = codegen
        .builder
        .build_int_compare(
            IntPredicate::EQ,
            res,
            codegen.context.i64_type().const_int(1, false),
            "file_write_bool",
        )
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((ok.into(), Type::Bool))
}

fn compile_env_get<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 { return Err(CodegenErrorKind::Internal("Env::get requires 1 argument".to_string()).into()); }
    let fn_val = codegen.module.get_function("tl_env_get").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_env_get not found".to_string())))?;
    let (key_val, key_ty) = &args[0];
    let ptr_int = if matches!(key_ty, Type::String(_)) {
        codegen.load_struct_i64_field(*key_val, &Type::String("String".to_string()), "ptr")?.into_int_value()
    } else {
        return Err(CodegenErrorKind::Internal("Env::get key must be String".to_string()).into());
    };
    let key_ptr = codegen.builder.build_int_to_ptr(ptr_int, codegen.context.ptr_type(inkwell::AddressSpace::default()), "key_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let call = codegen.builder.build_call(fn_val, &[key_ptr.into()], "env_get").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err(CodegenErrorKind::Internal("Invalid return from Env::get".to_string()).into()),
    };
    Ok((res, Type::String("String".to_string())))
}


    // ... existing compile functions ...

pub fn compile_path_new<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 { return Err(CodegenErrorKind::Internal("Path::new requires 1 argument".to_string()).into()); }
    // 1. Convert String (i8*) to PathBuf
    let tl_path_new = codegen.module.get_function("tl_path_new").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_path_new not found".to_string())))?;
    let (path_val, path_ty) = &args[0]; // String (i8*)
    
    let path_ptr_val = if matches!(path_ty, Type::String(_)) {
        let struct_ty = Type::String("String".to_string());
        // Extract ptr field from StringStruct
        let ptr_val = codegen.load_struct_i64_field(*path_val, &struct_ty, "ptr")?;
        ptr_val.into_int_value()
    } else {
         return Err(TlError::from(CodegenErrorKind::Internal("Path::new argument must be String".to_string())));
    };
    let path_ptr = codegen.builder.build_int_to_ptr(
        path_ptr_val,
        codegen.context.ptr_type(inkwell::AddressSpace::default()),
        "path_ptr_pb"
    ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

    let path_buf_call = codegen.builder.build_call(tl_path_new, &[path_ptr.into()], "path_new").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    
    let path_buf_ptr = match path_buf_call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err(CodegenErrorKind::Internal("Invalid return from tl_path_new".to_string()).into()),
    };
    
    Ok((path_buf_ptr, Type::Struct("Path".to_string(), vec![])))
}

pub fn compile_env_set<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 2 { return Err(CodegenErrorKind::Internal("Env::set requires 2 arguments".to_string()).into()); }
    let fn_val = codegen.module.get_function("tl_env_set").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_env_set not found (runtime)".to_string())))?;

    // helper to extract string ptr
    let mut extract_ptr = |v: BasicValueEnum<'ctx>, t: &Type| -> Result<inkwell::values::BasicMetadataValueEnum<'ctx>, TlError> {
         if matches!(t, Type::String(_)) {
              let v_struct = codegen.load_struct_i64_field(v, t, "ptr")?;
              let ptr = codegen.builder.build_int_to_ptr(v_struct.into_int_value(), codegen.context.ptr_type(inkwell::AddressSpace::default()), "str_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
              Ok(ptr.into())
         } else {
              Err(TlError::from(CodegenErrorKind::Internal("Expected String".to_string())))
         }
    };

    let key_ptr = extract_ptr(args[0].0, &args[0].1)?;
    let val_ptr = extract_ptr(args[1].0, &args[1].1)?;

    codegen.builder.build_call(fn_val, &[key_ptr, val_ptr], "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((codegen.context.i64_type().const_int(0, false).into(), Type::Void))
}

pub fn compile_http_download<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    // Reuse compile_file_download or same logic
    compile_file_download(codegen, args, _target)
}

pub fn compile_http_get<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 { return Err(CodegenErrorKind::Internal("Http::get requires 1 argument".to_string()).into()); }
    let fn_val = codegen.module.get_function("tl_http_get").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_http_get not found".to_string())))?;
    let (url_val, url_ty) = &args[0];
    let ptr_int = if matches!(url_ty, Type::String(_)) {
        codegen.load_struct_i64_field(*url_val, &Type::String("String".to_string()), "ptr")?.into_int_value()
    } else {
        return Err(CodegenErrorKind::Internal("Http::get url must be String".to_string()).into());
    };
    let url_ptr = codegen.builder.build_int_to_ptr(ptr_int, codegen.context.ptr_type(inkwell::AddressSpace::default()), "url_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let call = codegen.builder.build_call(fn_val, &[url_ptr.into()], "http_get").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err(CodegenErrorKind::Internal("Invalid return from Http::get".to_string()).into()),
    };
    Ok((res, Type::String("String".to_string())))
}

pub fn compile_path_exists<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 { return Err(CodegenErrorKind::Internal("Path::exists requires 1 argument".to_string()).into()); }
    
    // 1. Convert String (i8*) to PathBuf
    // 1. Convert String (i8*) to PathBuf
    let tl_path_new = codegen.module.get_function("tl_path_new").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_path_new not found".to_string())))?;
    let (path_val, path_ty) = &args[0]; // String (i8*)
    
    let path_ptr_val = if matches!(path_ty, Type::String(_)) {
        let struct_ty = Type::String("String".to_string());
        let v = codegen.load_struct_i64_field(*path_val, &struct_ty, "ptr")?;
        v.into_int_value()
    } else {
        return Err(TlError::from(CodegenErrorKind::Internal("Path::exists argument must be String".to_string())));
    };
    let path_ptr = codegen.builder.build_int_to_ptr(
        path_ptr_val,
        codegen.context.ptr_type(inkwell::AddressSpace::default()),
        "path_ptr_pb"
    ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

    let path_buf_call = codegen.builder.build_call(tl_path_new, &[path_ptr.into()], "path_new").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    
    let path_buf_ptr = match path_buf_call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
        _ => return Err(CodegenErrorKind::Internal("Invalid return from tl_path_new".to_string()).into()),
    };
    
    // 2. Call tl_path_exists
    let tl_path_exists = codegen.module.get_function("tl_path_exists").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_path_exists not found".to_string())))?;
    let exists_call = codegen.builder.build_call(tl_path_exists, &[path_buf_ptr.into()], "path_exists").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    
    let exists_val = match exists_call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err(CodegenErrorKind::Internal("Invalid return from tl_path_exists".to_string()).into()),
    };
    
    // 3. Free PathBuf
    let tl_path_free = codegen.module.get_function("tl_path_free").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_path_free not found".to_string())))?;
    codegen.builder.build_call(tl_path_free, &[path_buf_ptr.into()], "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

    // Return bool
    // tl_path_exists usually returns i1 or i8 (bool). 
    // We can just return it directly if it matches Type::Bool.
    Ok((exists_val, Type::Bool))
}

pub fn compile_file_open<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 2 { return Err(CodegenErrorKind::Internal("File::open requires 2 arguments".to_string()).into()); }
    let fn_val = codegen.module.get_function("tl_file_open").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_file_open not found".to_string())))?;
    
    // Helper to extract char* from String
    let extract_ptr = |codegen: &mut CodeGenerator<'ctx>, val: BasicValueEnum<'ctx>, ty: &Type| -> Result<inkwell::values::BasicMetadataValueEnum<'ctx>, TlError> {
        let ptr_int_val = if matches!(ty, Type::String(_)) {
             let struct_ty = Type::String("String".to_string());
             let v = codegen.load_struct_i64_field(val, &struct_ty, "ptr")?;
             v.into_int_value()
        } else {
             return Err(TlError::from(CodegenErrorKind::Internal(format!("Expected String argument, got {:?}", ty))));
        };
        let ptr = codegen.builder.build_int_to_ptr(
            ptr_int_val,
            codegen.context.ptr_type(inkwell::AddressSpace::default()),
            "str_ptr"
        ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        Ok(ptr.into())
    };

    // Arg 0: Path
    let (path_val, path_ty) = &args[0];
    let path_ptr = extract_ptr(codegen, *path_val, path_ty)?;

    // Arg 1: Mode
    let (mode_val, mode_ty) = &args[1];
    let mode_ptr = extract_ptr(codegen, *mode_val, mode_ty)?;

    let call = codegen.builder.build_call(fn_val, &[path_ptr, mode_ptr], "file_open").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err(CodegenErrorKind::Internal("Invalid return from File::open".to_string()).into()),
    };
    Ok((res, Type::Struct("File".to_string(), vec![])))
}

pub fn compile_file_exists<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 { return Err(CodegenErrorKind::Internal("File::exists requires 1 argument".to_string()).into()); }
    let fn_val = codegen.module.get_function("tl_file_exists_i64").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_file_exists_i64 not found".to_string())))?;
    
    let (path_val, path_ty) = &args[0];
    
    let ptr_int_val = if matches!(path_ty, Type::String(_)) {
         let struct_ty = Type::String("String".to_string());
         let v = codegen.load_struct_i64_field(*path_val, &struct_ty, "ptr")?;
         v.into_int_value()
    } else {
         return Err(TlError::from(CodegenErrorKind::Internal(format!("Expected String argument, got {:?}", path_ty))));
    };
    let path_ptr = codegen.builder.build_int_to_ptr(
        ptr_int_val,
        codegen.context.ptr_type(inkwell::AddressSpace::default()),
        "str_ptr"
    ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

    let call = codegen.builder.build_call(fn_val, &[path_ptr.into()], "file_exists").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v.into_int_value(),
        _ => return Err(CodegenErrorKind::Internal("Invalid return from File::exists".to_string()).into()),
    };
    let ok = codegen.builder.build_int_compare(inkwell::IntPredicate::EQ, res, codegen.context.i64_type().const_int(1, false), "exists_bool").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((ok.into(), Type::Bool))
}

pub fn compile_file_read_static<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 { return Err(CodegenErrorKind::Internal("File::read requires 1 argument".to_string()).into()); }
    
    let (path_val, path_ty) = &args[0];
    let path_ptr_val = if matches!(path_ty, Type::String(_)) {
         let struct_ty = Type::String("String".to_string());
         let v = codegen.load_struct_i64_field(*path_val, &struct_ty, "ptr")?;
         v.into_int_value()
    } else {
         return Err(TlError::from(CodegenErrorKind::Internal(format!("Expected String argument, got {:?}", path_ty))));
    };

    let path_ptr = codegen.builder.build_int_to_ptr(
        path_ptr_val,
        codegen.context.ptr_type(inkwell::AddressSpace::default()),
        "path_ptr"
    ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

    let fn_val = codegen.module.get_function("tl_read_file").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_read_file not found".to_string())))?;
    let call = codegen.builder.build_call(fn_val, &[path_ptr.into()], "file_read").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err(CodegenErrorKind::Internal("Invalid return from File::read".to_string()).into()),
    };
    Ok((res, Type::String("String".to_string())))
}

pub fn compile_file_download<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 2 { return Err(CodegenErrorKind::Internal("File::download requires 2 arguments".to_string()).into()); }
    
    let extract_ptr = |codegen: &mut CodeGenerator<'ctx>, val: BasicValueEnum<'ctx>, ty: &Type| -> Result<inkwell::values::BasicMetadataValueEnum<'ctx>, TlError> {
        let ptr_int_val = if matches!(ty, Type::String(_)) {
             let struct_ty = Type::String("String".to_string());
             let v = codegen.load_struct_i64_field(val, &struct_ty, "ptr")?;
             v.into_int_value()
        } else {
             return Err(TlError::from(CodegenErrorKind::Internal(format!("Expected String argument, got {:?}", ty))));
        };
        let ptr = codegen.builder.build_int_to_ptr(
            ptr_int_val,
            codegen.context.ptr_type(inkwell::AddressSpace::default()),
            "str_ptr"
        ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        Ok(ptr.into())
    };

    let (url_val, url_ty) = &args[0];
    let (path_val, path_ty) = &args[1];

    let url_ptr = extract_ptr(codegen, *url_val, url_ty)?;
    let path_ptr = extract_ptr(codegen, *path_val, path_ty)?;

    let fn_val = codegen.module.get_function("tl_download_file").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_download_file not found".to_string())))?;
    let call = codegen.builder.build_call(fn_val, &[url_ptr, path_ptr], "file_download").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v.into_int_value(),
        _ => return Err(CodegenErrorKind::Internal("Invalid return from File::download".to_string()).into()),
    };
    let ok = codegen.builder.build_int_compare(inkwell::IntPredicate::EQ, res, codegen.context.i64_type().const_int(1, false), "download_bool").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((ok.into(), Type::Bool))
}

#[allow(deprecated)]
pub fn compile_file_read_binary<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 { return Err(CodegenErrorKind::Internal("File::read_binary requires 1 argument".to_string()).into()); }
    
    // Extract char* from String struct (same pattern as compile_file_read_static)
    let (path_val, path_ty) = &args[0];
    let path_ptr_val = if matches!(path_ty, Type::String(_)) {
        let struct_ty = Type::String("String".to_string());
        let v = codegen.load_struct_i64_field(*path_val, &struct_ty, "ptr")?;
        v.into_int_value()
    } else {
        return Err(TlError::from(CodegenErrorKind::Internal(format!("Expected String argument, got {:?}", path_ty))));
    };
    let path_ptr = codegen.builder.build_int_to_ptr(
        path_ptr_val,
        codegen.context.ptr_type(inkwell::AddressSpace::default()),
        "path_ptr"
    ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    
    let fn_val = codegen.module.get_function("tl_file_read_binary_all").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_file_read_binary_all not found".to_string())))?;
    let call = codegen.builder.build_call(fn_val, &[path_ptr.into()], "file_read_binary_all").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let raw_ptr = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
        _ => return Err(CodegenErrorKind::Internal("Invalid return from File::read_binary".to_string()).into()),
    };

    let vec_ty = Type::Struct("Vec".to_string(), vec![Type::U8]);
    let err_ty = Type::String("String".to_string());
    let res_ty = Type::Enum("Result".to_string(), vec![vec_ty.clone(), err_ty.clone()]);
    
    let ok_ty_llvm = codegen.get_llvm_type(&vec_ty)?;
    let err_ty_llvm = codegen.get_llvm_type(&err_ty)?;

    let ok_variant_ty = codegen.context.struct_type(&[ok_ty_llvm], false);
    let err_variant_ty = codegen.context.struct_type(&[err_ty_llvm], false);

    // Get Struct type for Enum Result[Vec[u8],String]
    let mangled = codegen.mangle_type_name("Result", &[vec_ty.clone(), err_ty.clone()]);
    let enum_llvm_ty = if let Some(ty) = codegen.enum_types.get(&mangled) {
        *ty
    } else {
        codegen.monomorphize_enum("Result", &[vec_ty.clone(), err_ty.clone()]).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        *codegen.enum_types.get(&mangled).expect("just monomorphized")
    };

    let res_alloca = codegen.create_entry_block_alloca(
        codegen.current_function()?,
        "res_alloca",
        &res_ty
    )?;

    let is_null = codegen.builder.build_is_null(raw_ptr, "is_null").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let func = codegen.current_function()?;
    let ok_block = codegen.context.append_basic_block(func, "read_ok");
    let err_block = codegen.context.append_basic_block(func, "read_err");
    let merge_block = codegen.context.append_basic_block(func, "read_merge");

    codegen.builder.build_conditional_branch(is_null, err_block, ok_block).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

    // === OK BLOCK ===
    codegen.builder.position_at_end(ok_block);
    let tag_ptr_ok = codegen.builder.build_struct_gep(enum_llvm_ty, res_alloca, 0, "tag_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    codegen.builder.build_store(tag_ptr_ok, codegen.context.i32_type().const_int(0, false)).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?; // 0 is Ok

    let payload_ptr_raw_ok = codegen.builder.build_struct_gep(enum_llvm_ty, res_alloca, 1, "payload_ptr_raw").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let payload_ptr_ok = codegen.builder.build_pointer_cast(payload_ptr_raw_ok, codegen.context.ptr_type(inkwell::AddressSpace::default()), "payload_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

    let field_ptr_ok = codegen.builder.build_struct_gep(ok_variant_ty, payload_ptr_ok, 0, "field_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    codegen.builder.build_store(field_ptr_ok, raw_ptr).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    codegen.builder.build_unconditional_branch(merge_block).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

    // === ERR BLOCK ===
    codegen.builder.position_at_end(err_block);
    let tag_ptr_err = codegen.builder.build_struct_gep(enum_llvm_ty, res_alloca, 0, "tag_ptr_err").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    codegen.builder.build_store(tag_ptr_err, codegen.context.i32_type().const_int(1, false)).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?; // 1 is Err

    // Create Error String
    let (err_str_val, _) = codegen.compile_string_literal("Failed to read binary file")?;

    let payload_ptr_raw_err = codegen.builder.build_struct_gep(enum_llvm_ty, res_alloca, 1, "payload_ptr_raw_err").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let payload_ptr_err = codegen.builder.build_pointer_cast(payload_ptr_raw_err, codegen.context.ptr_type(inkwell::AddressSpace::default()), "payload_ptr_err").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

    let field_ptr_err = codegen.builder.build_struct_gep(err_variant_ty, payload_ptr_err, 0, "field_ptr_err").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    codegen.builder.build_store(field_ptr_err, err_str_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    codegen.builder.build_unconditional_branch(merge_block).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

    // === MERGE BLOCK ===
    codegen.builder.position_at_end(merge_block);

    Ok((res_alloca.into(), res_ty))
}

pub fn compile_file_read_string<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    instance_val: BasicValueEnum<'ctx>,
    _instance_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if !args.is_empty() { return Err(CodegenErrorKind::Internal("File::read_string takes no arguments".to_string()).into()); }
    let fn_val = codegen.module.get_function("tl_file_read_string").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_file_read_string not found".to_string())))?;
    let call = codegen.builder.build_call(fn_val, &[instance_val.into()], "file_read_str").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err(CodegenErrorKind::Internal("Invalid return from File::read_string".to_string()).into()),
    };
    Ok((res, Type::String("String".to_string())))
}

pub fn compile_file_write_string<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    instance_val: BasicValueEnum<'ctx>,
    _instance_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 {
        return Err(TlError::from(CodegenErrorKind::Internal("File::write_string requires 1 argument".to_string())));
    }
    // tl_file_write_string(f: *mut c_void, s: *mut StringStruct) -> bool
    // Pass the String struct pointer directly (not the char data pointer inside it)
    let content_val = args[0].0;
    let fn_val = codegen.module.get_function("tl_file_write_string").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_file_write_string not found".to_string())))?;
    codegen.builder.build_call(
        fn_val,
        &[instance_val.into(), content_val.into()],
        "file_write_str",
    ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
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
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if !args.is_empty() { return Err(CodegenErrorKind::Internal("File::close takes no arguments".to_string()).into()); }
    let fn_val = codegen.module.get_function("tl_file_close").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_file_close not found".to_string())))?;
    codegen.builder.build_call(fn_val, &[instance_val.into()], "file_close").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

pub fn compile_file_append<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 2 { return Err(CodegenErrorKind::Internal("File::append requires 2 arguments".to_string()).into()); }
    let extract_ptr = |codegen: &mut CodeGenerator<'ctx>, val: BasicValueEnum<'ctx>, ty: &Type| -> Result<inkwell::values::BasicMetadataValueEnum<'ctx>, TlError> {
        let ptr_int_val = if matches!(ty, Type::String(_)) {
            let struct_ty = Type::String("String".to_string());
            codegen.load_struct_i64_field(val, &struct_ty, "ptr")?.into_int_value()
        } else {
            return Err(TlError::from(CodegenErrorKind::Internal("Expected String".to_string())));
        };
        Ok(codegen.builder.build_int_to_ptr(
            ptr_int_val, codegen.context.ptr_type(inkwell::AddressSpace::default()), "str_ptr"
        ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into())
    };

    let path_ptr = extract_ptr(codegen, args[0].0, &args[0].1)?;
    let content_ptr = extract_ptr(codegen, args[1].0, &args[1].1)?;

    let fn_val = codegen.module.get_function("tl_file_append").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_file_append not found".to_string())))?;
    let call = codegen.builder.build_call(fn_val, &[path_ptr, content_ptr], "file_append").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v.into_int_value(),
        _ => return Err(CodegenErrorKind::Internal("Invalid return from File::append".to_string()).into()),
    };
    Ok((res.into(), Type::Bool))
}

pub fn compile_file_delete<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 { return Err(CodegenErrorKind::Internal("File::delete requires 1 argument".to_string()).into()); }
    let fn_val = codegen.module.get_function("tl_file_delete").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_file_delete not found".to_string())))?;
    let (path_val, path_ty) = &args[0];
    let ptr_int = if matches!(path_ty, Type::String(_)) {
        codegen.load_struct_i64_field(*path_val, &Type::String("String".to_string()), "ptr")?.into_int_value()
    } else { return Err(CodegenErrorKind::Internal("Expected String".to_string()).into()); };
    let path_ptr = codegen.builder.build_int_to_ptr(ptr_int, codegen.context.ptr_type(inkwell::AddressSpace::default()), "path_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

    let call = codegen.builder.build_call(fn_val, &[path_ptr.into()], "file_delete").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v.into_int_value(),
        _ => return Err(CodegenErrorKind::Internal("Invalid return from File::delete".to_string()).into()),
    };
    Ok((res.into(), Type::Bool))
}

pub fn compile_file_create_dir<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 { return Err(CodegenErrorKind::Internal("File::create_dir requires 1 argument".to_string()).into()); }
    let fn_val = codegen.module.get_function("tl_file_create_dir").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_file_create_dir not found".to_string())))?;
    let (path_val, path_ty) = &args[0];
    let ptr_int = if matches!(path_ty, Type::String(_)) {
        codegen.load_struct_i64_field(*path_val, &Type::String("String".to_string()), "ptr")?.into_int_value()
    } else { return Err(CodegenErrorKind::Internal("Expected String".to_string()).into()); };
    let path_ptr = codegen.builder.build_int_to_ptr(ptr_int, codegen.context.ptr_type(inkwell::AddressSpace::default()), "path_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

    let call = codegen.builder.build_call(fn_val, &[path_ptr.into()], "file_create_dir").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v.into_int_value(),
        _ => return Err(CodegenErrorKind::Internal("Invalid return from File::create_dir".to_string()).into()),
    };
    Ok((res.into(), Type::Bool))
}

pub fn compile_file_list_dir<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 { return Err(CodegenErrorKind::Internal("File::list_dir requires 1 argument".to_string()).into()); }
    let fn_val = codegen.module.get_function("tl_file_list_dir").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_file_list_dir not found".to_string())))?;
    let (path_val, path_ty) = &args[0];
    let ptr_int = if matches!(path_ty, Type::String(_)) {
        codegen.load_struct_i64_field(*path_val, &Type::String("String".to_string()), "ptr")?.into_int_value()
    } else { return Err(CodegenErrorKind::Internal("Expected String".to_string()).into()); };
    let path_ptr = codegen.builder.build_int_to_ptr(ptr_int, codegen.context.ptr_type(inkwell::AddressSpace::default()), "path_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

    let call = codegen.builder.build_call(fn_val, &[path_ptr.into()], "file_list_dir").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
        _ => return Err(CodegenErrorKind::Internal("Invalid return from File::list_dir".to_string()).into()),
    };
    
    // Cast appropriately
    let vec_ty_name = "Vec[String]";
    let _vec_struct_ty = if let Some(ty) = codegen.struct_types.get(vec_ty_name) {
        *ty
    } else {
        let vec_str_generics = vec![Type::String("String".to_string())];
        codegen.monomorphize_struct("Vec", &vec_str_generics).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
    };
    let vec_ptr_ty = codegen.context.ptr_type(inkwell::AddressSpace::default());
    let vec_ptr = codegen.builder.build_pointer_cast(res, vec_ptr_ty, "vec_cast").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

    Ok((vec_ptr.into(), Type::Struct("Vec".to_string(), vec![Type::String("String".to_string())])))
}

// Helper for Path instance methods
fn compile_path_instance_method<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    instance_val: BasicValueEnum<'ctx>,
    fn_name: &str,
    res_name: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = codegen.module.get_function(fn_name).ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("{} not found", fn_name))))?;
    
    let ptr = codegen.builder.build_ptr_to_int(instance_val.into_pointer_value(), codegen.context.i64_type(), "path_ptr_int").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let path_ptr = codegen.builder.build_int_to_ptr(ptr, codegen.context.ptr_type(inkwell::AddressSpace::default()), "path_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

    let call = codegen.builder.build_call(fn_val, &[path_ptr.into()], res_name).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err(CodegenErrorKind::Internal("Invalid return from Path method".to_string()).into()),
    };
    Ok((res, Type::String("String".to_string())))
}

pub fn compile_path_parent<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    instance_val: BasicValueEnum<'ctx>,
    _instance_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if !args.is_empty() { return Err(CodegenErrorKind::Internal("Path.parent takes no arguments".to_string()).into()); }
    compile_path_instance_method(codegen, instance_val, "tl_path_parent", "path_parent_str")
}

pub fn compile_path_file_name<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    instance_val: BasicValueEnum<'ctx>,
    _instance_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if !args.is_empty() { return Err(CodegenErrorKind::Internal("Path.file_name takes no arguments".to_string()).into()); }
    compile_path_instance_method(codegen, instance_val, "tl_path_file_name", "path_filename_str")
}

pub fn compile_path_extension<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    instance_val: BasicValueEnum<'ctx>,
    _instance_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if !args.is_empty() { return Err(CodegenErrorKind::Internal("Path.extension takes no arguments".to_string()).into()); }
    compile_path_instance_method(codegen, instance_val, "tl_path_extension", "path_ext_str")
}

/// File::write_binary(path: String, data: Vec<u8>) -> Bool
pub fn compile_file_write_binary<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 2 {
        return Err(CodegenErrorKind::Internal("File::write_binary requires 2 arguments (path, data)".to_string()).into());
    }

    // Extract char* from path String struct
    let (path_val, path_ty) = &args[0];
    let path_ptr_val = if matches!(path_ty, Type::String(_)) {
        let struct_ty = Type::String("String".to_string());
        let v = codegen.load_struct_i64_field(*path_val, &struct_ty, "ptr")?;
        v.into_int_value()
    } else {
        return Err(TlError::from(CodegenErrorKind::Internal(format!("Expected String argument, got {:?}", path_ty))));
    };
    let path_ptr = codegen.builder.build_int_to_ptr(
        path_ptr_val,
        codegen.context.ptr_type(inkwell::AddressSpace::default()),
        "path_ptr"
    ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

    // Get Vec<u8> pointer (2nd argument)
    let (data_val, _data_ty) = &args[1];

    let fn_val = codegen.module.get_function("tl_file_write_binary_all")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_file_write_binary_all not found".to_string())))?;
    let call = codegen.builder.build_call(
        fn_val,
        &[path_ptr.into(), (*data_val).into()],
        "file_write_binary_all"
    ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let result = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err(CodegenErrorKind::Internal("Invalid return from File::write_binary".to_string()).into()),
    };
    Ok((result, Type::Bool))
}
