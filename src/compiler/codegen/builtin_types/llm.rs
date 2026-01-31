use crate::compiler::codegen::type_manager::{CodeGenType, TypeManager, InstanceMethod, StaticMethod};
use crate::compiler::codegen::CodeGenerator;
use crate::compiler::ast::Type;
use inkwell::values::BasicValueEnum;


pub const SOURCE: &str = include_str!("llm.tl");

pub fn register_llm_types(manager: &mut TypeManager) {
    // Register Tokenizer
    let mut tokenizer = CodeGenType::new("Tokenizer");
    tokenizer.register_static_method("new", StaticMethod::Evaluated(compile_tokenizer_new));
    tokenizer.register_instance_method("encode", InstanceMethod::Evaluated(compile_tokenizer_encode));
    tokenizer.register_instance_method("decode", InstanceMethod::Evaluated(compile_tokenizer_decode));
    manager.register_type(tokenizer);

    // Register KVCache
    let mut kv_cache = CodeGenType::new("KVCache");
    kv_cache.register_static_method("new", StaticMethod::Evaluated(compile_kv_cache_new));
    kv_cache.register_instance_method("free", InstanceMethod::Evaluated(compile_kv_cache_free));
    kv_cache.register_instance_method("get_k", InstanceMethod::Evaluated(compile_kv_cache_get_k));
    kv_cache.register_instance_method("get_v", InstanceMethod::Evaluated(compile_kv_cache_get_v));
    kv_cache.register_instance_method("update", InstanceMethod::Evaluated(compile_kv_cache_update));
    manager.register_type(kv_cache);
}

// ... existing static methods ...

// ... existing instance methods ...



fn compile_kv_cache_update<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    instance_val: BasicValueEnum<'ctx>,
    instance_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 3 { return Err("KVCache::update requires 3 arguments".into()); }
    let handle = codegen.load_struct_i64_field(instance_val, &instance_ty, "handle")?;
    
    let (layer_val, _) = args[0];
    let (k_val, k_ty) = &args[1];
    let (v_val, v_ty) = &args[2];

    // Helper to extract tensor pointer
    let mut get_tensor_ptr = |val: BasicValueEnum<'ctx>, ty: &Type| -> Result<inkwell::values::BasicMetadataValueEnum<'ctx>, String> {
        let ptr_val = if let Type::Struct(name, _) = ty {
            if name == "Tensor" {
                 let h = codegen.load_struct_i64_field(val, ty, "handle")?;
                 codegen.builder.build_int_to_ptr(h.into_int_value(), codegen.context.ptr_type(inkwell::AddressSpace::default()), "handle_ptr").map_err(|e| e.to_string())?
            } else {
                 return Err("Expected Tensor argument".into());
            }
        } else if let Type::Tensor(_, _) = ty {
             if val.is_pointer_value() {
                 val.into_pointer_value()
             } else {
                 return Err("Expected Tensor pointer".into());
             }
        } else {
             return Err(format!("Expected Tensor argument, got {:?}", ty));
        };
        Ok(ptr_val.into())
    };

    let k_ptr = get_tensor_ptr(*k_val, k_ty)?;
    let v_ptr = get_tensor_ptr(*v_val, v_ty)?;

    let fn_val = codegen.module.get_function("tl_kv_cache_update").ok_or("tl_kv_cache_update not found")?;
    codegen.builder.build_call(fn_val, &[handle.into(), layer_val.into(), k_ptr, v_ptr], "").map_err(|e| e.to_string())?;
    
    Ok((codegen.context.i64_type().const_int(0, false).into(), Type::Void))
}
// Tokenizer Static Methods
fn compile_tokenizer_new<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target_type: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err("Tokenizer::new requires 1 argument".into());
    }
    let (path_val, path_ty) = &args[0];
    
    let path_ptr_val = match path_ty {
        Type::Struct(name, _) if name == "String" => {
            let val = codegen.load_struct_i64_field(*path_val, path_ty, "ptr")?;
            val.into_int_value()
        },
        Type::String => {
             let struct_ty = Type::Struct("String".to_string(), vec![]);
             let val = codegen.load_struct_i64_field(*path_val, &struct_ty, "ptr")?;
             val.into_int_value()
        },
        _ => return Err(format!("Tokenizer::new expects String argument, got {:?}", path_ty)),
    };

    let path_ptr = codegen.builder.build_int_to_ptr(
        path_ptr_val,
        codegen.context.ptr_type(inkwell::AddressSpace::default()),
        "path_ptr"
    ).map_err(|e| e.to_string())?;

    let fn_val = codegen.module.get_function("tl_tokenizer_new").ok_or("tl_tokenizer_new not found")?;
    let call = codegen.builder.build_call(fn_val, &[path_ptr.into()], "tok_new").map_err(|e| e.to_string())?;
    let handle = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from Tokenizer::new".into()),
    };

    let struct_type = *codegen.struct_types.get("Tokenizer").ok_or("Struct type Tokenizer not found")?;
    let struct_def = codegen.struct_defs.get("Tokenizer").ok_or("Struct definition Tokenizer not found")?;
    let size = struct_type.size_of().ok_or("Cannot determine size of Tokenizer")?;
    
    let size_int = size;
    let size_i64 = if size_int.get_type() == codegen.context.i32_type() {
        codegen.builder.build_int_z_extend(size_int, codegen.context.i64_type(), "size_i64").unwrap()
    } else {
        size_int
    };

    let malloc_fn = codegen.module.get_function("malloc").ok_or("malloc not found (declare in builtins)")?;
    let call = codegen.builder.build_call(malloc_fn, &[size_i64.into()], "tokenizer_malloc").map_err(|e| e.to_string())?;
    let raw_ptr = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
        _ => return Err("malloc returned invalid value".into()),
    };
    
    if let Some(register_fn) = codegen.module.get_function("tl_mem_register_struct") {
        let cast_ptr = codegen.builder.build_pointer_cast(raw_ptr, codegen.context.ptr_type(inkwell::AddressSpace::default()), "cast_ptr").unwrap();
        codegen.builder.build_call(register_fn, &[cast_ptr.into()], "").map_err(|e| e.to_string())?;
    }
    
    let struct_ptr = codegen.builder.build_pointer_cast(raw_ptr, codegen.context.ptr_type(inkwell::AddressSpace::default()), "tokenizer_ptr").map_err(|e| e.to_string())?;
    
    let field_idx = struct_def.fields.iter().position(|(n, _)| n == "handle").ok_or("Field handle not found in Tokenizer")?;
    let field_ptr = codegen.builder.build_struct_gep(struct_type, struct_ptr, field_idx as u32, "tokenizerhandle").map_err(|e| e.to_string())?;
    codegen.builder.build_store(field_ptr, handle).map_err(|e| e.to_string())?;
    
    Ok((struct_ptr.into(), Type::Struct("Tokenizer".to_string(), vec![])))
}

// Tokenizer Instance Methods
fn compile_tokenizer_encode<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    instance_val: BasicValueEnum<'ctx>,
    instance_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 { return Err("Tokenizer::encode requires 1 argument".into()); }
    let handle = codegen.load_struct_i64_field(instance_val, &instance_ty, "handle")?;
    let (prompt_val, prompt_ty) = &args[0];
    
    let prompt_ptr_val = match prompt_ty {
        Type::Struct(name, _) if name == "String" => {
            let val = codegen.load_struct_i64_field(*prompt_val, prompt_ty, "ptr")?;
            val.into_int_value()
        },
        Type::String => {
             let struct_ty = Type::Struct("String".to_string(), vec![]);
             let val = codegen.load_struct_i64_field(*prompt_val, &struct_ty, "ptr")?;
             val.into_int_value()
        },
        _ => return Err("Tokenizer::encode expects String argument".into()),
    };

    let prompt_ptr = codegen.builder.build_int_to_ptr(
        prompt_ptr_val,
        codegen.context.ptr_type(inkwell::AddressSpace::default()),
        "prompt_ptr"
    ).map_err(|e| e.to_string())?;
    let fn_val = codegen.module.get_function("tl_tokenizer_encode").ok_or("tl_tokenizer_encode not found")?;
    let call = codegen.builder.build_call(fn_val, &[handle.into(), prompt_ptr.into()], "tok_encode").map_err(|e| e.to_string())?;
    // check_tensor_result(val, msg) is method of codegen.
    let res = codegen.check_tensor_result(call, "tok_encode_error")?;
    Ok((res, Type::Tensor(Box::new(Type::I64), 0)))
}

fn compile_tokenizer_decode<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    instance_val: BasicValueEnum<'ctx>,
    instance_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 { return Err("Tokenizer::decode requires 1 argument".into()); }
    let handle = codegen.load_struct_i64_field(instance_val, &instance_ty, "handle")?;
    let (ids_val, _) = args[0];
    let fn_val = codegen.module.get_function("tl_tokenizer_decode").ok_or("tl_tokenizer_decode not found")?;
    let call = codegen.builder.build_call(fn_val, &[handle.into(), ids_val.into()], "tok_decode").map_err(|e| e.to_string())?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from Tokenizer::decode".into()),
    };
    Ok((res, Type::String))
}

// KVCache Static Methods
fn compile_kv_cache_new<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target_type: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err("KVCache::new requires 1 argument".into());
    }
    let layers_val = args[0].0;

    let fn_val = codegen.module.get_function("tl_kv_cache_new").ok_or("tl_kv_cache_new not found")?;
    let call = codegen.builder.build_call(fn_val, &[layers_val.into()], "kv_new").map_err(|e| e.to_string())?;
    let handle = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from KVCache::new".into()),
    };

    let struct_type = *codegen.struct_types.get("KVCache").ok_or("Struct type KVCache not found")?;
    let struct_def = codegen.struct_defs.get("KVCache").ok_or("Struct definition KVCache not found")?;
    let size = struct_type.size_of().ok_or("Cannot determine size of KVCache")?;
    
    let size_int = size;
    let size_i64 = if size_int.get_type() == codegen.context.i32_type() {
        codegen.builder.build_int_z_extend(size_int, codegen.context.i64_type(), "size_i64").unwrap()
    } else {
        size_int
    };

    let malloc_fn = codegen.module.get_function("malloc").ok_or("malloc not found (declare in builtins)")?;
    let call = codegen.builder.build_call(malloc_fn, &[size_i64.into()], "kvcache_malloc").map_err(|e| e.to_string())?;
    let raw_ptr = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
        _ => return Err("malloc returned invalid value".into()),
    };
    
    if let Some(register_fn) = codegen.module.get_function("tl_mem_register_struct") {
        let cast_ptr = codegen.builder.build_pointer_cast(raw_ptr, codegen.context.ptr_type(inkwell::AddressSpace::default()), "cast_ptr").unwrap();
        codegen.builder.build_call(register_fn, &[cast_ptr.into()], "").map_err(|e| e.to_string())?;
    }
    
    let struct_ptr = codegen.builder.build_pointer_cast(raw_ptr, codegen.context.ptr_type(inkwell::AddressSpace::default()), "kvcache_ptr").map_err(|e| e.to_string())?;
    
    let field_idx = struct_def.fields.iter().position(|(n, _)| n == "handle").ok_or("Field handle not found in KVCache")?;
    let field_ptr = codegen.builder.build_struct_gep(struct_type, struct_ptr, field_idx as u32, "kvcachehandle").map_err(|e| e.to_string())?;
    codegen.builder.build_store(field_ptr, handle).map_err(|e| e.to_string())?;
    
    Ok((struct_ptr.into(), Type::Struct("KVCache".to_string(), vec![])))
}

// KVCache Instance Methods
fn compile_kv_cache_free<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    instance_val: BasicValueEnum<'ctx>,
    instance_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let handle = codegen.load_struct_i64_field(instance_val, &instance_ty, "handle")?;
    let fn_val = codegen.module.get_function("tl_kv_cache_free").ok_or("tl_kv_cache_free not found")?;
    codegen.builder.build_call(fn_val, &[handle.into()], "kv_free").map_err(|e| e.to_string())?;
    Ok((codegen.context.i64_type().const_int(0, false).into(), Type::Void))
}

fn compile_kv_cache_get_k<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    instance_val: BasicValueEnum<'ctx>,
    instance_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 { return Err("KVCache::get_k requires 1 argument".into()); }
    let handle = codegen.load_struct_i64_field(instance_val, &instance_ty, "handle")?;
    let (layer_val, _) = args[0];
    let fn_val = codegen.module.get_function("tl_kv_cache_get_k").ok_or("tl_kv_cache_get_k not found")?;
    let call = codegen.builder.build_call(fn_val, &[handle.into(), layer_val.into()], "kv_get_k").map_err(|e| e.to_string())?;
    let res = codegen.check_tensor_result(call, "kv_get_k_error")?;
    Ok((res, Type::Tensor(Box::new(Type::F32), 2))) // Assuming 2D tensor
}

fn compile_kv_cache_get_v<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    instance_val: BasicValueEnum<'ctx>,
    instance_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 { return Err("KVCache::get_v requires 1 argument".into()); }
    let handle = codegen.load_struct_i64_field(instance_val, &instance_ty, "handle")?;
    let (layer_val, _) = args[0];
    let fn_val = codegen.module.get_function("tl_kv_cache_get_v").ok_or("tl_kv_cache_get_v not found")?;
    let call = codegen.builder.build_call(fn_val, &[handle.into(), layer_val.into()], "kv_get_v").map_err(|e| e.to_string())?;
    let res = codegen.check_tensor_result(call, "kv_get_v_error")?;
    Ok((res, Type::Tensor(Box::new(Type::F32), 2))) // Assuming 2D tensor
}
