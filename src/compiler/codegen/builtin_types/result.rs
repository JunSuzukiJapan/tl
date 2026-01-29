use crate::compiler::codegen::type_manager::{CodeGenType, TypeManager, InstanceMethod};
use crate::compiler::codegen::CodeGenerator;
use crate::compiler::ast::{Type, EnumDef, VariantDef, VariantKind};
use inkwell::values::BasicValueEnum;
use inkwell::IntPredicate;

pub fn get_result_enum_def() -> EnumDef {
    // enum Result<T, E> { Ok(T), Err(E) }
    let t = Type::UserDefined("T".to_string(), vec![]);
    let e = Type::UserDefined("E".to_string(), vec![]);
    
    EnumDef {
        name: "Result".to_string(),
        generics: vec!["T".to_string(), "E".to_string()],
        variants: vec![
            VariantDef {
                name: "Ok".to_string(),
                kind: VariantKind::Tuple(vec![t.clone()]),
            },
            VariantDef {
                name: "Err".to_string(),
                kind: VariantKind::Tuple(vec![e.clone()]),
            },
        ],
    }
}

pub fn register_result_types(manager: &mut TypeManager) {
    let mut result = CodeGenType::new("Result");

    result.register_instance_method("is_ok", InstanceMethod::Evaluated(compile_is_ok));
    result.register_instance_method("is_err", InstanceMethod::Evaluated(compile_is_err));
    result.register_instance_method("unwrap", InstanceMethod::Evaluated(compile_unwrap));
    
    manager.register_type(result);
}

fn get_result_inner_types(ty: &Type) -> Option<(Type, Type)> {
    match ty {
        Type::Enum(name, args) if name == "Result" => {
            if args.len() == 2 {
                Some((args[0].clone(), args[1].clone()))
            } else {
                Some((Type::I64, Type::I64)) // Fallback
            }
        }
        _ => None,
    }
}

fn compile_is_ok<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    instance_val: BasicValueEnum<'ctx>,
    instance_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let (_t_type, _e_type) = get_result_inner_types(&instance_ty).ok_or("Invalid Result type for is_ok")?;
    let i64_type = codegen.context.i64_type();
    
    // Enum layout: { tag: i64, payload: Union(T, E) }
    // We only need the tag.
    // However, to get the GEP correct, we need basic knowlege of layout or just use element 0.
    // Enum is generated as { i64, MaxSizeArray }. 
    // Wait, codegen.compile_enum_defs generates the struct type.
    // We can just ask for the struct type from codegen?
    // Or simpler: access element 0 (tag) which is always i64.
    
    // We need the pointer to the struct.
    let result_ptr = instance_val.into_pointer_value();
    
    // We construct distinct struct type for GEP: { i64, [array] } or similar.
    // Actually `codegen.get_llvm_type` should return the correct struct type for the Enum.
    // But `instance_val` might be an opaque pointer or already typed?
    // `instance_val` coming from checked_expr is usually a PointerValue.
    
    // Safest way: Cast to { i64 }* or use get_llvm_type.
    let result_struct_type = codegen.get_llvm_type(&instance_ty)?;
    
    // The struct type returned by get_llvm_type for Enum is correct.
    // But it might be complex.
    // However, element 0 is ALWAYS tag.
    let struct_ty = result_struct_type.into_struct_type();
    
    let tag_ptr = codegen.builder.build_struct_gep(struct_ty, result_ptr, 0, "tag_ptr")
        .map_err(|e| e.to_string())?;
        
    let tag_val = codegen.builder.build_load(i64_type, tag_ptr, "tag")
        .map_err(|e| e.to_string())?;
        
    // Ok is Variant 0
    let is_ok = codegen.builder.build_int_compare(
        IntPredicate::EQ,
        tag_val.into_int_value(),
        i64_type.const_int(0, false),
        "is_ok"
    ).map_err(|e| e.to_string())?;
    
    Ok((is_ok.into(), Type::Bool))
}

fn compile_is_err<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    instance_val: BasicValueEnum<'ctx>,
    instance_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let (_t_type, _e_type) = get_result_inner_types(&instance_ty).ok_or("Invalid Result type for is_err")?;
    let i64_type = codegen.context.i64_type();
    
    let result_struct_type = codegen.get_llvm_type(&instance_ty)?;
    let struct_ty = result_struct_type.into_struct_type();
    
    let result_ptr = instance_val.into_pointer_value();
    let tag_ptr = codegen.builder.build_struct_gep(struct_ty, result_ptr, 0, "tag_ptr")
        .map_err(|e| e.to_string())?;
    let tag_val = codegen.builder.build_load(i64_type, tag_ptr, "tag")
        .map_err(|e| e.to_string())?;
        
    // Err is Variant 1
    let is_err = codegen.builder.build_int_compare(
        IntPredicate::EQ,
        tag_val.into_int_value(),
        i64_type.const_int(1, false),
        "is_err"
    ).map_err(|e| e.to_string())?;
    
    Ok((is_err.into(), Type::Bool))
}

fn compile_unwrap<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    instance_val: BasicValueEnum<'ctx>,
    instance_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let (t_type, _e_type) = get_result_inner_types(&instance_ty).ok_or("Invalid Result type for unwrap")?;
    
    // For unwrap, we assume it is Ok (Tag 0).
    // The payload is at index 1.
    // The payload is a Union (byte array). We need to bitcast it to T.
    
    let result_struct_type = codegen.get_llvm_type(&instance_ty)?;
    let struct_ty = result_struct_type.into_struct_type();
    
    let result_ptr = instance_val.into_pointer_value();
    let payload_ptr = codegen.builder.build_struct_gep(struct_ty, result_ptr, 1, "payload_ptr")
        .map_err(|e| e.to_string())?;
        
    // Now we cast payload_ptr to T*
    let llvm_t_type = codegen.get_llvm_type(&t_type)?;
    
    // NOTE: Elements in Variant Tuple are wrapped in a struct?
    // VariantDef for Ok is Tuple(vec![T]).
    // So the payload is a struct { T }.
    // So payload_ptr is pointing to storage for { T }.
    
    // We treat it as pointer to T for simplicity if it's single element.
    // But strictly it is { T }.
    // Let's bitcast payload_ptr to { T }* (pointer to struct containing T).
    let tuple_inner_type = codegen.context.struct_type(&[llvm_t_type], false);
    
    let _cast_ptr = codegen.builder.build_pointer_cast(
        payload_ptr, 
        codegen.context.ptr_type(inkwell::AddressSpace::default()), 
        "cast_payload"
    ).map_err(|e| e.to_string())?;
    
    // Wait, build_load needs typed pointer or explicit type.
    // We can load the T directly if we assume layout.
    // BUT! Since Ok(T) is a Tuple variant, it is stored as a Struct { T }.
    // So we need to access index 0 of that inner struct.
    
    // Actually, let's just use the `Type::Tuple(vec![t])` logic.
    // But here we know it is specifically Ok (T).
    
    // Let's cast payload storage to { T }*.
    let typed_payload_ptr = codegen.builder.build_pointer_cast(
        payload_ptr,
        codegen.context.ptr_type(inkwell::AddressSpace::default()),
        "typed_payload"
    ).map_err(|e| e.to_string())?;
    
    // Now GEP to 0 to get T*.
    let item_ptr = codegen.builder.build_struct_gep(
        tuple_inner_type,
        typed_payload_ptr,
        0,
        "item_ptr"
    ).map_err(|e| e.to_string())?;
    
    // Load T
    let value = codegen.builder.build_load(llvm_t_type, item_ptr, "unwrap_value")
        .map_err(|e| e.to_string())?;
        
    Ok((value, t_type))
}
