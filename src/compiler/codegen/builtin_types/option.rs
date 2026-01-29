use crate::compiler::codegen::type_manager::{CodeGenType, TypeManager, InstanceMethod};
use crate::compiler::codegen::CodeGenerator;
use crate::compiler::ast::{Type, EnumDef, VariantDef, VariantKind};
use inkwell::values::BasicValueEnum;
use inkwell::IntPredicate;

pub fn get_option_enum_def() -> EnumDef {
    // enum Option<T> { Some(T), None }
    let t = Type::UserDefined("T".to_string(), vec![]);
    
    EnumDef {
        name: "Option".to_string(),
        generics: vec!["T".to_string()],
        variants: vec![
            VariantDef {
                name: "None".to_string(),
                kind: VariantKind::Unit,
            },
            VariantDef {
                name: "Some".to_string(),
                kind: VariantKind::Tuple(vec![t.clone()]),
            },
        ],
    }
}

pub fn register_option_types(manager: &mut TypeManager) {
    let mut option = CodeGenType::new("Option");

    option.register_instance_method("is_some", InstanceMethod::Evaluated(compile_is_some));
    option.register_instance_method("is_none", InstanceMethod::Evaluated(compile_is_none));
    option.register_instance_method("unwrap", InstanceMethod::Evaluated(compile_unwrap));
    option.register_instance_method("unwrap_or", InstanceMethod::Evaluated(compile_unwrap_or));
    manager.register_type(option);
}

fn get_option_inner_type(ty: &Type) -> Option<Type> {
    match ty {
        Type::Struct(name, args) | Type::UserDefined(name, args) | Type::Enum(name, args) if name == "Option" => {
            if args.len() == 1 {
                Some(args[0].clone())
            } else {
                Some(Type::I64) // Default fallback
            }
        }
        _ => None,
    }
}







fn compile_is_some<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    instance_val: BasicValueEnum<'ctx>,
    instance_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let inner_type = get_option_inner_type(&instance_ty).ok_or("Invalid Option type for is_some")?;
    let i64_type = codegen.context.i64_type();
    
    // Manual struct construction: { i64, T }
    let llvm_inner_type = codegen.get_llvm_type(&inner_type)?;
    let option_struct_type = codegen.context.struct_type(&[i64_type.into(), llvm_inner_type], false);
    
    let option_ptr = instance_val.into_pointer_value();
    let tag_ptr = codegen.builder.build_struct_gep(option_struct_type, option_ptr, 0, "tag_ptr")
        .map_err(|e| e.to_string())?;
    let tag_val = codegen.builder.build_load(i64_type, tag_ptr, "tag")
        .map_err(|e| e.to_string())?;
        
    let is_some = codegen.builder.build_int_compare(
        IntPredicate::EQ,
        tag_val.into_int_value(),
        i64_type.const_int(1, false),
        "is_some"
    ).map_err(|e| e.to_string())?;
    
    Ok((is_some.into(), Type::Bool))
}

fn compile_is_none<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    instance_val: BasicValueEnum<'ctx>,
    instance_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let inner_type = get_option_inner_type(&instance_ty).ok_or("Invalid Option type for is_none")?;
    let i64_type = codegen.context.i64_type();

    let llvm_inner_type = codegen.get_llvm_type(&inner_type)?;
    let option_struct_type = codegen.context.struct_type(&[i64_type.into(), llvm_inner_type], false);
    
    let option_ptr = instance_val.into_pointer_value();
    let tag_ptr = codegen.builder.build_struct_gep(option_struct_type, option_ptr, 0, "tag_ptr")
        .map_err(|e| e.to_string())?;
    let tag_val = codegen.builder.build_load(i64_type, tag_ptr, "tag")
        .map_err(|e| e.to_string())?;
        
    let is_none = codegen.builder.build_int_compare(
        IntPredicate::EQ,
        tag_val.into_int_value(),
        i64_type.const_int(0, false),
        "is_none"
    ).map_err(|e| e.to_string())?;
    
    Ok((is_none.into(), Type::Bool))
}

fn compile_unwrap<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    instance_val: BasicValueEnum<'ctx>,
    instance_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let inner_type = get_option_inner_type(&instance_ty).ok_or("Invalid Option type for unwrap")?;
    let i64_type = codegen.context.i64_type();
    let llvm_inner_type = codegen.get_llvm_type(&inner_type)?;
    let option_struct_type = codegen.context.struct_type(&[i64_type.into(), llvm_inner_type], false);
    
    let option_ptr = instance_val.into_pointer_value();
    let value_ptr = codegen.builder.build_struct_gep(option_struct_type, option_ptr, 1, "value_ptr")
        .map_err(|e| e.to_string())?;
    let value = codegen.builder.build_load(llvm_inner_type, value_ptr, "unwrap_value")
        .map_err(|e| e.to_string())?;
        
    Ok((value, inner_type))
}

fn compile_unwrap_or<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    instance_val: BasicValueEnum<'ctx>,
    instance_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 1 {
        return Err("unwrap_or requires 1 argument".into());
    }
    let inner_type = get_option_inner_type(&instance_ty).ok_or("Invalid Option type for unwrap_or")?;
    let (default_val, _) = args.into_iter().next().unwrap();
    
    let i64_type = codegen.context.i64_type();
    let llvm_inner_type = codegen.get_llvm_type(&inner_type)?;
    let option_struct_type = codegen.context.struct_type(&[i64_type.into(), llvm_inner_type], false);
    let option_ptr = instance_val.into_pointer_value();

    // Check tag
    let tag_ptr = codegen.builder.build_struct_gep(option_struct_type, option_ptr, 0, "tag_ptr")
        .map_err(|e| e.to_string())?;
    let tag_val = codegen.builder.build_load(i64_type, tag_ptr, "tag")
        .map_err(|e| e.to_string())?;
    let is_some = codegen.builder.build_int_compare(
        IntPredicate::EQ,
        tag_val.into_int_value(),
        i64_type.const_int(1, false),
        "is_some"
    ).map_err(|e| e.to_string())?;

    // Load value
    let value_ptr = codegen.builder.build_struct_gep(option_struct_type, option_ptr, 1, "value_ptr")
        .map_err(|e| e.to_string())?;
    let value = codegen.builder.build_load(llvm_inner_type, value_ptr, "option_value")
        .map_err(|e| e.to_string())?;

    // Select
    let result = codegen.builder.build_select(is_some, value, default_val, "unwrap_or_result")
        .map_err(|e| e.to_string())?;
        
    Ok((result, inner_type))
}

pub const SOURCE: &str = include_str!("option.tl");
