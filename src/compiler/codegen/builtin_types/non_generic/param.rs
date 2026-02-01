use crate::compiler::codegen::type_manager::{CodeGenType, TypeManager};
use crate::compiler::codegen::CodeGenerator;
use crate::compiler::ast::{Type};
use inkwell::values::{BasicValueEnum, ValueKind};
use std::collections::HashMap;

pub fn register_param_types(manager: &mut TypeManager) {
    let mut param = CodeGenType::new("Param");

    // Dummy signatures. Semantics check is skipped for Param to allow overloading.
    let any_tensor = Type::Tensor(Box::new(Type::F32), 0);
    
    // save(tensor: Tensor, path: String) -> Void
    // save(struct: Struct, path: String) -> Void
    param.register_evaluated_static_method("save", compile_param_save, vec![any_tensor.clone(), Type::String("String".into())], Type::Void);
    
    // load(path) -> Tensor 
    // load(struct, path) -> Void
    param.register_evaluated_static_method("load", compile_param_load, vec![Type::String("String".into())], any_tensor.clone());

    manager.register_type(param);
}

fn compile_param_save<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    if args.len() != 2 {
        return Err("Param::save requires 2 arguments".into());
    }
    
    let (arg0_val, arg0_ty) = &args[0];
    let (path_val, _) = &args[1]; 
    
    // 1. Create Map
    let map_new_fn = codegen.module.get_function("tl_tensor_map_new").ok_or("tl_tensor_map_new not found")?;
    let map_call = codegen.builder.build_call(map_new_fn, &[], "map").map_err(|e| e.to_string())?;
    
    let map_ptr = match map_call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("tl_tensor_map_new returned void".into()),
    };

    // 2. Traverse and Insert
    traverse_and_save(codegen, *arg0_val, arg0_ty.clone(), "".to_string(), map_ptr)?;

    // 3. Save Map
    let save_fn = codegen.module.get_function("tl_tensor_map_save").ok_or("tl_tensor_map_save not found")?;
    codegen.builder.build_call(save_fn, &[map_ptr.into(), (*path_val).into()], "").map_err(|e| e.to_string())?;

    // 4. Free Map
    let free_fn = codegen.module.get_function("tl_tensor_map_free").ok_or("tl_tensor_map_free not found")?;
    codegen.builder.build_call(free_fn, &[map_ptr.into()], "").map_err(|e| e.to_string())?;

    Ok((codegen.context.i64_type().const_int(0, false).into(), Type::Void))
}

fn traverse_and_save<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    val: BasicValueEnum<'ctx>,
    ty: Type,
    prefix: String,
    map_ptr: BasicValueEnum<'ctx>,
) -> Result<(), String> {
    match ty {
        Type::Tensor(_, _) => {
            // Insert into map
            // key = prefix
            let key_str = if prefix.is_empty() { "tensor".to_string() } else { prefix };
            let key_global_ptr = codegen.builder.build_global_string_ptr(&key_str, "key_str").map_err(|e| e.to_string())?; // i8*
            
            // tl_string_new(i8*) -> StringStruct*
            let str_new_fn = codegen.module.get_function("tl_string_new").ok_or("tl_string_new not found")?;
            let key_struct_call = codegen.builder.build_call(str_new_fn, &[key_global_ptr.as_pointer_value().into()], "key_struct").map_err(|e| e.to_string())?;
            
            let key_struct_ptr = match key_struct_call.try_as_basic_value() {
                ValueKind::Basic(v) => v,
                _ => return Err("tl_string_new returned void".into()),
            };

            // tl_tensor_map_insert(map, key, val)
            let insert_fn = codegen.module.get_function("tl_tensor_map_insert").ok_or("tl_tensor_map_insert not found")?;
            codegen.builder.build_call(insert_fn, &[map_ptr.into(), key_struct_ptr.into(), val.into()], "").map_err(|e| e.to_string())?;
        }
        Type::Struct(name, generics) => {
            // Note: We need a copy of struct_def. Cloning from codegen.struct_defs requires borrowing codegen.
            // We can't hold reference to struct_def while calling recursive which needs &mut codegen.
            let struct_def = codegen.struct_defs.get(&name).cloned();
            
            if let Some(def) = struct_def {
                // Prepare substitution map
                let mut subst = HashMap::new();
                for (i, param) in def.generics.iter().enumerate() {
                    if i < generics.len() {
                        subst.insert(param.clone(), generics[i].clone());
                    }
                }

                if !val.is_pointer_value() {
                    // It might be possible that Val is not a pointer if it's a zero-sized struct?
                    // But usually passed by pointer.
                    // Or if Type::Struct is treated as value by inkwell?
                    // In TL, structs are passed by pointer in LLVM IR mostly.
                    return Err(format!("Expected pointer for struct {}, got {:?}", name, val));
                }
                let ptr = val.into_pointer_value();

                for (idx, (field_name, field_type)) in def.fields.iter().enumerate() {
                     let concrete_type = substitute(&field_type, &subst);
                     let new_prefix = if prefix.is_empty() {
                         field_name.clone()
                     } else {
                         format!("{}.{}", prefix, field_name)
                     };
                     
                     let struct_llvm_type = *codegen.struct_types.get(&name).ok_or(format!("LLVM type for {} not found", name))?;
                     
                     let field_ptr = codegen.builder.build_struct_gep(struct_llvm_type, ptr, idx as u32, "field_ptr").map_err(|e| e.to_string())?;
                     
                     match concrete_type {
                         Type::Tensor(_,_) => {
                             // Load the OpaqueTensor*
                             let loaded_val = codegen.builder.build_load(codegen.context.ptr_type(inkwell::AddressSpace::default()), field_ptr, "tensor_val").map_err(|e| e.to_string())?;
                             traverse_and_save(codegen, loaded_val, concrete_type, new_prefix, map_ptr)?;
                         },
                         Type::Struct(_,_) => {
                             // Recurse with pointer to the sub-struct
                             // Structs are reference types/pointers in this backend, so we must load the pointer from the field.
                             let loaded_val = codegen.builder.build_load(codegen.context.ptr_type(inkwell::AddressSpace::default()), field_ptr, "struct_ptr").map_err(|e| e.to_string())?;
                             traverse_and_save(codegen, loaded_val, concrete_type, new_prefix, map_ptr)?;
                         },
                         _ => {
                             // Skip primitives
                         }
                     }
                }
            } else {
                return Err(format!("Struct definition for {} not found", name));
            }
        }
        _ => {
            // Ignore primitives
        }
    }
    Ok(())
}

fn substitute(ty: &Type, subst: &HashMap<String, Type>) -> Type {
    match ty {
        Type::Struct(n, args) => {
             // Check if n is a parameter
             if let Some(replacement) = subst.get(n) {
                 return replacement.clone();
             }
             let new_args = args.iter().map(|a| substitute(a, subst)).collect();
             Type::Struct(n.clone(), new_args)
        }
        Type::Tensor(dt, rank) => Type::Tensor(Box::new(substitute(dt, subst)), *rank),
        _ => ty.clone(),
    }
}

fn compile_param_load<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    
    if args.len() == 1 {
        // load(path) -> Tensor
        let (path_val, _) = &args[0];
        let fn_val = codegen.module.get_function("tl_tensor_load").ok_or("tl_tensor_load not found")?;
        let call = codegen.builder.build_call(fn_val, &[(*path_val).into()], "load_res").map_err(|e| e.to_string())?;
        
        let res = match call.try_as_basic_value() {
             ValueKind::Basic(v) => v,
             _ => return Err("Invalid return from load".into()),
        };
        return Ok((res, Type::Tensor(Box::new(Type::F32), 1))); 
    } else if args.len() == 2 {
        // load(struct, path) -> Void
        let (arg0_val, arg0_ty) = &args[0];
        let (path_val, _) = &args[1]; 
        
        let load_fn = codegen.module.get_function("tl_tensor_map_load").ok_or("tl_tensor_map_load not found")?;
        let map_call = codegen.builder.build_call(load_fn, &[(*path_val).into()], "map_loaded").map_err(|e| e.to_string())?;
        
        let map_ptr = match map_call.try_as_basic_value() {
             ValueKind::Basic(v) => v,
             _ => return Err("tl_tensor_map_load returned void".into()),
        };

        traverse_and_load(codegen, *arg0_val, arg0_ty.clone(), "".to_string(), map_ptr)?;

        let free_fn = codegen.module.get_function("tl_tensor_map_free").ok_or("tl_tensor_map_free not found")?;
        codegen.builder.build_call(free_fn, &[map_ptr.into()], "").map_err(|e| e.to_string())?;
        
        return Ok((codegen.context.i64_type().const_int(0, false).into(), Type::Void));
    } else {
         return Err("Param::load requires 1 or 2 args".into());
    }
}

fn traverse_and_load<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    val: BasicValueEnum<'ctx>,
    ty: Type,
    prefix: String,
    map_ptr: BasicValueEnum<'ctx>,
) -> Result<(), String> {
    match ty {
        Type::Tensor(_, _) => {
            // key = prefix
            let key_str = if prefix.is_empty() { "tensor".to_string() } else { prefix };
            let key_global_ptr = codegen.builder.build_global_string_ptr(&key_str, "key_str").map_err(|e| e.to_string())?; 
            
            let str_new_fn = codegen.module.get_function("tl_string_new").ok_or("tl_string_new not found")?;
            let key_struct_call = codegen.builder.build_call(str_new_fn, &[key_global_ptr.as_pointer_value().into()], "key_struct").map_err(|e| e.to_string())?;
            
            let key_struct_ptr = match key_struct_call.try_as_basic_value() {
                ValueKind::Basic(v) => v,
                _ => return Err("tl_string_new returned void".into()),
            };

            // get(map, key) -> *mut Tensor
            let get_fn = codegen.module.get_function("tl_tensor_map_get").ok_or("tl_tensor_map_get not found")?;
            let get_call = codegen.builder.build_call(get_fn, &[map_ptr.into(), key_struct_ptr.into()], "loaded_tensor").map_err(|e| e.to_string())?;
            
            let loaded_tensor_ptr = match get_call.try_as_basic_value() {
                 ValueKind::Basic(v) => v,
                 _ => return Err("tl_tensor_map_get returned void".into()),
            };
            
            let replace_fn = codegen.module.get_function("tl_tensor_replace_data").ok_or("tl_tensor_replace_data not found")?;
            codegen.builder.build_call(replace_fn, &[val.into(), loaded_tensor_ptr.into()], "").map_err(|e| e.to_string())?;
        }
        Type::Struct(name, generics) => {
            let struct_def = codegen.struct_defs.get(&name).cloned();
            
            if let Some(def) = struct_def {
                let mut subst = HashMap::new();
                for (i, param) in def.generics.iter().enumerate() {
                    if i < generics.len() {
                        subst.insert(param.clone(), generics[i].clone());
                    }
                }

                if !val.is_pointer_value() {
                    return Err(format!("Expected pointer for struct {}, got {:?}", name, val));
                }
                let ptr = val.into_pointer_value();
                let struct_llvm_type = *codegen.struct_types.get(&name).ok_or(format!("LLVM type for {} not found", name))?;

                for (idx, (field_name, field_type)) in def.fields.iter().enumerate() {
                     let concrete_type = substitute(&field_type, &subst);
                     let new_prefix = if prefix.is_empty() {
                         field_name.clone()
                     } else {
                         format!("{}.{}", prefix, field_name)
                     };
                     
                     let field_ptr = codegen.builder.build_struct_gep(struct_llvm_type, ptr, idx as u32, "field_ptr").map_err(|e| e.to_string())?;
                     
                     match concrete_type {
                         Type::Tensor(_,_) => {
                             let loaded_val = codegen.builder.build_load(codegen.context.ptr_type(inkwell::AddressSpace::default()), field_ptr, "tensor_val").map_err(|e| e.to_string())?;
                             traverse_and_load(codegen, loaded_val, concrete_type, new_prefix, map_ptr)?;
                         },
                         Type::Struct(_,_) => {
                             let loaded_val = codegen.builder.build_load(codegen.context.ptr_type(inkwell::AddressSpace::default()), field_ptr, "struct_ptr").map_err(|e| e.to_string())?;
                             traverse_and_load(codegen, loaded_val, concrete_type, new_prefix, map_ptr)?;
                         },
                         _ => {}
                     }
                }
            }
        }
        _ => {}
    }
    Ok(())
}
