// src/compiler/codegen/mono.rs
//! On-demand monomorphization for generic types and methods.
//!
//! This module provides utilities to generate specialized LLVM functions and types
//! when a generic type (like `Vec<T>`) or a user-defined generic struct (like `Rect<T>`)
//! is instantiated with concrete type arguments.

use super::CodeGenerator;
use crate::compiler::ast::Type;
use inkwell::types::{BasicTypeEnum, StructType};
use inkwell::AddressSpace;
use std::collections::HashMap;

impl<'ctx> CodeGenerator<'ctx> {
    /// Generate a mangled name for a monomorphized type.
    /// Example: `Vec` + `[i64]` -> `Vec_i64`
    pub fn mangle_type_name(&self, base_name: &str, type_args: &[Type]) -> String {
        if type_args.is_empty() {
            base_name.to_string()
        } else {
            let args_str: Vec<String> = type_args.iter().map(|t| self.type_to_suffix(t)).collect();
            format!("{}_{}", base_name, args_str.join("_"))
        }
    }

    /// Convert a Type to a string suffix for mangling.
    fn type_to_suffix(&self, ty: &Type) -> String {
        match ty {
            Type::I64 => "i64".to_string(),
            Type::I32 => "i32".to_string(),
            Type::F32 => "f32".to_string(),
            Type::F64 => "f64".to_string(),
            Type::Bool => "bool".to_string(),
            Type::Usize => "usize".to_string(),
            Type::Void => "void".to_string(),
            Type::Struct(name, args) | Type::UserDefined(name, args) => {
                if args.is_empty() {
                    name.clone()
                } else {
                    self.mangle_type_name(name, args)
                }
            }
            Type::Enum(name, args) => {
                if args.is_empty() {
                    name.clone()
                } else {
                    self.mangle_type_name(name, args)
                }
            }
            Type::Vec(inner) => format!("Vec_{}", self.type_to_suffix(inner)),
            Type::Tensor(inner, rank) => format!("Tensor_{}_{}", self.type_to_suffix(inner), rank),
            Type::Tuple(types) => {
                let parts: Vec<String> = types.iter().map(|t| self.type_to_suffix(t)).collect();
                format!("Tuple_{}", parts.join("_"))
            }
            _ => "unknown".to_string(),
        }
    }

    /// Get or create the LLVM type for the given AST Type.
    /// This is the single source of truth for Type -> BasicTypeEnum conversion.
    /// Note: This version uses &self and does NOT perform on-demand monomorphization.
    /// For monomorphization, use `get_or_monomorphize_type` which takes &mut self.
    pub fn get_llvm_type(&self, ty: &Type) -> Result<BasicTypeEnum<'ctx>, String> {
        match ty {
            Type::I64 | Type::Entity => Ok(self.context.i64_type().into()),
            Type::I32 => Ok(self.context.i32_type().into()),
            Type::F32 => Ok(self.context.f32_type().into()),
            Type::F64 => Ok(self.context.f64_type().into()),
            Type::Bool => Ok(self.context.bool_type().into()),
            Type::Usize => Ok(self.context.i64_type().into()), // usize as i64
            Type::Void => {
                panic!("Void type encountered in get_llvm_type");
                // Err("Void type has no LLVM representation".to_string())
            },
            
            Type::Tensor(_, _) | Type::TensorShaped(_, _) => {
                Ok(self.context.ptr_type(AddressSpace::default()).into())
            }
            
            Type::Vec(_) => {
                // All Vec types are represented as opaque pointers to Rust Vec
                Ok(self.context.ptr_type(AddressSpace::default()).into())
            }

            Type::Struct(name, _args) | Type::UserDefined(name, _args) => {
                // Handle special types
                if name == "String" || name == "File" || name == "Path" || name == "Env" || name == "Http" {
                    return Ok(self.context.ptr_type(AddressSpace::default()).into());
                }
                // All structs are pointer types
                Ok(self.context.ptr_type(AddressSpace::default()).into())
            }
            
            Type::Enum(_name, _args) => {
                Ok(self.context.ptr_type(AddressSpace::default()).into())
            }
            
            Type::ScalarArray(_, _) => {
                Ok(self.context.ptr_type(AddressSpace::default()).into())
            }
            
            Type::Tuple(types) => {
                let mut field_types = Vec::new();
                for inner_ty in types {
                    field_types.push(self.get_llvm_type(inner_ty)?);
                }
                let struct_ty = self.context.struct_type(&field_types, false);
                Ok(struct_ty.into())
            }
            
            _ => {
                // Default to i64 for unknown types
                Ok(self.context.i64_type().into())
            }
        }
    }
    
    /// Get or create the LLVM type, performing on-demand monomorphization if needed.
    /// This version takes &mut self and can create new struct definitions.
    pub fn get_or_monomorphize_type(&mut self, ty: &Type) -> Result<BasicTypeEnum<'ctx>, String> {
        match ty {
            Type::Struct(name, args) | Type::UserDefined(name, args) if !args.is_empty() => {
                // Generic struct: monomorphize
                let _ = self.monomorphize_struct(name, args)?;
                Ok(self.context.ptr_type(AddressSpace::default()).into())
            }
            _ => self.get_llvm_type(ty),
        }
    }

    /// Monomorphize a generic struct definition with concrete type arguments.
    /// Returns the LLVM StructType for the specialized struct.
    pub fn monomorphize_struct(
        &mut self,
        base_name: &str,
        type_args: &[Type],
    ) -> Result<StructType<'ctx>, String> {
        let mangled_name = self.mangle_type_name(base_name, type_args);
        
        // Check if already monomorphized
        if let Some(existing) = self.struct_types.get(&mangled_name) {
            return Ok(*existing);
        }
        
        // Get the generic struct definition
        let struct_def = self.struct_defs.get(base_name).cloned()
            .ok_or_else(|| format!("Generic struct definition not found: {}", base_name))?;
        
        // Build type parameter substitution map
        let mut subst: HashMap<String, Type> = HashMap::new();
        for (i, param_name) in struct_def.generics.iter().enumerate() {
            if let Some(arg) = type_args.get(i) {
                subst.insert(param_name.clone(), arg.clone());
            }
        }
        
        // Create opaque struct
        let opaque_struct = self.context.opaque_struct_type(&mangled_name);
        self.struct_types.insert(mangled_name.clone(), opaque_struct);
        
        // Build field types with substitution
        let mut field_llvm_types = Vec::new();
        for (_field_name, field_ty) in &struct_def.fields {
            let substituted_ty = self.substitute_type(field_ty, &subst);
            let llvm_ty = self.get_llvm_type(&substituted_ty)?;
            field_llvm_types.push(llvm_ty);
        }
        
        // Set struct body
        opaque_struct.set_body(&field_llvm_types, false);
        
        // Store the specialized struct def for later use
        let mut specialized_def = struct_def.clone();
        specialized_def.name = mangled_name.clone();
        specialized_def.generics = vec![]; // No longer generic
        // Substitute field types
        specialized_def.fields = struct_def.fields.iter().map(|(name, ty)| {
            (name.clone(), self.substitute_type(ty, &subst))
        }).collect();
        self.struct_defs.insert(mangled_name.clone(), specialized_def);
        
        Ok(opaque_struct)
    }
    
    /// Substitute type parameters in a type using the given substitution map.
    fn substitute_type(&self, ty: &Type, subst: &HashMap<String, Type>) -> Type {
        match ty {
            Type::UserDefined(name, args) => {
                // Check if this is a type parameter
                if args.is_empty() {
                    if let Some(replacement) = subst.get(name) {
                        return replacement.clone();
                    }
                }
                // Recursively substitute in args
                let new_args: Vec<Type> = args.iter().map(|a| self.substitute_type(a, subst)).collect();
                Type::UserDefined(name.clone(), new_args)
            }
            Type::Struct(name, args) => {
                let new_args: Vec<Type> = args.iter().map(|a| self.substitute_type(a, subst)).collect();
                Type::Struct(name.clone(), new_args)
            }
            Type::Vec(inner) => Type::Vec(Box::new(self.substitute_type(inner, subst))),
            Type::Tensor(inner, rank) => Type::Tensor(Box::new(self.substitute_type(inner, subst)), *rank),
            Type::Tuple(types) => Type::Tuple(types.iter().map(|t| self.substitute_type(t, subst)).collect()),
            _ => ty.clone(),
        }
    }

    /// Ensure a specialized Vec method exists for the given element type.
    /// Returns the mangled function name.
    pub fn ensure_vec_method(
        &mut self,
        element_type: &Type,
        method_name: &str,
    ) -> Result<String, String> {
        let suffix = self.type_to_suffix(element_type);
        let mangled_fn_name = format!("tl_vec_{}_{}", suffix, method_name);
        
        // Check if already declared
        if self.module.get_function(&mangled_fn_name).is_some() {
            return Ok(mangled_fn_name);
        }
        
        // Determine which core runtime function to delegate to
        let core_fn_name = match element_type {
            Type::I64 => format!("tl_vec_i64_{}", method_name),
            Type::F32 => format!("tl_vec_f32_{}", method_name),
            _ => format!("tl_vec_ptr_{}", method_name), // Pointer-based for structs/strings
        };
        
        // If core function exists but mangled doesn't, create a wrapper function
        if let Some(core_fn) = self.module.get_function(&core_fn_name) {
             // Create mangled function
             // Note: Params might need casting if we are strictly using matched types.
             // But for now, we just want a function accessible by mangled name.
             // Since ensure_vec_method callers will cast arguments to what the wrapper expects?
             // No, the wrapper should adapt types if possible, or just have same signature as core fn?
             // The core fn `tl_vec_ptr_push` takes `(i8* vec, i8* elem)`.
             // `tl_vec_i64_push` takes `(i8* vec, i64 elem)`.
             
             // If we duplicate the signature of core_fn, we don't change anything for the caller, except the name.
             // Caller currently does casting in compile_method_call.
             
             let fn_type = core_fn.get_type();
             let wrapper_fn = self.module.add_function(&mangled_fn_name, fn_type, None);
             
             // Generate body: call core_fn
             let entry_bb = self.context.append_basic_block(wrapper_fn, "entry");
             let saved_block = self.builder.get_insert_block(); // Save current block
             self.builder.position_at_end(entry_bb);
             
             let mut args = Vec::new();
             for i in 0..wrapper_fn.count_params() {
                 args.push(wrapper_fn.get_nth_param(i).unwrap().into());
             }
             
             let call = self.builder.build_call(core_fn, &args, "");
             
             if core_fn.get_type().get_return_type().is_none() {
                 self.builder.build_return(None);
             } else {
                 match call.unwrap().try_as_basic_value() {
                     inkwell::values::ValueKind::Basic(v) => {
                         self.builder.build_return(Some(&v));
                     }
                     _ => {
                        // Should not happen if return type is not none?
                        self.builder.build_return(None);
                     }
                 }
             }
             
             // Restore builder position
             if let Some(bb) = saved_block {
                 self.builder.position_at_end(bb);
             }
            
            return Ok(mangled_fn_name);
        }
        
        // Fallback: use core function name directly
        Ok(core_fn_name)
    }
}
