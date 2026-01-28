use super::CodeGenerator;
use crate::compiler::ast::Type;
use crate::compiler::ast_subst::TypeSubstitutor;
use inkwell::types::{BasicTypeEnum, StructType};
use inkwell::AddressSpace;
use std::collections::HashMap;

impl<'ctx> CodeGenerator<'ctx> {
    /// On-demand monomorphization of a method for a generic struct.
    pub fn monomorphize_method(
        &mut self,
        struct_name: &str,
        method_name: &str,
        generic_args: &[Type],
    ) -> Result<String, String> {
        let impls = self.generic_impls.get(struct_name)
             .ok_or_else(|| format!("No generic impls found for struct {}", struct_name))?;

        // Find method in impls
        let mut target_method = None;
        let mut target_impl = None;
        
        for imp in impls {
            for method in &imp.methods {
                 if method.name == method_name {
                     target_method = Some(method);
                     target_impl = Some(imp);
                     break;
                 }
            }
            if target_method.is_some() { break; }
        }
        
        let method = target_method.ok_or_else(|| format!("Method {} not found in generic impls of {}", method_name, struct_name))?;
        let imp = target_impl.unwrap();

        // Check generic count
        if imp.generics.len() != generic_args.len() {
             return Err(format!("Generic arg count mismatch for impl {}: expected {}, got {}", 
                 struct_name, imp.generics.len(), generic_args.len()));
        }

        // Build substitution map
        let mut subst_map = HashMap::new();
        for (param, arg) in imp.generics.iter().zip(generic_args) {
             subst_map.insert(param.clone(), arg.clone());
        }

        // Mangle name
        // tl_Rect_i64_area
        let suffix = generic_args.iter().map(|t| self.type_to_suffix(t)).collect::<Vec<_>>().join("_");
        let mangled_name = format!("tl_{}_{}_{}", struct_name, suffix, method_name);
        
        if self.module.get_function(&mangled_name).is_some() {
            return Ok(mangled_name);
        }

        // Instantiate
        let substitutor = TypeSubstitutor::new(subst_map);
        let mut new_method = method.clone();
        new_method.name = mangled_name.clone(); 
        new_method.generics = vec![]; // Concrete
        
        // Fix "Self" usage and Substitute
        // The return type, args, and body might use T or Self.
        // Self -> Struct<Args> (UserDefined)
        // Wait, ASTSubstitutor handles explicit Type mapping.
        // But Self is implicit in some contexts?
        // In valid TL AST, Self is usually pre-resolved or UserDefined("Self", [])?
        // Let's assume UserDefined("Self") is handled by substitution if we map "Self" -> ConcreteType.
        
        let concrete_self = Type::UserDefined(struct_name.to_string(), generic_args.to_vec());
        // Add Self to substitution map if not already present
        // Actually TypeSubstitutor might need special handling for Self?
        // Or we just add it to the map.
        // Let's create a new substitutor with Self included.
        let mut full_map = substitutor.subst.clone();
        full_map.insert("Self".to_string(), concrete_self);
        let full_substitutor = TypeSubstitutor::new(full_map);

        // Substitute
        for (_, ty) in &mut new_method.args {
            *ty = full_substitutor.substitute_type(ty);
        }
        new_method.return_type = full_substitutor.substitute_type(&new_method.return_type);
        new_method.body = new_method.body.iter().map(|s| full_substitutor.substitute_stmt(s)).collect();
        
        // Compile
        // Compile
        
        // Add to module
        // We need to compile it as a function.
        // Note: compile_impl_blocks usually handles name mangling "tl_Struct_method".
        // Here we pre-mangled it to "tl_Struct_Args_method", so we can just use compile_fn?
        // Yes, as long as compile_fn uses the name in function def.
        
        self.compile_fn_proto(&new_method)?;
        self.compile_fn(&new_method, &[])?;
        
        Ok(mangled_name)
    }

    /// On-demand monomorphization of a user-defined generic function.
    pub fn monomorphize_generic_function(
        &mut self,
        func_name: &str,
        arg_types: &[Type],
    ) -> Result<String, String> {
        // 1. Check if the function exists in generic registry
        if !self.generic_fn_defs.contains_key(func_name) {
             // Not a generic function, or not found.
             return Ok(func_name.to_string());
        }
        
        // 2. Retrieve definition
        let func_def = self.generic_fn_defs.get(func_name).cloned().unwrap();
        
        // 3. Unify argument types to infer type parameters
        // func_def.generics vs func_def.args vs arg_types
        if func_def.args.len() != arg_types.len() {
             return Err(format!("Argument count mismatch for generic function {}: expected {}, got {}", 
                 func_name, func_def.args.len(), arg_types.len()));
        }

        let mut subst_map: HashMap<String, Type> = HashMap::new();
        for ((_, expected_ty), actual_ty) in func_def.args.iter().zip(arg_types) {
            self.unify_types(expected_ty, actual_ty, &mut subst_map)?;
        }
        
        // Ensure all generics are inferred
        for param in &func_def.generics {
            if !subst_map.contains_key(param) {
                 return Err(format!("Could not infer type parameter {} for function {}", param, func_name));
            }
        }
        
        // 4. Mangle name based on concrete types
        // Create a list of type args in the order of declaration
        let type_args: Vec<Type> = func_def.generics.iter().map(|g| subst_map[g].clone()).collect();
        let mangled_name = format!("{}_{}", func_name, 
             type_args.iter().map(|t| self.type_to_suffix(t)).collect::<Vec<_>>().join("_"));
             
        if self.module.get_function(&mangled_name).is_some() {
            return Ok(mangled_name);
        }
        
        // 6. Instantiate
        let substitutor = TypeSubstitutor::new(subst_map);
        
        let mut new_func = func_def.clone();
        new_func.name = mangled_name.clone();
        new_func.generics = vec![]; // Concrete now
        
        // Substitute args
        for (_, ty) in &mut new_func.args {
            *ty = substitutor.substitute_type(ty);
        }
        // Substitute return type
        new_func.return_type = substitutor.substitute_type(&new_func.return_type);
        // Substitute body
        new_func.body = new_func.body.iter().map(|s| substitutor.substitute_stmt(s)).collect();
        
        // 7. Compile
        // Need to register proto first (for recursion support etc)
        self.compile_fn_proto(&new_func)?;
        self.compile_fn(&new_func, &[])?;
        
        Ok(mangled_name)
    }

    fn unify_types(
        &self,
        expected: &Type,
        actual: &Type,
        map: &mut HashMap<String, Type>,
    ) -> Result<(), String> {
         match (expected, actual) {
             (Type::UserDefined(name, args), _) => {
                 // If It's a Type Parameter (no args), infer it.
                 // Need to know if 'name' is a generic param.
                 // For now assume if it's in the generic list it is.
                 // But here we don't have scope.
                 // However, UserDefined type params usually have empty args.
                 if args.is_empty() {
                     // Check if already mapped
                     if let Some(existing) = map.get(name) {
                         if existing != actual {
                              return Err(format!("Type mismatch for generic {}: expected {:?}, got {:?}", name, existing, actual));
                         }
                     } else {
                         // Map it!
                         // But wait, what if 'name' is "Vec"? We shouldn't map "Vec" to actual.
                         // We should only map if 'name' is in Fn generics.
                         // But unification helper doesn't know the list.
                         // We can assume valid AST would prevent Shadowing of "Vec" by "T".
                         // So if we see UserDefined("T"), we map it.
                         // Ideally we should pass the set of generic params to safe guard.
                         map.insert(name.clone(), actual.clone());
                     }
                     return Ok(());
                 }
                 // If recursive generic (e.g. MyStruct<T>)
                 if let Type::UserDefined(act_name, act_args) = actual {
                     if name != act_name || args.len() != act_args.len() {
                          return Err("Type mismatch or arity mismatch".into());
                     }
                     for (e, a) in args.iter().zip(act_args) {
                         self.unify_types(e, a, map)?;
                     }
                     return Ok(());
                 }
                 // If structural match fails
                 // Might handle Struct vs UserDefined aliases?
             }
             (Type::Vec(e), Type::Vec(a)) => self.unify_types(e, a, map)?,
             (Type::Tensor(e, r), Type::Tensor(a, ar)) => {
                 if r != ar { return Err("Rank mismatch".into()); }
                 self.unify_types(e, a, map)?;
             }
             // Add other structural recursions as needed
             _ => {
                 if expected != actual {
                     // If they are different concrete types, error.
                     // But wait, what if expected is concrete? (e.g. Fn(i64, T))
                     return Err(format!("Type mismatch: expected {:?}, got {:?}", expected, actual));
                 }
             }
         }
         Ok(())
    }

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
    pub fn type_to_suffix(&self, ty: &Type) -> String {
        match ty {
            Type::I64 => "i64".to_string(),
            Type::I32 => "i32".to_string(),
            Type::U8 => "u8".to_string(),
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

    /// Mangle a method name for a generic type.
    /// Example: mangle_generic_method("Vec", [U8], "to_tensor_2d") -> "tl_vec_u8_to_tensor_2d"
    pub fn mangle_generic_method(
        &self,
        base_type: &str,
        type_args: &[Type],
        method: &str,
    ) -> String {
        let suffix = if type_args.is_empty() {
            String::new()
        } else {
            format!("_{}", type_args.iter()
                .map(|t| self.type_to_suffix(t).to_lowercase())
                .collect::<Vec<_>>()
                .join("_"))
        };
        format!("tl_{}{}_{}", base_type.to_lowercase(), suffix, method)
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
            },
            
            Type::Tensor(_, _) | Type::TensorShaped(_, _) => {
                Ok(self.context.ptr_type(AddressSpace::default()).into())
            }
            
            Type::Vec(_) => {
                // All Vec types are represented as opaque pointers to Rust Vec
                Ok(self.context.ptr_type(AddressSpace::default()).into())
            }

            Type::Struct(name, _args) | Type::UserDefined(name, _args) => {
                // Compatibility: Handle primitives parsed as UserDefined
                match name.as_str() {
                    "bool" => return Ok(self.context.bool_type().into()),
                    "i64" => return Ok(self.context.i64_type().into()),
                    "i32" => return Ok(self.context.i32_type().into()),
                    "f32" => return Ok(self.context.f32_type().into()),
                    "f64" => return Ok(self.context.f64_type().into()),
                    "usize" => return Ok(self.context.i64_type().into()),
                    _ => {}
                }

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
                 let _ = self.builder.build_return(None);
             } else {
                 match call.unwrap().try_as_basic_value() {
                     inkwell::values::ValueKind::Basic(v) => {
                         let _ = self.builder.build_return(Some(&v));
                     }
                     _ => {
                        // Should not happen if return type is not none?
                        let _ = self.builder.build_return(None);
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
