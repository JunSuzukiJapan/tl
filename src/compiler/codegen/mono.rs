use super::CodeGenerator;
use crate::compiler::ast::Type;
use crate::compiler::ast_subst::TypeSubstitutor;
use inkwell::types::{BasicTypeEnum, StructType};
use inkwell::AddressSpace;
use std::collections::HashMap;

impl<'ctx> CodeGenerator<'ctx> {
    /// On-demand monomorphization of a method for a generic struct.
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

        // Use standard mangling resolution
        let mangled_name = crate::compiler::codegen::builtin_types::resolver::resolve_static_method_name(struct_name, method_name, generic_args);
        
        if let Some(f) = self.module.get_function(&mangled_name) {
             if !f.verify(true) {
                 f.print_to_stderr();
             }
            return Ok(mangled_name);
        }

        // Instantiate
        let substitutor = TypeSubstitutor::new(subst_map);
        let mut new_method = method.clone();
        new_method.name = mangled_name.clone(); 
        new_method.generics = vec![]; // Concrete
        
        let concrete_self = Type::Struct(struct_name.to_string(), generic_args.to_vec());
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
        // Save current builder position
        let previous_block = self.builder.get_insert_block();

        self.compile_fn_proto(&new_method)?;
        
        // NEW: If extern, we are done (declaration only). Otherwise compile body.
        if !new_method.is_extern {
            self.pending_functions.push(new_method);
        }
        
        // Restore builder position
        if let Some(block) = previous_block {
            self.builder.position_at_end(block);
        }
        
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
        self.pending_functions.push(new_func);
        
        Ok(mangled_name)
    }

    /// On-demand monomorphization of a generic enum.
    pub fn monomorphize_enum(
        &mut self,
        enum_name: &str,
        generic_args: &[Type],
    ) -> Result<String, String> {
        // 1. Check if the enum exists in generic registry
        let enum_def = self.enum_defs.get(enum_name)
            .ok_or_else(|| format!("Enum {} not found", enum_name))?.clone();

        // 2. Check generics
        if enum_def.generics.len() != generic_args.len() {
             return Err(format!("Generic count mismatch for enum {}: expected {}, got {}", 
                 enum_name, enum_def.generics.len(), generic_args.len()));
        }

        // 3. Mangle
        let mangled_name = self.mangle_type_name(enum_name, generic_args);
        
        // 4. Check if already instantiated
        if self.enum_types.contains_key(&mangled_name) {
             return Ok(mangled_name);
        }

        // 5. Instantiate
        // Build substitution map
        let mut subst_map = HashMap::new();
        for (param, arg) in enum_def.generics.iter().zip(generic_args) {
             subst_map.insert(param.clone(), arg.clone());
        }
        let substitutor = TypeSubstitutor::new(subst_map);

        let mut new_def = enum_def.clone();
        new_def.name = mangled_name.clone();
        new_def.generics = vec![];

        // Substitute variants
        for variant in &mut new_def.variants {
             match &mut variant.kind {
                 crate::compiler::ast::VariantKind::Unit => {},
                 crate::compiler::ast::VariantKind::Tuple(types) => {
                     for t in types {
                         *t = substitutor.substitute_type(t);
                     }
                 }
                 crate::compiler::ast::VariantKind::Struct(fields) => {
                     for (_, t) in fields {
                         *t = substitutor.substitute_type(t);
                     }
                 }
             }
        }

        // 6. Compile/Register
        // We can reuse compile_enum_defs. It handles LLVM struct creation and map insertion.
        self.compile_enum_defs(&[new_def.clone()])
            .map_err(|e| e.to_string())?;

        // Note: compile_enum_defs adds it to `enum_defs` and `enum_types`
        
        Ok(mangled_name)
    }

    fn unify_types(
        &self,
        expected: &Type,
        actual: &Type,
        map: &mut HashMap<String, Type>,
    ) -> Result<(), String> {
         match (expected, actual) {
             (Type::Struct(name, args), _) => {
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
                         // But wait, what if 'name' is a concrete type? We shouldn't map it to actual.
                         // We should only map if 'name' is in Fn generics.
                         // But unification helper doesn't know the list.
                         // We can assume valid AST would prevent Shadowing of Types by generic params.
                         // So if we see UserDefined("T"), we map it.
                         // Ideally we should pass the set of generic params to safe guard.
                         map.insert(name.clone(), actual.clone());
                     }
                     return Ok(());
                 }
                 // If recursive generic (e.g. MyStruct<T>)
                 if let Type::Struct(act_name, act_args) = actual {
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
                 // Check if it's a generic struct instance matching a concrete struct
                 // This logic might be redundant if the above arm handles it, 
                 // but kept for "Any" matching or partial unification if needed.
                 // For now, if we are UNIFYING, we usually want exact structure.
                 // If `expected` is generic T and `actual` is Struct, handled by (Type::Generic, _)
                 
                 // If we are here, we have Struct vs something else (not Struct).
                 // e.g. Struct vs Tensor? Error.
                 return Err(format!("Type mismatch: Expected Struct {}, found {:?}", name, actual));
             }
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
            Type::String(_) => "String".to_string(),
            Type::Char(_) => "Char".to_string(),
            Type::Struct(name, args) => {
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

            Type::Tensor(inner, rank) => format!("Tensor_{}_{}", self.type_to_suffix(inner), rank),
            Type::Tuple(types) => {
                let parts: Vec<String> = types.iter().map(|t| self.type_to_suffix(t)).collect();
                format!("Tuple_{}", parts.join("_"))
            }
            _ => "unknown".to_string(),
        }
    }

    /// Mangle a method name for a generic type.
    /// Example: mangle_generic_method("Container", [U8], "process") -> "tl_container_u8_process"
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
            Type::U8 => Ok(self.context.i8_type().into()), // Added U8 support
            Type::Usize => Ok(self.context.i64_type().into()), // usize as i64
            Type::Void => panic!("Void type encountered in get_llvm_type"),
            
            Type::Tensor(_, _) | Type::TensorShaped(_, _) => {
                Ok(self.context.ptr_type(AddressSpace::default()).into())
            }
            
            Type::String(_) => {
                Ok(self.context.ptr_type(AddressSpace::default()).into())
            }
            Type::Char(_) => {
                Ok(self.context.i32_type().into())
            }

            Type::Struct(name, _args) => {
                // Compatibility: Handle primitives parsed as UserDefined
                match name.as_str() {
                    "bool" => return Ok(self.context.bool_type().into()),
                    "i64" => return Ok(self.context.i64_type().into()),
                    "i32" => return Ok(self.context.i32_type().into()),
                    "f32" => return Ok(self.context.f32_type().into()),
                    "f64" => return Ok(self.context.f64_type().into()),
                    "usize" => return Ok(self.context.i64_type().into()),
                    "u8" => return Ok(self.context.i8_type().into()), // Added u8 support
                    "String" => return Ok(self.context.ptr_type(inkwell::AddressSpace::default()).into()),
                    _ => {}
                }

                // Handle special types
                if name == "File" || name == "Path" || name == "Env" || name == "Http" {
                    return Ok(self.context.ptr_type(AddressSpace::default()).into());
                }

                // Check for ZSTs
                let simple_name = if name.contains("::") {
                    name.split("::").last().unwrap()
                } else {
                    name.as_str()
                };

                if let Some(def) = self.struct_defs.get(simple_name) {
                    if def.fields.is_empty() {
                        // ZST = Pointer (NULL)
                        return Ok(self.context.ptr_type(AddressSpace::default()).into());
                    }
                }
                
                if name.contains("PhantomData") {
                }

                // All other structs are pointer types (Reference Semantics)
                Ok(self.context.ptr_type(AddressSpace::default()).into())
            }
            
            Type::Enum(_name, _args) => {
                Ok(self.context.ptr_type(AddressSpace::default()).into())
            }
            
            Type::Tuple(_) => {
                Ok(self.context.ptr_type(AddressSpace::default()).into())
            }

            Type::Ref(_) => {
                Ok(self.context.ptr_type(AddressSpace::default()).into())
            }

            Type::Path(segments, _) => {
                 panic!("Unresolved Type::Path in codegen: {:?}. This should have been resolved to Type::Struct by semantics/monomorphizer.", segments);
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
            Type::Struct(name, args) | Type::Struct(name, args) if !args.is_empty() => {
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
        for (field_name, field_ty) in &struct_def.fields {
            let substituted_ty = self.substitute_type(field_ty, &subst);
            let llvm_ty = self.get_llvm_type(&substituted_ty).map_err(|e| format!("Error compiling field {} of {}: {}", field_name, mangled_name, e))?;
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
            Type::Struct(name, args) => {
                // Check if this is a type parameter
                if args.is_empty() {
                    if let Some(replacement) = subst.get(name) {
                        return replacement.clone();
                    }
                }
                // Recursively substitute in args
                let new_args: Vec<Type> = args.iter().map(|a| self.substitute_type(a, subst)).collect();
                
                // match name.as_str() { ... } removed
                Type::Struct(name.clone(), new_args)
            }


            Type::Tensor(inner, rank) => Type::Tensor(Box::new(self.substitute_type(inner, subst)), *rank),
            Type::Tuple(types) => Type::Tuple(types.iter().map(|t| self.substitute_type(t, subst)).collect()),
            Type::Path(segments, args) => {
                 if segments.len() == 1 {
                     if let Some(replacement) = subst.get(&segments[0]) {
                         return replacement.clone();
                     }
                 }
                 let new_args: Vec<Type> = args.iter().map(|a| self.substitute_type(a, subst)).collect();
                 Type::Path(segments.clone(), new_args)
            },
            _ => ty.clone(),
        }
    }



}

