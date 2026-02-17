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

        // Check and fix generic count - pad with default type if insufficient
        let mut final_generic_args = generic_args.to_vec();
        if imp.generics.len() > generic_args.len() {
            // Pad with I64 as default type for missing generics
            for _ in 0..(imp.generics.len() - generic_args.len()) {
                final_generic_args.push(Type::I64);
            }
        } else if imp.generics.len() < generic_args.len() {
            // Too many generics - truncate to expected count
            final_generic_args.truncate(imp.generics.len());
        }
        let generic_args = &final_generic_args;

        // Build substitution map
        let mut subst_map = HashMap::new();
        for (param, arg) in imp.generics.iter().zip(generic_args) {
             subst_map.insert(param.clone(), arg.clone());
        }

        // Use standard mangling resolution
        let mangled_name = crate::compiler::codegen::builtin_types::resolver::resolve_static_method_name(struct_name, method_name, generic_args);
        
        if self.module.get_function(&mangled_name).is_some() {

            return Ok(mangled_name);
        }

        // Instantiate
        let substitutor = TypeSubstitutor::new(subst_map);
        let mut new_method = method.clone();
        new_method.name = mangled_name.clone(); 
        new_method.generics = vec![]; // Concrete
        
        let concrete_self = if self.enum_defs.contains_key(struct_name) {
            Type::Enum(struct_name.to_string(), generic_args.to_vec())
        } else {
            Type::Struct(struct_name.to_string(), generic_args.to_vec())
        };
        let mut full_map = substitutor.subst.clone();
        full_map.insert("Self".to_string(), concrete_self);
        let full_substitutor = TypeSubstitutor::new(full_map);

        // Substitute
        for (_, ty) in &mut new_method.args {
            *ty = full_substitutor.substitute_type(ty);
            *ty = self.normalize_type(ty);
        }
        new_method.return_type = full_substitutor.substitute_type(&new_method.return_type);
        new_method.return_type = self.normalize_type(&new_method.return_type);
        new_method.body = new_method.body.iter().map(|s| full_substitutor.substitute_stmt(s)).collect();
        
        // Transform StaticMethodCall to EnumInit for enum variant constructors

        self.transform_method_body_enum_inits(&mut new_method.body);
        
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
        // First, try direct lookup
        let enum_def = if let Some(def) = self.enum_defs.get(enum_name) {
            def.clone()
        } else if enum_name.contains('_') && !enum_name.contains('<') {
            // Try extracting base name from underscore-mangled name (e.g., "Option_i64" -> "Option")
            let base_name = enum_name.split('_').next().unwrap_or(enum_name);
            if let Some(def) = self.enum_defs.get(base_name) {
                // If generic_args is empty but the def needs generics, try to parse from name
                if generic_args.is_empty() && !def.generics.is_empty() {
                    // This is already monomorphized with a different naming scheme
                    // Try to find the angle-bracket version
                    let suffix = enum_name.strip_prefix(base_name).unwrap_or("");
                    let suffix_clean = suffix.trim_start_matches('_');
                    let inferred_generics: Vec<Type> = suffix_clean.split('_')
                        .filter_map(|s| {
                            match s.to_lowercase().as_str() {
                                "i64" => Some(Type::I64),
                                "i32" => Some(Type::I32),
                                "f32" => Some(Type::F32),
                                "f64" => Some(Type::F64),
                                "bool" => Some(Type::Bool),
                                "u8" => Some(Type::U8),
                                "" => None,
                                _ => Some(Type::Struct(s.to_string(), vec![])),
                            }
                        })
                        .collect();
                    // Try to find or create the angle-bracket version
                    let angle_name = self.mangle_type_name(base_name, &inferred_generics);
                    if let Some(_existing) = self.enum_types.get(&angle_name) {
                        // Already exists
                        return Ok(angle_name);
                    }
                    // Recursively monomorphize with inferred generics
                    return self.monomorphize_enum(base_name, &inferred_generics);
                }
                def.clone()
            } else {
                return Err(format!("Enum {} not found (tried base {})", enum_name, base_name));
            }
        } else {
            return Err(format!("Enum {} not found", enum_name));
        };

        // 2. Check generics
        if enum_def.generics.len() != generic_args.len() {
             return Err(format!("Generic count mismatch for enum {}: expected {}, got {}", 
                 enum_name, enum_def.generics.len(), generic_args.len()));
        }

        // 3. Mangle
        let mangled_name = self.mangle_type_name(enum_name, generic_args);
        
        // Register reverse lookup for mangled name -> (base_name, generic_args)
        self.register_mangled_type(&mangled_name, enum_name, generic_args);
        
        // 4. Check if already instantiated
        if self.enum_types.contains_key(&mangled_name) {
             return Ok(mangled_name);
        }
        
        // Register this specialization
        self.specialization_registry.register(enum_name, generic_args);

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
                     for t in types.iter_mut() {
                         *t = substitutor.substitute_type(t);
                     }
                     eprintln!("[DEBUG] monomorphize_enum: {} variant {} = {:?}", mangled_name, variant.name, types);
                 }
                 crate::compiler::ast::VariantKind::Struct(fields) => {
                     for (_, t) in fields.iter_mut() {
                         *t = substitutor.substitute_type(t);
                     }
                     eprintln!("[DEBUG] monomorphize_enum: {} variant {} = {:?}", mangled_name, variant.name, fields);
                 }
                 crate::compiler::ast::VariantKind::Array(ty, _size) => {
                     *ty = substitutor.substitute_type(ty);
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
             (Type::Array(e_inner, e_size), Type::Array(a_inner, a_size)) => {
                 if e_size != a_size { return Err(format!("Array size mismatch: expected {}, got {}", e_size, a_size)); }
                 self.unify_types(e_inner, a_inner, map)?;
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
    /// Example: `Vec` + `[i64]` -> `Vec<i64>`
    /// Note: Call register_mangled_type separately when the mangled name is used for type registration.
    pub fn mangle_type_name(&self, base_name: &str, type_args: &[Type]) -> String {
        if type_args.is_empty() {
            base_name.to_string()
        } else {
            // Use underscore notation for consistency with frontend monomorphizer
            let args_str: Vec<String> = type_args.iter().map(|t| self.type_to_suffix(t)).collect();
            format!("{}_{}", base_name, args_str.join("_"))
        }
    }
    
    /// Register a mangled type name with its original base name and generic args.
    /// This enables reverse lookup from mangled name to original type information.
    pub fn register_mangled_type(&mut self, mangled_name: &str, base_name: &str, type_args: &[Type]) {
        self.reified_types.register_from_parts(mangled_name, base_name, type_args);
    }
    
    /// Lookup original type information from a mangled name.
    /// Returns (base_name, generic_args) if found.
    pub fn lookup_mangled_type(&self, mangled_name: &str) -> Option<(String, Vec<Type>)> {
        self.reified_types.lookup(mangled_name)
            .map(|r| (r.base_name.clone(), r.type_args.clone()))
    }
    
    /// Lookup a reified type by mangled name (full info)
    pub fn lookup_reified_type(&self, mangled_name: &str) -> Option<&super::reified_type::ReifiedType> {
        self.reified_types.lookup(mangled_name)
    }

    /// Convert a Type to a string representation for mangling/display.
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
            Type::Path(path, args) => {
                // Path types represent generic parameters or type references
                // Use the last segment of the path
                if let Some(name) = path.last() {
                    if args.is_empty() {
                        name.clone()
                    } else {
                        self.mangle_type_name(name, args)
                    }
                } else {
                    "unknown_path".to_string()
                }
            }
            Type::Undefined(id) => format!("undefined_{}", id),
            Type::Ptr(inner) => format!("ptr_{}", self.type_to_suffix(inner)),
            Type::Array(inner, size) => format!("Array_{}_{}", self.type_to_suffix(inner), size),
            Type::Entity => "entity".to_string(),
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
            Type::Ptr(_) => Ok(self.context.ptr_type(AddressSpace::default()).into()),
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
                let simple_name = name.as_str();

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

            Type::Array(inner, size) => {
                let elem_ty = self.get_llvm_type(inner)?;
                // Build LLVM array type from element type
                match elem_ty {
                    BasicTypeEnum::IntType(t) => Ok(t.array_type(*size as u32).into()),
                    BasicTypeEnum::FloatType(t) => Ok(t.array_type(*size as u32).into()),
                    BasicTypeEnum::PointerType(t) => Ok(t.array_type(*size as u32).into()),
                    BasicTypeEnum::StructType(t) => Ok(t.array_type(*size as u32).into()),
                    BasicTypeEnum::ArrayType(t) => Ok(t.array_type(*size as u32).into()),
                    BasicTypeEnum::VectorType(t) => Ok(t.array_type(*size as u32).into()),
                    _ => Err(format!("Unsupported array element type: {:?}", elem_ty)),
                }
            }
            
            Type::Tuple(_) => {
                Ok(self.context.ptr_type(AddressSpace::default()).into())
            }

            // Type::Ref(_) => { // REMOVED - Ref not in spec
            //     Ok(self.context.ptr_type(AddressSpace::default()).into())
            // }

            Type::Path(_segments, _) => {
                 // Assume Path resolves to Struct/Enum which are pointers (in this phase)
                 Ok(self.context.ptr_type(AddressSpace::default()).into())
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
            Type::Struct(name, args) if !args.is_empty() => {
                // Generic struct: monomorphize
                let _ = self.monomorphize_struct(name, args)?;
                Ok(self.context.ptr_type(AddressSpace::default()).into())
            }
            Type::Enum(name, args) if !args.is_empty() => {
                // Generic enum: monomorphize
                let _ = self.monomorphize_enum(name, args)?;
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
        
        // Register this specialization
        self.specialization_registry.register(base_name, type_args);
        
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
            let substituted = self.substitute_type(ty, &subst);
            // Fix: Convert Struct to Enum if it exists in enum_defs
            // Check only exact name match (not base_name) to avoid false positives like Vec_u8 -> Vec
            let corrected = if let Type::Struct(s_name, s_args) = &substituted {
                // Only convert if exact name exists in enum_defs
                if self.enum_defs.contains_key(s_name) {
                    Type::Enum(s_name.clone(), s_args.clone())
                } else {
                    substituted
                }
            } else {
                substituted
            };
            (name.clone(), corrected)
        }).collect();
        self.struct_defs.insert(mangled_name.clone(), specialized_def);
        
        Ok(opaque_struct)
    }
    
    /// Substitute type parameters in a type using the given substitution map.
    pub fn substitute_type(&self, ty: &Type, subst: &HashMap<String, Type>) -> Type {
        match ty {
            Type::Struct(name, args) => {
                // Check if this is a type parameter
                if args.is_empty() {
                    if let Some(replacement) = subst.get(name) {
                        return replacement.clone();
                    }
                }
                let new_args: Vec<Type> = args.iter().map(|a| self.substitute_type(a, subst)).collect();
                
                // Fix: Check if this is actually an Enum (exact match only)
                if self.enum_defs.contains_key(name) {
                    Type::Enum(name.clone(), new_args)
                } else {
                    Type::Struct(name.clone(), new_args)
                }
            }


            Type::Tensor(inner, rank) => Type::Tensor(Box::new(self.substitute_type(inner, subst)), *rank),
            Type::Tuple(types) => Type::Tuple(types.iter().map(|t| self.substitute_type(t, subst)).collect()),
             Type::Path(segments, args) => {
                 if segments.len() == 1 {
                     let name = &segments[0];
                     // Check if this is a type parameter being substituted
                     if let Some(replacement) = subst.get(name) {
                         return replacement.clone();
                     }
                     // Otherwise, convert to Struct or Enum based on definitions
                     let new_args: Vec<Type> = args.iter().map(|a| self.substitute_type(a, subst)).collect();
                     if self.enum_defs.contains_key(name) {
                         return Type::Enum(name.clone(), new_args);
                     } else {
                         return Type::Struct(name.clone(), new_args);
                     }
                 }
                 let new_args: Vec<Type> = args.iter().map(|a| self.substitute_type(a, subst)).collect();
                 Type::Path(segments.clone(), new_args)
            },
            Type::Ptr(inner) => Type::Ptr(Box::new(self.substitute_type(inner, subst))),
            Type::Array(inner, size) => Type::Array(Box::new(self.substitute_type(inner, subst)), *size),
            _ => ty.clone(),
        }
    }

    /// Normalize Type::Path to Type::Struct or Type::Enum based on definitions.
    /// This is called after TypeSubstitutor.substitute_type which doesn't have access to enum_defs.
    pub fn normalize_type(&self, ty: &Type) -> Type {
        match ty {
            Type::Path(segments, args) => {
                if segments.len() == 1 {
                    let name = &segments[0];
                    let normalized_args: Vec<Type> = args.iter().map(|a| self.normalize_type(a)).collect();
                    // Check if it's an enum and generic counts match
                    if let Some(enum_def) = self.enum_defs.get(name) {
                        if enum_def.generics.len() == normalized_args.len() || enum_def.generics.is_empty() {
                            return Type::Enum(name.clone(), normalized_args);
                        }
                        // Generic count mismatch - return as-is (don't convert)
                        return ty.clone();
                    }
                    // Not an enum, treat as Struct
                    Type::Struct(name.clone(), normalized_args)
                } else {
                    let normalized_args: Vec<Type> = args.iter().map(|a| self.normalize_type(a)).collect();
                    Type::Path(segments.clone(), normalized_args)
                }
            }
            Type::Struct(name, args) => {
                let normalized_args: Vec<Type> = args.iter().map(|a| self.normalize_type(a)).collect();
                // Only convert Struct to Enum if it's actually an enum AND generic counts match
                if let Some(enum_def) = self.enum_defs.get(name) {
                    if enum_def.generics.len() == normalized_args.len() || enum_def.generics.is_empty() {
                        return Type::Enum(name.clone(), normalized_args);
                    }
                    // Generic count mismatch - keep as Struct
                }
                Type::Struct(name.clone(), normalized_args)
            }
            Type::Enum(name, args) => {
                let normalized_args: Vec<Type> = args.iter().map(|a| self.normalize_type(a)).collect();
                Type::Enum(name.clone(), normalized_args)
            }
            Type::Tensor(inner, rank) => Type::Tensor(Box::new(self.normalize_type(inner)), *rank),
            Type::Tuple(types) => Type::Tuple(types.iter().map(|t| self.normalize_type(t)).collect()),
            Type::Ptr(inner) => Type::Ptr(Box::new(self.normalize_type(inner))),
            Type::Array(inner, size) => Type::Array(Box::new(self.normalize_type(inner)), *size),
            _ => ty.clone(),
        }
    }

    // ========== AST Transformation for Enum Variant Constructors ==========
    
    fn transform_method_body_enum_inits(&self, stmts: &mut Vec<crate::compiler::ast::Stmt>) {
        for stmt in stmts.iter_mut() {
            self.transform_stmt_enum_inits(stmt);
        }
    }
    
    fn transform_stmt_enum_inits(&self, stmt: &mut crate::compiler::ast::Stmt) {
        use crate::compiler::ast::StmtKind;
        match &mut stmt.inner {
            StmtKind::Let { value, .. } => self.transform_expr_enum_inits(value),
            StmtKind::Expr(e) => self.transform_expr_enum_inits(e),
            StmtKind::Return(Some(e)) => self.transform_expr_enum_inits(e),
            StmtKind::While { cond, body } => {
                self.transform_expr_enum_inits(cond);
                self.transform_method_body_enum_inits(body);
            }
            StmtKind::For { iterator, body, .. } => {
                self.transform_expr_enum_inits(iterator);
                self.transform_method_body_enum_inits(body);
            }
            StmtKind::Loop { body } => {
                self.transform_method_body_enum_inits(body);
            }
            StmtKind::Assign { value, .. } => {
                self.transform_expr_enum_inits(value);
            }
            _ => {}
        }
    }
    
    fn transform_expr_enum_inits(&self, expr: &mut crate::compiler::ast::Expr) {
        use crate::compiler::ast::{ExprKind, EnumVariantInit};
        
        // First, recurse into children
        match &mut expr.inner {
            ExprKind::BinOp(l, _, r) => {
                self.transform_expr_enum_inits(l);
                self.transform_expr_enum_inits(r);
            }
            ExprKind::UnOp(_, e) => {
                self.transform_expr_enum_inits(e);
            }
            ExprKind::MethodCall(obj, _, args) => {
                self.transform_expr_enum_inits(obj);
                for arg in args.iter_mut() {
                    self.transform_expr_enum_inits(arg);
                }
            }
            ExprKind::FnCall(_, args) => {
                for arg in args.iter_mut() {
                    self.transform_expr_enum_inits(arg);
                }
            }
            ExprKind::IndexAccess(e, indices) => {
                self.transform_expr_enum_inits(e);
                for idx in indices.iter_mut() {
                    self.transform_expr_enum_inits(idx);
                }
            }
            ExprKind::FieldAccess(obj, _) => {
                self.transform_expr_enum_inits(obj);
            }
            ExprKind::Match { expr: subject, arms } => {
                self.transform_expr_enum_inits(subject);
                for (_, arm_expr) in arms.iter_mut() {
                    self.transform_expr_enum_inits(arm_expr);
                }
            }
            ExprKind::Block(stmts) => {
                self.transform_method_body_enum_inits(stmts);
            }
            ExprKind::IfExpr(cond, then_block, else_block) => {
                self.transform_expr_enum_inits(cond);
                self.transform_method_body_enum_inits(then_block);
                if let Some(else_stmts) = else_block {
                    self.transform_method_body_enum_inits(else_stmts);
                }
            }
            ExprKind::Tuple(exprs) => {
                for e in exprs.iter_mut() {
                    self.transform_expr_enum_inits(e);
                }
            }
            ExprKind::StructInit(_, fields) => {
                for (_, e) in fields.iter_mut() {
                    self.transform_expr_enum_inits(e);
                }
            }
            ExprKind::EnumInit { payload, .. } => {
                match payload {
                    EnumVariantInit::Tuple(exprs) => {
                        for e in exprs.iter_mut() {
                            self.transform_expr_enum_inits(e);
                        }
                    }
                    EnumVariantInit::Struct(fields) => {
                        for (_, e) in fields.iter_mut() {
                            self.transform_expr_enum_inits(e);
                        }
                    }
                    EnumVariantInit::Unit => {}
                }
            }
            ExprKind::StaticMethodCall(ty, method, args) => {
                // Transform args first
                for arg in args.iter_mut() {
                    self.transform_expr_enum_inits(arg);
                }
                
                // Check if this is an enum variant constructor
                let enum_name = match ty {
                    Type::Struct(name, _) | Type::Enum(name, _) => name.clone(),
                    Type::Path(segments, _) => segments.last().cloned().unwrap_or_default(),
                    _ => String::new(),
                };
                
                if let Some(enum_def) = self.enum_defs.get(&enum_name) {
                    if let Some(variant) = enum_def.variants.iter().find(|v| &v.name == method) {
                        use crate::compiler::ast::VariantKind;
                        
                        // Build payload
                        let payload = match &variant.kind {
                            VariantKind::Unit => EnumVariantInit::Unit,
                            VariantKind::Tuple(_) => EnumVariantInit::Tuple(std::mem::take(args)),
                            VariantKind::Struct(_) => EnumVariantInit::Unit, // TODO: struct variant
                            VariantKind::Array(_, _) => EnumVariantInit::Tuple(std::mem::take(args)),
                        };
                        
                        // Extract generics from type
                        let generics = match ty {
                            Type::Struct(_, g) | Type::Enum(_, g) | Type::Path(_, g) => g.clone(),
                            _ => vec![],
                        };
                        
                        // Replace with EnumInit
                        expr.inner = ExprKind::EnumInit {
                            enum_name,
                            variant_name: method.clone(),
                            generics,
                            payload,
                        };
                        return;
                    }
                }
            }
            _ => {}
        }
    }

    /// Generate all methods for a specialized type.
    /// This is called at the end of compilation to ensure all methods are generated.
    pub fn generate_methods_for_specialized_type(
        &mut self,
        base_name: &str,
        type_args: &[Type],
    ) -> Result<(), String> {
        // Get impl blocks for this base type
        let impls = match self.generic_impls.get(base_name) {
            Some(impls) => impls.clone(),
            None => return Ok(()), // No impl blocks for this type
        };
        
        // Generate each method
        for imp in &impls {
            for method in &imp.methods {
                // Skip if already generated
                let mangled_name = crate::compiler::codegen::builtin_types::resolver::resolve_static_method_name(
                    base_name, &method.name, type_args
                );
                if self.module.get_function(&mangled_name).is_some() {
                    continue;
                }
                
                // Generate this method
                match self.monomorphize_method(base_name, &method.name, type_args) {
                    Ok(_) => {}
                    Err(e) => {
                        // Log but continue - some methods may not be needed
                        log::debug!("Could not generate method {}.{}: {}", 
                            base_name, method.name, e);
                    }
                }
            }
        }
        
        Ok(())
    }

}
