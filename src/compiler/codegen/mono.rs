use super::CodeGenerator;
use crate::compiler::ast::{Type, Expr, mangle_wrap_args, mangle_base_name, mangle_has_args};
use crate::compiler::mangler::MANGLER;
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
        let args_str = format!("{:?}", generic_args);
        if struct_name == "Vec" && method_name == "pop" && args_str.contains("K") {
            panic!("Vec_pop_K_monomorphized!!");
        }

        if generic_args.is_empty() {
            let mangled = format!("tl_{}_{}", struct_name, method_name);
            if self.module.get_function(&mangled).is_some() {
                return Ok(mangled);
            }
        }

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

        // Check trait bounds before monomorphization
        if !self.check_method_trait_bounds(method, generic_args, &imp.generics) {
            return Err(format!("Method {}.{} skipped: trait bounds not satisfied for {:?}", struct_name, method_name, generic_args));
        }

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
        let substitutor = TypeSubstitutor::new(subst_map.clone());
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
        new_method.return_type = full_substitutor.substitute_type(&new_method.return_type);
        new_method.return_type = self.normalize_type(&new_method.return_type);
        new_method.body = new_method.body.iter().map(|s| {
            let ns = full_substitutor.substitute_stmt(s);
            if let crate::compiler::ast::StmtKind::Let { name, type_annotation, .. } = &s.inner {
                 if name == "result" {
                      eprintln!("[MONO] var={} orig_ta={:?}", name, type_annotation);
                      if let crate::compiler::ast::StmtKind::Let { type_annotation: n_ta, .. } = &ns.inner {
                           eprintln!("[MONO] var={} new_ta={:?} subst={:?}", name, n_ta, subst_map);
                      }
                 }
            }
            ns
        }).collect();
        
        // Transform StaticMethodCall to EnumInit for enum variant constructors
        self.transform_method_body_enum_inits(&mut new_method.body);

        // Transform StructInit names: resolve generic struct names to mangled concrete names
        // e.g., StructInit(Struct("Container", []), ...) -> StructInit(Struct("Container[i64]", []), ...)
        self.transform_method_body_struct_inits(&mut new_method.body, &full_substitutor);

        // Compile
        // Save current builder position
        let previous_block = self.builder.get_insert_block();

        self.compile_fn_proto(&new_method)?;
        
        // NEW: If extern, we are done (declaration only). Otherwise compile body.
        if !new_method.is_extern {
            self.pending_functions.push((new_method, Some(subst_map)));
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
        let args_str: Vec<String> = type_args.iter().map(|t| self.type_to_suffix(t)).collect();
        let mangled_name = mangle_wrap_args(func_name, &args_str);
             
        if self.module.get_function(&mangled_name).is_some() {
            return Ok(mangled_name);
        }
        
        // 6. Instantiate
        let substitutor = TypeSubstitutor::new(subst_map.clone());
        
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

        // Transform StructInit names in generic function body
        self.transform_method_body_struct_inits(&mut new_func.body, &substitutor);

        // 7. Compile
        // Need to register proto first (for recursion support etc)
        self.compile_fn_proto(&new_func)?;
        self.pending_functions.push((new_func, Some(subst_map)));

        Ok(mangled_name)
    }

    /// On-demand monomorphization of a generic enum.
    pub fn monomorphize_enum(
        &mut self,
        enum_name: &str,
        generic_args: &[Type],
    ) -> Result<String, String> {
        // Early return: if enum_name is already a mangled name that exists in enum_types, done.
        if self.enum_types.contains_key(enum_name) {
            return Ok(enum_name.to_string());
        }
        // Also check if we can mangle from base + args and find it
        if !generic_args.is_empty() {
            let base = mangle_base_name(enum_name);
            let candidate = self.mangle_type_name(base, generic_args);
            if self.enum_types.contains_key(&candidate) {
                return Ok(candidate);
            }
        }

        // 1. Check if the enum exists in generic registry
        // First, try direct lookup
        let enum_def = if let Some(def) = self.enum_defs.get(enum_name) {
            def.clone()
        } else if enum_name.contains('[') && !enum_name.contains('<') {
            // Try extracting base name from mangled name (e.g., "Option[i64]" -> "Option")
            let base_name = mangle_base_name(enum_name);
            if let Some(def) = self.enum_defs.get(base_name) {
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

        // 3. Mangle - use base name to avoid double-mangling (e.g. Entry[i64][i64] -> Entry[i64][i64][i64][i64])
        let base = mangle_base_name(enum_name);
        let mangled_name = self.mangle_type_name(base, generic_args);
        
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

        // Substitute variants and convert generic types to UnifiedType
        for variant in &mut new_def.variants {
             match &mut variant.kind {
                 crate::compiler::ast::VariantKind::Unit => {},
                 crate::compiler::ast::VariantKind::Tuple(types) => {
                     for t in types.iter_mut() {
                         *t = substitutor.substitute_type(t);
                         *t = self.to_unified_type_if_generic(t.clone());
                     }
                 }
                 crate::compiler::ast::VariantKind::Struct(fields) => {
                     for (_, t) in fields.iter_mut() {
                         *t = substitutor.substitute_type(t);
                         *t = self.to_unified_type_if_generic(t.clone());
                     }
                 }
                 crate::compiler::ast::VariantKind::Array(ty, _size) => {
                     *ty = substitutor.substitute_type(ty);
                     *ty = self.to_unified_type_if_generic(ty.clone());
                 }

             }
        }


        // 6. Pre-monomorphize nested generic types in variant payloads
        // e.g., for Option<Pair<i64>>, ensure Pair_i64 is monomorphized before compiling the enum
        for variant in &new_def.variants {
            let types_to_check: Vec<&Type> = match &variant.kind {
                crate::compiler::ast::VariantKind::Unit => vec![],
                crate::compiler::ast::VariantKind::Tuple(types) => types.iter().collect(),
                crate::compiler::ast::VariantKind::Struct(fields) => fields.iter().map(|(_, t)| t).collect(),
                crate::compiler::ast::VariantKind::Array(ty, size) => vec![ty; *size],
            };
            for ty in types_to_check {
                let _ = self.get_or_monomorphize_type(ty)?;
            }
        }

        // 7. Compile/Register
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
    /// Example: `Vec` + `[i64]` -> `Vec[i64]`
    pub fn mangle_type_name(&self, base_name: &str, type_args: &[Type]) -> String {
        if type_args.is_empty() {
            base_name.to_string()
        } else if base_name.contains('[') {
            // Already mangled: don't double-mangle
            base_name.to_string()
        } else {
            let args_str: Vec<String> = type_args.iter().map(|t| self.type_to_suffix(t)).collect();
            mangle_wrap_args(base_name, &args_str)
        }
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
                if args.is_empty() || mangle_has_args(name) {
                    // Already mangled or no args: use name directly
                    name.clone()
                } else {
                    self.mangle_type_name(name, args)
                }
            }
            Type::Enum(name, args) => {
                if args.is_empty() || mangle_has_args(name) {
                    name.clone()
                } else {
                    self.mangle_type_name(name, args)
                }
            }
            Type::SpecializedType { gen_type, .. } => {
                gen_type.mangled_name_or_name().unwrap_or("specialized").to_string()
            }

            Type::Tensor(inner, rank) => {
                let args = vec![self.type_to_suffix(inner), rank.to_string()];
                mangle_wrap_args("Tensor", &args)
            }
            Type::Tuple(types) => {
                let parts: Vec<String> = types.iter().map(|t| self.type_to_suffix(t)).collect();
                mangle_wrap_args("Tuple", &parts)
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
            Type::Undefined(id) => format!("undefined{}", MANGLER.wrap_single(&id.to_string())),
            Type::Ptr(inner) => format!("ptr{}", MANGLER.wrap_single(&self.type_to_suffix(inner))),
            Type::Array(inner, size) => {
                let args = vec![self.type_to_suffix(inner), size.to_string()];
                mangle_wrap_args("Array", &args)
            }
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
            type_args.iter()
                .map(|t| MANGLER.wrap_single(&self.type_to_suffix(t).to_lowercase()))
                .collect::<String>()
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
            Type::Void => Ok(self.context.i8_type().into()), // Prevent LLVM crashes on Void Assignments using a dummy 1-byte allocation
            
            Type::Tensor(_, _) | Type::TensorShaped(_, _) | Type::GradTensor(_, _) => {
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
            
            Type::Fn(_, _) => {
                // Fat pointer: {fn_ptr, env_ptr} struct
                let ptr_ty = self.context.ptr_type(AddressSpace::default());
                Ok(self.context.struct_type(&[ptr_ty.into(), ptr_ty.into()], false).into())
            }
            
            Type::TraitObject(_) => {
                // Fat pointer: {data_ptr, vtable_ptr} struct
                let ptr_ty = self.context.ptr_type(AddressSpace::default());
                Ok(self.context.struct_type(&[ptr_ty.into(), ptr_ty.into()], false).into())
            }
            
            Type::SpecializedType { gen_type, .. } => {
                // Monomorphized generic type - dispatch based on original kind
                if gen_type.is_enum_type() || gen_type.is_struct_type() {
                    Ok(self.context.ptr_type(AddressSpace::default()).into())
                } else {
                    // Fallback: try to resolve via mangled_name
                    Ok(self.context.ptr_type(AddressSpace::default()).into())
                }
            }
            
            _ => {
                // strict NO IMPLICIT FALLBACK rule: returning explicitly failed types to prevent undefined behavior
                Err(format!("get_llvm_type: compilation error, unhandled or unresolved type {:?}", ty))
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
        // Substitute field types and convert generic types to UnifiedType
        specialized_def.fields = struct_def.fields.iter().map(|(name, ty)| {
            let substituted = self.substitute_type(ty, &subst);
            let unified = self.to_unified_type_if_generic(substituted);
            (name.clone(), unified)
        }).collect();
        
        self.struct_defs.insert(mangled_name.clone(), specialized_def);
        
        Ok(opaque_struct)
    }
    
    /// Check if a list of type arguments contains unresolved generic parameters.
    /// Generic parameters appear as Struct("K", []) or Struct("V", []) etc.
    #[allow(dead_code)]
    fn contains_unresolved_generics(args: &[Type]) -> bool {
        args.iter().any(|arg| Self::is_unresolved_generic(arg))
    }
    
    /// Check if a single type is (or contains) an unresolved generic parameter.
    #[allow(dead_code)]
    fn is_unresolved_generic(ty: &Type) -> bool {
        match ty {
            Type::Struct(name, inner_args) => {
                // A generic parameter is typically a short uppercase name with no args
                // e.g., Struct("K", []), Struct("V", []), Struct("T", [])
                if inner_args.is_empty() && name.len() <= 2 && name.chars().all(|c| c.is_ascii_uppercase()) {
                    return true;
                }
                // Check nested args recursively
                inner_args.iter().any(|a| Self::is_unresolved_generic(a))
            }
            Type::Enum(_, inner_args) => inner_args.iter().any(|a| Self::is_unresolved_generic(a)),
            Type::SpecializedType { type_args, .. } => type_args.iter().any(|a| Self::is_unresolved_generic(a)),
            Type::Tuple(types) => types.iter().any(|a| Self::is_unresolved_generic(a)),
            Type::Undefined(_) => true,
            _ => false,
        }
    }

    /// Convert a Type to UnifiedType if it has generic arguments.
    /// Also corrects Struct→Enum misclassification using enum_defs.
    /// Non-generic types (e.g. I64, Struct("File", [])) are returned as-is.
    pub fn to_unified_type_if_generic(&self, ty: Type) -> Type {
        match &ty {
            Type::Struct(name, args) if !args.is_empty() => {
                let is_enum = self.enum_defs.contains_key(name)
                    || self.enum_defs.contains_key(mangle_base_name(name));
                // Avoid double-mangling: if name already contains brackets, use it directly
                let (base, mangled) = if mangle_has_args(name) {
                    (mangle_base_name(name).to_string(), name.clone())
                } else {
                    (name.clone(), self.mangle_type_name(name, args))
                };
                // Recursively convert inner type_args too
                let unified_args: Vec<Type> = args.iter()
                    .map(|a| self.to_unified_type_if_generic(a.clone()))
                    .collect();
                Type::SpecializedType {
                    gen_type: Box::new(if is_enum { Type::Enum(base, vec![]) } else { Type::Struct(base, vec![]) }),
                    type_args: unified_args,
                    type_map: vec![], // Type map is not always available here, but we pass unified_args
                    mangled_name: mangled,
                }
            }
            Type::Struct(name, args) if args.is_empty() => {
                // Correct Struct→Enum for non-generic types
                if self.enum_defs.contains_key(name) {
                    Type::Enum(name.clone(), vec![])
                } else {
                    ty
                }
            }
            Type::Enum(name, args) if !args.is_empty() => {
                // Avoid double-mangling
                let (base, mangled) = if mangle_has_args(name) {
                    (mangle_base_name(name).to_string(), name.clone())
                } else {
                    (name.clone(), self.mangle_type_name(name, args))
                };
                let unified_args: Vec<Type> = args.iter()
                    .map(|a| self.to_unified_type_if_generic(a.clone()))
                    .collect();
                Type::SpecializedType {
                    gen_type: Box::new(Type::Enum(base, vec![])),
                    type_args: unified_args,
                    type_map: vec![],
                    mangled_name: mangled,
                }
            }
            Type::Tuple(types) => {
                let unified_types: Vec<Type> = types.iter()
                    .map(|t| self.to_unified_type_if_generic(t.clone()))
                    .collect();
                Type::Tuple(unified_types)
            }
            _ => ty,
        }
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

            Type::Enum(name, args) => {
                // Check if this is a type parameter (e.g. Enum("K", []))
                if args.is_empty() {
                    if let Some(replacement) = subst.get(name) {
                        return replacement.clone();
                    }
                }
                let new_args: Vec<Type> = args.iter().map(|a| self.substitute_type(a, subst)).collect();
                Type::Enum(name.clone(), new_args)
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
            Type::SpecializedType { gen_type, type_args, type_map, mangled_name } => {
                let new_args: Vec<Type> = type_args.iter().map(|a| self.substitute_type(a, subst)).collect();
                Type::SpecializedType {
                    gen_type: gen_type.clone(),
                    type_args: new_args,
                    type_map: type_map.clone(),
                    mangled_name: mangled_name.clone(),
                }
            }
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
            Type::SpecializedType { gen_type, type_args, type_map: _, mangled_name } => {
                let normalized_args: Vec<Type> = type_args.iter().map(|a| self.normalize_type(a)).collect();
                if gen_type.is_enum_type() {
                    Type::Enum(mangled_name.clone(), normalized_args)
                } else {
                    Type::Struct(mangled_name.clone(), normalized_args)
                }
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
                            VariantKind::Struct(fields) => {
                                let field_pairs: Vec<(String, Expr)> = fields.iter()
                                    .zip(std::mem::take(args).into_iter())
                                    .map(|((name, _), expr)| (name.clone(), expr))
                                    .collect();
                                EnumVariantInit::Struct(field_pairs)
                            }
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

    // ========== AST Transformation for StructInit in monomorphized methods ==========

    fn transform_method_body_struct_inits(&self, stmts: &mut Vec<crate::compiler::ast::Stmt>, substitutor: &TypeSubstitutor) {
        for stmt in stmts.iter_mut() {
            self.transform_stmt_struct_inits(stmt, substitutor);
        }
    }

    fn transform_stmt_struct_inits(&self, stmt: &mut crate::compiler::ast::Stmt, substitutor: &TypeSubstitutor) {
        use crate::compiler::ast::StmtKind;
        match &mut stmt.inner {
            StmtKind::Let { value, .. } => self.transform_expr_struct_inits(value, substitutor),
            StmtKind::Expr(e) => self.transform_expr_struct_inits(e, substitutor),
            StmtKind::Return(Some(e)) => self.transform_expr_struct_inits(e, substitutor),
            StmtKind::While { cond, body } => {
                self.transform_expr_struct_inits(cond, substitutor);
                self.transform_method_body_struct_inits(body, substitutor);
            }
            StmtKind::For { iterator, body, .. } => {
                self.transform_expr_struct_inits(iterator, substitutor);
                self.transform_method_body_struct_inits(body, substitutor);
            }
            StmtKind::Loop { body } => {
                self.transform_method_body_struct_inits(body, substitutor);
            }
            StmtKind::Assign { value, .. } => {
                self.transform_expr_struct_inits(value, substitutor);
            }
            _ => {}
        }
    }

    fn transform_expr_struct_inits(&self, expr: &mut crate::compiler::ast::Expr, substitutor: &TypeSubstitutor) {
        use crate::compiler::ast::ExprKind;

        match &mut expr.inner {
            ExprKind::BinOp(l, _, r) => {
                self.transform_expr_struct_inits(l, substitutor);
                self.transform_expr_struct_inits(r, substitutor);
            }
            ExprKind::UnOp(_, e) => self.transform_expr_struct_inits(e, substitutor),
            ExprKind::MethodCall(obj, _, args) => {
                self.transform_expr_struct_inits(obj, substitutor);
                for arg in args.iter_mut() {
                    self.transform_expr_struct_inits(arg, substitutor);
                }
            }
            ExprKind::FnCall(_, args) => {
                for arg in args.iter_mut() {
                    self.transform_expr_struct_inits(arg, substitutor);
                }
            }
            ExprKind::StaticMethodCall(ty, _, args) => {
                // Recurse into arguments
                for arg in args.iter_mut() {
                    self.transform_expr_struct_inits(arg, substitutor);
                }
                // Mangle the receiver type (e.g., Vec<T> -> Vec[i64])
                self.transform_generic_type(ty, substitutor);
            }
            ExprKind::EnumInit { payload, .. } => {
                match payload {
                    crate::compiler::ast::EnumVariantInit::Tuple(args) => {
                        for arg in args.iter_mut() {
                            self.transform_expr_struct_inits(arg, substitutor);
                        }
                    }
                    crate::compiler::ast::EnumVariantInit::Struct(fields) => {
                        for (_, e) in fields.iter_mut() {
                            self.transform_expr_struct_inits(e, substitutor);
                        }
                    }
                    crate::compiler::ast::EnumVariantInit::Unit => {}
                }
            }
            ExprKind::Match { expr: match_expr, arms } => {
                self.transform_expr_struct_inits(match_expr, substitutor);
                for (_, arm_body) in arms.iter_mut() {
                    self.transform_expr_struct_inits(arm_body, substitutor);
                }
            }
            ExprKind::Try(inner) => {
                self.transform_expr_struct_inits(inner, substitutor);
            }
            ExprKind::IndexAccess(obj, indices) => {
                self.transform_expr_struct_inits(obj, substitutor);
                for idx in indices.iter_mut() {
                    self.transform_expr_struct_inits(idx, substitutor);
                }
            }
            ExprKind::Tuple(elems) => {
                for e in elems.iter_mut() {
                    self.transform_expr_struct_inits(e, substitutor);
                }
            }
            ExprKind::TupleAccess(obj, _) => {
                self.transform_expr_struct_inits(obj, substitutor);
            }
            ExprKind::Closure { body, .. } => {
                self.transform_method_body_struct_inits(body, substitutor);
            }
            ExprKind::FieldAccess(obj, _) => self.transform_expr_struct_inits(obj, substitutor),
            ExprKind::Block(stmts) => self.transform_method_body_struct_inits(stmts, substitutor),
            ExprKind::IfExpr(cond, then_block, else_block) => {
                self.transform_expr_struct_inits(cond, substitutor);
                self.transform_method_body_struct_inits(then_block, substitutor);
                if let Some(else_stmts) = else_block {
                    self.transform_method_body_struct_inits(else_stmts, substitutor);
                }
            }
            ExprKind::StructInit(ty, fields) => {
                // Recurse into field expressions first
                for (_, e) in fields.iter_mut() {
                    self.transform_expr_struct_inits(e, substitutor);
                }
                // Resolve generic struct/enum name to concrete mangled name
                self.transform_generic_type(ty, substitutor);
            }
            _ => {}
        }
    }

    /// Resolve a generic type name to its concrete mangled name using the substitutor.
    /// Works for Struct, Enum, Path, and SpecializedType variants.
    fn transform_generic_type(&self, ty: &mut Type, substitutor: &TypeSubstitutor) {
        // Handle Path types: convert Path(["Vec"], []) to Struct("Vec", []) first
        if let Type::Path(segments, generics) = ty {
            if segments.len() == 1 {
                let name = &segments[0];
                // Check struct_defs, enum_defs, and generic_impls (for stdlib types like Vec, HashMap)
                if self.struct_defs.contains_key(name) || self.generic_impls.contains_key(name) {
                    *ty = Type::Struct(name.clone(), generics.clone());
                } else if self.enum_defs.contains_key(name) {
                    *ty = Type::Enum(name.clone(), generics.clone());
                }
            }
        }
        let name = match ty {
            Type::Struct(n, _) | Type::Enum(n, _) => n.clone(),
            _ => return,
        };

        // Get generic parameter names from struct_defs, enum_defs, or generic_impls
        let generic_params = self.get_type_generic_params(&name);
        if generic_params.is_empty() {
            return;
        }

        // Substitute generic params to get concrete types
        let concrete_args: Vec<Type> = generic_params.iter()
            .map(|g| {
                let param_ty = Type::Path(vec![g.clone()], vec![]);
                let substituted = substitutor.substitute_type(&param_ty);
                self.normalize_type(&substituted)
            })
            .collect();

        // Check if any param remained unresolved (same as input)
        let all_resolved = concrete_args.iter().zip(generic_params.iter()).all(|(resolved, param)| {
            !matches!(resolved, Type::Path(segs, _) if segs.len() == 1 && segs[0] == *param)
        });

        if !all_resolved {
            return; // Some params couldn't be resolved, don't mangle
        }

        let mangled = self.mangle_type_name(&name, &concrete_args);
        match ty {
            Type::Struct(n, g) => { *n = mangled; g.clear(); }
            Type::Enum(n, g) => { *n = mangled; g.clear(); }
            _ => {}
        }
    }

    /// Get the generic parameter names for a type from struct_defs, enum_defs, or generic_impls.
    fn get_type_generic_params(&self, name: &str) -> Vec<String> {
        // Check struct_defs first
        if let Some(def) = self.struct_defs.get(name) {
            if !def.generics.is_empty() {
                return def.generics.clone();
            }
        }
        // Check enum_defs
        if let Some(def) = self.enum_defs.get(name) {
            if !def.generics.is_empty() {
                return def.generics.clone();
            }
        }
        // Check generic_impls (for stdlib types like Vec, HashMap, Option)
        if let Some(impls) = self.generic_impls.get(name) {
            for imp in impls {
                if !imp.generics.is_empty() {
                    return imp.generics.clone();
                }
            }
        }
        vec![]
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
            if base_name == "Vec" {
                let _method_names: Vec<String> = imp.methods.iter().map(|m| m.name.clone()).collect();
            }
            for method in &imp.methods {
                // Check trait bounds before attempting monomorphization
                if !self.check_method_trait_bounds(method, type_args, &imp.generics) {
                    log::debug!("Skipping method {}.{}: trait bounds not satisfied for {:?}",
                        base_name, method.name, type_args);
                    continue;
                }

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

    /// Check if a method's trait bounds are satisfied for the given type arguments.
    /// Returns true if all bounds are satisfied (or if there are no bounds).
    fn check_method_trait_bounds(
        &self,
        method: &crate::compiler::ast::FunctionDef,
        type_args: &[Type],
        impl_generics: &[String],
    ) -> bool {
        // DEBUG
        if method.name == "index_of" || method.name == "contains" {
        }
        
        // If no bounds, always satisfied
        if method.generic_bounds.is_empty() {
            if let Some(ref wc) = method.where_clause {
                // Check where clause predicates
                for pred in &wc.predicates {
                    // Map the type param to a concrete type
                    let concrete_type = self.resolve_generic_param(&pred.type_param, type_args, impl_generics);
                    let type_name = self.concrete_type_to_trait_key(&concrete_type);
                    
                    for bound in &pred.bounds {
                        let trait_impls = self.trait_registry.get(&type_name);
                        let mut satisfied = trait_impls.map_or(false, |traits| traits.contains(&bound.trait_name));
                        
                        // Primitive trait bound whitelist fallback
                        if !satisfied {
                            satisfied = match type_name.as_str() {
                                "i64" | "f64" | "i32" | "f32" | "u8" | "char" | "bool" | "String" => {
                                    match bound.trait_name.as_str() {
                                        "PartialEq" | "Clone" => true,
                                        "PartialOrd" => type_name != "String" && type_name != "bool",
                                        _ => false,
                                    }
                                }
                                _ => false,
                            };
                        }
                        
                        if !satisfied {
                            return false;
                        }
                    }
                }
            }
            return true;
        }

        // Check generic_bounds: Vec<(String, Vec<TraitBound>)>
        for (param_name, bounds) in &method.generic_bounds {
            let concrete_type = self.resolve_generic_param(param_name, type_args, impl_generics);
            let type_name = self.concrete_type_to_trait_key(&concrete_type);
            
            for bound in bounds {
                let trait_impls = self.trait_registry.get(&type_name);
                let mut satisfied = trait_impls.map_or(false, |traits| traits.contains(&bound.trait_name));
                
                // Primitive trait bound whitelist fallback
                if !satisfied {
                    satisfied = match type_name.as_str() {
                        "i64" | "f64" | "i32" | "f32" | "u8" | "char" | "bool" | "String" => {
                            match bound.trait_name.as_str() {
                                "PartialEq" | "Clone" => true,
                                "PartialOrd" => type_name != "String" && type_name != "bool",
                                _ => false,
                            }
                        }
                        _ => false,
                    };
                }
                
                if !satisfied {
                    return false;
                }
            }
        }
        true
    }

    /// Resolve a generic parameter name to its concrete type.
    pub(crate) fn resolve_generic_param(&self, param: &str, type_args: &[Type], impl_generics: &[String]) -> Type {
        for (i, generic_name) in impl_generics.iter().enumerate() {
            if generic_name == param {
                if let Some(ty) = type_args.get(i) {
                    return ty.clone();
                }
            }
        }
        // Not found — return as unresolved struct
        Type::Struct(param.to_string(), vec![])
    }

    /// Convert a concrete type to a string key for trait_registry lookup.
    pub(crate) fn concrete_type_to_trait_key(&self, ty: &Type) -> String {
        match ty {
            Type::I64 => "i64".to_string(),
            Type::I32 => "i32".to_string(),
            Type::F32 => "f32".to_string(),
            Type::F64 => "f64".to_string(),
            Type::Bool => "bool".to_string(),
            Type::U8 => "u8".to_string(),
            Type::Usize => "usize".to_string(),
            Type::String(_) => "String".to_string(),
            Type::Struct(name, _) => name.clone(),
            Type::Enum(name, _) => name.clone(),
            Type::SpecializedType { gen_type, .. } => gen_type.get_base_name(),
            _ => format!("{:?}", ty),
        }
    }

}
