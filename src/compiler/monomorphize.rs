use crate::compiler::ast::*;
use crate::compiler::error::{TlError, MonomorphizationErrorKind};
use std::collections::{HashMap, VecDeque};


pub struct Monomorphizer {
    // Map from (OriginalName, ConcreteTypes) -> MangledName
    struct_instances: HashMap<(String, Vec<Type>), String>,
    function_instances: HashMap<(String, Vec<Type>), String>,
    
    // Original definitions to instantiate from
    generic_structs: HashMap<String, StructDef>,
    generic_functions: HashMap<String, FunctionDef>,
    generic_enums: HashMap<String, EnumDef>,
    generic_impls: Vec<ImplBlock>,
    
    // Map from mangled name to (original_name, args)
    reverse_struct_instances: HashMap<String, (String, Vec<Type>)>,
    
    // Queue of pending instantiations: (OriginalName, ConcreteTypes, IsStruct)
    pending_queue: VecDeque<(String, Vec<Type>, bool)>,
    
    // Scopes for variable type tracking during rewrite
    scopes: Vec<HashMap<String, Type>>,
    current_return_type: Option<Type>,
    
    // Resulting concrete definitions
    concrete_structs: Vec<StructDef>,
    concrete_functions: Vec<FunctionDef>,
    concrete_enums: Vec<EnumDef>,
    concrete_impls: Vec<ImplBlock>,
    
    // Track visited functions to avoid cycles or re-scanning? 
    // Actually we only scan reachable functions.
    // If we reach a generic function, we instantiate it and queue it.
    // The instantiated function will be scanned later when popped from queue.
    // Non-generic functions are scanned once at start.
}

impl Monomorphizer {
    pub fn new() -> Self {
        Self {
            struct_instances: HashMap::new(),
            reverse_struct_instances: HashMap::new(),
            function_instances: HashMap::new(),
            generic_structs: HashMap::new(),
            generic_functions: HashMap::new(),
            generic_enums: HashMap::new(),
            generic_impls: Vec::new(),
            pending_queue: VecDeque::new(),
            scopes: Vec::new(),
            current_return_type: None,
            concrete_structs: Vec::new(),
            concrete_functions: Vec::new(),
            concrete_enums: Vec::new(),
            concrete_impls: Vec::new(),
        }
    }

    pub fn run(&mut self, module: &mut Module) -> Result<(), TlError> {
        log::debug!("Starting Monomorphization");
        // 1. Collect generic definitions
        self.collect_generics(module);
        
        let initial_structs = module.structs.len();
        log::debug!("Generic structs collected: {}", self.generic_structs.len());
        
        // 2. Scan for initial usages in main and non-generic functions
        log::debug!("Scanning module...");
        self.scan_module(module);

        
        // 3. Process queue until empty
        let mut steps = 0;
        while let Some((name, types, is_struct)) = self.pending_queue.pop_front() {
            log::trace!("Processing pending: {} {:?}", name, types);

            steps += 1;
            if steps > 10000 {
                return Err(TlError::Monomorphization {
                    kind: MonomorphizationErrorKind::RecursionLimitReached(format!("Processing {} with args {:?}", name, types)),
                    span: None,
                });
            }
            if is_struct {
                self.instantiate_struct(&name, &types)?;
            } else {
                self.instantiate_function(&name, &types)?;
            }
        }
        
        // 4. Update module with concrete definitions
        // 4. Update module with concrete definitions
        log::debug!("Adding {} concrete structs", self.concrete_structs.len());
        module.structs.extend(self.concrete_structs.drain(..));
        module.functions.extend(self.concrete_functions.drain(..));
        module.enums.extend(self.concrete_enums.drain(..));
        module.impls.extend(self.concrete_impls.drain(..));
        
        // 5. Remove original generic definitions to avoid codegen errors
        module.structs.retain(|s| s.generics.is_empty());
        module.functions.retain(|f| f.generics.is_empty()); 
        module.enums.retain(|e| e.generics.is_empty());
        
        log::info!("Monomorphization done. Structs: {} -> {}", initial_structs, module.structs.len());
        Ok(())
    }



    fn collect_generics(&mut self, module: &Module) {
        for s in &module.structs {
            if !s.generics.is_empty() {
                self.generic_structs.insert(s.name.clone(), s.clone());
            }
        }
        for f in &module.functions {
            if !f.generics.is_empty() {
                self.generic_functions.insert(f.name.clone(), f.clone());
            }
        }
        for e in &module.enums {
            if !e.generics.is_empty() {

                self.generic_enums.insert(e.name.clone(), e.clone());
            }
        }
    }

    fn scan_module(&mut self, module: &mut Module) {
        // 1. Process impls logic FIRST so they are available for resolution
        for mut impl_block in module.impls.drain(..) {
            let is_generic_target = if let Type::Struct(name, _) | Type::Enum(name, _) = &impl_block.target_type {
                self.generic_structs.contains_key(name) || self.generic_enums.contains_key(name)
            } else {
                false
            };

            if !impl_block.generics.is_empty() || is_generic_target {
                self.generic_impls.push(impl_block);
            } else {
                // Rewrite methods in non-generic impl
                // Check if this is a specialized impl (e.g., HashMap_i64_i64) that needs substitution
                let subst = if let Type::Struct(name, _) | Type::Enum(name, _) = &impl_block.target_type {
                    self.build_subst_from_mangled_name(name)
                } else {
                    HashMap::new()
                };
                
                for method in &mut impl_block.methods {
                    for stmt in &mut method.body {
                        self.rewrite_stmt(&mut stmt.inner, &subst, None);
                    }
                }
                self.concrete_impls.push(impl_block);
            }
        }


        // 2. Scan all non-generic function bodies (roots) e.g. main
        let empty_subst = HashMap::new();
        for f in &mut module.functions {
            if f.generics.is_empty() {
                 self.rewrite_function_body(f, &empty_subst);
            }
        }
        
        // 3. Scan global statements
        for stmt in &mut module.tensor_decls {
            self.rewrite_stmt(&mut stmt.inner, &empty_subst, None);
        }
    }


    
    fn resolve_type(&mut self, ty: &Type) -> Type {
        match ty {
            Type::Struct(name, args) | Type::Enum(name, args) => {
                 // Track if original type is Enum
                 let is_enum = matches!(ty, Type::Enum(_, _));
                 
                 if !args.is_empty() {
                     // Check if this is a generic struct instantiation
                     if self.generic_structs.contains_key(name) {
                         let concrete_args: Vec<Type> = args.iter().map(|a| self.resolve_type(a)).collect();
                         let mangled = self.request_struct_instantiation(name, concrete_args);
                         // Preserve original type kind
                         return Type::Struct(mangled, vec![]);
                     }
                     // Check enum
                     if self.generic_enums.contains_key(name) {
                          let concrete_args: Vec<Type> = args.iter().map(|a| self.resolve_type(a)).collect();
                          let mangled = self.request_enum_instantiation(name, concrete_args);
                          return Type::Enum(mangled, vec![]);
                     }
                 }
                 // Recurse for args even if not generic struct (e.g. unknown type?)
                 let new_args: Vec<Type> = args.iter().map(|a| self.resolve_type(a)).collect();
                 
                 // Preserve original type kind OR check base name for mangled names
                 if is_enum {
                     Type::Enum(name.clone(), new_args)
                 } else {
                     // Check if base name (before _) is a known enum (e.g., Entry_i64_i64 -> Entry)
                     let base_name = name.split('_').next().unwrap_or(name);
                     if self.generic_enums.contains_key(base_name) {
                         Type::Enum(name.clone(), new_args)
                     } else {
                         Type::Struct(name.clone(), new_args)
                     }
                 }
                 }

            // Recursive resolution
            Type::Tensor(inner, r) => Type::Tensor(Box::new(self.resolve_type(inner)), *r),

            Type::Tuple(types) => Type::Tuple(types.iter().map(|t| self.resolve_type(t)).collect()),
            Type::TensorShaped(inner, dims) => Type::TensorShaped(Box::new(self.resolve_type(inner)), dims.clone()),
            
            // Fix for builtins using Path
            Type::Path(segments, args) => {
                let name = segments.join("::");
                // Resolve args first (they may be Paths like K, V)
                let resolved_args: Vec<Type> = args.iter().map(|a| self.resolve_type(a)).collect();
                // Check if this is a generic enum first (e.g., Entry<K, V>)
                if self.generic_enums.contains_key(&name) {
                    let new_ty = Type::Enum(name, resolved_args);
                    return self.resolve_type(&new_ty);
                }
                // Otherwise treat as struct
                let new_ty = Type::Struct(name, resolved_args);
                self.resolve_type(&new_ty)
            }
            
            Type::Ptr(inner) => Type::Ptr(Box::new(self.resolve_type(inner))),


            _ => ty.clone()
        }
    }

    fn rewrite_stmt(&mut self, stmt: &mut StmtKind, subst: &HashMap<String, Type>, expected_type: Option<&Type>) {
        match stmt {
            StmtKind::Let { name, type_annotation, value, .. } => {
                let context_ty = if let Some(ty) = type_annotation {
                    *ty = self.substitute_type(ty, subst);
                    *ty = self.resolve_type(ty);
                    Some(ty.clone())
                } else {
                    None
                };
                self.rewrite_expr(&mut value.inner, subst, context_ty.as_ref());
                
                // If type inferred or annotated, add to scope
                let var_type = context_ty.or_else(|| self.infer_expr_type(&value.inner));
                if let Some(ty) = var_type {
                    if type_annotation.is_none() {
                        *type_annotation = Some(ty.clone());
                    }
                    if let Some(scope) = self.scopes.last_mut() {
                        scope.insert(name.clone(), ty);
                    }
                }
            }
            StmtKind::Expr(e) => {
                self.rewrite_expr(&mut e.inner, subst, expected_type);
            }
            StmtKind::Return(Some(e)) => {
                // Use current_return_type as expected type
                let ctx = self.current_return_type.clone();
                self.rewrite_expr(&mut e.inner, subst, ctx.as_ref());
            }
            StmtKind::Assign { lhs, value, .. } => {
                self.rewrite_expr(&mut value.inner, subst, None);
                // Rewrite index expressions in LValue (recursive structure)
                fn rewrite_lvalue_indices(mono: &mut Monomorphizer, lv: &mut crate::compiler::ast::LValue, subst: &HashMap<String, Type>) {
                    match lv {
                        crate::compiler::ast::LValue::Variable(_) => {},
                        crate::compiler::ast::LValue::FieldAccess(inner, _) => {
                            rewrite_lvalue_indices(mono, inner, subst);
                        },
                        crate::compiler::ast::LValue::IndexAccess(base, indices) => {
                            rewrite_lvalue_indices(mono, base, subst);
                            for idx in indices {
                                mono.rewrite_expr(&mut idx.inner, subst, None);
                            }
                        },
                    }
                }
                rewrite_lvalue_indices(self, lhs, subst);
            }
            StmtKind::For { iterator, body, .. } => {
                self.rewrite_expr(&mut iterator.inner, subst, None);
                for s in body {
                    self.rewrite_stmt(&mut s.inner, subst, None);
                }
            }
            StmtKind::While { cond, body } => {
                self.rewrite_expr(&mut cond.inner, subst, None);
                for s in body {
                    self.rewrite_stmt(&mut s.inner, subst, None);
                }
            }
             StmtKind::Loop { body } => {
                 for s in body {
                    self.rewrite_stmt(&mut s.inner, subst, None);
                }
             }
            StmtKind::TensorDecl { type_annotation, init, .. } => {
                 *type_annotation = self.substitute_type(type_annotation, subst);
                 *type_annotation = self.resolve_type(type_annotation);
                 let ctx = Some(type_annotation.clone());
                 if let Some(e) = init {
                     self.rewrite_expr(&mut e.inner, subst, ctx.as_ref());
                 }
            }
             _ => {}
        }
    }

     fn rewrite_expr(&mut self, expr: &mut ExprKind, subst: &HashMap<String, Type>, expected_type: Option<&Type>) {
         match expr {
             ExprKind::StructInit(ty, fields) => {
                 // Fix for builtins: resolve Path -> Struct if semantics didn't run
                 if let Type::Path(segments, args) = ty {
                     let name = segments.join("::");
                     *ty = Type::Struct(name, args.clone());
                 }

                let (name, explicit_generics) = match ty {
                    Type::Struct(n, g) => (n, g),
                    Type::Enum(n, g) => (n, g),
                    _ => panic!("StructInit must have Struct or Enum type in Monomorphizer, found {:?}", ty),
                };


                 // 0. Resolve explicit generics
                 let mut resolved_generics = Vec::new();
                 if !explicit_generics.is_empty() {
                     for g in explicit_generics {
                         *g = self.substitute_type(g, subst);
                         *g = self.resolve_type(g);
                         resolved_generics.push(g.clone());
                     }
                 }

                // Try to resolve generic struct instantiation             ExprKind::StructInit(name, fields) => {
                 // 1. Explicit Generics
                 if !resolved_generics.is_empty() {
                     if self.generic_structs.contains_key(name) {
                         let concrete_name = self.request_struct_instantiation(name, resolved_generics);
                         *name = concrete_name;
                     }
                 }
                 // 2. Try context inference if not resolved
                  else if let Some(Type::Struct(expected_name, _)) = expected_type {
                     if expected_name.starts_with(name.as_str()) {
                         *name = expected_name.clone();
                     } else {
                     }
                 } else {
                 }
                 
                 // Look up the concrete struct definition (renamed) to propagate field types
                 let mut field_types_map = HashMap::new();
                 
                 // 1. Try concrete (if already compiled/available)
                 if let Some(def) = self.concrete_structs.iter().find(|s| s.name == *name) {
                     for (fname, fty) in &def.fields {
                         field_types_map.insert(fname.clone(), fty.clone());
                     }
                 } 
                 // 2. Try generic (if not renamed? but name is likely mangled here if we renamed it)
                 // 2. Try generic Struct
                 else if self.generic_structs.contains_key(name) {
                     // Pre-rewrite fields to resolve generic arguments in nested structures
                     for (_fname, val) in fields.iter_mut() {
                         self.rewrite_expr(&mut val.inner, subst, None);
                     }

                     if let Some(def) = self.generic_structs.get(name) {
                         // Try to infer generics from fields!
                         let mut inferred_map = HashMap::new();
                     let mut all_inferred = true;
                     
                     for (fname, val) in fields.iter() {
                         if let Some(field_ty) = def.fields.iter().find(|(f, _)| f == fname).map(|(_, t)| t) {
                             // Infer type of value expression
                             if let Some(val_ty) = self.infer_expr_type(&val.inner) {
                                 // Unify field_ty (e.g. T) with val_ty (e.g. I64)
                                 self.unify_types(field_ty, &val_ty, &mut inferred_map);
                             }
                         }
                     }
                     
                     // Construct args
                     let mut type_args = Vec::new();
                     for param in &def.generics {
                         if let Some(ty) = inferred_map.get(param) {
                             type_args.push(ty.clone());
                         } else {
                             all_inferred = false;
                         }
                     }
                     
                     if all_inferred && !type_args.is_empty() {
                         let concrete_name = self.request_struct_instantiation(name, type_args.clone());
                         *name = concrete_name;
                         
                         if let Some(c_def) = self.concrete_structs.iter().find(|s| s.name == *name) {
                             for (fname, fty) in &c_def.fields {
                                 field_types_map.insert(fname.clone(), fty.clone());
                             }
                         }
                     } else {
                         for (fname, fty) in &def.fields {
                             field_types_map.insert(fname.clone(), fty.clone());
                         }
                     }
                 }
                 }
                 // 3. Try reverse lookup (Mangled -> Original + Args)
                 else if let Some((orig_name, args)) = self.reverse_struct_instances.get(name) {
                     if let Some(def) = self.generic_structs.get(orig_name) {
                         // Create substitution map
                         let mut inner_subst = HashMap::new();
                         for (i, param) in def.generics.iter().enumerate() {
                             if i < args.len() {
                                 inner_subst.insert(param.clone(), args[i].clone());
                             }
                         }
                         for (fname, fty) in &def.fields {
                             let resolved_fty = self.substitute_type(fty, &inner_subst);
                             field_types_map.insert(fname.clone(), resolved_fty);
                         }
                     }
                 }
                 // 4. Try generic Enum (e.g. MyEnum::Variant) - LEGACY? 
                 // Semantics should receive StructInit and transform to EnumInit.
                 // But if Monomorphizer runs after Semantics, this case might be unreachable for valid Enums?
                 // Or maybe for un-resolved?
                 // Let's comment this out or just fix it?
                 // The compilation error forces me to fix it.
                 // But since semantics converts to EnumInit, this branch should strictly not trigger for Enums unless semantics failed or logic differs.
                 // Given strict compilation check, I'll update it to be safe.
                 else {
                     // ... Update logic to use VariantKind ...
                     let (type_name_str, variant_name_str) = {
                        let parts: Vec<&str> = name.split("::").collect();
                        let t = if parts.len() > 1 { parts[0].to_string() } else { name.clone() };
                        let v = if parts.len() > 1 { parts[1].to_string() } else { "".to_string() };
                        (t, v)
                     };

                     if let Some(def) = self.generic_enums.get(&type_name_str) {
                          if let Some(v_def) = def.variants.iter().find(|v| v.name == variant_name_str) {
                               // Assuming StructInit syntax implies Struct Variant
                               if let crate::compiler::ast::VariantKind::Struct(def_fields) = &v_def.kind {
                                    let mut inferred_map = HashMap::new();
                                    let mut all_inferred = true;
                                    
                                    for (fname, val) in fields.iter() {
                                        if let Some(field_ty) = def_fields.iter().find(|(f, _)| f == fname).map(|(_, t)| t) {
                                            if let Some(val_ty) = self.infer_expr_type(&val.inner) {
                                                self.unify_types(field_ty, &val_ty, &mut inferred_map);
                                            }
                                        }
                                    }
                                    // ... existing instantiation logic ...
                                     let mut type_args = Vec::new();
                                     for param in &def.generics {
                                         if let Some(ty) = inferred_map.get(param) {
                                             type_args.push(ty.clone());
                                         } else {
                                             all_inferred = false;
                                         }
                                     }
                                     
                                     if all_inferred && !type_args.is_empty() {
                                          let concrete_enum_name = self.request_struct_instantiation(&type_name_str, type_args.clone());
                                          *name = format!("{}::{}", concrete_enum_name, variant_name_str);
                                          
                                          if let Some(c_def) = self.concrete_enums.iter().find(|e| e.name == concrete_enum_name) {
                                               if let Some(c_v) = c_def.variants.iter().find(|v| v.name == variant_name_str) {
                                                    if let crate::compiler::ast::VariantKind::Struct(c_fields) = &c_v.kind {
                                                         for (fname, fty) in c_fields {
                                                             field_types_map.insert(fname.clone(), fty.clone());
                                                         }
                                                    }
                                               }
                                          }
                                     } else {
                                          for (fname, fty) in def_fields {
                                              field_types_map.insert(fname.clone(), fty.clone());
                                          }
                                     }
                               }
                          }
                     }
                 }

                 for (fname, val) in fields {
                     let ctx_ty = field_types_map.get(fname);
                     self.rewrite_expr(&mut val.inner, subst, ctx_ty);
                 }
                 return;
             }
              ExprKind::EnumInit { enum_name, variant_name, generics, payload } => {
                   // Process EnumInit for monomorphization

                   
                   // Check if all generics are unresolved (empty or all Undefined)
                   let needs_inference = generics.is_empty() || 
                       generics.iter().all(|t| matches!(t, Type::Undefined(_)));
                   
                   if needs_inference {
                        // Handle both Type::Enum and Type::Struct (mangled enums may come as Struct)
                        if let Some(Type::Enum(expected_name, expected_args)) | Some(Type::Struct(expected_name, expected_args)) = expected_type {
                            if expected_name == enum_name && !expected_args.is_empty() {
                                // Case 1: Expected type matches our enum and has generics
                                *generics = expected_args.clone();

                            } else if expected_name.starts_with(&format!("{}_", enum_name)) && expected_args.is_empty() {
                                // Case 2: Expected type is already mangled (e.g. Option_i64 or Entry_i64_i64)
                                // Use the mangled name directly
                                *enum_name = expected_name.clone();
                                generics.clear();

                            }
                        }
                    }
                   
                   // Substitute and resolve any generic type params in generics
                   for ty in generics.iter_mut() {
                       *ty = self.substitute_type(ty, subst);
                       *ty = self.resolve_type(ty);
                   }
                   
                   // Filter out any remaining Undefined generics after resolution
                   generics.retain(|t| !matches!(t, Type::Undefined(_)));
                   
                   // Request enum instantiation if we have resolved generics
                   if !generics.is_empty() {
                       let concrete_name = self.request_enum_instantiation(enum_name, generics.clone());
                       *enum_name = concrete_name;
                       // Clear generics since name is now mangled
                       generics.clear();
                   }
                   
                   // Process payload
                   match payload {
                       crate::compiler::ast::EnumVariantInit::Unit => {},
                       crate::compiler::ast::EnumVariantInit::Tuple(exprs) => {
                           for e in exprs { 
                               self.rewrite_expr(&mut e.inner, subst, None); 
                           }
                       },
                       crate::compiler::ast::EnumVariantInit::Struct(fields) => {
                           for (_, e) in fields { 
                               self.rewrite_expr(&mut e.inner, subst, None); 
                           }
                       }
                   }
          }
          ExprKind::FnCall(name, args) => {
                  // Rewrite args first
                  for arg in args.iter_mut() {
                      self.rewrite_expr(&mut arg.inner, subst, None);
                  }
                  
                  // Check if this is a generic function call
                  if let Some(def) = self.generic_functions.get(name).cloned() {
                      // Infer type arguments
                      let mut type_args = Vec::new();
                      let mut inference_map: HashMap<String, Type> = HashMap::new();
                      
                      // Match arguments to parameters to infer types
                      for (i, (arg_val, _)) in args.iter().zip(&def.args).enumerate() {
                          let param_ty = &def.args[i].1;
                          let arg_expr_ty = self.infer_expr_type(&arg_val.inner);
                          
                          if let Some(concrete_ty) = arg_expr_ty {
                              // unify param_ty (e.g. T) with concrete_ty (e.g. I64)
                              self.unify_types(param_ty, &concrete_ty, &mut inference_map);
                          }
                      }
                      
                      // Construct type args vector in order of generics
                      for param_name in &def.generics {
                          if let Some(ty) = inference_map.get(param_name) {
                              type_args.push(ty.clone());
                          } else {
                              // Default or Error?
                              // If un-inferable, we might have issues. Assume resolved for now or skip.
                          }
                      }
                      
                      if type_args.len() == def.generics.len() {
                          // Instantiate!
                          let new_name = self.request_function_instantiation(name, type_args);
                          *name = new_name;
                      }
                  }
              }
              ExprKind::StaticMethodCall(type_ty, method_name, args) => {
                          *type_ty = self.substitute_type(type_ty, subst);
                          *type_ty = self.resolve_type(type_ty);
                          for arg in args.iter_mut() {
                              self.rewrite_expr(&mut arg.inner, subst, None);
                          }
                          
                          // Check if it's a generic struct constructor/method that needs instantiation?
                          // Check if it's a generic struct constructor/method that needs instantiation?
                          let type_name_str = type_ty.get_base_name();
                          if let Some(def) = self.generic_structs.get(&type_name_str).cloned() {
                              // Find implementation handling this call
                              let mut best_impl: Option<&ImplBlock> = None;
                              for impl_block in &self.generic_impls {
                                  if let Type::Struct(target, _) = &impl_block.target_type {
                                      if target.as_str() == type_name_str.as_str() {
                                          if impl_block.methods.iter().any(|m| m.name == *method_name) {
                                              best_impl = Some(impl_block);
                                              break;
                                          }
                                      }
                                  }
                              }
                              
                              if let Some(impl_block) = best_impl {
                                  if let Some(method) = impl_block.methods.iter().find(|m| m.name == *method_name) {
                                      // Infer generic args from arguments
                                      let mut type_args = Vec::new();
                                      let mut inference_map = HashMap::new();
                                      let mut all_inferred = true;
                                      
                                      for (i, (arg_val, _)) in args.iter().zip(&method.args).enumerate() {
                                          let param_ty = &method.args[i].1;
                                          if let Some(val_ty) = self.infer_expr_type(&arg_val.inner) {
                                              self.unify_types(param_ty, &val_ty, &mut inference_map);
                                          }
                                      }
                                      
                                      // Construct impl generics
                                      for param in &def.generics {
                                          if let Some(ty) = inference_map.get(param) {
                                              type_args.push(ty.clone());
                                          } else {
                                              // Context-based Inference (Required for static methods like Vec::new() where args don't imply generics)
                                              let mut inferred_via_context = false;
                                              if let Some(Type::Struct(n, args)) = expected_type {
                                                  // Case A: Generic Struct (Vec<i32>)
                                                  if n == &type_name_str && args.len() == def.generics.len() {
                                                     if let Some(idx) = def.generics.iter().position(|r| r == param) {
                                                         type_args.push(args[idx].clone());
                                                         inferred_via_context = true;
                                                     }
                                                  }
                                                  // Case B: Already Monomorphized Struct (Vec_i32)
                                                  else if let Some((orig_name, orig_args)) = self.reverse_struct_instances.get(n) {
                                                      if orig_name == &type_name_str && orig_args.len() == def.generics.len() {
                                                          if let Some(idx) = def.generics.iter().position(|r| r == param) {
                                                              type_args.push(orig_args[idx].clone());
                                                              inferred_via_context = true;
                                                          }
                                                      }
                                                  }
                                              }
                                              
                                              if !inferred_via_context {
                                                  all_inferred = false;
                                              }
                                          }
                                      }
                                      
                                      if all_inferred && !type_args.is_empty() {
                                          let concrete_name = self.request_struct_instantiation(&type_name_str, type_args);
                                          *type_ty = Type::Struct(concrete_name, vec![]);
                                       }
                                  }
                              }
                           } else if let Some(def) = self.generic_enums.get(&type_name_str).cloned() {
                               // Enum Logic - Rewrite to EnumInit

                               let mut best_impl: Option<&ImplBlock> = None;
                               
                               // 1. Try to find if it matches a Variant (Constructor)
                               if let Some(variant) = def.variants.iter().find(|v| v.name == *method_name) {

                                   // It is a variant constructor! Rewrite to EnumInit.
                                   // First infer generics from args
                                   let mut type_args = Vec::new();
                                   let mut inference_map = HashMap::new();
                                   // Map args to variant fields to infer
                                   let all_inferred = true;
                                   
                                   match &variant.kind {
                                        crate::compiler::ast::VariantKind::Unit => {},
                                        crate::compiler::ast::VariantKind::Tuple(types) => {
                                             for (i, arg_expr) in args.iter().enumerate() {
                                                 if i < types.len() {
                                                     let param_ty = &types[i];
                                                     if let Some(val_ty) = self.infer_expr_type(&arg_expr.inner) {
                                                         self.unify_types(param_ty, &val_ty, &mut inference_map);
                                                     }
                                                 }
                                             }
                                        },
                                        crate::compiler::ast::VariantKind::Struct(_fields) => {
                                             // Positional args to struct definitions? 
                                             // For now assume matching order or skip inference for struct variants called as methods
                                        }
                                   }

                                   
                                   for param in &def.generics {
                                       if let Some(ty) = inference_map.get(param) {
                                           type_args.push(ty.clone());
                                       } else {
                                           // Fallback: Infer from expected_type

                                           let mut inferred_from_expected = None;
                                           if let Some(Type::Enum(n, args)) | Some(Type::Struct(n, args)) = expected_type {
                                               // Case 1: Unmangled name with generics (e.g. Option with [Entry_i64_i64])
                                               if n == &type_name_str && args.len() == def.generics.len() {
                                                    // Find index of param
                                                    if let Some(idx) = def.generics.iter().position(|r| r == param) {
                                                         inferred_from_expected = Some(args[idx].clone());
                                                    }
                                               }
                                               // Case 2: Already mangled name (e.g. Option_Entry_i64_i64 with args=[])
                                               // In this case, we use the mangled name directly and don't need to infer generics
                                               // since codegen will look up the type by mangled name
                                               else if n.starts_with(&format!("{}_", type_name_str)) && args.is_empty() {
                                                   // Signal that we should use the mangled name directly
                                                   // Mark "skip" by using a special sentinel (we'll handle this below)
                                                   inferred_from_expected = Some(Type::Void);
                                               }
                                           }

                                           if let Some(ty) = inferred_from_expected.as_ref() {
                                                type_args.push(ty.clone());
                                           } else {
                                                // Default to Unit (Tuple([])) if partially inferred (e.g. Result::Ok(x) implies T=Typeof(x), E=Unit)
                                                // Void is not valid in LLVM structs.
                                                type_args.push(Type::Tuple(vec![]));
                                           }
                                       }
                                   }
                                   
                                   // Try to infer concrete name if arguments inference failed
                                   let mut concrete_name = String::new();
                                   let mut resolved_generics = Vec::new();

                                   // Check if we should use mangled name directly (Type::Void sentinel)
                                   let has_void_sentinel = type_args.iter().any(|t| matches!(t, Type::Void));

                                   if all_inferred && !type_args.is_empty() && !has_void_sentinel {
                                       resolved_generics = type_args.clone();
                                       concrete_name = self.request_enum_instantiation(&type_name_str, type_args);
                                   } else if let Some(Type::Enum(n, args)) | Some(Type::Struct(n, args)) = expected_type {
                                       // Fallback 1: If expected type is already concrete (e.g. Option_Entry_i64_i64) matching this generic
                                       if n.starts_with(&format!("{}_", type_name_str)) {
                                            resolved_generics = args.clone();
                                            concrete_name = n.clone();
                                       }
                                       // Fallback 2: If expected type has same base name with generics (e.g. Option with [Entry_i64_i64])
                                       else if n == &type_name_str && !args.is_empty() {
                                            resolved_generics = args.clone();
                                            concrete_name = self.request_enum_instantiation(&type_name_str, args.clone());
                                       }
                                   }

                                   if !concrete_name.is_empty() {
                                        // Construct Payload
                                       let payload = match &variant.kind {
                                            crate::compiler::ast::VariantKind::Unit => EnumVariantInit::Unit,
                                            crate::compiler::ast::VariantKind::Tuple(_) => {
                                                EnumVariantInit::Tuple(args.clone())
                                            },
                                            crate::compiler::ast::VariantKind::Struct(_) => {
                                                // Hard to map positional to struct fields without names
                                                // Fallback to Tuple? No, syntax error.
                                                // For arbitrary Tuple variants, this is fine.
                                                EnumVariantInit::Unit // Placeholder/Error
                                            }
                                       };
                                       
                                       // REWRITE EXPR
                                       *expr = ExprKind::EnumInit {
                                           enum_name: concrete_name,
                                           variant_name: method_name.clone(),
                                           generics: resolved_generics,
                                           payload: payload,
                                       };
                                       return; // Done
                                   }
                               }

                               // 2. If not variant, look for impl methods (as before)
                               for impl_block in &self.generic_impls {
                                  if let Type::Struct(target, _) | Type::Enum(target, _) = &impl_block.target_type {
                                      if target.as_str() == type_name_str.as_str() {
                                          if impl_block.methods.iter().any(|m| m.name == *method_name) {
                                              best_impl = Some(impl_block);
                                              break;
                                          }
                                      }
                                  }
                               }

                               if let Some(impl_block) = best_impl {
                                   if let Some(method) = impl_block.methods.iter().find(|m| m.name == *method_name) {
                                       let mut type_args = Vec::new();
                                       let mut inference_map = HashMap::new();
                                       let mut all_inferred = true;
                                       
                                       for (i, (arg_val, _)) in args.iter().zip(&method.args).enumerate() {
                                           let param_ty = &method.args[i].1;
                                           if let Some(val_ty) = self.infer_expr_type(&arg_val.inner) {
                                               self.unify_types(param_ty, &val_ty, &mut inference_map);
                                           }
                                       }
                                       
                                       for param in &def.generics {
                                           if let Some(ty) = inference_map.get(param) {
                                               type_args.push(ty.clone());
                                           } else {
                                               all_inferred = false;
                                           }
                                       }
                                       
                                       if all_inferred && !type_args.is_empty() {
                                           let concrete_name = self.request_enum_instantiation(&type_name_str, type_args);
                                           *type_ty = Type::Struct(concrete_name, vec![]);
                                       }
                                   }
                               }
                           }
                          
                          if type_name_str.as_str() == "Tensor" {
                               // ignore
                          }
                      }
              ExprKind::MethodCall(receiver, method_name, args) => {
                  // First, rewrite the receiver
                  self.rewrite_expr(&mut receiver.inner, subst, None);
                  
                  // Try to infer receiver type to get expected_type for arguments
                  let receiver_ty = self.infer_expr_type(&receiver.inner);
                  
                  // Apply substitution to resolve generic type parameters
                  let resolved_receiver_ty = receiver_ty.map(|ty| {
                      let substituted = self.substitute_type(&ty, subst);
                      self.resolve_type(&substituted)
                  });
                  
                  // For Vec<T>.push(item), the item should have expected_type T
                  // For other generic methods, we could do similar logic
                  let arg_expected_type: Option<Type> = match (&resolved_receiver_ty, method_name.as_str()) {
                      (Some(Type::Struct(name, generics)), "push") if name == "Vec" || name.starts_with("Vec_") => {
                          // Vec<T>.push(item: T) - expected_type is T
                          if !generics.is_empty() {
                              Some(generics[0].clone())
                          } else if name.starts_with("Vec_") {
                              // Already mangled name like Vec_Entry_i64_String
                              // Extract the inner type from the mangled name
                              let inner = name.strip_prefix("Vec_").unwrap_or("");
                              if !inner.is_empty() {
                                  Some(Type::Struct(inner.to_string(), vec![]))
                              } else {
                                  None
                              }
                          } else {
                              None
                          }
                      }
                      _ => None,
                  };
                  
                  // Rewrite args with expected_type if available
                  for arg in args {
                      self.rewrite_expr(&mut arg.inner, subst, arg_expected_type.as_ref());
                  }
              }
              ExprKind::BinOp(l, _, r) => {
                  self.rewrite_expr(&mut l.inner, subst, None);
                  self.rewrite_expr(&mut r.inner, subst, None);
              }
              ExprKind::As(expr, ty) => {
                  self.rewrite_expr(&mut expr.inner, subst, None);
                  *ty = self.substitute_type(ty, subst);
                  *ty = self.resolve_type(ty);
              }
              ExprKind::Block(stmts) => {
                  self.scopes.push(HashMap::new());
                  for s in stmts {
                      self.rewrite_stmt(&mut s.inner, subst, None);
                  }
                  self.scopes.pop();
              }
              ExprKind::IfExpr(cond, then_block, else_block) => {
                   self.rewrite_expr(&mut cond.inner, subst, None);
                   for s in then_block {
                       self.rewrite_stmt(&mut s.inner, subst, None);
                   }
                   if let Some(eb) = else_block {
                       for s in eb {
                           self.rewrite_stmt(&mut s.inner, subst, None);
                       }
                   }
              }
              ExprKind::Match { expr, arms } => {
                  // Rewrite the matched expression
                  self.rewrite_expr(&mut expr.inner, subst, None);
                  
                  // Rewrite each arm: substitute pattern types and rewrite body
                  for (pattern, body) in arms.iter_mut() {
                      // Substitute types in pattern (e.g., Entry<K, V> -> Entry_i64_i64)
                      if let Pattern::EnumPattern { enum_name, variant_name: _, bindings: _ } = pattern {
                          // Parse the enum_name to extract base name and type args
                          // e.g., "Entry<K, V>" -> base="Entry", args=["K", "V"]
                          if let Some(angle_idx) = enum_name.find('<') {
                              let base = &enum_name[..angle_idx];
                              let args_str = &enum_name[angle_idx+1..enum_name.len()-1];
                              // Parse type args and substitute
                              let arg_names: Vec<&str> = args_str.split(", ").collect();
                              let mut substituted_args = Vec::new();
                              for arg_name in &arg_names {
                                  let arg_name = arg_name.trim();
                                  if let Some(replacement) = subst.get(arg_name) {
                                      substituted_args.push(self.mangle_type(replacement));
                                  } else {
                                      substituted_args.push(arg_name.to_string());
                                  }
                              }
                              // Build new mangled enum name
                              if !substituted_args.is_empty() {
                                  let new_name = format!("{}_{}", base, substituted_args.join("_"));
                                  *enum_name = new_name;
                              }
                          }
                      }
                      // Rewrite arm body
                      self.rewrite_expr(&mut body.inner, subst, expected_type);
                  }
              }
              _ => {}
         }
     }

     fn instantiate_impls(&mut self, name: &str, args: &[Type]) {
         let concrete_struct_name = if let Some(mangled) = self.struct_instances.get(&(name.to_string(), args.to_vec())) {
             mangled.clone()
         } else {
             format!("{}_{}", name, self.mangle_types(args))
         };

         let mut new_impls = Vec::new();
         let generic_impls = self.generic_impls.clone();
         
         for impl_block in &generic_impls {
             // Check compatibility
             let mut subst = HashMap::new();
             let mut matches = false;
             
             if let Type::Struct(target, target_args) | Type::Enum(target, target_args) = &impl_block.target_type {
                 if target == name {
                     // Check args match length
                     // If target has generic params, unifying them with concrete args gives substitution
                     // But target_args in impl might be specific e.g. impl Box<i64>.
                     // Or generic: impl<T> Box<T>.
                     // We need to map impl generics to concrete types.
                     
                     // Case 1: Impl args match Generic Struct definition args (canonical).
                     // Usually impl<U> Struct<U>.
                     if target_args.len() == args.len() {
                         matches = true;
                         for (impl_arg, concrete_arg) in target_args.iter().zip(args) {
                             self.unify_types(impl_arg, concrete_arg, &mut subst);
                         }
                     }
                 }
             }
             
             if matches {
                  // Preserve whether this is Struct or Enum
                  let is_enum = matches!(&impl_block.target_type, Type::Enum(_, _));
                  
                  let concrete_target_type = if is_enum {
                      Type::Enum(concrete_struct_name.clone(), vec![])
                  } else {
                      Type::Struct(concrete_struct_name.clone(), vec![])
                  };
                  
                  subst.insert("Self".to_string(), concrete_target_type.clone());
                  
                  let mut new_impl = impl_block.clone();
                  new_impl.generics.clear();
                  new_impl.target_type = concrete_target_type;
                 
                 // Rewrite methods
                 for method in &mut new_impl.methods {
                    // Substitute signature and then RESOLVE to concrete names
                    for (_, ty) in &mut method.args {
                        *ty = self.substitute_type(ty, &subst);
                        *ty = self.resolve_type(ty);
                    }
                    method.return_type = self.substitute_type(&method.return_type, &subst);
                    method.return_type = self.resolve_type(&method.return_type);
                    
                    // Rewrite body using helper (pushes scope)
                    self.rewrite_function_body(method, &subst);
                 }
                 
                 new_impls.push(new_impl);
             }
         }
         
         self.concrete_impls.extend(new_impls);
     }

     
     fn request_struct_instantiation(&mut self, name: &str, args: Vec<Type>) -> String {
        let key = (name.to_string(), args.clone());
        if let Some(mangled) = self.struct_instances.get(&key) {
            return mangled.clone();
        }
        
        let mangled = format!("{}_{}", name, self.mangle_types(&args));
        self.struct_instances.insert(key, mangled.clone());
        self.reverse_struct_instances.insert(mangled.clone(), (name.to_string(), args.clone()));
        self.pending_queue.push_back((name.to_string(), args, true));
        mangled
    }

    fn request_enum_instantiation(&mut self, name: &str, args: Vec<Type>) -> String {
        // Similar to struct
        let key = (name.to_string(), args.clone());
        // Reuse struct_instances map or create new one?
        // Enums and Structs share Type::UserDefined namespace effectively.
        // Let's use separate map for safety or reuse struct_instances if we treat them same.
        // But we need to know if it is struct or enum for `pending_queue`?
        // Actually pending_queue boolean `is_struct` handles it.
        // But for storage... using `struct_instances` is fine if names are unique.
        if let Some(mangled) = self.struct_instances.get(&key) {
            return mangled.clone();
        }
        let mangled = format!("{}_{}", name, self.mangle_types(&args));
        self.struct_instances.insert(key, mangled.clone());
         // We need to know it IS an enum to set `is_struct=false`? No `is_struct` param in queue is binary.
         // We need `is_enum`?
         // Let's assume queue: (name, types, kind) where kind=0:struct, 1:fn, 2:enum
         // For now hack: check `generic_enums`.
         self.pending_queue.push_back((name.to_string(), args, true)); // Reuse true for "Type" (Struct/Enum)?
         // But `instantiate_struct` handles structs. We need `instantiate_enum`.
         // Let's modify pending_queue to be `(String, Vec<Type>, ItemKind)`
         mangled
    }
    
    fn request_function_instantiation(&mut self, name: &str, args: Vec<Type>) -> String {
         let key = (name.to_string(), args.clone());
         if let Some(mangled) = self.function_instances.get(&key) {
             return mangled.clone();
         }
         let mangled = format!("{}_{}", name, self.mangle_types(&args));
         self.function_instances.insert(key, mangled.clone());
         self.pending_queue.push_back((name.to_string(), args, false));
         mangled
    }
    
    fn instantiate_struct(&mut self, name: &str, args: &[Type]) -> Result<(), TlError> {
        if self.generic_enums.contains_key(name) {
             self.instantiate_enum(name, args)?;
             return Ok(());
        }

        if let Some(def) = self.generic_structs.get(name) {
            let mangled = self.struct_instances.get(&(name.to_string(), args.to_vec()))
                .ok_or_else(|| TlError::Monomorphization {
                    kind: MonomorphizationErrorKind::GenericItemNotFound(format!("Struct instance {} {:?} not found in map", name, args)),
                    span: None
                })?.clone();
            log::debug!("Instantiating struct {} -> {} with args {:?}", name, mangled, args);
            
            // Substitution map
            let mut subst = HashMap::new();
            for (i, param) in def.generics.iter().enumerate() {
                if i < args.len() {
                     subst.insert(param.clone(), args[i].clone());
                }
            }
            
            let mut new_def = def.clone();
            new_def.name = mangled;
            new_def.generics.clear(); // Concrete now
            for (_fname, ty) in &mut new_def.fields {
                *ty = self.substitute_type(ty, &subst);
                *ty = self.resolve_type(ty);
            }
            
            self.concrete_structs.push(new_def);
            
            // Also instantiate associated impl blocks
            self.instantiate_impls(name, args);
        }
        Ok(())
    }


    fn instantiate_enum(&mut self, name: &str, args: &[Type]) -> Result<(), TlError> {
         if let Some(def) = self.generic_enums.get(name) {
            let mangled = self.struct_instances.get(&(name.to_string(), args.to_vec()))
                .ok_or_else(|| TlError::Monomorphization {
                    kind: MonomorphizationErrorKind::GenericItemNotFound(format!("Enum instance {} {:?} not found in map", name, args)),
                    span: None
                })?.clone();
            
            let mut subst = HashMap::new();
            for (i, param) in def.generics.iter().enumerate() {
                if i < args.len() {
                     subst.insert(param.clone(), args[i].clone());
                }
            }
            
            let mut new_def = def.clone();
            new_def.name = mangled;
            new_def.generics.clear();
            
            for variant in &mut new_def.variants {
                match &mut variant.kind {
                    crate::compiler::ast::VariantKind::Unit => {},
                    crate::compiler::ast::VariantKind::Tuple(types) => {
                        for ty in types {
                            *ty = self.substitute_type(ty, &subst);
                            *ty = self.resolve_type(ty);
                        }
                    },
                    crate::compiler::ast::VariantKind::Struct(fields) => {
                        for (_, ty) in fields {
                            *ty = self.substitute_type(ty, &subst);
                            *ty = self.resolve_type(ty);
                        }
                    }
                }
            }
            
            self.concrete_enums.push(new_def);
            
            // Also instantiate associated impl blocks
            self.instantiate_impls(name, args);
         }
         Ok(())
    }

    
    fn instantiate_function(&mut self, name: &str, args: &[Type]) -> Result<(), TlError> {
         if let Some(def) = self.generic_functions.get(name) {
            let mangled = self.function_instances.get(&(name.to_string(), args.to_vec()))
                .ok_or_else(|| TlError::Monomorphization {
                    kind: MonomorphizationErrorKind::GenericItemNotFound(format!("Function instance {} {:?} not found in map", name, args)),
                    span: None
                })?.clone();
            
            let mut subst = HashMap::new();
            for (i, param) in def.generics.iter().enumerate() {
                if i < args.len() {
                     subst.insert(param.clone(), args[i].clone());
                }
            }
            
            let mut new_def = def.clone();
            new_def.name = mangled;
            new_def.generics.clear();
            
            // Substitute signature
            for (_, ty) in &mut new_def.args {
                *ty = self.substitute_type(ty, &subst);
                *ty = self.resolve_type(ty);
            }
            new_def.return_type = self.substitute_type(&new_def.return_type, &subst);
            new_def.return_type = self.resolve_type(&new_def.return_type);
            
            // Rewrite Body
            // Push new scope for arguments
            // self.scopes.push(HashMap::new()); // Moved to rewrite_function_body
            // for (arg_name, arg_ty) in &new_def.args { // Moved to rewrite_function_body
            //     self.scopes.last_mut().unwrap().insert(arg_name.clone(), arg_ty.clone()); // Moved to rewrite_function_body
            // }

            self.rewrite_function_body(&mut new_def, &subst);
            
            // Pop scope // Moved to rewrite_function_body
            // self.scopes.pop();

            self.concrete_functions.push(new_def);
         }
         Ok(())
    }

    
    fn rewrite_function_body(&mut self, func: &mut FunctionDef, subst: &HashMap<String, Type>) {
        self.scopes.push(HashMap::new());
        // Register params
        for (name, ty) in &func.args {
            if let Some(scope) = self.scopes.last_mut() {
                scope.insert(name.clone(), ty.clone());
            }
        }
        
        let old_ret = self.current_return_type.clone();
        self.current_return_type = Some(func.return_type.clone());

        let len = func.body.len();
        for (i, stmt) in func.body.iter_mut().enumerate() {
            let is_last = i == len - 1;
            // Clone the type to avoid borrowing `self` which is needed mutably for rewrite_stmt
            let expected_owned = if is_last { self.current_return_type.clone() } else { None };
            self.rewrite_stmt(&mut stmt.inner, subst, expected_owned.as_ref());
        }
        
        self.current_return_type = old_ret;
        self.scopes.pop();
    }
    


    fn substitute_type(&self, ty: &Type, subst: &HashMap<String, Type>) -> Type {
        match ty {

             Type::Tensor(inner, r) => Type::Tensor(Box::new(self.substitute_type(inner, subst)), *r),
             Type::Struct(name, args) => {
                 if let Some(replacement) = subst.get(name) {
                     return replacement.clone();
                 }
                 let new_args: Vec<Type> = args.iter().map(|a| self.substitute_type(a, subst)).collect();
                 match name.as_str() {
                     "String" if new_args.is_empty() => Type::String("String".to_string()),
                     "I64" if new_args.is_empty() => Type::I64,
                     "Bool" if new_args.is_empty() => Type::Bool,
                     "F32" if new_args.is_empty() => Type::F32,
                     "Char" if new_args.is_empty() => Type::Char("Char".to_string()),
                     _ => Type::Struct(name.clone(), new_args)
                 }
             },
             // Handle Enum type arguments recursively
             Type::Enum(name, args) => {
                 if let Some(replacement) = subst.get(name) {
                     return replacement.clone();
                 }
                 let new_args: Vec<Type> = args.iter().map(|a| self.substitute_type(a, subst)).collect();
                 Type::Enum(name.clone(), new_args)
             },
             // ... other recursive cases
             Type::Path(segments, args) => {
                 if segments.len() == 1 {
                     if let Some(replacement) = subst.get(&segments[0]) {
                         return replacement.clone();
                     }
                 }
                 let new_args: Vec<Type> = args.iter().map(|a| self.substitute_type(a, subst)).collect();
                 Type::Path(segments.clone(), new_args)
             },
             Type::Ptr(inner) => {
                 Type::Ptr(Box::new(self.substitute_type(inner, subst)))
             },
             Type::Ref(inner) => Type::Ref(Box::new(self.substitute_type(inner, subst))),

             _ => ty.clone()
        }
    }

    fn mangle_types(&self, types: &[Type]) -> String {
        types.iter().map(|t| self.mangle_type(t)).collect::<Vec<_>>().join("_")
    }

    fn mangle_type(&self, ty: &Type) -> String {
        match ty {
            Type::I64 => "i64".to_string(),
            Type::F64 => "f64".to_string(),
            Type::F32 => "f32".to_string(),
            Type::Bool => "bool".to_string(),
            Type::String(_) => "String".to_string(),
            Type::Char(_) => "Char".to_string(),
            Type::I32 => "i32".to_string(),
            Type::I8 => "i8".to_string(),
            Type::U8 => "u8".to_string(),
            Type::U16 => "u16".to_string(),
            Type::U32 => "u32".to_string(),
            Type::Usize => "usize".to_string(),
            Type::Entity => "entity".to_string(),
            Type::Void => "void".to_string(),

            Type::Struct(name, args) => {
                 if args.is_empty() {
                     name.clone()
                 } else {
                     format!("{}_{}", name, self.mangle_types(args))
                 }
            }
            Type::Tensor(inner, _) => format!("Tensor_{}", self.mangle_type(inner)),
            Type::Enum(name, args) => {
                if args.is_empty() {
                    name.clone()
                } else {
                    format!("{}_{}", name, self.mangle_types(args))
                }
            }
            Type::Path(segments, args) => {
                let base = segments.join("_");
                if args.is_empty() {
                    base
                } else {
                    format!("{}_{}", base, self.mangle_types(args))
                }
            }
            Type::Tuple(types) => {
                if types.is_empty() {
                    "unit".to_string() // Empty tuple is unit type
                } else {
                    format!("Tuple_{}", self.mangle_types(types))
                }
            }
            Type::Ptr(inner) => format!("Ptr_{}", self.mangle_type(inner)),
            _ => {
                "unknown".to_string()
            },
        }
    }

    fn infer_expr_type(&self, expr: &ExprKind) -> Option<Type> {
        match expr {
            ExprKind::Int(_) => Some(Type::I64),
            ExprKind::Float(_) => Some(Type::F32), // Match Semantics default (F32)
            ExprKind::Bool(_) => Some(Type::Bool),
            ExprKind::StringLiteral(_) => Some(Type::String("String".to_string())),
            ExprKind::Variable(name) => {
                for scope in self.scopes.iter().rev() {
                    if let Some(ty) = scope.get(name) {
                        return Some(ty.clone());
                    }
                }
                None
            }
            ExprKind::BinOp(lhs, _, _) => self.infer_expr_type(&lhs.inner), // Assume LHS determines type (mostly true)
            ExprKind::StructInit(ty, _) => Some(ty.clone()),
            ExprKind::EnumInit { enum_name, generics, .. } => Some(Type::Enum(enum_name.clone(), generics.clone())),
            // FnCall return type inference requires looking up the function
            ExprKind::FnCall(name, _) => {
                // If it's a concrete function we already processed
                if let Some(f) = self.concrete_functions.iter().find(|f| f.name == *name) {
                    return Some(f.return_type.clone());
                }
                // If it's a generic function (not instantiated yet? or name is original?)
                // If it's a generic function (not instantiated yet? or name is original?)
                if let Some(f) = self.generic_functions.get(name) {
                    return Some(f.return_type.clone()); // Returns potentially generic type T
                }
                // If it's a regular function
                // We need to look up in module functions? We assume we have all defs.
                // We don't have list of all non-generic functions readily available in struct?
                // We scan module at start. Maybe we should store signature of all functions?
                None
            }
            _ => None,
        }
    }

    // Helper to unify generic param with concrete type
    fn unify_types(&self, generic: &Type, concrete: &Type, map: &mut HashMap<String, Type>) {
        match (generic, concrete) {
            (Type::Struct(name, args), _) => {
                // ...
                if args.is_empty() {
                    if !map.contains_key(name) {
                         map.insert(name.clone(), concrete.clone());
                    }
                } else if let Type::Struct(c_name, c_args) = concrete {
                    if name == c_name && args.len() == c_args.len() {
                        for (a, b) in args.iter().zip(c_args) {
                            self.unify_types(a, b, map);
                        }
                    }
                }
            }
             (Type::Path(segments, args), _) => {
                 if segments.len() == 1 && args.is_empty() {
                     let name = &segments[0];
                     if !map.contains_key(name) {
                         map.insert(name.clone(), concrete.clone());
                     }
                 }
                 // Handle nested args?
             }
            (Type::Tensor(inner_g, _), Type::Tensor(inner_c, _)) => {
                self.unify_types(inner_g, inner_c, map);
            }

            _ => {}
        }
    }


    /// Build a substitution map from a mangled type name like "HashMap_i64_i64"
    /// by extracting the type arguments from the suffix and mapping them to the
    /// original generic parameters (K, V) from the generic struct/enum definition.
    fn build_subst_from_mangled_name(&self, mangled_name: &str) -> HashMap<String, Type> {
        let mut subst = HashMap::new();
        
        // Try to find the base generic struct/enum name
        for (base_name, def) in &self.generic_structs {
            if mangled_name.starts_with(&format!("{}_", base_name)) {
                let suffix = &mangled_name[base_name.len() + 1..]; // +1 for underscore
                let type_args = self.parse_mangled_type_args(suffix);
                
                if type_args.len() == def.generics.len() {
                    for (param, arg) in def.generics.iter().zip(type_args.iter()) {
                        subst.insert(param.clone(), arg.clone());
                    }
                    return subst;
                }
            }
        }
        
        // Also check generic enums
        for (base_name, def) in &self.generic_enums {
            if mangled_name.starts_with(&format!("{}_", base_name)) {
                let suffix = &mangled_name[base_name.len() + 1..];
                let type_args = self.parse_mangled_type_args(suffix);
                
                if type_args.len() == def.generics.len() {
                    for (param, arg) in def.generics.iter().zip(type_args.iter()) {
                        subst.insert(param.clone(), arg.clone());
                    }
                    return subst;
                }
            }
        }
        
        subst
    }
    
    /// Parse mangled type arguments from a suffix like "i64_i64" -> [I64, I64]
    fn parse_mangled_type_args(&self, suffix: &str) -> Vec<Type> {
        let mut args = Vec::new();
        for part in suffix.split('_') {
            let ty = match part {
                "i64" => Type::I64,
                "f32" => Type::F32,
                "f64" => Type::F64,
                "bool" => Type::Bool,
                "string" => Type::String("String".to_string()),
                "char" => Type::Char("Char".to_string()),
                other => {
                    // Check if it's a known struct/enum
                    if self.generic_structs.contains_key(other) {
                        Type::Struct(other.to_string(), vec![])
                    } else if self.generic_enums.contains_key(other) {
                        Type::Enum(other.to_string(), vec![])
                    } else {
                        // Default to Struct for unknown types
                        Type::Struct(other.to_string(), vec![])
                    }
                }
            };
            args.push(ty);
        }
        args
    }


}

