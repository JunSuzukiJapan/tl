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
                eprintln!("DEBUG: Collected generic enum {}", e.name);
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
                let empty_subst = HashMap::new();
                for method in &mut impl_block.methods {
                    for stmt in &mut method.body {
                        self.rewrite_stmt(&mut stmt.inner, &empty_subst, None);
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
                 if !args.is_empty() {
                     // Check if this is a generic struct instantiation
                     if self.generic_structs.contains_key(name) {
                         let concrete_args: Vec<Type> = args.iter().map(|a| self.resolve_type(a)).collect();
                         let mangled = self.request_struct_instantiation(name, concrete_args);
                         // Return as UserDefined with NO args (concrete name)
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
                     Type::Struct(name.clone(), new_args)
                 }

            // Recursive resolution
            Type::Tensor(inner, r) => Type::Tensor(Box::new(self.resolve_type(inner)), *r),
            Type::Vec(inner) => Type::Vec(Box::new(self.resolve_type(inner))),
            Type::Tuple(types) => Type::Tuple(types.iter().map(|t| self.resolve_type(t)).collect()),
            Type::TensorShaped(inner, dims) => Type::TensorShaped(Box::new(self.resolve_type(inner)), dims.clone()),
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
            StmtKind::Assign { value, indices, .. } => {
                self.rewrite_expr(&mut value.inner, subst, None);
                if let Some(idxs) = indices {
                    for idx in idxs {
                        self.rewrite_expr(&mut idx.inner, subst, None);
                    }
                }
            }
            StmtKind::FieldAssign { obj, value, .. } => {
                self.rewrite_expr(&mut obj.inner, subst, None);
                self.rewrite_expr(&mut value.inner, subst, None);
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
         // eprintln!("DEBUG: rewrite_expr {:?}", expr);
         match expr {
             ExprKind::StructInit(name, explicit_generics, fields) => {
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
                 // 4. Try generic Enum (e.g. Option::Some) - LEGACY? 
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
              ExprKind::EnumInit { enum_name, variant_name, generics: _, payload } => {
                  eprintln!("DEBUG: EnumInit entry name={} variant={}", enum_name, variant_name);
                  if let Some(def) = self.generic_enums.get(enum_name) {
                      if let Some(v_def) = def.variants.iter().find(|v| v.name == *variant_name) {
                          let mut inferred_map = HashMap::new();
                          let mut all_inferred = true;
                          
                          // Infer generics
                          match (&*payload, &v_def.kind) {
                              (EnumVariantInit::Unit, VariantKind::Unit) => {},
                              (EnumVariantInit::Tuple(exprs), VariantKind::Tuple(types)) => {
                                  for (e_val, field_ty) in exprs.iter().zip(types.iter()) {
                                      if let Some(val_ty) = self.infer_expr_type(&e_val.inner) {
                                          self.unify_types(field_ty, &val_ty, &mut inferred_map);
                                      }
                                  }
                              },
                              (EnumVariantInit::Struct(fields), VariantKind::Struct(def_fields)) => {
                                  for (fname, val) in fields.iter() {
                                     if let Some(field_ty) = def_fields.iter().find(|(f, _)| f == fname).map(|(_, t)| t) {
                                         if let Some(val_ty) = self.infer_expr_type(&val.inner) {
                                             self.unify_types(field_ty, &val_ty, &mut inferred_map);
                                         }
                                     }
                                  }
                              }
                              _ => {} // Mismatch handled in semantics
                          }
                          
                          let mut type_args = Vec::new();
                          for param in &def.generics {
                              if let Some(ty) = inferred_map.get(param) {
                                  type_args.push(ty.clone());
                              } else {
                                  all_inferred = false;
                              }
                          }
                          
                          if all_inferred && !type_args.is_empty() {
                              eprintln!("DEBUG: instantiate generic enum {} with {:?}", enum_name, type_args);
                              let concrete_enum_name = self.request_struct_instantiation(enum_name, type_args.clone());
                              *enum_name = concrete_enum_name;
                              
                              // Rewrite fields with concrete types
                              // Need to fetch concrete variant definition
                              // Clone the variant kind to avoid borrowing self during rewrite
                              let concrete_variant_kind = if let Some(c_def) = self.concrete_enums.iter().find(|e| e.name == *enum_name) {
                                  if let Some(c_v) = c_def.variants.iter().find(|v| v.name == *variant_name) {
                                      Some(c_v.kind.clone())
                                  } else { None }
                              } else { None };

                              if let Some(kind) = concrete_variant_kind {
                                      match (&mut *payload, &kind) {
                                           (EnumVariantInit::Tuple(exprs), VariantKind::Tuple(types)) => {
                                                for (e, ty) in exprs.iter_mut().zip(types.iter()) {
                                                     self.rewrite_expr(&mut e.inner, subst, Some(ty));
                                                }
                                                return;
                                           },
                                           (EnumVariantInit::Struct(fields), VariantKind::Struct(def_fields)) => {
                                                let mut field_types_map = HashMap::new();
                                                for (fname, fty) in def_fields {
                                                    field_types_map.insert(fname.clone(), fty.clone());
                                                }
                                                for (fname, val) in fields {
                                                     let ctx_ty = field_types_map.get(fname);
                                                     self.rewrite_expr(&mut val.inner, subst, ctx_ty);
                                                }
                                                return;
                                           },
                                           _ => {}
                                      }
                                  }
                          // Else generic default handling (rewrite children)
                  
                  }
                  
                  // Recurse children
                  match payload {
                      EnumVariantInit::Unit => {},
                      EnumVariantInit::Tuple(exprs) => {
                          for e in exprs { self.rewrite_expr(&mut e.inner, subst, None); }
                      },
                      EnumVariantInit::Struct(fields) => {
                          for (_, e) in fields { self.rewrite_expr(&mut e.inner, subst, None); }
                      }
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
                          eprintln!("DEBUG: StaticMethodCall entry type={:?} method={}", type_ty, method_name);
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
                                              all_inferred = false;
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
                                               if n == &type_name_str && args.len() == def.generics.len() {
                                                    // Find index of param
                                                    if let Some(idx) = def.generics.iter().position(|r| r == param) {
                                                         inferred_from_expected = Some(args[idx].clone());
                                                    }
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

                                   if all_inferred && !type_args.is_empty() {
                                       concrete_name = self.request_enum_instantiation(&type_name_str, type_args);
                                   } else if let Some(Type::Enum(n, _args)) | Some(Type::Struct(n, _args)) = expected_type {
                                       // Fallback: If expected type is already concrete (e.g. Option_I64) matching this generic
                                       if n.starts_with(&format!("{}_", type_name_str)) {
                                            concrete_name = n.clone();
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
                                                // For Option/Result (Tuple variants), this is fine.
                                                EnumVariantInit::Unit // Placeholder/Error
                                            }
                                       };
                                       
                                       // REWRITE EXPR
                                       *expr = ExprKind::EnumInit {
                                           enum_name: concrete_name,
                                           variant_name: method_name.clone(),
                                           generics: vec![],
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
              ExprKind::MethodCall(_, _, args) => {
                  for arg in args {
                      self.rewrite_expr(&mut arg.inner, subst, None);
                  }
              }
              ExprKind::BinOp(l, _, r) => {
                  self.rewrite_expr(&mut l.inner, subst, None);
                  self.rewrite_expr(&mut r.inner, subst, None);
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
                 subst.insert("Self".to_string(), Type::Struct(concrete_struct_name.clone(), vec![]));
                 
                 let mut new_impl = impl_block.clone();
                 new_impl.generics.clear();
                 new_impl.target_type = Type::Struct(concrete_struct_name.clone(), vec![]);
                 
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
                        }
                    },
                    crate::compiler::ast::VariantKind::Struct(fields) => {
                        for (_, ty) in fields {
                            *ty = self.substitute_type(ty, &subst);
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
            eprintln!("DEBUG: instantiate_function {} return_type {:?}", new_def.name, new_def.return_type);
            
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
        eprintln!("DEBUG: substitute_type {:?} with subst keys {:?}", ty, subst.keys());
        match ty {

             Type::Tensor(inner, r) => Type::Tensor(Box::new(self.substitute_type(inner, subst)), *r),
             Type::Struct(name, args) => {
                 if let Some(replacement) = subst.get(name) {
                     eprintln!("DEBUG: substitute_type FOUND {} -> {:?}", name, replacement);
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
             // ... other recursive cases
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
            Type::Vec(inner) => format!("vec_{}", self.mangle_type(inner)),
            Type::Struct(name, args) => {
                 if args.is_empty() {
                     name.clone()
                 } else {
                     format!("{}_{}", name, self.mangle_types(args))
                 }
            }
            Type::Tensor(inner, _) => format!("Tensor_{}", self.mangle_type(inner)),
            _ => "unknown".to_string(),
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
            ExprKind::StructInit(name, _, _) => Some(Type::Struct(name.clone(), vec![])), // Could be Generic?
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
                // If this UserDefined is actually a generic parameter (T)
                // We assume generic parameters are represented as UserDefined in current AST for "T"
                
                // If simple T -> Concrete
                if args.is_empty() {
                    // Check if 'name' is in knowledge base of generics?
                    // Currently unbound check: we just insert.
                    if !map.contains_key(name) {
                        eprintln!("DEBUG: unify_types insert {} -> {:?}", name, concrete);
                        map.insert(name.clone(), concrete.clone());
                    } else {
                        // verify consistency?
                    }
                } else if let Type::Struct(c_name, c_args) = concrete {
                    // T<A> vs List<B> ?
                    if name == c_name && args.len() == c_args.len() {
                        for (a, b) in args.iter().zip(c_args) {
                            self.unify_types(a, b, map);
                        }
                    }
                }
            }
            (Type::Tensor(inner_g, _), Type::Tensor(inner_c, _)) => {
                self.unify_types(inner_g, inner_c, map);
            }

            (Type::Vec(inner_g), Type::Vec(inner_c)) => {
                self.unify_types(inner_g, inner_c, map);
            }
            _ => {}
        }
    }


}

