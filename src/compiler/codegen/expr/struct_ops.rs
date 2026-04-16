//! codegen/expr/struct_ops.rs
//!
//! 構造体の初期化・アロケーション。
//! compile_struct_init, compile_struct_alloc。
use crate::compiler::error::TlError;

use inkwell::values::*;

use crate::compiler::ast::*;
use crate::compiler::codegen::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {

    pub(super) fn compile_struct_init(
        &mut self,
        name: &str,
        generics: &[Type],
        fields: &[(String, Expr)],
    ) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
        // Early detection: If `name` is already a mangled name that exists in struct_types,
        // use it directly and ignore generics (avoids double-mangling)
        if !generics.is_empty() && mangle_has_args(name) {
            if let Some(&existing_type) = self.struct_types.get(name) {
                let struct_def = self.struct_defs.get(name)
                    .ok_or_else(|| format!("Struct definition {} not found", name))?
                    .clone();
                return self.compile_struct_alloc(name, &[], &existing_type, &struct_def, fields);
            }
        }
        
        if !generics.is_empty() {
             // Generate mangled name first
             let mangled_name = self.mangle_type_name(name, generics);
             
             // Try to get existing or monomorphize on-demand
             let struct_type = if let Some(t) = self.struct_types.get(&mangled_name) {
                 *t
             } else {
                 // Attempt monomorphization
                 if self.monomorphize_struct(name, generics).is_ok() {
                     // Try again after monomorphization
                     if let Some(t) = self.struct_types.get(&mangled_name) {
                         *t
                     } else {
                         return Err(format!("Struct type {} not found after monomorphization", mangled_name).into());
                     }
                 } else {
                     // Recovery for double-mangled names (e.g. HashMap_i64_i64 -> HashMap)
                     let def_names: Vec<String> = self.struct_defs.keys().cloned().collect();
                      
                     let mut recovered = false;
                     for def_name in def_names {
                         if name.starts_with(&def_name) && name != def_name {
                             if self.monomorphize_struct(&def_name, generics).is_ok() {
                                 recovered = true;
                                 break;
                             }
                         }
                     }
                     
                     if recovered {
                         if let Some(t) = self.struct_types.get(&mangled_name) {
                             *t
                         } else {
                             // Try with base name mangled
                             let base = mangle_base_name(name);
                             let base_mangled = self.mangle_type_name(base, generics);
                             *self.struct_types.get(&base_mangled)
                                 .ok_or(format!("Struct type {} not found (tried {} and {})", name, mangled_name, base_mangled))?
                         }
                     } else {
                         return Err(format!("Monomorphization failed for {} with generics {:?}", name, generics).into());
                     }
                 }
             };
             
             let struct_def = self.struct_defs.get(&mangled_name)
                 .or_else(|| {
                     // Try base name mangled
                     let base = mangle_base_name(name);
                     let base_mangled = self.mangle_type_name(base, generics);
                     self.struct_defs.get(&base_mangled)
                 })
                 .ok_or(format!("Struct definition {} not found", mangled_name))?
                 .clone();
             
             return self.compile_struct_alloc(name, generics, &struct_type, &struct_def, fields);
        }

        // Non-generic case
        let lookup_name = name.to_string();

        // If the struct is not found in struct_types, try resolving from context.
        // This handles generic structs (e.g., Container) used inside their own monomorphized impl methods
        // where the StructInit uses the base name without type arguments.
        if !self.struct_types.contains_key(&lookup_name) {
            // Infer from current function name (e.g., tl_Container[i64]_new -> Container[i64]).
            if let Some(block) = self.builder.get_insert_block() {
                if let Some(func) = block.get_parent() {
                    let fn_name = func.get_name().to_str().unwrap_or("");
                    let prefix = format!("tl_{}", lookup_name);
                    if fn_name.starts_with(&prefix) {
                        let after_prefix = &fn_name[prefix.len()..];
                        if after_prefix.starts_with('[') {
                            if let Some(bracket_end) = after_prefix.rfind(']') {
                                let mangled = format!("{}{}", lookup_name, &after_prefix[..=bracket_end]);
                                if let (Some(&st), Some(sd)) = (self.struct_types.get(&mangled), self.struct_defs.get(&mangled).cloned()) {
                                    return self.compile_struct_alloc(&mangled, &[], &st, &sd, fields);
                                }
                            }
                        }
                    }
                }
            }
        }

        let struct_type = *self
            .struct_types
            .get(&lookup_name)
            .ok_or_else(|| {
                let fn_name = self.builder.get_insert_block()
                    .and_then(|b| b.get_parent())
                    .map(|f| f.get_name().to_str().unwrap_or("?").to_string())
                    .unwrap_or_else(|| "?".to_string());
                let struct_keys: Vec<String> = self.struct_types.keys()
                    .filter(|k| k.starts_with(&lookup_name))
                    .take(5)
                    .cloned()
                    .collect();
                format!("Struct type {} not found in codegen (in function {}, similar keys: {:?})", lookup_name, fn_name, struct_keys)
            })?;

        let struct_def = self
            .struct_defs
            .get(&lookup_name)
            .ok_or_else(|| {
                 format!("Struct definition {} not found", lookup_name)
            })?
            .clone();

        self.compile_struct_alloc(name, generics, &struct_type, &struct_def, fields)
    }

    fn compile_struct_alloc(
        &mut self,
        _original_name: &str,
        generics: &[Type],
        struct_type: &inkwell::types::StructType<'ctx>,
        struct_def: &crate::compiler::ast::StructDef,
        fields: &[(String, Expr)],
    ) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
        let name = &struct_def.name; // Use resolved name


        // Determine allocation strategy: Arena or Heap
        let size = struct_type
            .size_of()
            .ok_or(format!("Cannot determine size of struct {}", name))?;

        // ZST Optimization (PhantomData etc): Return NULL, not an aggregate value.
        // The Runtime handles NULL pointers gracefully (ignores them).
        if struct_def.fields.is_empty() {
            let null_ptr = self.context.ptr_type(inkwell::AddressSpace::default()).const_null();
            return Ok((null_ptr.into(), Type::Struct(name.to_string(), generics.to_vec())));
        }

        // 1. Heap Allocation
        let malloc_fn = self
            .module
            .get_function("malloc")
            .ok_or("malloc not found (declare in builtins)")?;
        
        let size_int = size;
        let size_i64 = if size_int.get_type() == self.context.i32_type() {
             self.builder.build_int_z_extend(size_int, self.context.i64_type(), "size_i64").map_err(|e| e.to_string())?
        } else {
             size_int
        };

        let call = self
            .builder
            .build_call(malloc_fn, &[size_i64.into()], "struct_malloc")
            .map_err(|e| e.to_string())?;
        let raw_ptr = match call.try_as_basic_value() {
            inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
            _ => return Err("malloc returned invalid value".into()),
        };

        // 2. Register with MemoryManager
        // Force named registration
        let register_fn = self
            .module
            .get_function("tl_mem_register_struct_named")
            .expect("tl_mem_register_struct_named not found in module");

        let cast_ptr = self
            .builder
            .build_pointer_cast(
                raw_ptr,
                self.context.ptr_type(inkwell::AddressSpace::default()),
                "cast_ptr",
            )
            .map_err(|e| e.to_string())?;

        let name_global = self
            .builder
            .build_global_string_ptr(name, "struct_name")
            .map_err(|e| e.to_string())?;

        self.builder
            .build_call(register_fn, &[cast_ptr.into(), name_global.as_pointer_value().into()], "")
            .map_err(|e| e.to_string())?;

        // Cast to Struct Pointer (opaque pointer in modern LLVM, but typed for GEP)
        let struct_ptr = self
            .builder
            .build_pointer_cast(
                raw_ptr,
                self.context.ptr_type(inkwell::AddressSpace::default()),
                "struct_ptr",
            )
            .map_err(|e| e.to_string())?;

        for (field_name, field_expr) in fields {
            let field_idx = struct_def
                .fields
                .iter()
                .position(|(n, _)| n == field_name)
                .ok_or(format!("Field {} not found in struct {}", field_name, name))?;

            let (val, _ty) = self.compile_expr(field_expr)?;
            
            // Move Semantics for pointer types:
            // When a variable is used as a struct field initializer, we MOVE ownership.
            // This means:
            // 1. If it's a temporary: remove from cleanup list (already done by try_consume_temp)
            // 2. If it's a variable: set its cleanup_mode to CLEANUP_NONE (ownership transferred)
            
            let moved = self.try_consume_temp(val);
            
            // Check if field_expr is a direct variable access - if so, mark it as moved
            let is_moveable_type = matches!(
                _ty,
                Type::Tensor(_, _) | Type::Struct(_, _) | Type::Tuple(_) | Type::Enum(_, _)
            );
            
            if !moved && is_moveable_type {
                // If field_expr is a Variable, we should transfer ownership (move semantics)
                // by disabling cleanup for the source variable
                if let ExprKind::Variable(var_name) = &field_expr.inner {
                    // Find the variable in scope and set cleanup_mode to CLEANUP_NONE
                    for scope in self.variables.iter_mut().rev() {
                        if let Some((_, _, cleanup_mode)) = scope.get_mut(var_name) {
                            *cleanup_mode = crate::compiler::codegen::CLEANUP_NONE;
                            break;
                        }
                    }
                }
            }

            let field_ptr = self
                .builder
                .build_struct_gep(
                    *struct_type,
                    struct_ptr,
                    field_idx as u32,
                    &format!("{}.{}", name, field_name),
                )
                .map_err(|e| e.to_string())?;

            // Store the value directly (move semantics - no deep clone needed since we're transferring ownership)
            // For scalar types, just store the value.
            // For pointer types (Tensor, Struct, etc.), store the pointer - ownership is transferred.
            self.builder
                .build_store(field_ptr, val)
                .map_err(|e| e.to_string())?;
        }


        // Return the pointer directly (no load)
        Ok((struct_ptr.into(), Type::Struct(name.to_string(), generics.to_vec())))
    }

}
