use crate::compiler::error::{TlError, CodegenErrorKind};
pub mod types;
mod tensor_methods;
mod builtin_fns;
mod struct_ops;
mod method_registry;

pub use types::*;
pub(crate) use builtin_fns::cast_value_to_i64;
use builtin_fns::*;

use super::CodeGenerator;
use crate::compiler::ast::*;

use inkwell::values::*;
use inkwell::types::BasicType;
use std::collections::{HashMap, HashSet};

// ── BuiltinManager: new() / register_all() ──────────────────────────────────
// 構造体・get・register_eval・register_uneval は types.rs で定義済み。
// new() は register_all() 経由で関数を登録するため mod.rs 側に置く。
impl BuiltinManager {
    pub fn new() -> Self {
        let mut manager = BuiltinManager {
            functions: HashMap::new(),
        };
        manager.register_all();
        manager
    }

    fn register_all(&mut self) {
        // IO functions
        self.register_uneval("print", compile_print_uneval);
        self.register_uneval("println", compile_println_uneval);
        self.register_uneval("format", compile_format_uneval);
        self.register_uneval("read_line", compile_read_line_uneval);

        // Panic function - diverging, never returns
        self.register_uneval("panic", compile_panic_uneval);

        // Assert function
        self.register_uneval("assert", compile_assert_uneval);

        // Command line arguments
        self.register_eval("args_count", compile_args_count);
        self.register_eval("args_get", compile_args_get);

        // String functions
        self.register_eval("char_at", compile_string_char_at);
        self.register_eval("len", compile_string_len);

        self.register_uneval("varbuilder_get", compile_varbuilder_get);
    }
}


// System Static Methods
fn compile_system_method<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    fn_name: &str,
    ret_type: Type,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = codegen.module.get_function(fn_name).ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("{} not found", fn_name))))?;
    let call = codegen.builder.build_call(fn_val, &[], "sys_call").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err(TlError::from(CodegenErrorKind::Internal(format!("Invalid return from {}", fn_name)))),
    };
    Ok((res, ret_type))
}

pub fn compile_system_memory_mb<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if !args.is_empty() { return Err(CodegenErrorKind::Internal("System::memory_mb takes no arguments".to_string()).into()); }
    compile_system_method(codegen, "tl_get_memory_mb", Type::I64)
}

pub fn compile_system_metal_pool_bytes<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if !args.is_empty() { return Err(CodegenErrorKind::Internal("System::metal_pool_bytes takes no arguments".to_string()).into()); }
    compile_system_method(codegen, "tl_get_metal_pool_bytes", Type::I64)
}

pub fn compile_system_metal_pool_mb<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if !args.is_empty() { return Err(CodegenErrorKind::Internal("System::metal_pool_mb takes no arguments".to_string()).into()); }
    compile_system_method(codegen, "tl_get_metal_pool_mb", Type::I64)
}

pub fn compile_system_metal_pool_count<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if !args.is_empty() { return Err(CodegenErrorKind::Internal("System::metal_pool_count takes no arguments".to_string()).into()); }
    compile_system_method(codegen, "tl_get_metal_pool_count", Type::I64)
}

pub fn compile_system_metal_sync<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if !args.is_empty() { return Err(CodegenErrorKind::Internal("System::metal_sync takes no arguments".to_string()).into()); }
    compile_system_method(codegen, "tl_metal_synchronize", Type::Void)
}

pub fn compile_system_pool_count<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if !args.is_empty() { return Err(CodegenErrorKind::Internal("System::pool_count takes no arguments".to_string()).into()); }
    compile_system_method(codegen, "tl_get_pool_count", Type::I64)
}

pub fn compile_system_refcount_count<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if !args.is_empty() { return Err(CodegenErrorKind::Internal("System::refcount_count takes no arguments".to_string()).into()); }
    compile_system_method(codegen, "tl_get_refcount_count", Type::I64)
}

pub fn compile_system_scope_depth<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if !args.is_empty() { return Err(CodegenErrorKind::Internal("System::scope_depth takes no arguments".to_string()).into()); }
    compile_system_method(codegen, "tl_get_scope_depth", Type::I64)
}


pub fn compile_path_exists<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 { return Err(CodegenErrorKind::Internal("Path::exists requires 1 argument".to_string()).into()); }
    let fn_val = codegen.module.get_function("tl_path_exists").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_path_exists not found".to_string())))?;
    let call = codegen.builder.build_call(fn_val, &[args[0].0.into()], "path_exists").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v, // Bool return type?
        _ => return Err(CodegenErrorKind::Internal("Invalid return from Path::exists".to_string()).into()),
    };
    // Original code checks return loop handling for Bool?
    // Line 4886 says: return Ok((res, Type::Bool));
    // So tl_path_exists returns i1 (bool) directly? 
    // Wait, checked the earlier view - yes, it returned res (v) directly as Type::Bool.
    Ok((res, Type::Bool))
}


// ======================================================================================
//                                  Registered Methods
// ======================================================================================

impl<'ctx> CodeGenerator<'ctx> {

    #[allow(deprecated)]
    pub fn load_struct_i64_field(
        &mut self,
        obj_val: BasicValueEnum<'ctx>,
        obj_ty: &Type,
        field_name: &str,
    ) -> Result<BasicValueEnum<'ctx>, TlError> {
        let struct_name = match obj_ty {
            Type::Struct(name, _) => name.clone(),

            Type::String(_) => "String".to_string(),
            _ => return Err(TlError::from(CodegenErrorKind::Internal(format!("Expected struct type for field {} (got {:?})", field_name, obj_ty)))),
        };

        let simple_struct_name = struct_name.as_str();

        if simple_struct_name == "String" {
            if field_name == "ptr" {
                 // StringStruct is { ptr: *mut c_char, len: i64 }
                 // We need to load the ptr field (index 0) as a pointer
                 let ptr = obj_val.into_pointer_value();
                 
                 // Define StringStruct layout
                 let str_struct_ty = self.context.struct_type(&[
                     self.context.ptr_type(inkwell::AddressSpace::default()).into(), // ptr
                     self.context.i64_type().into(), // len
                 ], false);
                 
                 // GEP to get pointer to ptr field (index 0)
                 let ptr_field_ptr = self.builder
                     .build_struct_gep(str_struct_ty, ptr, 0, "str_ptr_field")
                     .map_err(|_| TlError::from(CodegenErrorKind::Internal("Failed to GEP String.ptr".to_string())))?;
                 
                 // Load the ptr value as pointer type
                 let ptr_val = self.builder
                     .build_load(self.context.ptr_type(inkwell::AddressSpace::default()), ptr_field_ptr, "str_ptr_val")
                     .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                 
                 // Convert pointer to i64 for callers that expect i64
                 let ptr_as_i64 = self.builder
                     .build_ptr_to_int(ptr_val.into_pointer_value(), self.context.i64_type(), "str_ptr_i64")
                     .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                 
                 return Ok(ptr_as_i64.into());
            } else if field_name == "len" || field_name == "cap" {
                 // Basic String struct might have len/cap? 
                 // Current implementation seems to rely on runtime functions for len.
                 // io.rs only asks for "ptr".
                 return Err(TlError::from(CodegenErrorKind::Internal("String len/cap not directly accessible via field".to_string())));
            }
        }

        let struct_def = self
            .struct_defs
            .get(simple_struct_name)
            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Struct definition for {} not found", struct_name))))?;

        let field_idx = struct_def
            .fields
            .iter()
            .position(|(n, _)| n == field_name)
            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!(
                "Field {} not found in struct {}",
                field_name, struct_name
            ))))?;

        if obj_val.is_pointer_value() {
            let ptr = obj_val.into_pointer_value();
            let st_llvm_ty = self
                .struct_types
                .get(simple_struct_name)
                .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Struct type {} not found", struct_name))))?;

            let field_ptr = self
                .builder
                .build_struct_gep(
                    *st_llvm_ty,
                    ptr,
                    field_idx as u32,
                    &format!("ptr_{}", field_name),
                )
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            let loaded = self
                .builder
                .build_load(self.context.i64_type(), field_ptr, field_name)
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            Ok(loaded)
        } else if obj_val.is_struct_value() {
            let struct_val = obj_val.into_struct_value();
            let extracted = self
                .builder
                .build_extract_value(struct_val, field_idx as u32, field_name)
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            Ok(extracted)
        } else {
            Err(TlError::from(CodegenErrorKind::Internal("Cannot access field of non-pointer and non-struct value".to_string())))
        }
    }

    pub(crate) fn is_safe_to_free(&self, expr: &Expr, ty: &Type) -> bool {
        match ty {
            Type::Tensor(_, _) => {
                // Tensors originating from expressions (not variables) are always new allocations (R-values)
                match &expr.inner {
                    ExprKind::Variable(_) | ExprKind::FieldAccess(_, _) => false,
                    _ => true,
                }
            }
            Type::Struct(_, _) => {
                match &expr.inner {
                    // Fresh allocations are safe to free
                    ExprKind::StaticMethodCall(_, _, _) | ExprKind::StructInit(_, _) => true,
                    // Variables and Fields are L-values, not safe to free
                    ExprKind::Variable(_) | ExprKind::FieldAccess(_, _) => false,
                    // Method calls: Check recursively if the receiver was safe to free.
                    // If obj is temporary, obj.method() result is treated as temporary (part of obj).
                    // If obj is strictly safe (fresh), propagation says result is safe.
                    // BUT: We will apply runtime check too.
                    ExprKind::MethodCall(obj, _, _) => self.is_safe_to_free(obj, ty),
                    // Other expressions?
                    _ => false,
                }
            }
            _ => false,
        }
    }

    /// Register a tensor with the memory manager for automatic cleanup
    /// This should be called for intermediate expression results (BinOp, MethodCall, etc.)
    /// The tensor will be freed when exit_scope is called, unless it's unregistered first
    pub(crate) fn emit_register_tensor(
        &self,
        val: BasicValueEnum<'ctx>,
        ty: &Type,
    ) -> Result<(), TlError> {
        // Only register tensors
        if !matches!(ty, Type::Tensor(_, _)) {
            return Ok(());
        }

        // V4.5: Use tl_tensor_register (mapped to backend-specific implementation)
        let reg_fn = self
            .module
            .get_function("tl_tensor_register")
            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_tensor_register not found".to_string())))?;

        let ptr = val.into_pointer_value();
        let cast_ptr = self
            .builder
            .build_pointer_cast(
                ptr,
                self.context.ptr_type(inkwell::AddressSpace::default()),
                "reg_tensor_ptr",
            )
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        self.builder
            .build_call(reg_fn, &[cast_ptr.into()], "")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        Ok(())
    }

    /// Unregister a tensor from the memory manager (ownership transfer)
    /// This should be called when a tensor is moved to a variable, struct field, or return value
    pub(crate) fn gen_save_struct(
        &mut self,
        map: inkwell::values::BasicValueEnum<'ctx>,
        struct_ptr: inkwell::values::BasicValueEnum<'ctx>,
        struct_name: &str,
        prefix: String,
    ) -> Result<(), TlError> {
        // Extract simple name from module path (e.g., "mnist_common::Linear" -> "Linear")
        let simple_name = struct_name;

        let def = self
            .struct_defs
            .get(simple_name)
            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Struct definition '{}' not found", struct_name))))?;

        let struct_ty = *self
            .struct_types
            .get(simple_name)
            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("Struct LLVM type not found".to_string())))?;

        let fields = def.fields.clone();
        for (i, (field_name, field_type)) in fields.iter().enumerate() {
            let full_key = if prefix.is_empty() {
                field_name.clone()
            } else {
                format!("{}.{}", prefix, field_name)
            };

            let ptr = struct_ptr.into_pointer_value();
            let field_ptr = self
                .builder
                .build_struct_gep(struct_ty, ptr, i as u32, field_name)
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            match field_type {
                Type::Tensor(_, _) => {
                    // Save Tensor
                    let tensor_ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
                    let t_val = self
                        .builder
                        .build_load(tensor_ptr_ty, field_ptr, field_name)
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    let (key_val, _) = self.compile_string_literal(&full_key)
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(format!("Failed to compile key literal: {}", e))))?;

                    let insert_fn = self
                        .module
                        .get_function("tl_tensor_map_insert")
                        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_tensor_map_insert not found".to_string())))?;
                    let _ = self
                        .builder
                        .build_call(insert_fn, &[map.into(), key_val.into(), t_val.into()], "")
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                }
                Type::Struct(sub_name, _) => {
                    if sub_name == "String" { panic!("Struct(String) found in codegen save"); }
                    // Recurse
                    let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
                    let sub_val = self
                        .builder
                        .build_load(ptr_ty, field_ptr, "sub_ptr")
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    self.gen_save_struct(map, sub_val, sub_name, full_key)?;
                }
                _ => {
                    // Skip primitives
                }
            }
        }
        Ok(())
    }

    pub(crate) fn gen_load_struct(
        &mut self,
        map: inkwell::values::BasicValueEnum<'ctx>,
        struct_ptr: inkwell::values::BasicValueEnum<'ctx>,
        struct_name: &str,
        prefix: String,
    ) -> Result<(), TlError> {
        // Extract simple name from module path
        let simple_name = struct_name;

        let def = self
            .struct_defs
            .get(simple_name)
            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Struct definition '{}' not found", struct_name))))?;

        let struct_ty = *self
            .struct_types
            .get(simple_name)
            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("Struct LLVM type not found".to_string())))?;

        let fields = def.fields.clone();
        for (i, (field_name, field_type)) in fields.iter().enumerate() {
            let full_key = if prefix.is_empty() {
                field_name.clone()
            } else {
                format!("{}.{}", prefix, field_name)
            };

            let ptr = struct_ptr.into_pointer_value();
            let field_ptr = self
                .builder
                .build_struct_gep(struct_ty, ptr, i as u32, field_name)
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            match field_type {
                Type::Tensor(_, _) => {
                    // Load Tensor
                    let (key_val, _) = self.compile_string_literal(&full_key)
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(format!("Failed to compile key literal: {}", e))))?;

                    let get_fn = self
                        .module
                        .get_function("tl_tensor_map_get")
                        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_tensor_map_get not found".to_string())))?;
                    let call = self
                        .builder
                        .build_call(get_fn, &[map.into(), key_val.into()], "t_val")
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                    let t_val = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err(CodegenErrorKind::Internal("tl_tensor_map_get returned inst/void".to_string()).into()),
                    };

                    self.builder
                        .build_store(field_ptr, t_val)
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                }
                Type::Struct(sub_name, _) => {
                    if sub_name == "String" { panic!("Struct(String) found in codegen load"); }
                    // Recurse: load the pointer to inner struct
                    let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
                    let sub_val = self
                        .builder
                        .build_load(ptr_ty, field_ptr, "sub_ptr")
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    self.gen_load_struct(map, sub_val, sub_name, full_key)?;
                }
                _ => {
                    // Skip primitives
                }
            }
        }
        Ok(())
    }

    pub(crate) fn gen_register_params(
        &mut self,
        struct_ptr: inkwell::values::BasicValueEnum<'ctx>,
        struct_name: &str,
        prefix: String,
    ) -> Result<(), TlError> {
        let simple_name = struct_name;

        let def = self
            .struct_defs
            .get(simple_name)
            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Struct definition '{}' not found", struct_name))))?;

        let struct_ty = *self
            .struct_types
            .get(simple_name)
            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("Struct LLVM type not found".to_string())))?;

        let fields = def.fields.clone();
        for (i, (field_name, field_type)) in fields.iter().enumerate() {
            let full_key = if prefix.is_empty() {
                field_name.clone()
            } else {
                format!("{}.{}", prefix, field_name)
            };

            let ptr = struct_ptr.into_pointer_value();
            let field_ptr = self
                .builder
                .build_struct_gep(struct_ty, ptr, i as u32, field_name)
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            match field_type {
                Type::Tensor(_, _) => {
                    // Register Tensor: tl_add_parameter(name, tensor)
                    let tensor_ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
                    let t_val = self
                        .builder
                        .build_load(tensor_ptr_ty, field_ptr, field_name)
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                    let (key_val, _) = self.compile_string_literal(&full_key)
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(format!("Failed to compile key literal: {}", e))))?;

                    let add_fn = self
                        .module
                        .get_function("tl_add_parameter")
                        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_add_parameter not found".to_string())))?;
                    self.builder
                        .build_call(add_fn, &[key_val.into(), t_val.into()], "")
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                }
                Type::Struct(sub_name, _) => {
                    // Recurse for Type::Struct as well (e.g. from generic instantiation)
                    let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
                    let sub_val = self
                        .builder
                        .build_load(ptr_ty, field_ptr, "sub_ptr")
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    self.gen_register_params(sub_val, sub_name, full_key)?;
                }
                _ => {
                    // Skip primitives
                }
            }
        }
        Ok(())
    }
    pub(crate) fn is_last_use(&self, name: &str) -> bool {
        for scope in self.variable_liveness.iter().rev() {
            if let Some(&last_use) = scope.get(name) {
                let res = last_use != 0 && self.current_time >= last_use;
                return res;
            }
        }
        false
    }


    fn emit_retain(&mut self, val: BasicValueEnum<'ctx>, ty: &Type) -> Result<(), TlError> {
        self.emit_recursive_retain(val, ty)?;
        Ok(())
    }

    fn extract_inner_ty(&self, obj_ty: &Type) -> Type {
        let (name, targs) = match obj_ty {
            Type::Struct(n, t) => (n.clone(), t.clone()),
            Type::SpecializedType { gen_type, type_args, .. } => (gen_type.get_base_name(), type_args.clone()),
            _ => return Type::I64,
        };
        if targs.is_empty() && name.contains('[') {
            let parsed_ty = crate::compiler::mangler::MANGLER.parse_type_str(&name);
            match parsed_ty {
                Type::Struct(_, parsed_args) => parsed_args.first().cloned().unwrap_or(Type::I64),
                _ => Type::I64,
            }
        } else {
            targs.first().cloned().unwrap_or(Type::I64)
        }
    }

    pub(crate) fn compile_expr(
        &mut self,
        expr: &Expr,
    ) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
        self.current_time += 1;
        let prev_span = self.current_span.clone();
        self.current_span = Some(expr.span.clone());
        let mut result = self.compile_expr_inner(expr)?;
        
        // V6.1: Dynamically concretize type leakage from semantics phase inside generic monomorphization.
        if let Some(subst) = &self.current_method_generics {
            let orig_ty = result.1.clone();
            result.1 = self.substitute_type_simple_bind(&result.1, subst);
            if orig_ty != result.1 {
            }
        }
        
        self.current_span = prev_span;
        Ok(result)
    }

    pub(crate) fn compile_expr_inner(
        &mut self,
        expr: &Expr,
    ) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
        match &expr.inner {
            ExprKind::Block(stmts) => {
                self.enter_scope();
                let mut last_val = None;
                for (i, stmt) in stmts.iter().enumerate() {
                    if i == stmts.len() - 1 {
                        if let StmtKind::Expr(e) = &stmt.inner {
                            last_val = Some(self.compile_expr(e)?);
                        } else {
                            self.compile_stmt(stmt)?;
                        }
                    } else {
                        self.compile_stmt(stmt)?;
                    }
                }

                let final_res = last_val.unwrap_or((
                    self.context.i64_type().const_int(0, false).into(),
                    Type::Void,
                ));

                // FIX UAF: Unregister block result before scope cleanup
                if matches!(
                    final_res.1,
                    Type::Tensor(_, _) | Type::Struct(_, _) | Type::Tuple(_)
                ) {
                    if let Some(unreg_fn) = self.module.get_function("tl_mem_unregister") {
                        let ptr = final_res.0.into_pointer_value();
                        let cast_ptr = self
                            .builder
                            .build_pointer_cast(
                                ptr,
                                self.context.ptr_type(inkwell::AddressSpace::default()),
                                "cast_unreg",
                            )
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        self.builder
                            .build_call(unreg_fn, &[cast_ptr.into()], "")
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    }
                }

                self.exit_scope();

                Ok(final_res)
            }
            ExprKind::Int(i) => {
                let i64_type = self.context.i64_type();
                Ok((i64_type.const_int(*i as u64, true).into(), Type::I64))
            }
            ExprKind::Float(f) => {
                let f32_type = self.context.f32_type();
                Ok((f32_type.const_float(*f).into(), Type::F32))
            }
            ExprKind::Bool(b) => {
                let bool_type = self.context.bool_type();
                Ok((
                    bool_type.const_int(if *b { 1 } else { 0 }, false).into(),
                    Type::Bool,
                ))
            }
            ExprKind::CharLiteral(c) => {
                let i32_type = self.context.i32_type();
                Ok((i32_type.const_int(*c as u32 as u64, false).into(), Type::Char("Char".to_string())))
            }
            ExprKind::StaticConstAccess(ty, method_or_const) => {
                match (ty, method_or_const.as_str()) {
                    (Type::F64, "INFINITY") => Ok((self.context.f64_type().const_float(std::f64::INFINITY).into(), Type::F64)),
                    (Type::F64, "NEG_INFINITY") => Ok((self.context.f64_type().const_float(std::f64::NEG_INFINITY).into(), Type::F64)),
                    (Type::F32, "INFINITY") => Ok((self.context.f32_type().const_float(std::f64::INFINITY).into(), Type::F32)),
                    (Type::F32, "NEG_INFINITY") => Ok((self.context.f32_type().const_float(std::f64::NEG_INFINITY).into(), Type::F32)),
                    (Type::I64, "MAX") => Ok((self.context.i64_type().const_int(std::i64::MAX as u64, true).into(), Type::I64)),
                    (Type::I64, "MIN") => Ok((self.context.i64_type().const_int(std::i64::MIN as u64, true).into(), Type::I64)),
                    (Type::I32, "MAX") => Ok((self.context.i32_type().const_int(std::i32::MAX as u64, true).into(), Type::I32)),
                    (Type::I32, "MIN") => Ok((self.context.i32_type().const_int(std::i32::MIN as u64, true).into(), Type::I32)),
                    _ => Err(TlError::from(CodegenErrorKind::UnsupportedOperation(format!("Unsupported static constant access: {}::{}", ty.get_base_name(), method_or_const))))
                }
            }
            ExprKind::StringLiteral(s) => self.compile_string_literal(s),
            ExprKind::Symbol(name) => {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                std::hash::Hash::hash(name, &mut hasher);
                let seed = std::hash::Hasher::finish(&hasher);
                let i64_type = self.context.i64_type();
                Ok((i64_type.const_int(seed, false).into(), Type::Entity))
            }
            ExprKind::LogicVar(name) => {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                std::hash::Hash::hash(name, &mut hasher);
                let seed = std::hash::Hasher::finish(&hasher);
                let i64_type = self.context.i64_type();
                Ok((i64_type.const_int(seed, false).into(), Type::Entity))
            }
            ExprKind::EnumInit {
                enum_name,
                variant_name,
                generics: original_generics,
                payload,
            } => {
                let generics: Vec<Type> = original_generics.clone();
                // 0. specialized name handling
                // Extract base name from mangled name (e.g., "Option_i64" -> "Option")
                // and parse generics from mangled suffix if generics is empty
                let (base_name, inferred_generics) = if generics.is_empty() && enum_name.contains('[') {
                    let base = mangle_base_name(enum_name).to_string();
                    (base, generics.clone())
                } else {
                    (enum_name.clone(), generics.clone())
                };
                
                let mangled_name = if inferred_generics.is_empty() {
                    enum_name.clone()
                } else {
                    self.mangle_type_name(&base_name, &inferred_generics)
                };

                // 1. On-demand Monomorphization
                // First, try to find already-monomorphized enum by mangled_name
                let mut enum_def = if let Some(def) = self.enum_defs.get(&mangled_name) {
                    def.clone()
                } else if let Some(def) = self.enum_defs.get(enum_name) {
                    // Found exact enum_name (might be mangled already)
                    def.clone()
                } else if let Some(def) = self.enum_defs.get(&base_name) {
                    def.clone()
                } else {
                    return Err(TlError::from(CodegenErrorKind::Internal(format!("Enum def {} not found (tried: {}, {}, {})",
                        enum_name, mangled_name, base_name, enum_name))));
                };
                
                // If the found enum_def is still generic, monomorphize with inferred or default types
                if !enum_def.generics.is_empty() {
                    let actual_generics = inferred_generics.clone();
                    // Generate generic array logic removed. Assuming AOT provides concrete types.
                    if actual_generics.is_empty() {
                         return Err(TlError::from(CodegenErrorKind::Internal(format!("codegen error: Enum {}::{} lacks generic parameters and type inference could not resolve them. AOT is missing generics.", base_name, variant_name))));
                    }

                    if actual_generics.is_empty() {
                        // NOTE: If generic arguments are completely missing, they may have failed to propagate during AST substitution.
                        // See semantics.rs `StmtKind::Return` for correct resolution propagation.
                        eprintln!("Codegen ERROR EnumInit: base_name={}, variant={}, original_generics.len()={}", base_name, variant_name, original_generics.len()); 
                        return Err(TlError::from(CodegenErrorKind::Internal(format!("Enum {}::{} lacks generic parameters and type inference could not resolve them. Implicit fallback is strictly prohibited.", base_name, variant_name))));
                    };
                    let actual_mangled = self.mangle_type_name(&base_name, &actual_generics);
                    
                    // Try to find already-monomorphized version first
                    if let Some(specialized_def) = self.enum_defs.get(&actual_mangled) {
                        enum_def = specialized_def.clone();
                    } else {
                        // Monomorphize on-demand
                        self.monomorphize_enum(&base_name, &actual_generics).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        enum_def = self.enum_defs.get(&actual_mangled)
                            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Monomorphization failed for {} -> {}", base_name, actual_mangled))))?
                            .clone();
                    }
                };

                // 2. Ensure enum_type exists
                // Note: After default type monomorphization, enum_def.name may be "Option<i64>" style
                let enum_ty = if let Some(ty) = self.enum_types.get(&enum_def.name) {
                    // Use the name from enum_def (might be monomorphized name)
                    *ty
                } else if let Some(ty) = self.enum_types.get(&mangled_name) {
                    *ty
                } else if let Some(ty) = self.enum_types.get(enum_name) {
                    *ty
                } else {
                    // Need to compile this enum type on-demand
                    // This should work because enum_def has generics=[] after monomorphization
                    self.compile_enum_defs(&[enum_def.clone()])?;
                    *self.enum_types.get(&enum_def.name)
                        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Failed to compile enum type {} (from {}), generics={:?}", enum_def.name, enum_name, enum_def.generics))))?
                };

                // 2. Allocate Enum
                // Manual malloc(size)
                // Use mangled type for size calculation
                let size_ptr = unsafe {
                    self.builder.build_gep(
                        enum_ty,
                        self.context.ptr_type(inkwell::AddressSpace::default()).const_null(),
                        &[self.context.i64_type().const_int(1, false)],
                        "size_ptr",
                    ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                };
                let size = self.builder
                    .build_ptr_to_int(size_ptr, self.context.i64_type(), "size")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                let malloc_fn = self.module.get_function("malloc").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("malloc not found".to_string())))?;
                let malloc_ptr = match self.builder
                    .build_call(malloc_fn, &[size.into()], &format!("enum_{}", mangled_name))
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                    .try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
                        _ => return Err(CodegenErrorKind::Internal("malloc returned void".to_string()).into()),
                    };
                // malloc_ptr is *EnumStruct
                
                // Cast to EnumType* (which is WaitStruct*)
                let alloca = self.builder.build_pointer_cast(
                    malloc_ptr,
                    self.context.ptr_type(inkwell::AddressSpace::default()),
                    "enum_cast"
                ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // 2. Store Tag
                let variant_idx = enum_def
                    .variants
                    .iter()
                    .position(|v| v.name == *variant_name)
                    .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("unexpected None".to_string())))?;
                let tag_ptr = self
                    .builder
                    .build_struct_gep(enum_ty, alloca, 0, "tag_ptr")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder
                    .build_store(
                        tag_ptr,
                        self.context.i32_type().const_int(variant_idx as u64, false),
                    )
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // 3. Store Fields
                let variant_def = &enum_def.variants[variant_idx];
                
                // Helper to compile field storage
                let _compile_fields = |fields_def: &Vec<Type>, exprs: &Vec<Expr>, field_names: Option<&Vec<String>>| -> Result<(), TlError> {
                    let payload_ptr_raw = self
                        .builder
                        .build_struct_gep(enum_ty, alloca, 1, "payload_ptr_raw")
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                    // Reconstruct Variant Struct Type
                    let mut field_types: Vec<inkwell::types::BasicTypeEnum> = vec![];
                    for ty in fields_def {
                        let llvm_ty = match ty {
                            Type::F32 => self.context.f32_type().into(),
                            Type::F64 => self.context.f64_type().into(),
                            Type::I64 | Type::Usize | Type::Entity => self.context.i64_type().into(),
                            Type::I32 | Type::Char(_) => self.context.i32_type().into(),
                            Type::U8 => self.context.i8_type().into(),
                            Type::Bool => self.context.bool_type().into(),
                            Type::Tensor(_, _) | Type::TensorShaped(_, _) | Type::GradTensor(_, _)
                            | Type::Struct(_, _) | Type::Enum(_, _)
                            | Type::String(_) | Type::Ptr(_) | Type::Tuple(_) | Type::SpecializedType { .. } => self
                                .context
                                .ptr_type(inkwell::AddressSpace::default())
                                .into(),
                            Type::Void => self.context.i8_type().into(),
                            Type::Array(inner, size) => self.get_llvm_type(&Type::Array(inner.clone(), *size)).unwrap_or(self.context.i64_type().into()),
                            Type::Path(_, _) | Type::Fn(_, _) | Type::I8 | Type::I16 | Type::U16 | Type::U32 | Type::U64 | Type::F16 | Type::BF16 | Type::TypeVar(_) | Type::Never | Type::Undefined(_) | Type::Range => self.context.i64_type().into(), Type::TraitObject(_) => { let p = self.context.ptr_type(inkwell::AddressSpace::default()); self.context.struct_type(&[p.into(), p.into()], false).into() },
                        };
                        field_types.push(llvm_ty);
                    }
                    let variant_struct_ty = self.context.struct_type(&field_types, false);

                    let payload_ptr = self
                        .builder
                        .build_pointer_cast(
                            payload_ptr_raw,
                            self.context.ptr_type(inkwell::AddressSpace::default()),
                            "payload_cast",
                        )
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                    // Iterate over DEFINITION fields
                    for (idx, f_ty) in fields_def.iter().enumerate() {
                        // Find expression
                        let expr = if let Some(_names) = field_names {
                             // Struct variant: find by name
                             // exprs is actually not Vec<Expr>, but we pass it as such? 
                             // Wait, payload closure arg needs adjustment.
                             // Actually, let's handle struct/tuple separately in the match below to avoid closure complexity.
                             // Placeholder to compile
                             return Err(TlError::from(CodegenErrorKind::UnsupportedOperation("Struct variant not supported in closure helper".to_string())));
                        } else {
                             // Tuple variant: by index
                             &exprs[idx]
                        };

                        let (val, _) = self.compile_expr(expr)?;
                        
                        // Deep clone logic
                        let is_rvalue = matches!(
                            &expr.inner,
                            ExprKind::FnCall(_, _)
                                | ExprKind::MethodCall(_, _, _)
                                | ExprKind::StaticMethodCall(_, _, _)
                                | ExprKind::BinOp(_, _, _)
                                | ExprKind::UnOp(_, _)
                                | ExprKind::TensorLiteral(_)
                                | ExprKind::Block(_)
                                | ExprKind::Int(_)
                                | ExprKind::Float(_)
                                | ExprKind::Bool(_)
                                | ExprKind::StringLiteral(_)
                        );
                        let mut stored_val = val;
                        let should_deep_clone = match f_ty {
                            Type::Tensor(_, _) => !is_rvalue,
                            Type::Struct(_, _) => !is_rvalue,
                            _ => false,
                        };
                        if should_deep_clone {
                             stored_val = self.emit_deep_clone(val, f_ty)?;
                        }

                        let f_ptr = self
                            .builder
                            .build_struct_gep(
                                variant_struct_ty,
                                payload_ptr,
                                idx as u32,
                                "field_ptr",
                            )
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                        self.builder.build_store(f_ptr, stored_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    }
                    Ok(())
                };

                match (&variant_def.kind, payload) {
                     (crate::compiler::ast::VariantKind::Unit, crate::compiler::ast::EnumVariantInit::Unit) => {
                         // No fields to store
                     },
                     (crate::compiler::ast::VariantKind::Tuple(types), crate::compiler::ast::EnumVariantInit::Tuple(exprs)) => {
                        let payload_ptr_raw = self
                            .builder
                            .build_struct_gep(enum_ty, alloca, 1, "payload_ptr_raw")
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                        // Reconstruct Variant Struct Type
                        let mut field_types_llvm: Vec<inkwell::types::BasicTypeEnum> = vec![];
                        for ty in types {
                             let llvm_ty = match ty {
                                Type::F32 => self.context.f32_type().into(),
                                Type::F64 => self.context.f64_type().into(),
                                Type::I64 | Type::Usize | Type::Entity => self.context.i64_type().into(),
                                Type::I32 | Type::Char(_) => self.context.i32_type().into(),
                                Type::U8 => self.context.i8_type().into(),
                                Type::Bool => self.context.bool_type().into(),
                                Type::Struct(_, _) | Type::Enum(_, _)
                                | Type::String(_) | Type::Tensor(_, _)
                                | Type::TensorShaped(_, _) | Type::GradTensor(_, _)
                                | Type::Ptr(_) | Type::Tuple(_) | Type::SpecializedType { .. } => self
                                    .context
                                    .ptr_type(inkwell::AddressSpace::default())
                                    .into(),
                                Type::Void => self.context.i8_type().into(),
                                Type::Array(inner, size) => self.get_llvm_type(&Type::Array(inner.clone(), *size)).unwrap_or(self.context.i64_type().into()),
                                Type::Path(_, _) | Type::Fn(_, _) | Type::I8 | Type::I16 | Type::U16 | Type::U32 | Type::U64 | Type::F16 | Type::BF16 | Type::TypeVar(_) | Type::Never | Type::Undefined(_) | Type::Range => self.context.i64_type().into(), Type::TraitObject(_) => { let p = self.context.ptr_type(inkwell::AddressSpace::default()); self.context.struct_type(&[p.into(), p.into()], false).into() },
                            };
                            field_types_llvm.push(llvm_ty);
                        }

                        let variant_struct_ty = self.context.struct_type(&field_types_llvm, false);

                        let payload_ptr = self
                            .builder
                            .build_pointer_cast(
                                payload_ptr_raw,
                                self.context.ptr_type(inkwell::AddressSpace::default()),
                                "payload_cast",
                            )
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        
                        for (idx, (f_ty, expr)) in types.iter().zip(exprs.iter()).enumerate() {
                             let (val, _) = self.compile_expr(expr)?;
                             
                            // Move semantics for pointer types:
                            // When a variable is used as enum payload, we MOVE ownership.
                            let is_moveable_type = match f_ty {
                                Type::Tensor(_, _) => true,
                                Type::Struct(n, args) => {
                                    // Primitive types don't need move semantics
                                    !(args.is_empty() && 
                                      (n == "I64" || n == "F64" || n == "I32" || n == "F32" || n == "Bool" || 
                                       n == "i64" || n == "f64" || n == "i32" || n == "f32" || n == "bool"))
                                }
                                Type::Enum(_, _) | Type::Tuple(_) | Type::SpecializedType { .. } => true,
                                _ => false,
                            };

                            
                            // If field_expr is a Variable, mark it as moved (disable cleanup)
                            if is_moveable_type {
                                if let ExprKind::Variable(var_name) = &expr.inner {
                                    for scope in self.variables.iter_mut().rev() {
                                        if let Some((_, _, cleanup_mode)) = scope.get_mut(var_name) {
                                            *cleanup_mode = crate::compiler::codegen::CLEANUP_NONE;
                                            break;
                                        }
                                    }
                                }
                            }
                            
                            let f_ptr = self.builder.build_struct_gep(variant_struct_ty, payload_ptr, idx as u32, "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                            self.builder.build_store(f_ptr, val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        }

                     },
                     (crate::compiler::ast::VariantKind::Struct(fields_def), crate::compiler::ast::EnumVariantInit::Struct(exprs)) => {
                        let payload_ptr_raw = self
                            .builder
                            .build_struct_gep(enum_ty, alloca, 1, "payload_ptr_raw")
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        
                        let mut field_types_llvm: Vec<inkwell::types::BasicTypeEnum> = vec![];
                        for (_, ty) in fields_def {
                             let llvm_ty = match ty {
                                Type::F32 => self.context.f32_type().into(),
                                Type::I64 => self.context.i64_type().into(),
                                Type::Bool => self.context.bool_type().into(),
                                Type::Tensor(_, _)
                                | Type::Struct(_, _)
                                | Type::Enum(_, _)
                                | Type::SpecializedType { .. } => self
                                    .context
                                    .ptr_type(inkwell::AddressSpace::default())
                                    .into(),
                                Type::String(_) => self
                                    .context
                                    .ptr_type(inkwell::AddressSpace::default())
                                    .into(),
                                Type::Char(_) => self.context.i32_type().into(),
                                Type::U8 => self.context.i8_type().into(),
                                Type::F64 => self.context.f64_type().into(),
                                Type::Usize | Type::Entity => self.context.i64_type().into(),
                                Type::I32 => self.context.i32_type().into(),
                                Type::Ptr(_) | Type::Tuple(_)
                                | Type::TensorShaped(_, _) | Type::GradTensor(_, _) => self
                                    .context
                                    .ptr_type(inkwell::AddressSpace::default())
                                    .into(),
                                Type::Void => self.context.i8_type().into(),
                                Type::Array(inner, size) => self.get_llvm_type(&Type::Array(inner.clone(), *size)).unwrap_or(self.context.i64_type().into()),
                                Type::Path(_, _) | Type::Fn(_, _) | Type::I8 | Type::I16 | Type::U16 | Type::U32 | Type::U64 | Type::F16 | Type::BF16 | Type::TypeVar(_) | Type::Never | Type::Undefined(_) | Type::Range => self.context.i64_type().into(), Type::TraitObject(_) => { let p = self.context.ptr_type(inkwell::AddressSpace::default()); self.context.struct_type(&[p.into(), p.into()], false).into() },
                            };
                            field_types_llvm.push(llvm_ty);
                        }
                        let variant_struct_ty = self.context.struct_type(&field_types_llvm, false);

                        let payload_ptr = self
                            .builder
                            .build_pointer_cast(
                                payload_ptr_raw,
                                self.context.ptr_type(inkwell::AddressSpace::default()),
                                "payload_cast",
                            )
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                            
                        for (idx, (f_name, f_ty)) in fields_def.iter().enumerate() {
                             let (_, expr) = exprs.iter().find(|(n, _)| n == f_name).ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Missing field {}", f_name))))?;
                             
                             let (val, _) = self.compile_expr(expr)?;
                             
                            let is_rvalue = matches!(
                                &expr.inner,
                                ExprKind::FnCall(_, _)
                                    | ExprKind::MethodCall(_, _, _)
                                    | ExprKind::StaticMethodCall(_, _, _)
                                    | ExprKind::BinOp(_, _, _)
                                    | ExprKind::UnOp(_, _)
                                    | ExprKind::TensorLiteral(_)
                                    | ExprKind::Block(_)
                                    | ExprKind::Int(_)
                                    | ExprKind::Float(_)
                                    | ExprKind::Bool(_)
                                    | ExprKind::StringLiteral(_)
                            );
                            let mut stored_val = val;
                            let should_deep_clone = match f_ty {
                                Type::Tensor(_, _) => !is_rvalue,
                                Type::Struct(_, _) => !is_rvalue,
                                _ => false,
                            };
                            if should_deep_clone {
                                 stored_val = self.emit_deep_clone(val, f_ty)?;
                            }
                            
                            let f_ptr = self.builder.build_struct_gep(variant_struct_ty, payload_ptr, idx as u32, "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                            self.builder.build_store(f_ptr, stored_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        }
                     },
                     _ => return Err(TlError::from(CodegenErrorKind::Internal(format!("Mismatch between variant definition {:?} and init payload {:?}", variant_def.kind, payload)))),
                }

                // Return the monomorphized type so downstream processing (like MethodCall) 
                // gets the correct generics. After monomorphization, enum_def.generics is empty
                // and enum_def.name contains the full name like "Option<i64>".
                // For proper type matching, we return (enum_def.name, []) as generics are baked into the name.
                Ok((alloca.into(), Type::Enum(enum_def.name.clone(), inferred_generics.clone())))
            }
            ExprKind::Match {
                expr: subject_expr,
                arms,
            } => self.compile_match_like(subject_expr, arms),
            ExprKind::IfLet {
                pattern,
                expr,
                then_block,
                else_block,
            } => {
                let mut arms: Vec<(Pattern, Expr)> = Vec::with_capacity(2);
                arms.push((
                    pattern.clone(),
                    Spanned::dummy(ExprKind::Block(then_block.clone())),
                ));
                let fallback =
                    Spanned::dummy(ExprKind::Block(else_block.clone().unwrap_or_default()));
                arms.push((Pattern::Wildcard, fallback));
                self.compile_match_like(expr, &arms)
            }

            ExprKind::Wildcard => {
                Err(TlError::from(CodegenErrorKind::Internal("ExprKind::Wildcard should only appear in logic rules".to_string())))
            }
            ExprKind::Range(_, _) => {
                Err(TlError::from(CodegenErrorKind::Internal("ExprKind::Range should only appear in For loops".to_string())))
            }

            ExprKind::TypeOf(_inner_expr, opt_ty) => {
                let inner_ty = opt_ty.as_ref().expect("typeof type must be inferred by semantic analyzer");
                
                // Pack into Type struct
                let type_name = format!("{:?}", inner_ty);
                let is_primitive = matches!(inner_ty, Type::I64 | Type::I32 | Type::F64 | Type::F32 | Type::Bool | Type::String(_));
                let is_ref_counted = matches!(inner_ty, Type::String(_) | Type::Tensor(_, _) | Type::TensorShaped(_, _) | Type::GradTensor(_, _));
                let size = 8; // placeholder size

                let struct_type = self.module.get_struct_type("Type").ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("struct type not found"))))?;
                
                // Heap allocate the struct
                let size_val_alloc = struct_type.size_of().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("type has no size".to_string())))?;
                let size_i64 = if size_val_alloc.get_type() == self.context.i32_type() {
                     self.builder.build_int_z_extend(size_val_alloc, self.context.i64_type(), "size_i64").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                } else {
                     size_val_alloc
                };
                let malloc_fn = self.get_fn("malloc")?;
                let call = self.builder.build_call(malloc_fn, &[size_i64.into()], "typeof_malloc").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let struct_ptr = match call.try_as_basic_value() {
                     inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
                     _ => panic!("malloc returned void"),
                 };

                // Build name (String)
                let (str_val, _) = self.compile_string_literal(&type_name).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // size
                let size_val = self.context.i64_type().const_int(size as u64, false);
                
                // is_primitive
                let is_prim_val = self.context.bool_type().const_int(if is_primitive { 1 } else { 0 }, false);

                // is_ref_counted
                let is_ref_val = self.context.bool_type().const_int(if is_ref_counted { 1 } else { 0 }, false);

                // Set fields
                // name: 0, size: 1, is_primitive: 2, is_ref_counted: 3
                let name_ptr = self.builder.build_struct_gep(struct_type, struct_ptr, 0, "name_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(name_ptr, str_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                let size_ptr = self.builder.build_struct_gep(struct_type, struct_ptr, 1, "size_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(size_ptr, size_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                let prim_ptr = self.builder.build_struct_gep(struct_type, struct_ptr, 2, "prim_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(prim_ptr, is_prim_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                let ref_ptr = self.builder.build_struct_gep(struct_type, struct_ptr, 3, "ref_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(ref_ptr, is_ref_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                Ok((struct_ptr.into(), Type::Struct("Type".to_string(), vec![])))
            }

            ExprKind::As(expr, target_type) => {
                let target_type = target_type.clone();
                let (val, source_type) = self.compile_expr(expr)?;
                if source_type == target_type {
                    return Ok((val, source_type));
                }

                match (&source_type, &target_type) {
                    (Type::I64, Type::F32) => {
                        let i = val.into_int_value();
                        let f = self
                            .builder
                            .build_signed_int_to_float(i, self.context.f32_type(), "cast")
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        Ok((f.into(), Type::F32))
                    }
                    (Type::F32, Type::I64) => {
                        let f = val.into_float_value();
                        let i = self
                            .builder
                            .build_float_to_signed_int(f, self.context.i64_type(), "cast")
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        Ok((i.into(), Type::I64))
                    }
                    (Type::I64, Type::F64) => {
                        let i = val.into_int_value();
                        let f = self
                            .builder
                            .build_signed_int_to_float(i, self.context.f64_type(), "cast")
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        Ok((f.into(), Type::F64))
                    }
                    (Type::F64, Type::I64) => {
                        let f = val.into_float_value();
                        let i = self
                            .builder
                            .build_float_to_signed_int(f, self.context.i64_type(), "cast")
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        Ok((i.into(), Type::I64))
                    }
                    (Type::F32, Type::F64) => {
                        let f = val.into_float_value();
                        let f64_val = self
                            .builder
                            .build_float_ext(f, self.context.f64_type(), "cast")
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        Ok((f64_val.into(), Type::F64))
                    }
                    (Type::F64, Type::F32) => {
                        let f = val.into_float_value();
                        let f32_val = self
                            .builder
                            .build_float_trunc(f, self.context.f32_type(), "cast")
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        Ok((f32_val.into(), Type::F32))
                    }
                    (Type::Bool, Type::I64) => {
                        let b = val.into_int_value();
                        let i = self
                            .builder
                            .build_int_z_extend(b, self.context.i64_type(), "cast")
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        Ok((i.into(), Type::I64))
                    }
                    (Type::I64, Type::Bool) => {
                        let i = val.into_int_value();
                        let zero = self.context.i64_type().const_zero();
                        let b = self
                            .builder
                            .build_int_compare(inkwell::IntPredicate::NE, i, zero, "cast")
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        Ok((b.into(), Type::Bool))
                    }
                    (Type::Bool, Type::F32) => {
                        let b = val.into_int_value();
                        let f = self
                            .builder
                            .build_unsigned_int_to_float(b, self.context.f32_type(), "cast")
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        Ok((f.into(), Type::F32))
                    }
                    (Type::F32, Type::Bool) => {
                        let f = val.into_float_value();
                        let zero = self.context.f32_type().const_zero();
                        let b = self
                            .builder
                            .build_float_compare(inkwell::FloatPredicate::UNE, f, zero, "cast")
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        Ok((b.into(), Type::Bool))
                    }
                    // Int to Ptr
                    (Type::I64, Type::Ptr(_)) => {
                        let i = val.into_int_value();
                        let ptr = self.builder.build_int_to_ptr(i, self.context.ptr_type(inkwell::AddressSpace::default()), "cast_int_to_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        Ok((ptr.into(), target_type.clone()))
                    }
                    // Ptr to Ptr (Bitcast)
                    (Type::Ptr(_), Type::Ptr(_)) => {
                        if val.is_pointer_value() {
                             let ptr = val.into_pointer_value();
                             let new_ptr = self.builder.build_pointer_cast(ptr, self.context.ptr_type(inkwell::AddressSpace::default()), "cast_ptr_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                             Ok((new_ptr.into(), target_type.clone()))
                        } else if val.is_int_value() {
                             // Handle case where Ptr is represented as I64 (e.g. malloc return if signature mismatch?)
                             let i = val.into_int_value();
                             let new_ptr = self.builder.build_int_to_ptr(i, self.context.ptr_type(inkwell::AddressSpace::default()), "cast_int_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                             Ok((new_ptr.into(), target_type.clone()))
                        } else {
                             return Err(TlError::from(CodegenErrorKind::Internal(format!("Invalid value kind for Ptr cast: {:?}", val))));
                        }
                    }
                    // Integer Casts
                    (Type::I64, Type::I32) => {
                         let i = val.into_int_value();
                         let t = self.builder.build_int_cast(i, self.context.i32_type(), "cast_i64_i32").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                         Ok((t.into(), Type::I32))
                    }
                    (Type::I32, Type::I64) => {
                         let i = val.into_int_value();
                         let t = self.builder.build_int_z_extend(i, self.context.i64_type(), "cast_i32_i64").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                         Ok((t.into(), Type::I64))
                    }
                    // U8 Casts
                    (Type::I64, Type::Struct(name, _)) | (Type::I32, Type::Struct(name, _)) if name == "u8" => {
                         let i = val.into_int_value();
                         let truncated = self.builder.build_int_cast(i, self.context.i8_type(), "cast_u8").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                         Ok((truncated.into(), target_type.clone()))
                    }
                    (Type::F32, Type::Struct(name, _)) if name == "u8" => {
                         let f = val.into_float_value();
                         let i = self.builder.build_float_to_unsigned_int(f, self.context.i8_type(), "cast_u8").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                         Ok((i.into(), target_type.clone()))
                    }
                    // u8 -> Int Casts
                    (Type::Struct(name, _), Type::I64) if name == "u8" => {
                         let i = val.into_int_value(); // i8
                         let extended = self.builder.build_int_z_extend(i, self.context.i64_type(), "cast_u8_i64").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                         Ok((extended.into(), Type::I64))
                    }
                     (Type::Struct(name, _), Type::I32) if name == "u8" => {
                         let i = val.into_int_value(); // i8
                         let extended = self.builder.build_int_z_extend(i, self.context.i32_type(), "cast_u8_i32").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                         Ok((extended.into(), Type::I32))
                    }
                     (Type::I64, Type::U8) | (Type::I32, Type::U8) => {
                         let i = val.into_int_value();
                         let truncated = self.builder.build_int_cast(i, self.context.i8_type(), "cast_u8").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                         Ok((truncated.into(), Type::U8))
                    }
                    (Type::F32, Type::U8) => {
                         let f = val.into_float_value();
                         let i = self.builder.build_float_to_unsigned_int(f, self.context.i8_type(), "cast_u8").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                         Ok((i.into(), Type::U8))
                    }
                    // U8 -> wider types
                    (Type::U8, Type::I64) => {
                         let i = val.into_int_value();
                         let extended = self.builder.build_int_z_extend(i, self.context.i64_type(), "cast_u8_i64").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                         Ok((extended.into(), Type::I64))
                    }
                    (Type::U8, Type::I32) => {
                         let i = val.into_int_value();
                         let extended = self.builder.build_int_z_extend(i, self.context.i32_type(), "cast_u8_i32").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                         Ok((extended.into(), Type::I32))
                    }
                    (Type::U8, Type::F32) => {
                         let i = val.into_int_value();
                         let extended = self.builder.build_int_z_extend(i, self.context.i32_type(), "cast_u8_i32").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                         let f = self.builder.build_unsigned_int_to_float(extended, self.context.f32_type(), "cast_u8_f32").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                         Ok((f.into(), Type::F32))
                    }
                    // Tensor -> GradTensor: enable_grad and return same ptr as GradTensor type
                    (Type::Tensor(inner, rank), Type::GradTensor(_, _)) => {
                        let enable_fn = self
                            .module
                            .get_function("tl_tensor_enable_grad")
                            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_tensor_enable_grad not found".to_string())))?;
                        self.builder
                            .build_call(enable_fn, &[val.into()], "enable_grad")
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        Ok((val, Type::GradTensor(inner.clone(), *rank)))
                    }
                    // GradTensor -> Tensor: detach (drops autograd, returns new ptr)
                    (Type::GradTensor(inner, rank), Type::Tensor(_, _)) => {
                        let detach_fn = self
                            .module
                            .get_function("tl_tensor_detach")
                            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_tensor_detach not found".to_string())))?;
                        let false_val = self.context.bool_type().const_int(0, false);
                        let call = self
                            .builder
                            .build_call(detach_fn, &[val.into(), false_val.into()], "detach")
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        let res = match call.try_as_basic_value() {
                            ValueKind::Basic(v) => v,
                            _ => return Err(CodegenErrorKind::Internal("Invalid return from detach".to_string()).into()),
                        };
                        Ok((res, Type::Tensor(inner.clone(), *rank)))
                    }
                    (Type::Tensor(_, _), Type::Tensor(inner_dst, _)) => match inner_dst.as_ref() {
                        Type::F32 => {
                            let cast_fn = self
                                .module
                                .get_function("tl_tensor_to_f32")
                                .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_tensor_to_f32 not found".to_string())))?;
                            let call = self
                                .builder
                                .build_call(cast_fn, &[val.into()], "cast_t")
                                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                            let res = match call.try_as_basic_value() {
                                ValueKind::Basic(v) => v,
                                _ => return Err(CodegenErrorKind::Internal("Invalid return".to_string()).into()),
                            };
                            // self.emit_register_tensor(res, target_type)?;

                            Ok((res, target_type.clone()))
                        }
                        Type::I64 => {
                            let cast_fn = self
                                .module
                                .get_function("tl_tensor_to_i64")
                                .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_tensor_to_i64 not found".to_string())))?;
                            let call = self
                                .builder
                                .build_call(cast_fn, &[val.into()], "cast_t")
                                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                            let res = match call.try_as_basic_value() {
                                ValueKind::Basic(v) => v,
                                _ => return Err(CodegenErrorKind::Internal("Invalid return".to_string()).into()),
                            };
                            // self.emit_register_tensor(res, target_type)?;

                            Ok((res, target_type.clone()))
                        }
                        _ => Err(TlError::from(CodegenErrorKind::UnsupportedOperation(format!("Unsupported tensor cast target: {:?}", inner_dst)))),
                    },
                    (Type::Struct(struct_name, _), Type::TraitObject(trait_name)) => {
                        let obj = self.emit_trait_object_upcast(val, struct_name, trait_name)?;
                        Ok((obj, target_type.clone()))
                    }
                    _ => {
                        // Trait fallback: `target_type::from(source)`
                        let target_base_name = target_type.get_base_name();
                        let target_generics = match &target_type {
                            Type::Struct(_, args) | Type::Enum(_, args) => args.clone(),
                            _ => vec![],
                        };
                        
                        if let Ok(mangled) = self.monomorphize_method(&target_base_name, "from", &target_generics) {
                            if let Some(fn_val) = self.module.get_function(&mangled) {
                                // Check if method returns via sret (hidden first pointer arg)
                                let returns_sret = fn_val.get_nth_param(0).map_or(false, |p| p.get_name().to_str().unwrap_or("") == "sret");
                                
                                if returns_sret {
                                    let sret_alloca = self.create_entry_block_alloca(
                                        self.current_function()?,
                                        "sret_from",
                                        &target_type,
                                    )?;
                                    
                                    self.builder.build_call(fn_val, &[sret_alloca.into(), val.into()], "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                                    return Ok((sret_alloca.into(), target_type.clone()));
                                } else {
                                    let call = self.builder.build_call(fn_val, &[val.into()], "trait_from").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                                    if let inkwell::values::ValueKind::Basic(res_val) = call.try_as_basic_value() {
                                        return Ok((res_val, target_type.clone()));
                                    }
                                }
                            }
                        }

                        Err(TlError::from(CodegenErrorKind::UnsupportedOperation(format!(
                            "Unsupported cast from {:?} to {:?}",
                            source_type, target_type
                        ))))
                    }
                }
            }
            ExprKind::Try(inner) => {
                let (val, raw_ty) = self.compile_expr(inner)?;

                // Normalize SpecializedType -> Enum for Result types
                let ty = self.normalize_type(&raw_ty);

                // We know it is Result<T, E> from semantics
                let (ok_ty, err_ty) = if let Type::Enum(_, generics) = &ty {
                    (generics[0].clone(), generics[1].clone())
                } else {
                    return Err(TlError::from(CodegenErrorKind::Internal(format!("Try operator on non-Result type in codegen: {:?} (should be caught by semantics)", ty))));
                };

                let ptr = val.into_pointer_value();

                // Get Struct Type from Type::Enum
                let struct_ty = if let Type::Enum(name, generics) = &ty {
                     let mangled = if generics.is_empty() {
                         name.clone()
                     } else {
                         self.mangle_type_name(name, generics)
                     };
                     
                     if !self.enum_types.contains_key(&mangled) {
                         // On-demand monomorphization
                         if let Some(_) = self.enum_defs.get(name) {
                             self.monomorphize_enum(name, generics).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                             // Identify the newly created enum def
                             let specialized_def = self.enum_defs.get(&mangled).ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Monomorphization failed for {}", mangled))))?.clone();
                             self.compile_enum_defs(&[specialized_def])?;
                         }
                     }
                     
                     *self.enum_types.get(&mangled).ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Enum type {} not found", mangled))))?
                } else {
                     return Err(TlError::from(CodegenErrorKind::Internal("Try on non-Enum type".to_string())));
                };
                
                let tag_ptr = self.builder.build_struct_gep(struct_ty, ptr, 0, "tag_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let tag = self.builder.build_load(self.context.i32_type(), tag_ptr, "tag").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value();
                
                let current_block = self.current_block()?;
                let func = current_block.get_parent().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("block has no parent function".to_string())))?;
                let ok_block = self.context.append_basic_block(func, "try_ok");
                let err_block = self.context.append_basic_block(func, "try_err");
                
                let zero = self.context.i32_type().const_zero();
                let is_ok = self.builder.build_int_compare(inkwell::IntPredicate::EQ, tag, zero, "is_ok").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_conditional_branch(is_ok, ok_block, err_block).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                
                // === ERR BLOCK ===
                self.builder.position_at_end(err_block);
                
                let payload_ptr_raw = self.builder.build_struct_gep(struct_ty, ptr, 1, "payload_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let payload_ptr = self.builder.build_pointer_cast(payload_ptr_raw, self.context.ptr_type(inkwell::AddressSpace::default()), "payload_cast").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                
                // Extract Err Value (E)
                // If primitive, load. If pointer, use pointer.
                let err_val = if matches!(err_ty, Type::I64 | Type::F64 | Type::I32 | Type::F32 | Type::Bool | Type::U8 | Type::Char(_)) {
                    let field_llvm_ty = self.get_llvm_type(&err_ty)?;
                    self.builder.build_load(field_llvm_ty, payload_ptr, "err_val").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                } else {
                     // Structs/Enums are stored as pointers in payload?
                     // Verify storage logic in EnumInit (struct variants use pointers, scalar types likely too?)
                     // Enum payload is `[i8; size]`.
                     // We cast it to `E*`.
                     // If E is ByValue (primitive), we already loaded it above.
                     // If E is ByRef/Pointer (Struct, Enum, String), the Payload IS the data? OR Payload contains the pointer?
                     // In `EnumInit`:
                     // `build_store(f_ptr, val)`.
                     // If `val` is pointer (struct), we store the pointer.
                     // So we just load the pointer from payload?
                     // Wait, Payload is a struct `{ field_types... }`.
                     // Since Result<T,E> has T or E in union.
                     // We cast payload to `{ E }` (struct with 1 field).
                     // Then access index 0.
                     let err_variant_ty = self.context.struct_type(&[self.get_llvm_type(&err_ty)?], false);
                     let err_variant_ptr = self.builder.build_pointer_cast(payload_ptr, self.context.ptr_type(inkwell::AddressSpace::default()), "err_variant_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                     let field_ptr = self.builder.build_struct_gep(err_variant_ty, err_variant_ptr, 0, "err_field_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                     self.builder.build_load(self.get_llvm_type(&err_ty)?, field_ptr, "err_val_loaded").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                };

                // FIX: Retain ownership of extracted error value because we are taking it out of Result
                self.emit_retain(err_val, &err_ty)?;

                // Construct Return Value: Result<RetOk, E>::Err(err_val)
                let raw_func_ret_ty = self.current_fn_return_type.clone().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("Unknown function return type".to_string())))?;
                let func_ret_ty = self.normalize_type(&raw_func_ret_ty);
                let func_ok_ty = if let Type::Enum(_, generics) = &func_ret_ty {
                    generics[0].clone()
                } else {
                     return Err(TlError::from(CodegenErrorKind::TypeError(format!("Function return type mismatch (expected Result, got {:?})", func_ret_ty))));
                };
                
                // We use compile_expr with EnumInit to construct the return value
                // We need to inject `err_val` into scope to use it in ExprKind::Variable
                let temp_name = "try_err_temp";
                // Insert into CURRENT scope (inner-most)
                self.variables.last_mut().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("no variable scope".to_string())))?.insert(temp_name.to_string(), (err_val, err_ty.clone(), crate::compiler::codegen::CLEANUP_NONE)); // CLEANUP_NONE because we consume/move it
                
                let span = expr.span.clone();
                let enum_init_expr = Spanned::new(
                    ExprKind::EnumInit {
                        enum_name: "Result".to_string(),
                        variant_name: "Err".to_string(),
                        generics: vec![func_ok_ty, err_ty.clone()],
                        payload: crate::compiler::ast::EnumVariantInit::Tuple(vec![
                            Spanned::new(ExprKind::Variable(temp_name.to_string()), span.clone())
                        ]),
                    },
                    span.clone()
                );
                
                let (ret_val, ret_ty_compiled) = self.compile_expr(&enum_init_expr)?;
                
                // Cleanup and Return
                self.mark_temp_no_cleanup(ret_val);

                if let Some(sret) = self.current_sret_dest {
                    let ret_ptr = ret_val.into_pointer_value();
                    // Avoid self-copy if EnumInit already wrote to sret
                    if ret_ptr != sret {
                        self.builder.build_store(sret, ret_ptr).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    }
                    self.emit_all_scopes_cleanup();
                    if let Some(exit_fn) = self.module.get_function("tl_mem_function_exit") {
                         self.builder.build_call(exit_fn, &[], "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    }
                    self.builder.build_return(None).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                } else {
                     // Direct return
                     if matches!(ret_ty_compiled, Type::Struct(_, _) | Type::Enum(_, _)) {
                        if let Some(unreg_fn) = self.module.get_function("tl_mem_unregister") {
                            let ptr = ret_val.into_pointer_value();
                            let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
                            let cast_ptr = self.builder.build_pointer_cast(ptr, ptr_type, "cast_unreg_ret").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                            self.builder.build_call(unreg_fn, &[cast_ptr.into()], "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        }
                     }

                     self.emit_all_scopes_cleanup();
                     if let Some(exit_fn) = self.module.get_function("tl_mem_function_exit") {
                          self.builder.build_call(exit_fn, &[], "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                     }
                     if let Some(_rt) = func.get_type().get_return_type() {
                         self.builder.build_return(Some(&ret_val)).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                     } else {
                         self.builder.build_return(None).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                     }
                }
                
                // === OK BLOCK ===
                self.builder.position_at_end(ok_block);
                
                // Extract Ok Value (T) - Same logic as Err
                let payload_ptr_raw_ok = self.builder.build_struct_gep(struct_ty, ptr, 1, "payload_ptr_ok").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let payload_ptr_ok = self.builder.build_pointer_cast(payload_ptr_raw_ok, self.context.ptr_type(inkwell::AddressSpace::default()), "payload_cast_ok").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                
                 let ok_val = if matches!(ok_ty, Type::I64 | Type::F64 | Type::I32 | Type::F32 | Type::Bool | Type::U8 | Type::Char(_)) {
                    let field_llvm_ty = self.get_llvm_type(&ok_ty)?;
                    self.builder.build_load(field_llvm_ty, payload_ptr_ok, "ok_val").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                } else {
                     let ok_variant_ty = self.context.struct_type(&[self.get_llvm_type(&ok_ty)?], false);
                     let ok_variant_ptr = self.builder.build_pointer_cast(payload_ptr_ok, self.context.ptr_type(inkwell::AddressSpace::default()), "ok_variant_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                     let field_ptr = self.builder.build_struct_gep(ok_variant_ty, ok_variant_ptr, 0, "ok_field_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                     self.builder.build_load(self.get_llvm_type(&ok_ty)?, field_ptr, "ok_val_loaded").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                };
                
                // FIX: Retain ownership of extracted ok value
                self.emit_retain(ok_val, &ok_ty)?;
                
                Ok((ok_val, ok_ty))
            }

            // `.await` 式: Phase 4 でステートマシン変換として完全実装する。
            // Phase 1〜3 では構文解析・型チェックのみ対応し、codegen はエラーを返す。
            ExprKind::Await(inner) => {
                // Phase 4 MVP: デシュガー `expr.await` → `expr.poll()` + Ready ペイロード抽出。
                // 現時点では Pending の場合はスピンループで再試行し、完了まで待機する。
                // (真のコルーチン変換は Phase 4 full で実装)
                //
                // 1. inner.poll() を呼び出す
                let (poll_val, raw_poll_ty) = self.compile_method_call(inner, "poll", &[])?;
                let poll_ty = self.normalize_type(&raw_poll_ty);

                // 2. Poll<T> の output 型を取得 (Poll<T> は Type::Enum か Type::Struct として現れる)
                let output_ty = match &poll_ty {
                    Type::Enum(name, generics) if name == "Poll" && !generics.is_empty() => {
                        generics[0].clone()
                    }
                    Type::Struct(name, generics) if name.starts_with("Poll") && !generics.is_empty() => {
                        generics[0].clone()
                    }
                    _ => {
                        return Err(TlError::from(CodegenErrorKind::Internal(format!(
                            ".await: poll() が Poll<T> を返しませんでした (got {:?})", poll_ty
                        ))));
                    }
                };

                // 3. Poll<T> の LLVM struct 型を取得 / モノモーフ化
                let (enum_name, enum_generics) = match &poll_ty {
                    Type::Enum(n, g) | Type::Struct(n, g) => (n.clone(), g.clone()),
                    _ => return Err(TlError::from(CodegenErrorKind::Internal("Poll type not enum".to_string()))),
                };
                let mangled = if enum_generics.is_empty() {
                    enum_name.clone()
                } else {
                    self.mangle_type_name(&enum_name, &enum_generics)
                };
                if !self.enum_types.contains_key(&mangled) {
                    if self.enum_defs.contains_key(&enum_name) {
                        self.monomorphize_enum(&enum_name, &enum_generics)
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    }
                }
                let enum_llvm_ty = *self.enum_types.get(&mangled)
                    .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(
                        format!(".await: Poll enum type {} not found", mangled)
                    )))?;

                // 4. poll_val はポインタとして受け取る
                let poll_ptr = poll_val.into_pointer_value();

                // 5. tag (index 0, i32) を読み込む
                let tag_ptr = self.builder
                    .build_struct_gep(enum_llvm_ty, poll_ptr, 0, "await_tag_ptr")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let tag = self.builder
                    .build_load(self.context.i32_type(), tag_ptr, "await_tag")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                    .into_int_value();

                // 6. tag == 0 なら Ready, tag == 1 なら Pending
                let current_block = self.current_block()?;
                let func = current_block.get_parent()
                    .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("block has no parent".to_string())))?;
                let ready_block = self.context.append_basic_block(func, "await_ready");
                let pending_block = self.context.append_basic_block(func, "await_pending");
                let merge_block = self.context.append_basic_block(func, "await_merge");

                let zero = self.context.i32_type().const_zero();
                let is_ready = self.builder
                    .build_int_compare(inkwell::IntPredicate::EQ, tag, zero, "is_ready")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder
                    .build_conditional_branch(is_ready, ready_block, pending_block)
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // === PENDING ブロック: Phase 4 MVP では panic ===
                self.builder.position_at_end(pending_block);
                if let Some(panic_fn) = self.module.get_function("tl_panic") {
                    let msg_ptr = self.builder.build_global_string_ptr(
                        "Future returned Poll::Pending - async executor not yet supported",
                        "await_pending_msg",
                    ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    self.builder.build_call(panic_fn, &[msg_ptr.as_pointer_value().into()], "")
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                }
                self.builder.build_unconditional_branch(merge_block)
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // === READY ブロック: payload (index 1) から Output 型の値を取り出す ===
                self.builder.position_at_end(ready_block);
                let payload_ptr = self.builder
                    .build_struct_gep(enum_llvm_ty, poll_ptr, 1, "await_payload_ptr")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                let output_llvm_ty = self.get_llvm_type(&output_ty)?;
                let ready_val = if matches!(output_ty, Type::I64 | Type::F64 | Type::I32 | Type::F32 | Type::Bool | Type::U8 | Type::Usize | Type::Entity | Type::Char(_)) {
                    // プリミティブ型: ペイロードから直接 load
                    self.builder
                        .build_load(output_llvm_ty, payload_ptr, "await_ready_val")
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                } else {
                    // 参照型 (Struct / String / Enum): ペイロードはポインタを保持
                    let variant_ty = self.context.struct_type(&[output_llvm_ty], false);
                    let variant_ptr = self.builder
                        .build_pointer_cast(payload_ptr, self.context.ptr_type(inkwell::AddressSpace::default()), "await_variant_ptr")
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    let field_ptr = self.builder
                        .build_struct_gep(variant_ty, variant_ptr, 0, "await_field_ptr")
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    self.builder
                        .build_load(output_llvm_ty, field_ptr, "await_ready_val")
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                };

                // ready_val を PHI ノードに渡すために alloca に保存
                let result_alloca = self.create_entry_block_alloca_manual(func, "await_result", &output_llvm_ty)?;
                self.builder
                    .build_store(result_alloca, ready_val)
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder
                    .build_unconditional_branch(merge_block)
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // === MERGE ブロック ===
                self.builder.position_at_end(merge_block);
                let final_val = self.builder
                    .build_load(output_llvm_ty, result_alloca, "await_final")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                Ok((final_val, output_ty))
            }

            ExprKind::Closure { args, return_type, body, captures } => {
                // Generate a unique anonymous function name using $ (invalid in TL identifiers)
                let closure_id = self.next_closure_id();
                let fn_name = format!("__tl_closure${}", closure_id);

                // Determine arg types from annotations or default to i64
                let mut param_types = Vec::new();
                let mut param_names = Vec::new();
                for (arg_name, arg_type_opt) in args {
                    let arg_ty = arg_type_opt.clone().ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Closure argument '{}' missing type annotation and could not be inferred", arg_name))))?;
                    param_types.push(arg_ty);
                    param_names.push(arg_name.clone());
                }

                // Determine return type
                let ret_ty = return_type.clone().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("Closure return type missing and could not be inferred".to_string())))?;

                let has_captures = !captures.is_empty();
                let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());

                // Build LLVM function type — ALL closures get env_ptr as first arg (fat pointer)
                let mut llvm_param_types: Vec<inkwell::types::BasicMetadataTypeEnum<'ctx>> = Vec::new();
                llvm_param_types.push(ptr_type.into()); // env_ptr always first
                for pt in &param_types {
                    let llvm_ty = self.get_llvm_type(pt)?;
                    llvm_param_types.push(llvm_ty.into());
                }

                let fn_type = if matches!(ret_ty, Type::Void) {
                    self.context.void_type().fn_type(&llvm_param_types, false)
                } else {
                    let ret_llvm_ty = self.get_llvm_type(&ret_ty)?;
                    ret_llvm_ty.fn_type(&llvm_param_types, false)
                };

                let fn_val = self.module.add_function(&fn_name, fn_type, None);

                let env_ptr_result: inkwell::values::PointerValue<'ctx> = if has_captures {
                    // Calculate size: sum of all captured value LLVM sizes
                    // For mutable captures, store a pointer (for indirection)
                    let mut field_types: Vec<inkwell::types::BasicTypeEnum<'ctx>> = Vec::new();
                    for (_, cty, is_mut) in captures.iter() {
                        if *is_mut {
                            field_types.push(ptr_type.into()); // pointer to original alloca
                        } else {
                            let llvm_ty = self.get_llvm_type(cty)?;
                            field_types.push(llvm_ty);
                        }
                    }
                    let env_struct_ty = self.context.struct_type(&field_types, false);
                    let env_size = env_struct_ty.size_of().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("type has no size".to_string())))?;

                    // malloc
                    let malloc_fn = self.module.get_function("malloc")
                        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("malloc not found".to_string())))?;
                    let i64_type = self.context.i64_type();
                    let env_size_i64 = self.builder.build_int_cast(env_size, i64_type, "env_size")
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    let raw_env = self.builder.build_call(malloc_fn, &[env_size_i64.into()], "env_malloc")
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    let env_ptr = match raw_env.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
                        _ => return Err(CodegenErrorKind::Internal("malloc returned no value".to_string()).into()),
                    };

                    // Store captured values into env struct
                    for (idx, (cap_name, cap_ty, is_mut)) in captures.iter().enumerate() {
                        // Look up captured variable in current scope
                        let mut found = None;
                        for scope in self.variables.iter().rev() {
                            if let Some((val_ptr, _, _)) = scope.get(cap_name) {
                                found = Some(*val_ptr);
                                break;
                            }
                        }
                        let cap_alloca = found.ok_or_else(|| {
                            TlError::from(CodegenErrorKind::Internal(format!("Captured variable '{}' not found in scope", cap_name)))
                        })?;

                        let field_ptr = self.builder.build_struct_gep(env_struct_ty, env_ptr, idx as u32, &format!("env_field_{}", cap_name))
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                        if *is_mut {
                            // Mutable: store the alloca pointer itself
                            self.builder.build_store(field_ptr, cap_alloca.into_pointer_value())
                                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        } else {
                            // Immutable: store a copy of the value
                            let cap_llvm_ty = self.get_llvm_type(cap_ty)?;
                            let cap_val = self.builder.build_load(cap_llvm_ty, cap_alloca.into_pointer_value(), &format!("load_cap_{}", cap_name))
                                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                            self.builder.build_store(field_ptr, cap_val)
                                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        }
                    }

                    env_ptr
                } else {
                    // No captures: env_ptr = null
                    ptr_type.const_null()
                };

                // Save current state
                let prev_block = self.builder.get_insert_block();
                let prev_vars = self.variables.clone();
                let prev_sret = self.current_sret_dest;

                // Set up new function
                self.current_sret_dest = None;
                let entry = self.context.append_basic_block(fn_val, "entry");
                self.builder.position_at_end(entry);

                // Create new variable scope
                self.variables = vec![std::collections::HashMap::new()];

                // === Callee side: extract captures from env_ptr ===
                // env_ptr is always param 0 (fat pointer convention)
                let param_offset = 1u32; // always offset by 1
                let env_param = fn_val.get_nth_param(0).ok_or_else(|| TlError::from(CodegenErrorKind::Internal("missing function parameter".to_string())))?;
                env_param.set_name("env_ptr");

                if has_captures {

                    // Build env struct type (same as caller side)
                    let mut field_types: Vec<inkwell::types::BasicTypeEnum<'ctx>> = Vec::new();
                    for (_, cty, is_mut) in captures.iter() {
                        if *is_mut {
                            field_types.push(ptr_type.into());
                        } else {
                            let llvm_ty = self.get_llvm_type(cty)?;
                            field_types.push(llvm_ty);
                        }
                    }
                    let env_struct_ty = self.context.struct_type(&field_types, false);
                    let env_ptr_pv = env_param.into_pointer_value();

                    // Extract each captured variable
                    for (idx, (cap_name, cap_ty, is_mut)) in captures.iter().enumerate() {
                        let cap_llvm_ty = self.get_llvm_type(cap_ty)?;
                        let field_ptr = self.builder.build_struct_gep(env_struct_ty, env_ptr_pv, idx as u32, &format!("env_{}", cap_name))
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                        if *is_mut {
                            // Mutable: field contains a pointer to the original alloca.
                            // Load the pointer and use it directly as the variable's alloca.
                            let original_alloca = self.builder.build_load(ptr_type, field_ptr, &format!("mut_ptr_{}", cap_name))
                                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                            self.variables.last_mut().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("no variable scope".to_string())))?.insert(
                                cap_name.clone(),
                                (original_alloca, cap_ty.clone(), crate::compiler::codegen::CLEANUP_NONE),
                            );
                        } else {
                            // Immutable: load value and store as local
                            let cap_val = self.builder.build_load(cap_llvm_ty, field_ptr, cap_name)
                                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                            let alloca = self.builder.build_alloca(cap_llvm_ty, cap_name)
                                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                            self.builder.build_store(alloca, cap_val)
                                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                            self.variables.last_mut().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("no variable scope".to_string())))?.insert(
                                cap_name.clone(),
                                (alloca.into(), cap_ty.clone(), crate::compiler::codegen::CLEANUP_NONE),
                            );
                        }
                    }
                }

                // Bind regular arguments
                for (i, (name, ty)) in param_names.iter().zip(param_types.iter()).enumerate() {
                    let param = fn_val.get_nth_param(i as u32 + param_offset).ok_or_else(|| TlError::from(CodegenErrorKind::Internal("missing function parameter".to_string())))?;
                    param.set_name(name);
                    let alloca = self.builder.build_alloca(param.get_type(), name)
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    self.builder.build_store(alloca, param)
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    self.variables.last_mut().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("no variable scope".to_string())))?.insert(
                        name.clone(),
                        (alloca.into(), ty.clone(), crate::compiler::codegen::CLEANUP_NONE),
                    );
                }

                // Compile body
                let mut last_val: Option<(inkwell::values::BasicValueEnum<'ctx>, Type)> = None;
                for (i, stmt) in body.iter().enumerate() {
                    if i == body.len() - 1 {
                        if let crate::compiler::ast::StmtKind::Expr(e) = &stmt.inner {
                            last_val = Some(self.compile_expr(e)?);
                        } else {
                            self.compile_stmt(stmt)?;
                        }
                    } else {
                        self.compile_stmt(stmt)?;
                    }
                }

                // Build return
                if matches!(ret_ty, Type::Void) {
                    self.builder.build_return(None).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                } else if let Some((val, _)) = last_val {
                    self.builder.build_return(Some(&val)).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                } else {
                    let zero = self.context.i64_type().const_zero();
                    self.builder.build_return(Some(&zero)).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                }

                // Restore previous state
                self.variables = prev_vars;
                self.current_sret_dest = prev_sret;
                if let Some(prev) = prev_block {
                    self.builder.position_at_end(prev);
                }

                // Build fat pointer: {fn_ptr, env_ptr} struct
                let fn_ptr = fn_val.as_global_value().as_pointer_value();
                let closure_struct_ty = self.context.struct_type(&[ptr_type.into(), ptr_type.into()], false);
                let fat_ptr = closure_struct_ty.const_zero();
                let fat_ptr = self.builder.build_insert_value(fat_ptr, fn_ptr, 0, "fat_fn")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let fat_ptr = self.builder.build_insert_value(fat_ptr, env_ptr_result, 1, "fat_env")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                Ok((fat_ptr.into_struct_value().into(), Type::Fn(param_types, Box::new(ret_ty))))
            }

            ExprKind::FieldAccess(obj, field) => {
                let (obj_val, obj_ty) = self.compile_expr(obj)?;

                // Auto-dereference Ref types - REMOVED (Ref not in spec)
                // while let Type::Ref(inner) = obj_ty.clone() {
                //     let ptr = obj_val.into_pointer_value();
                //     let loaded = self.builder.build_load(
                //         self.get_llvm_type(&inner)?,
                //         ptr,
                //         "deref"
                //     ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                //     obj_val = loaded.into();
                //     obj_ty = *inner;
                // }
                
                // Determine struct name and generic args
                // Note: Vec_u8 etc. may come as Enum due to monomorphize.rs conversion
                // Treat such types as Struct for field access purposes
                // Normalize obj_ty to resolve Path types in generic args
                let normalized_obj_ty = self.normalize_type(&obj_ty);

                let (base_name, generic_args) = match &normalized_obj_ty {
                    Type::Struct(name, args) => (name.clone(), args.clone()),
                    Type::Enum(name, args) => {
                        // Workaround: Some types like Vec_u8 are incorrectly classified as Enum
                        // If it has struct_defs entry, treat it as struct
                        (name.clone(), args.clone())
                    }
                    _ => return Err(TlError::from(CodegenErrorKind::Internal(format!("Field access on non-struct type {:?}", obj_ty)))),
                };


                // Determine mangled name for lookup
                let mangled_name = if generic_args.is_empty() {
                    base_name.clone()
                } else {
                    self.mangle_type_name(&base_name, &generic_args)
                };

                let simple_struct_name = mangled_name.clone();

                /*
                Logic:
                1. Try looking up exact specialized name (e.g. "MyStruct_i64").
                2. If not found, try base name (e.g. "MyStruct") effectively falling back to generic definition.
                   If base name is found, we must substitutions generic params in the field type.
                */
                
                let (struct_def, is_generic_base) = if let Some(def) = self.struct_defs.get(&simple_struct_name) {
                    (def, false)
                } else if let Some(def) = self.struct_defs.get(&base_name) {
                    (def, true)
                } else {
                    // Try extracting base name from underscore-mangled name (e.g., "Vec_u8" -> "Vec")
                    let underscore_base = mangle_base_name(&base_name).to_string();
                    if let Some(def) = self.struct_defs.get(&underscore_base) {
                        (def, true)
                    } else {
                        return Err(TlError::from(CodegenErrorKind::Internal(format!("Struct definition for {} not found (checked {}, {}, {})",
                            base_name, simple_struct_name, base_name, underscore_base))));
                    }
                };

                let field_idx = struct_def
                    .fields
                    .iter()
                    .position(|(n, _)| n == field)
                    .ok_or_else(|| {
                        TlError::from(CodegenErrorKind::Internal(format!(
                        "Field {} not found in struct {}. Available: {:?}",
                        field, base_name, struct_def.fields.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>()
                    )))})?;
                
                // Retrieve field type and substitute if necessary
                let (_, field_ty_raw) = &struct_def.fields[field_idx];
                
                let effective_generic_args = generic_args.clone();
                
                let field_ty = if is_generic_base && !effective_generic_args.is_empty() {
                     self.substitute_type_generic(field_ty_raw, &struct_def.generics, &effective_generic_args)
                } else {
                     field_ty_raw.clone()
                };
                
                // Continue with loading... use simple_struct_name or base_name for struct type lookup?
                // self.struct_types stores llvm type.
                // If we are using generic base, "Vec_i64" might NOT be in struct_types if it wasn't compiled.
                // But monomorphize_method usually generates it?
                // If it isn't in struct_defs, maybe it isn't in struct_types either?
                // If so, we need to register it?
                // Or maybe Vec is Opaque Pointer in struct_types?
                
                // For Vec, we know it translates to i8*.
                // But for general structs, we need correct offset.
                // If we use Generic definition, we assume layout matches logic.
                // For Vec, fields are fixed.
                
                // If struct_types doesn't have specialized name, try base name?
                // BUT LLVM types for specialized structs might differ if fields differ (size).
                // For Vec, size is fixed (ptr, i64, i64).
                // So base struct name in struct_types should be sufficient if registered?
                // CodeGenerator registers structs in init.
                
                let st_llvm_ty = if let Some(t) = self.struct_types.get(&simple_struct_name) {
                     *t
                } else if let Some(t) = self.struct_types.get(&base_name) {
                     *t 
                } else {
                    // Try underscore-based base name (e.g., "Vec_u8" -> "Vec")
                    // and monomorphize on-demand if generics can be inferred
                    let underscore_base = mangle_base_name(&base_name).to_string();
                    if let Some(t) = self.struct_types.get(&underscore_base) {
                        *t
                    } else if !generic_args.is_empty() {
                        // Try to monomorphize on-demand

                        match self.monomorphize_struct(&underscore_base, &generic_args) {
                            Ok(t) => {

                                t
                            }
                            Err(_) => {
                                return Err(TlError::from(CodegenErrorKind::Internal(format!("Failed to monomorphize struct {} for FieldAccess", base_name))));
                            }
                        }
                    } else {
                        // generic_args is empty but name is mangled (e.g., Vec_Entry_i64_i64)
                        // For Vec, HashMap, etc., the LLVM layout is fixed regardless of generic args
                        // So we can use the base type's LLVM layout
                        if underscore_base == "Vec" || underscore_base == "HashMap" || underscore_base == "HashSet" || underscore_base == "Option" || underscore_base == "Result" || underscore_base == "VecDeque" || underscore_base == "BTreeMap" || underscore_base == "BTreeNode" || underscore_base == "StringBuilder" {
                            // Try to get base type, then monomorphize with i64 as fallback
                            if let Some(t) = self.struct_types.get(&underscore_base) {
                                *t
                            } else {
                                self.monomorphize_struct(&underscore_base, &[Type::I64])
                                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(format!("Failed to monomorphize {} for FieldAccess: {}", underscore_base, e))))?
                            }
                        } else {
                            return Err(TlError::from(CodegenErrorKind::Internal(format!("LLVM struct type for {} not found (tried {}, {}, {})",
                                base_name, simple_struct_name, base_name, underscore_base))));
                        }
                    }
                };


                if obj_val.is_pointer_value() {
                    let ptr = obj_val.into_pointer_value();
                    // let st_llvm_ty = self.struct_types.get(simple_struct_name).ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("struct type not found"))))?; // Handled above

                    let field_ptr = self
                        .builder
                        .build_struct_gep(
                            st_llvm_ty,
                            ptr,
                            field_idx as u32,
                            &format!("ptr_{}", field),
                        )
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                    let llvm_ty: inkwell::types::BasicTypeEnum = match field_ty {
                        Type::I64 => self.context.i64_type().into(),
                        Type::F32 => self.context.f32_type().into(),
                        Type::Bool => self.context.bool_type().into(),
                        Type::Tensor(_, _) => self
                            .context
                            .ptr_type(inkwell::AddressSpace::default())
                            .into(),
                        Type::String(_) => self
                            .context
                            .ptr_type(inkwell::AddressSpace::default())
                            .into(),
                        Type::Char(_) | Type::I32 => self.context.i32_type().into(),
                        Type::Struct(_, _) 
                        | Type::Enum(_, _)
                        | Type::Ptr(_) // FIX: Handle Ptr fields
                        | Type::Tuple(_)
                        | Type::SpecializedType { .. } => self
                            .context
                            .ptr_type(inkwell::AddressSpace::default())
                            .into(),
                        Type::U8 => self.context.i8_type().into(),
                        Type::F64 => self.context.f64_type().into(),
                        Type::Usize | Type::Entity => self.context.i64_type().into(),
                        Type::TensorShaped(_, _) | Type::GradTensor(_, _)
                        | Type::Array(_, _) => self.get_llvm_type(&field_ty).unwrap_or(self.context.i64_type().into()),
                        Type::Void => self.context.i8_type().into(),
                        Type::Path(_, _) | Type::Fn(_, _) | Type::I8 | Type::I16 | Type::U16 | Type::U32 | Type::U64 | Type::F16 | Type::BF16 | Type::TypeVar(_) | Type::Never | Type::Undefined(_) | Type::Range => self.context.i64_type().into(), Type::TraitObject(_) => { let p = self.context.ptr_type(inkwell::AddressSpace::default()); self.context.struct_type(&[p.into(), p.into()], false).into() },
                    };

                    let loaded = self
                        .builder
                        .build_load(llvm_ty, field_ptr, field)
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    // フィールドは親構造体が所有 — 借用として RC 不変で返す。
                    // 消費側が必要に応じて retain/clone する。
                    Ok((loaded, field_ty.clone()))
                } else if obj_val.is_struct_value() {
                    let struct_val = obj_val.into_struct_value();
                    let extracted = self
                        .builder
                        .build_extract_value(struct_val, field_idx as u32, field)
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    // フィールドは親構造体が所有 — 借用として RC 不変で返す。
                    Ok((extracted, field_ty.clone()))
                } else {
                    Err(TlError::from(CodegenErrorKind::Internal("Cannot access field of non-pointer and non-struct value".to_string())))
                }
            }

            ExprKind::Variable(name) => {
                let mut found = None;
                for scope in self.variables.iter().rev() {
                    if let Some((val, ty, _)) = scope.get(name) {
                         found = Some((*val, ty.clone()));
                         break;
                    }
                }

                if let Some((val, ty)) = found {
                     if val.is_pointer_value() {
                          let ptr = val.into_pointer_value();
                          let llvm_ty = self.get_llvm_type(&ty).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                          let loaded = self
                                .builder
                                .build_load(llvm_ty, ptr, name)
                                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                          // Variable は借用として返す。RC は変更しない。
                          // 消費側（Let の emit_deep_clone, FnCall 引数等）が必要に応じて
                          // retain/clone を行う。
                          Ok((loaded, ty))
                     } else {
                          Ok((val, ty))
                     }
                } else {
                     Err(TlError::from(CodegenErrorKind::VariableNotFound(format!("Variable {} not found in scopes", name))))
                }
            }
            ExprKind::StructInit(ty, fields) => {
                 // Substitute with current method generics if available
                 let ty = ty.clone();

                 // Normalize Path types to Struct/Enum
                 let normalized_ty = self.normalize_type(&ty);
                 let (name, generics) = match &normalized_ty {
                      Type::Struct(name, generics) => (name.clone(), generics.clone()),
                      Type::Enum(name, generics) => (name.clone(), generics.clone()), // Enums might use struct-init syntax?
                      _ => panic!("StructInit type must be Struct or Enum (after normalization), found {:?} (original: {:?})", normalized_ty, ty),
                 };
                 
                 // [CONTEXT INHERITANCE] Logic removed. Assuming AOT resolving.

                 self.compile_struct_init(&name, &generics, fields)
            },
            ExprKind::StaticMethodCall(original_type_ty, method_name, args) => {
                let type_ty = original_type_ty.clone();
                if type_ty.get_base_name() == "Vec" && method_name == "new" {
                }
                
                if method_name == "sizeof" {
                     // For Enum types, we need to get the actual data struct size, not pointer size
                     if let Type::Enum(enum_name, generics) = &type_ty {
                         // Try direct lookup first, then mangled name if generics present
                         let lookup_name = if generics.is_empty() {
                             enum_name.clone()
                         } else if self.enum_types.contains_key(enum_name) {
                             enum_name.clone()
                         } else {
                             self.mangle_type_name(enum_name, generics)
                         };
                         
                         // Look up the actual LLVM struct type from enum_types
                         if let Some(enum_struct_type) = self.enum_types.get(&lookup_name) {
                             let size_val = enum_struct_type.size_of().ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Enum type {} has no size", lookup_name))))?;
                             return Ok((size_val.into(), Type::I64));
                         } else {
                             return Err(TlError::from(CodegenErrorKind::Internal(format!("Enum type {} not found in enum_types for sizeof", lookup_name))));
                         }
                     }
                     
                     // For Struct types, also check if it's actually an enum (mangled name)
                     if let Type::Struct(name, _) = &type_ty {
                         if let Some(enum_struct_type) = self.enum_types.get(name) {
                             // It's actually an enum with a Struct type wrapper
                             let size_val = enum_struct_type.size_of().ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Enum type {} has no size", name))))?;
                             return Ok((size_val.into(), Type::I64));
                         }
                     }
                     
                     // Generic T already substituted by Monomorphizer
                     let llvm_ty = self.get_llvm_type(&type_ty).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                     let size_val = llvm_ty.size_of().ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Type {:?} has no size (ZST not supported)", type_ty))))?;
                     // Cast to i64 if needed? IntValue is generic, but usually i64 for size_t on 64bit
                     // LLVM size_of returns integer type matching target's pointer width.
                     // Our Type::I64 expects LLVM i64.
                     return Ok((size_val.into(), Type::I64));
                 }

                // Normalize type for method dispatch
                let flat_type_ty = type_ty.flatten_specialized();
                let struct_name: String = match &flat_type_ty {
                    Type::Struct(name, args) if !args.is_empty() && !name.contains('[') => {
                        self.mangle_type_name(name, args)
                    }
                    Type::Enum(name, args) if !args.is_empty() && !name.contains('[') => {
                        self.mangle_type_name(name, args)
                    }
                    Type::Struct(name, _) => name.clone(),
                    Type::Enum(name, _) => name.clone(),
                    Type::F32 => "F32".to_string(),
                    Type::I64 => "I64".to_string(),
                    Type::Bool => "Bool".to_string(),
                    Type::String(_) => "String".to_string(),
                    Type::Tensor(_, _) => "Tensor".to_string(),
                    Type::GradTensor(_, _) => "GradTensor".to_string(),
                    Type::Path(segments, _) => segments.last().cloned().unwrap_or("Unknown".to_string()),
                    _ => return Err(TlError::from(CodegenErrorKind::TypeError(format!("Cannot call static method on type {:?}", type_ty)))),
                };
                let res = self.compile_static_method_call(&struct_name, method_name, args, &flat_type_ty)?;
                self.add_temp(res.0, res.1.clone());
                Ok(res)
            }
            ExprKind::BinOp(lhs, op, rhs) => {
                let left = self.compile_expr(lhs)?;
                let right = self.compile_expr(rhs)?;
                let res = self.compile_bin_op(
                    left.0,
                    left.1.clone(),
                    right.0,
                    right.1.clone(),
                    op.clone(),
                )?;

                // Register intermediate tensor result
                self.add_temp(res.0, res.1.clone());
                Ok(res)
            }

            ExprKind::Tuple(exprs) => self.compile_tuple(exprs),
            ExprKind::TupleAccess(expr, idx) => self.compile_tuple_access(expr, *idx),

            ExprKind::TensorComprehension {
                indices,
                clauses,
                body,
            } => {
                // Generate a unique name for the temporary result
                static NEXT_ID: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(0);
                let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let temp_name = format!("_comp_res_{}", id);

                // Compile as a tensor equation: let temp[indices] = body;
                // Implicit reduction analysis: detect indices that appear in body
                // but not in the LHS indices (Einstein summation convention).
                let reduction_indices: Vec<String> = if let Some(b) = body.as_deref() {
                    let (_analyzed_free, analyzed_reduction) = self.analyze_tensor_indices(b);
                    // Also detect generator-bound vars not in LHS as reduction
                    let mut all_reduction = analyzed_reduction;
                    for clause in clauses {
                        if let ComprehensionClause::Generator { name, .. } = clause {
                            if !indices.contains(name) && !all_reduction.contains(name) {
                                all_reduction.push(name.clone());
                            }
                        }
                    }
                    all_reduction
                } else {
                    vec![]
                };
                self.compile_tensor_equation(&temp_name, indices, &reduction_indices, clauses, body.as_deref())
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // After compilation, the tensor 'temp_name' is registered in the scope.
                // We need to load it to return it as an expression value.
                // It should be in the current scope.

                let (val_enum, val_ty, _) = self
                    .variables
                    .last()
                    .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("unexpected None".to_string())))?
                    .get(&temp_name)
                    .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Failed to retrieve temporary tensor {}", temp_name))))?
                    .clone();

                if let Type::Tensor(_, _) = val_ty {
                    // It's a pointer to the tensor struct
                    if val_enum.is_pointer_value() {
                        let ptr_to_ptr = val_enum.into_pointer_value();
                        let void_ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
                        let ptr = self
                            .builder
                            .build_load(void_ptr_type, ptr_to_ptr, "tensor_ptr")
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                        // FIX: テンソル方程式の中間変数 _comp_res_N の所有権を呼び出し元に移転。
                        // tl_tensor_acquire は no-op のため、_comp_res_N と代入先変数が同じポインタを共有する。
                        // 両方が CLEANUP_FULL のままだと exit_scope で二重解放が発生する。
                        if let Some(scope) = self.variables.last_mut() {
                            if let Some(entry) = scope.get_mut(&temp_name) {
                                entry.2 = super::CLEANUP_NONE;
                            }
                        }

                        Ok((ptr, val_ty))

                    } else {
                        Err(TlError::from(CodegenErrorKind::Internal("Tensor variable should be a pointer".to_string())))
                    }
                } else {
                    Err(TlError::from(CodegenErrorKind::Internal("Comprehension result must be a tensor".to_string())))
                }
            }
            ExprKind::TensorLiteral(elements) => self.compile_tensor_literal(elements),
            ExprKind::TensorConstLiteral(elements) => self.compile_tensor_const_literal(elements),
            ExprKind::MethodCall(obj, method, args) => self.compile_method_call(obj, method, args),
            ExprKind::FnCall(name, args) => self.compile_fn_call(name, args),
            ExprKind::IndexAccess(target, indices) => {
                let (val, raw_val_type) = self.compile_expr(target)?;
                let val_type = raw_val_type.flatten_specialized();
                match val_type.clone() {
                    Type::Struct(name, _) if name == "Tensor" => {
                        let rank = indices.len();
                        let i64_type = self.context.i64_type();
                        let array_type = i64_type.array_type(rank as u32);
                        let current_block = self.current_block()?;
                        let function = current_block.get_parent().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("block has no parent function".to_string())))?;
                        let entry_block = function.get_first_basic_block().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("function has no entry block".to_string())))?;
                        let entry_builder = self.context.create_builder();
                        if let Some(first_instr) = entry_block.get_first_instruction() {
                            entry_builder.position_before(&first_instr);
                        } else {
                            entry_builder.position_at_end(entry_block);
                        }

                        let array_alloca = entry_builder
                            .build_alloca(array_type, "idx_arr")
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                        for (i, idx_expr) in indices.iter().enumerate() {
                            let (compiled_idx, ty) = self.compile_expr(idx_expr)?;

                            // Ensure index is integer or float (cast if needed)
                            let idx_val = match ty {
                                Type::I64 => compiled_idx.into_int_value(),
                                Type::I32 => self
                                    .builder
                                    .build_int_z_extend(
                                        compiled_idx.into_int_value(),
                                        i64_type,
                                        "zext",
                                    )
                                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?,
                                Type::F64 | Type::F32 => self
                                    .builder
                                    .build_float_to_signed_int(
                                        compiled_idx.into_float_value(),
                                        i64_type,
                                        "f2i",
                                    )
                                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?,
                                _ => return Err(TlError::from(CodegenErrorKind::Internal(format!("Invalid index type {:?}", ty)))),
                            };
                            let idx_val = inkwell::values::BasicValueEnum::IntValue(idx_val);

                            let elem_ptr = unsafe {
                                self.builder
                                    .build_gep(
                                        array_type,
                                        array_alloca,
                                        &[
                                            i64_type.const_int(0, false),
                                            i64_type.const_int(i as u64, false),
                                        ],
                                        "idx_ptr",
                                    )
                                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                            };
                            self.builder
                                .build_store(elem_ptr, idx_val)
                                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        }

                        let get_fn_name = "tl_tensor_get_f32_md";
                        let get_fn = self.get_fn(get_fn_name)?;
                        let tensor_ptr = val.into_pointer_value();
                        let array_ptr = self
                            .builder
                            .build_pointer_cast(
                                array_alloca,
                                self.context.ptr_type(inkwell::AddressSpace::default()),
                                "arr_ptr",
                            )
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        let rank_val = i64_type.const_int(rank as u64, false);

                        let call = self
                            .builder
                            .build_call(
                                get_fn,
                                &[tensor_ptr.into(), array_ptr.into(), rank_val.into()],
                                "get_md_call",
                            )
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                        let res = match call.try_as_basic_value() {
                            ValueKind::Basic(v) => v,
                            _ => return Err(CodegenErrorKind::Internal("Invalid get return".to_string()).into()),
                        };
                        Ok((res, Type::F32))
                    }
                    Type::Struct(_, _) => {
                         // Generic Struct Indexing -> index() を優先、なければ get() にフォールバック
                         let method = self.resolve_index_method(&val_type);
                         self.emit_method_call(target, val, val_type, &method, indices)
                    }
                    Type::Tensor(inner, _) => {
                        // Prepare indices array
                        let rank = indices.len();
                        let i64_type = self.context.i64_type();

                        // Create array on stack in the ENTRY block to avoid stack overflow in loops
                        let array_type = i64_type.array_type(rank as u32);

                        let current_block = self.current_block()?;
                        let function = current_block.get_parent().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("block has no parent function".to_string())))?;
                        let entry_block = function.get_first_basic_block().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("function has no entry block".to_string())))?;

                        let entry_builder = self.context.create_builder();
                        if let Some(first_instr) = entry_block.get_first_instruction() {
                            entry_builder.position_before(&first_instr);
                        } else {
                            entry_builder.position_at_end(entry_block);
                        }

                        let array_alloca = entry_builder
                            .build_alloca(array_type, "idx_arr")
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                        for (i, idx_expr) in indices.iter().enumerate() {
                            let (compiled_idx, ty) = self.compile_expr(idx_expr)?;

                            // Ensure index is integer or float (cast if needed)
                            let idx_val = match ty {
                                Type::I64 => compiled_idx.into_int_value(),
                                Type::I32 => self
                                    .builder
                                    .build_int_z_extend(
                                        compiled_idx.into_int_value(),
                                        i64_type,
                                        "zext",
                                    )
                                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?,
                                Type::F64 | Type::F32 => self
                                    .builder
                                    .build_float_to_signed_int(
                                        compiled_idx.into_float_value(),
                                        i64_type,
                                        "f2i",
                                    )
                                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?,
                                _ => return Err(TlError::from(CodegenErrorKind::Internal(format!("Invalid index type {:?}", ty)))),
                            };
                            let idx_val = inkwell::values::BasicValueEnum::IntValue(idx_val);

                            let elem_ptr = unsafe {
                                self.builder
                                    .build_gep(
                                        array_type,
                                        array_alloca,
                                        &[
                                            i64_type.const_int(0, false),
                                            i64_type.const_int(i as u64, false),
                                        ],
                                        "idx_ptr",
                                    )
                                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                            };
                            self.builder
                                .build_store(elem_ptr, idx_val)
                                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        }

                        let (get_fn_name, res_ty) = match inner.as_ref() {
                            Type::I64 | Type::I32 => {
                                ("tl_tensor_get_i64_md", inner.as_ref().clone())
                            }
                            _ => ("tl_tensor_get_f32_md", Type::F32),
                        };
                        let get_fn = self.get_fn(get_fn_name)?;
                        let tensor_ptr = val.into_pointer_value();
                        let array_ptr = self
                            .builder
                            .build_pointer_cast(
                                array_alloca,
                                self.context.ptr_type(inkwell::AddressSpace::default()),
                                "arr_ptr",
                            )
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        let rank_val = i64_type.const_int(rank as u64, false);

                        let call = self
                            .builder
                            .build_call(
                                get_fn,
                                &[tensor_ptr.into(), array_ptr.into(), rank_val.into()],
                                "get_md_call",
                            )
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                        let res = match call.try_as_basic_value() {
                            ValueKind::Basic(v) => v,
                            _ => return Err(CodegenErrorKind::Internal("Invalid get return".to_string()).into()),
                        };

                        if let Type::I32 = res_ty {
                            let i32_val = self
                                .builder
                                .build_int_truncate(
                                    res.into_int_value(),
                                    self.context.i32_type(),
                                    "i32_trunc",
                                )
                                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                            Ok((i32_val.into(), Type::I32))
                        } else {
                            Ok((res, res_ty))
                        }
                    }


                    Type::Ptr(inner) => {
                         let ptr = val.into_pointer_value();
                         if indices.len() != 1 {
                             return Err(TlError::from(CodegenErrorKind::Internal("Ptr indexing must be 1D".to_string())));
                         }
                         let (idx_val, _) = self.compile_expr(&indices[0])?;
                         
                         unsafe {
                             // Use actual element type for correct GEP offset calculation
                             // (e.g. u8 = i8 = 1 byte, not ptr = 8 bytes)
                             let elem_llvm_ty = self.get_llvm_type(&inner)?;
                             let elem_ptr = self.builder.build_gep(
                                 elem_llvm_ty,
                                 ptr,
                                 &[idx_val.into_int_value()],
                                 "ptr_idx"
                             ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                             
                             let load_ty = self.get_or_monomorphize_type(&inner)?;
                             let val = self.builder.build_load(
                                 load_ty,
                                 elem_ptr,
                                 "ptr_val"
                             ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                             
                             Ok((val, *inner.clone()))
                         }
                    }
                    Type::Array(inner, size) => {
                         if indices.len() != 1 {
                             return Err(TlError::from(CodegenErrorKind::Internal("Array indexing must be 1D".to_string())));
                         }
                         let (idx_val, _) = self.compile_expr(&indices[0])?;
                         let idx_int = idx_val.into_int_value();

                         // Get LLVM array type for GEP
                         let llvm_arr_ty = self.get_llvm_type(&Type::Array(inner.clone(), size))?;
                         let llvm_elem_ty = self.get_llvm_type(&inner)?;

                         // The array value is on the stack as an alloca.
                         // We need to get the alloca pointer from the value.
                         // If val was loaded from an alloca, we need the alloca ptr.
                         // Actually, for stack arrays we should have the pointer.
                         // If val is a basic value (loaded), we need to store it to a temp alloca first.
                         let arr_ptr = if val.is_pointer_value() {
                             val.into_pointer_value()
                         } else {
                             // Store to temp alloca
                             let alloca = self.builder.build_alloca(llvm_arr_ty, "tmp_arr")
                                 .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                             self.builder.build_store(alloca, val)
                                 .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                             alloca
                         };
                         
                         let i64_type = self.context.i64_type();
                         let elem_ptr = unsafe {
                             self.builder.build_gep(
                                 llvm_arr_ty,
                                 arr_ptr,
                                 &[i64_type.const_int(0, false), idx_int],
                                 "arr_elem_ptr"
                             ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                         };
                         
                         let elem_val = self.builder.build_load(
                             llvm_elem_ty,
                             elem_ptr,
                             "arr_elem"
                         ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                         
                         Ok((elem_val, *inner))
                    }
                    _ => Err(CodegenErrorKind::Internal("Index access only on Tensor, Ptr, or Array".to_string()).into()),
                }
            }
            ExprKind::UnOp(op, expr) => {
                let (val, ty) = self.compile_expr(expr)?;
                match op {
                    UnOp::Neg => match &ty {
                        Type::I64 | Type::I32 => {
                            let i = val.into_int_value();
                            let res = self
                                .builder
                                .build_int_neg(i, "negtmp")
                                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                            Ok((res.into(), ty))
                        }
                        Type::F32 => {
                            let f = val.into_float_value();
                            let res = self
                                .builder
                                .build_float_neg(f, "negtmp")
                                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                            Ok((res.into(), Type::F32))
                        }
                        Type::F64 => {
                            let f = val.into_float_value();
                            let res = self
                                .builder
                                .build_float_neg(f, "negtmp")
                                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                            Ok((res.into(), Type::F64))
                        }
                        Type::Tensor(_inner, _rank) => {
                            let neg_fn = self.get_fn("tl_tensor_neg")?;
                            let call = self
                                .builder
                                .build_call(neg_fn, &[val.into()], "neg")
                                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                            let res = match call.try_as_basic_value() {
                                ValueKind::Basic(v) => v,
                                _ => return Err(CodegenErrorKind::Internal("Failed neg".to_string()).into()),
                            };
                            self.add_temp(res, ty.clone());

                            Ok((res, ty.clone()))

                        }
                        Type::U8 | Type::Bool | Type::String(_) | Type::Struct(_, _)
                        | Type::Enum(_, _) | Type::Tuple(_) | Type::Ptr(_) | Type::Void
                        | Type::Usize | Type::Entity | Type::Char(_) | Type::Array(_, _)
                        | Type::TensorShaped(_, _) | Type::GradTensor(_, _)
                        | Type::Path(_, _) | Type::Fn(_, _) | Type::I8 | Type::I16 | Type::U16 | Type::U32 | Type::U64 | Type::F16 | Type::BF16 | Type::TypeVar(_) | Type::SpecializedType { .. } | Type::Never | Type::Undefined(_) | Type::Range => Err(TlError::from(CodegenErrorKind::UnsupportedOperation(format!("Negation not supported for type {:?}", ty)))), Type::TraitObject(_) => Err(CodegenErrorKind::UnsupportedOperation("Negation not supported for TraitObject".to_string()).into()),
                    },

                    UnOp::Not => {
                        match &ty {
                            Type::Bool => {
                                let b = val.into_int_value(); // i1
                                let res = self
                                    .builder
                                    .build_not(b, "nottmp")
                                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                                Ok((res.into(), Type::Bool))
                            }
                            _ => Err(TlError::from(CodegenErrorKind::TypeError("Not only on bool".to_string()))),
                        }
                    }

                    UnOp::Query => {
                        // Logic Query: check if result tensor is non-empty
                        if let Type::Tensor(_, _) = ty {
                            let len_fn = self.get_fn("tl_tensor_len")?;
                            let cast_ptr = self
                                .builder
                                .build_pointer_cast(
                                    val.into_pointer_value(),
                                    self.context.ptr_type(inkwell::AddressSpace::default()),
                                    "cast_len",
                                )
                                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                            let call = self
                                .builder
                                .build_call(len_fn, &[cast_ptr.into()], "len")
                                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                            let len = match call.try_as_basic_value() {
                                inkwell::values::ValueKind::Basic(v) => v.into_int_value(),
                                _ => return Err(CodegenErrorKind::Internal("Failed len".to_string()).into()),
                            };
                            

                            // Check len > 0

                            let zero = self.context.i64_type().const_zero();
                            let bool_val = self
                                .builder
                                .build_int_compare(
                                    inkwell::IntPredicate::SGT,
                                    len,
                                    zero,
                                    "is_true",
                                )
                                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                            // Convert i1 bool to f32 (0.0 or 1.0)
                            let float_val = self.builder.build_unsigned_int_to_float(
                                bool_val,
                                self.context.f32_type(),
                                "bool_to_f32"
                            ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                            // Create scalar tensor using tl_tensor_new(data_ptr, rank=0, shape_ptr=NULL)
                            // 1. Alloca for f32 data
                            let current_block = self.current_block()?;
                            let func = current_block.get_parent().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("block has no parent function".to_string())))?;
                            let entry_block = func.get_first_basic_block().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("function has no entry block".to_string())))?;
                            
                            // Insert at the beginning of entry block to avoid being after terminator
                            if let Some(first_inst) = entry_block.get_first_instruction() {
                                self.builder.position_before(&first_inst);
                            } else {
                                self.builder.position_at_end(entry_block);
                            }
                            
                            let f32_ptr = self.builder.build_alloca(self.context.f32_type(), "scalar_data").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                            // Also allocate dummy shape pointer (i64/usize aligned) to satisfy slice::from_raw_parts alignment check
                            let shape_dummy_ptr = self.builder.build_alloca(self.context.i64_type(), "shape_dummy").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                            
                            self.builder.position_at_end(current_block); // Back to current
                            
                            // 2. Store value
                            self.builder.build_store(f32_ptr, float_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                            // 3. Call tl_tensor_new
                            let tensor_new_fn = self.module.get_function("tl_tensor_new").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_tensor_new not found".to_string())))?;
                            // tl_tensor_new(data: *const f32, rank: usize, shape: *const usize)
                            let zero_sz = self.context.i64_type().const_zero(); // rank = 0
                            
                            // Pass properly aligned dummy pointer
                            let shape_ptr = self.builder.build_pointer_cast(
                                shape_dummy_ptr,
                                self.context.ptr_type(inkwell::AddressSpace::default()),
                                "shape_ptr_cast"
                            ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                            let tensor_call = self.builder.build_call(
                                tensor_new_fn, 
                                &[f32_ptr.into(), zero_sz.into(), shape_ptr.into()], 
                                "pred_tensor"
                            ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                            
                            let res = match tensor_call.try_as_basic_value() {
                                inkwell::values::ValueKind::Basic(v) => v,
                                _ => return Err(CodegenErrorKind::Internal("Invalid return from tl_tensor_new".to_string()).into()),
                            };

                            // Register result
                            let res_ty = Type::Tensor(Box::new(Type::F32), 0);
                            self.add_temp(res, res_ty.clone());

                            Ok((res, res_ty))
                        } else {
                            Err(TlError::from(CodegenErrorKind::Internal("Query on non-tensor".to_string())))
                        }
                    }
                }
            }


            ExprKind::IfExpr(cond, then_stmts, else_stmts) => {
                let parent = self
                    .builder
                    .get_insert_block()
                    .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("unexpected None".to_string())))?
                    .get_parent()
                    .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("unexpected None".to_string())))?;

                let (cond_val, cond_ty) = self.compile_expr(cond)?;
                
                let cond_int = if let Type::Tensor(_, _) = cond_ty {
                    // Implicit conversion: Tensor -> Bool
                    // Assuming scalar tensor (0-dim or 1-element). extract item as int/bool.
                    // We can use `tl_tensor_item_i64` and cast to bool?
                    let item_fn = self.get_fn("tl_tensor_item_i64")?;
                    let call = self.builder.build_call(item_fn, &[cond_val.into()], "cond_item").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    let item_val = match call.try_as_basic_value() {
                         inkwell::values::ValueKind::Basic(v) => v,
                         _ => return Err(CodegenErrorKind::Internal("Expected basic value from tl_tensor_item_i64".to_string()).into()),
                    };
                    
                    self.builder.build_int_compare(
                        inkwell::IntPredicate::NE,
                        item_val.into_int_value(),
                        self.context.i64_type().const_int(0, false),
                        "cond_bool"
                    ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                } else if cond_val.is_pointer_value() {
                     // Condition is a Tensor (PointerValue). Extract boolean value.
                     // It is likely a Scalar Tensor from a comparison (e.g. gt, lt).
                     // use tl_tensor_item_i64 to extract value (0 or 1)
                     let item_fn = self.module.get_function("tl_tensor_item_i64").or_else(|| self.module.get_function("tl_tensor_item_f32").map(|f| {
                         // Fallback? f32 compare != 0.0?
                         // Ideally comparison returns i64-backed scalar tensor (0 or 1).
                         // Let's assume tl_tensor_item_i64 works for now or if missing try to cast.
                         // But wait, explicit check for tensor type was done via item_fn earlier?
                         // The panic says: cond_val is PointerValue.
                         // The unwrapped item_fn above was for "if cond_val is tensor but identified elsewhere?"
                         // Wait, let's look at lines 3046-3049 which I didn't see yet.
                         f
                     })).ok_or_else(|| TlError::from(CodegenErrorKind::Internal("Runtime function tl_tensor_item_i64 not found to resolve Tensor boolean condition".to_string())))?;
                     
                     let call = self.builder.build_call(item_fn, &[cond_val.into()], "cond_item").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                     let item_val = match call.try_as_basic_value() {
                          inkwell::values::ValueKind::Basic(v) => v,
                          _ => return Err(CodegenErrorKind::Internal("Expected basic value from tl_tensor_item_i64".to_string()).into()),
                     };
                     
                     // item_i64 returns i64. Comparison 0 or 1.
                     self.builder.build_int_compare(
                        inkwell::IntPredicate::NE,
                        item_val.into_int_value(),
                        self.context.i64_type().const_int(0, false),
                        "cond_bool"
                     ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                } else {
                     self
                    .builder
                    .build_int_cast(
                        cond_val.into_int_value(),
                        self.context.bool_type(),
                        "boolcast",
                    )
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                };


                let then_bb = self.context.append_basic_block(parent, "if_then");
                let else_bb = self.context.append_basic_block(parent, "if_else");
                let merge_bb = self.context.append_basic_block(parent, "if_merge");

                self.builder
                    .build_conditional_branch(cond_int, then_bb, else_bb)
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // Then branch
                self.builder.position_at_end(then_bb);
                self.enter_scope();
                let mut then_val: Option<(BasicValueEnum<'ctx>, Type)> = None;
                for (i, stmt) in then_stmts.iter().enumerate() {
                    if i == then_stmts.len() - 1 {
                        if let StmtKind::Expr(e) = &stmt.inner {
                            then_val = Some(self.compile_expr(e)?);
                        } else {
                            self.compile_stmt(stmt)?;
                        }
                    } else {
                        self.compile_stmt(stmt)?;
                    }
                }
                // Default value if no expression
                let then_result = then_val.unwrap_or((
                    self.context.i64_type().const_int(0, false).into(),
                    Type::Void,
                ));

                // Scope Promotion Logic
                // If result is in current scope (Temporary), remove it so it isn't freed. (Move)
                // If result is NOT in current scope (L-value), CLONE it. (Copy)

                let mut then_final_val = then_result.0;

                if matches!(
                    then_result.1,
                    Type::Tensor(_, _) | Type::Struct(_, _) | Type::Tuple(_) | Type::String(_)
                ) {
                    // Logic:
                    // 1. Check AST of last stmt.
                    let is_lvalue = if let Some(last) = then_stmts.last() {
                        match &last.inner {
                            StmtKind::Expr(e) => matches!(
                                &e.inner,
                                ExprKind::Variable(_)
                                    | ExprKind::FieldAccess(_, _)
                                    | ExprKind::IndexAccess(_, _)
                            ),
                            _ => false, // Stmt that isn't Expr? Void.
                        }
                    } else {
                        false
                    }; // Empty body -> Void

                    if is_lvalue {
                        // L-Value: Must Clone to return Owned.
                        then_final_val = self.emit_deep_clone(then_final_val, &then_result.1)?;
                        // Original remains in parent scope. Safe.
                    } else {
                        // R-Value (Temporary in this scope):
                        // We MUST prevent exit_scope from freeing it.
                        // FIX: Mark the temporary as CLEANUP_NONE in the compiler's temporaries list.
                        // This prevents emit_cleanup_vars_in_scope from calling emit_recursive_free.
                        self.mark_temp_no_cleanup(then_final_val);
                        
                        // Also call runtime UNREGISTER to remove from memory manager.
                        if let Some(unreg_fn) = self.module.get_function("tl_mem_unregister") {
                            let ptr = then_final_val.into_pointer_value();
                            let cast_ptr = self
                                .builder
                                .build_pointer_cast(
                                    ptr,
                                    self.context.ptr_type(inkwell::AddressSpace::default()),
                                    "cast",
                                )
                                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                            self.builder
                                .build_call(unreg_fn, &[cast_ptr.into()], "")
                                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        }
                    }
                }

                self.exit_scope();

                // Get the block after cleanup (important for PHI incoming)
                let then_final_bb = self.current_block()?;
                if then_final_bb.get_terminator().is_none() {
                    self.builder
                        .build_unconditional_branch(merge_bb)
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                }

                // Else branch
                self.builder.position_at_end(else_bb);
                self.enter_scope();
                let mut else_val: Option<(BasicValueEnum<'ctx>, Type)> = None;
                if let Some(else_body) = else_stmts {
                    for (i, stmt) in else_body.iter().enumerate() {
                        if i == else_body.len() - 1 {
                            if let StmtKind::Expr(e) = &stmt.inner {
                                else_val = Some(self.compile_expr(e)?);
                            } else {
                                self.compile_stmt(stmt)?;
                            }
                        } else {
                            self.compile_stmt(stmt)?;
                        }
                    }
                }
                let else_result = else_val.unwrap_or((
                    self.context.i64_type().const_int(0, false).into(),
                    Type::Void,
                ));

                let mut else_final_val = else_result.0;

                if matches!(
                    else_result.1,
                    Type::Tensor(_, _) | Type::Struct(_, _) | Type::Tuple(_) | Type::String(_)
                ) {
                    // Logic:
                    // 1. Check AST of last stmt.
                    let is_lvalue = if let Some(body) = &else_stmts {
                        if let Some(last) = body.last() {
                            match &last.inner {
                                StmtKind::Expr(e) => matches!(
                                    &e.inner,
                                    ExprKind::Variable(_)
                                        | ExprKind::FieldAccess(_, _)
                                        | ExprKind::IndexAccess(_, _)
                                ),
                                _ => false,
                            }
                        } else {
                            false
                        }
                    } else {
                        false
                    };

                    if is_lvalue {
                        else_final_val = self.emit_deep_clone(else_final_val, &else_result.1)?;
                    } else {
                        // FIX: Mark the temporary as CLEANUP_NONE in the compiler's temporaries list.
                        self.mark_temp_no_cleanup(else_final_val);
                        
                        if let Some(unreg_fn) = self.module.get_function("tl_mem_unregister") {
                            let ptr = else_final_val.into_pointer_value();
                            let cast_ptr = self
                                .builder
                                .build_pointer_cast(
                                    ptr,
                                    self.context.ptr_type(inkwell::AddressSpace::default()),
                                    "cast",
                                )
                                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                            self.builder
                                .build_call(unreg_fn, &[cast_ptr.into()], "")
                                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        }
                    }
                }

                self.exit_scope();

                // Get the block after cleanup
                let else_final_bb = self.current_block()?;
                if else_final_bb.get_terminator().is_none() {
                    self.builder
                        .build_unconditional_branch(merge_bb)
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                }

                // Merge block with PHI
                self.builder.position_at_end(merge_bb);

                // Only create PHI if both branches return non-void values
                if !matches!(then_result.1, Type::Void) && !matches!(else_result.1, Type::Void) {
                    let llvm_ty: inkwell::types::BasicTypeEnum = match &then_result.1 {
                        Type::I64 => self.context.i64_type().into(),
                        Type::F32 => self.context.f32_type().into(),
                        Type::Bool => self.context.bool_type().into(),
                        Type::Tensor(_, _)
                        | Type::Struct(_, _)
                        | Type::String(_)
                        | Type::Tuple(_) | Type::SpecializedType { .. } => self
                            .context
                            .ptr_type(inkwell::AddressSpace::default())
                            .into(),
                        Type::F64 => self.context.f64_type().into(),
                        Type::I32 | Type::Char(_) => self.context.i32_type().into(),
                        Type::U8 => self.context.i8_type().into(),
                        Type::Usize | Type::Entity => self.context.i64_type().into(),
                        Type::Enum(_, _) | Type::Ptr(_) | Type::GradTensor(_, _)
                        | Type::TensorShaped(_, _) => self
                            .context
                            .ptr_type(inkwell::AddressSpace::default())
                            .into(),
                        Type::Void => self.context.i8_type().into(),
                        Type::Array(inner, size) => self.get_llvm_type(&Type::Array(inner.clone(), *size)).unwrap_or(self.context.i64_type().into()),
                        Type::Path(_, _) | Type::Fn(_, _) | Type::I8 | Type::I16 | Type::U16 | Type::U32 | Type::U64 | Type::F16 | Type::BF16 | Type::TypeVar(_) | Type::Never | Type::Undefined(_) | Type::Range => self.context.i64_type().into(), Type::TraitObject(_) => { let p = self.context.ptr_type(inkwell::AddressSpace::default()); self.context.struct_type(&[p.into(), p.into()], false).into() },
                    };

                    let phi = self
                        .builder
                        .build_phi(llvm_ty, "if_result")
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    phi.add_incoming(&[
                        (&then_final_val, then_final_bb),
                        (&else_final_val, else_final_bb),
                    ]);

                    // FIX: Register the PHI result as a temporary
                    // The incoming values (then_result, else_result) were unregistered in their respective blocks
                    // to prevent freeing. Now we have a new value (phi) representing the result.
                    // We must register it so it is managed in the PARENT scope.
                    let res_val = phi.as_basic_value();
                    let res_ty = then_result.1.clone(); // Assume types match (semantics checks this)

                    if matches!(
                        res_ty,
                        Type::Tensor(_, _)
                            | Type::Struct(_, _)
                            | Type::String(_)
                            | Type::Tuple(_)
                    ) {
                        self.emit_register_tensor(res_val, &res_ty)?;
                        // Note: emit_register_tensor handles recursion for Structs/Tuples if implemented,
                        // but currently it checks for Type::Tensor.
                        // For structs/tuples we might need `gen_register_params` or similar if they are managed.
                        // However, `IfExpr` typically unregisters the *value pointer*.

                        // If we unregistered the *inputs*, they are effectively "moved" to the PHI node.
                        // But wait, `tl_mem_unregister` tells runtime "stop tracking this pointer".
                        // The PHI node returns the *same pointer* (incoming from blocks).
                        // So the pointer itself is valid, but runtime doesn't track it.
                        // We need to re-register it.

                        if let Type::Tensor(_, _) = res_ty {
                            // Already called emit_register_tensor above
                        } else if let Type::Struct(_, _) | Type::String(_) = &res_ty {
                            if let Some(reg_fn) = self.module.get_function("tl_mem_register_struct")
                            {
                                let ptr = res_val.into_pointer_value();
                                let cast_ptr = self
                                    .builder
                                    .build_pointer_cast(
                                        ptr,
                                        self.context.ptr_type(inkwell::AddressSpace::default()),
                                        "cast",
                                    )
                                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                                self.builder
                                    .build_call(reg_fn, &[cast_ptr.into()], "")
                                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                            }
                        }
                    }

                    Ok((res_val, res_ty))
                } else {
                    Ok((
                        self.context.i64_type().const_int(0, false).into(),
                        Type::Void,
                    ))
                }
            }
        }
    }

    // Refactored helper to allow calling from recovery path

    pub fn compile_string_literal(&self, s: &str) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
        let s_null_term = format!("{}\0", s);
        let str_val = self
            .builder
            .build_global_string_ptr(&s_null_term, "str_lit")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
            .as_pointer_value();

        // Create String object
        let new_fn = self
            .module
            .get_function("tl_string_new")
            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_string_new not found".to_string())))?;
        let call = self
            .builder
            .build_call(new_fn, &[str_val.into()], "string_obj")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        let ptr = match call.try_as_basic_value() {
            inkwell::values::ValueKind::Basic(v) => v,
            _ => return Err(CodegenErrorKind::Internal("tl_string_new returned void".to_string()).into()),
        };

        Ok((ptr, Type::String("String".to_string())))
    }

    fn compile_tuple(&mut self, elements: &[Expr]) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
        let mut vals = Vec::new();
        let mut types = Vec::new();
        let mut llvm_types = Vec::new();

        for e in elements {
            let (val, ty) = self.compile_expr(e)?;
            vals.push(val);
            types.push(ty.clone());
            llvm_types.push(self.get_llvm_type(&ty)?);
        }

        let tuple_struct_type = self.context.struct_type(&llvm_types, false);

        // Malloc
        // We use malloc to ensure it survives if returned (and managed by owner).
        // Similar to StructInit.
        let size = tuple_struct_type
            .size_of()
            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("Cannot get size of tuple".to_string())))?;
        
        let size_int = size;
        let size_i64 = if size_int.get_type() == self.context.i32_type() {
             self.builder.build_int_z_extend(size_int, self.context.i64_type(), "size_i64").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
        } else {
             size_int
        };

        let malloc_fn = self
            .module
            .get_function("malloc")
            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("malloc not found".to_string())))?;
        let call = self
            .builder
            .build_call(malloc_fn, &[size_i64.into()], "tuple_malloc")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        let raw_ptr = match call.try_as_basic_value() {
            ValueKind::Basic(v) => v.into_pointer_value(),
            _ => return Err(CodegenErrorKind::Internal("malloc returned instruction value".to_string()).into()),
        };

        // Cast to tuple struct pointer
        let tuple_ptr = self
            .builder
            .build_pointer_cast(
                raw_ptr,
                self.context.ptr_type(inkwell::AddressSpace::default()),
                "tuple_ptr",
            )
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        // Store elements
        for (i, (val, ty)) in vals.iter().zip(types.iter()).enumerate() {
            let field_ptr = self
                .builder
                .build_struct_gep(tuple_struct_type, tuple_ptr, i as u32, "tuple_field")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder
                .build_store(field_ptr, *val)
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            // Ownership transfer: Unregister from scope if it's a temp/managed type
            if matches!(
                ty,
                Type::Tensor(_, _) | Type::Struct(_, _) | Type::Tuple(_) | Type::String(_)
            ) {
                let unreg_fn = self
                    .module
                    .get_function("tl_mem_unregister")
                    .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_mem_unregister not found".to_string())))?;
                let val_ptr = val.into_pointer_value();
                let cast_ptr = self
                    .builder
                    .build_pointer_cast(
                        val_ptr,
                        self.context.ptr_type(inkwell::AddressSpace::default()),
                        "cast_unreg_elem",
                    )
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder
                    .build_call(unreg_fn, &[cast_ptr.into()], "")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // If element was a variable, we need to remove it from scope tracking appropriately?
                // compile_expr for variable returns a LOADed value usually, or the pointer?
                // compile_expr(Variable) returns (Load(alloca), Type).
                // So 'val' is the pointer to the object (if Struct/Tensor).
                // For Variables, we might need similar logic to StructInit to prevent double-free if we "move" it.
                // But generally tuple construction is: let t = (a, b); -> t owns copies of a and b (ref copy).
                // If a, b are variables, we are COPYING the reference. So we should ACQUIRE?
                // Wait, StructInit unregisters?
                // StructInit logic:
                // if arg is Variable: remove from scope (Move semantics?) -> Logic says "should_remove = true".
                // Yes, StructInit implements Move semantics for variables passed to it.
                // So let's replicate that for Tuple.
                if let ExprKind::Variable(var_name) = &elements[i].inner {
                    // Search and remove variable from scope to prevent automatic free at end of scope
                    // Effectively "Moving" ownership to the tuple.
                    for scope in self.variables.iter_mut().rev() {
                        let mut should_remove = false;
                        if let Some((_, var_ty, _)) = scope.get(var_name) {
                            if matches!(
                                var_ty,
                                Type::Tensor(_, _)
                                    | Type::Struct(_, _)
                                    | Type::Tuple(_)
                                    | Type::String(_)
                            ) {
                                should_remove = true;
                            }
                        }
                        if should_remove {
                            scope.remove(var_name); // Remove from tracking -> No free at exit_scope
                            break;
                        }
                    }
                }
            }
        }

        Ok((tuple_ptr.into(), Type::Tuple(types)))
    }

    fn compile_tuple_access(
        &mut self,
        expr: &Expr,
        idx: usize,
    ) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
        let (tuple_val, tuple_ty) = self.compile_expr(expr)?;

        let element_types = match tuple_ty {
            Type::Tuple(ts) => ts,
            _ => return Err(TlError::from(CodegenErrorKind::Internal(format!("Expected tuple type, found {:?}", tuple_ty)))),
        };

        if idx >= element_types.len() {
            return Err(TlError::from(CodegenErrorKind::Internal(format!(
                "Tuple index {} out of bounds (len {})",
                idx,
                element_types.len()
            ))));
        }
        let field_ty = element_types[idx].clone();

        // tuple_val should be a pointer to the tuple struct (i8*)
        if !tuple_val.is_pointer_value() {
            return Err(TlError::from(CodegenErrorKind::Internal("Tuple value is not a pointer".to_string())));
        }
        let tuple_ptr = tuple_val.into_pointer_value();

        // Reconstruct LLVM struct type for GEP
        let mut llvm_types = Vec::new();
        for ty in &element_types {
            llvm_types.push(self.get_llvm_type(ty)?);
        }
        let tuple_struct_type = self.context.struct_type(&llvm_types, false);

        // GEP
        let field_ptr = self
            .builder
            .build_struct_gep(tuple_struct_type, tuple_ptr, idx as u32, "tuple_access")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        // Load
        let llvm_field_ty = self.get_llvm_type(&field_ty)?;
        let val = self
            .builder
            .build_load(llvm_field_ty, field_ptr, "tuple_elem")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        // FIX: Retain ownership (TupleAccess returns new reference)
        self.emit_retain(val, &field_ty)?;
        // FIX: Register as temporary
        self.add_temp(val, field_ty.clone());

        Ok((val, field_ty))
    }

    fn compile_tuple_struct_init(
        &mut self,
        name: &str,
        args: &[Expr],
    ) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
        // Extract simple name from module path
        let simple_name = name;

        let struct_type = *self
            .struct_types
            .get(simple_name)
            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Struct type {} not found in codegen", name))))?;

        let struct_def = self
            .struct_defs
            .get(simple_name)
            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Struct definition {} not found", name))))?
            .clone();

        if args.len() != struct_def.fields.len() {
            return Err(TlError::from(CodegenErrorKind::Internal(format!(
                "Field count mismatch for struct {}: expected {}, found {}",
                name,
                struct_def.fields.len(),
                args.len()
            ))));
        }

        // 1. Heap Allocation
        let malloc_fn = self
            .module
            .get_function("malloc")
            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("malloc not found".to_string())))?;
        let size = struct_type
            .size_of()
            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Cannot determine size of struct {}", name))))?;
            
        let size_int = size;
        let size_i64 = if size_int.get_type() == self.context.i32_type() {
             self.builder.build_int_z_extend(size_int, self.context.i64_type(), "size_i64").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
        } else {
             size_int
        };

        let call = self
            .builder
            .build_call(malloc_fn, &[size_i64.into()], "struct_malloc")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        let raw_ptr = match call.try_as_basic_value() {
            inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
            _ => return Err(CodegenErrorKind::Internal("malloc returned invalid value".to_string()).into()),
        };

        // 2. Register
        // 2. Register
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
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        let name_global = self
            .builder
            .build_global_string_ptr(name, "struct_name")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        self.builder
            .build_call(register_fn, &[cast_ptr.into(), name_global.as_pointer_value().into()], "")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        // Debug trace
        if name.contains("GPT") {
            let size_val = self.context.i64_type().const_int(0, false);
            self.emit_log_alloc(cast_ptr.into(), size_val).ok();
        }

        let struct_ptr = self
            .builder
            .build_pointer_cast(
                raw_ptr,
                self.context.ptr_type(inkwell::AddressSpace::default()),
                "struct_ptr",
            )
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        for (i, arg_expr) in args.iter().enumerate() {
            let (val, _ty) = self.compile_expr(arg_expr)?;
            let field_ptr = self
                .builder
                .build_struct_gep(
                    struct_type,
                    struct_ptr,
                    i as u32,
                    &format!("{}.{}", name, i),
                )
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            let store_val = if matches!(
                _ty,
                Type::Tensor(_, _) | Type::Struct(_, _) | Type::Tuple(_) | Type::String(_)
            ) {
                self.emit_deep_clone(val, &_ty)?
            } else {
                val
            };
            self.builder
                .build_store(field_ptr, store_val)
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            // Move Semantics Removed (same as compile_struct_init):
            // We now use RefCounting/DeepCopy. Source variable should remain valid (Shared ownership).
            // Cleanup at end of scope will decrement refcount (balancing the Acquire in emit_deep_clone).
            // No manual unregister. No removal from scope.
        }

        // Return the pointer directly (no load)
        // Struct remains registered in scope; caller (StmtKind::Let) will deep_clone it
        // Return the pointer directly (no load)
        // Struct remains registered in scope; caller (StmtKind::Let) will deep_clone it
        
        Ok((struct_ptr.into(), Type::Struct(name.to_string(), vec![])))
    }

    #[allow(deprecated)]
    pub(crate) fn compile_static_method_call(
        &mut self,
        struct_name: &str,
        method: &str,
        args: &[Expr],
        target_type: &Type,
    ) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
        // Normalize target_type to resolve Path types to Struct/Enum
        let normalized_target = self.normalize_type(target_type);
        let target_type = normalized_target.clone();

        let target_type = &target_type;
        
        // Compatibility aliases for existing logic
        let is_grad_tensor_call = struct_name == "GradTensor";
        let type_name = if is_grad_tensor_call { "Tensor" } else { struct_name };

        // GradTensor factory: delegate to Tensor, then enable_grad and retype
        if is_grad_tensor_call {
            let (val, ty) = self.compile_static_method_call("Tensor", method, args, target_type)?;
            // enable_grad on the newly created tensor
            let enable_fn = self
                .module
                .get_function("tl_tensor_enable_grad")
                .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_tensor_enable_grad not found".to_string())))?;
            self.builder
                .build_call(enable_fn, &[val.into()], "enable_grad")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            // Convert Tensor type to GradTensor type
            let grad_ty = match ty {
                Type::Tensor(inner, rank) => Type::GradTensor(inner, rank),
                other => other,
            };
            return Ok((val, grad_ty));
        }
        
        let is_channel_call = struct_name.starts_with("Channel") || type_name == "Channel";
        if is_channel_call && method == "new" {
            if args.len() != 1 {
                return Err(TlError::from(CodegenErrorKind::Internal("Channel::new requires 1 argument (capacity)".to_string())));
            }
            let (capacity_val, _) = self.compile_expr(&args[0])?;
            let capacity_i64 = capacity_val.into_int_value();
            
            let new_fn = self.module.get_function("tl_channel_new").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_channel_new not found".to_string())))?;
            let call = self.builder.build_call(new_fn, &[capacity_i64.into()], "call_channel_new").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let id_val = match call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v.into_int_value(),
                _ => return Err(CodegenErrorKind::Internal("tl_channel_new returned void".to_string()).into()),
            };
            let type_args_vec = if let Type::Struct(_, targs) = target_type { targs.clone() } else { vec![] };
            let type_args = type_args_vec.as_slice();
            let struct_name = self.mangle_type_name("Channel", type_args);
            let llvm_struct_ty = self.context.get_struct_type(&struct_name).ok_or_else(|| TlError::from(CodegenErrorKind::Internal("unexpected None".to_string())))?;
            let malloc_fn = self.module.get_function("malloc").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("malloc not found".to_string())))?;
            let size_val = llvm_struct_ty.size_of().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("type has no size".to_string())))?;
            let call_malloc = self.builder.build_call(malloc_fn, &[size_val.into()], "malloc_ch").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let raw_ptr = match call_malloc.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
                _ => return Err(CodegenErrorKind::Internal("malloc returned void".to_string()).into()),
            };
            let struct_ptr = self.builder.build_pointer_cast(raw_ptr, self.context.ptr_type(inkwell::AddressSpace::default()), "struct_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let id_ptr = self.builder.build_struct_gep(llvm_struct_ty, struct_ptr, 0, "id_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_store(id_ptr, id_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            return Ok((struct_ptr.into(), target_type.clone()));
        }

        let is_thread_call = struct_name.starts_with("Thread") || type_name == "Thread";
        if is_thread_call && method == "spawn" {
            if args.len() != 1 {
                return Err(TlError::from(CodegenErrorKind::Internal("Thread::spawn requires 1 argument (closure)".to_string())));
            }
            
            // Expected element type T
            let inner_ty = self.extract_inner_ty(&self.normalize_type(target_type));
            
            let (closure_val, _closure_ty) = self.compile_expr(&args[0])?;
            let closure_struct = closure_val.into_struct_value();
            
            let fn_ptr = self.builder.build_extract_value(closure_struct, 0, "fn_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_pointer_value();
            let env_ptr = self.builder.build_extract_value(closure_struct, 1, "env_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_pointer_value();
            
            // To support arbitrary returning closures without ABI mismatch, we must generate a wrapper function
            // that invokes fn_ptr(env_ptr), allocates memory using tl_alloc_tmp, stores the result, 
            // and returns *mut c_void. BUT we can't easily dynamically invoke an unknown LLVM function pointer
            // without knowing its exact LLVM signature in the trampoline!
            // Wait, actually, fn_ptr IS statically known at this point if it's evaluated, but in TL closures
            // are always `fn(*mut c_void) -> T`. We CAN create a trampoline because we know `T`'s LLVM Type!
            
            let llvm_ret_ty = self.get_llvm_type(&inner_ty)?;
            let uses_sret = matches!(&inner_ty, Type::Struct(name, _) if name != "Tensor" && name != "String");
            let closure_llvm_ty = if uses_sret {
                // If it uses sret, the signature is `void (T*, i8*)`
                self.context.void_type().fn_type(&[
                    self.context.ptr_type(inkwell::AddressSpace::default()).into(),
                    self.context.ptr_type(inkwell::AddressSpace::default()).into()
                ], false)
            } else {
                llvm_ret_ty.fn_type(&[self.context.ptr_type(inkwell::AddressSpace::default()).into()], false)
            };
            
            static mut TRAMPOLINE_ID: u64 = 0;
            let tid = unsafe { TRAMPOLINE_ID += 1; TRAMPOLINE_ID };
            let trampoline_name = format!("tl_thread_trampoline_{:x}", tid);
            
            let trampoline_sig = self.context.ptr_type(inkwell::AddressSpace::default()).fn_type(&[
                self.context.ptr_type(inkwell::AddressSpace::default()).into()
            ], false);
            
            let trampoline_fn = self.module.add_function(&trampoline_name, trampoline_sig, None);
            let prev_bb = self.current_block()?;
            
            let basic_block = self.context.append_basic_block(trampoline_fn, "entry");
            self.builder.position_at_end(basic_block);
            
            let env_param = trampoline_fn.get_first_param().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("unexpected None".to_string())))?.into_pointer_value();
            
            // Re-cast fn_ptr inside trampoline? We don't have fn_ptr inside trampoline because it's dynamic!
            // Ah! We must pass BOTH fn_ptr and env_ptr to the trampoline.
            // But tl_thread_spawn only takes `env_ptr`.
            // We can allocate a `{ fn_ptr, env_ptr }` bundle on the heap, and pass THAT as env_ptr!
            let bundle_ty = self.context.struct_type(&[
                self.context.ptr_type(inkwell::AddressSpace::default()).into(),
                self.context.ptr_type(inkwell::AddressSpace::default()).into()
            ], false);
            
            let ext_env_param = self.builder.build_pointer_cast(env_param, self.context.ptr_type(inkwell::AddressSpace::default()), "bundle_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let dyn_fn_ptr = self.builder.build_load(self.context.ptr_type(inkwell::AddressSpace::default()), self.builder.build_struct_gep(bundle_ty, ext_env_param, 0, "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?, "dyn_fn").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_pointer_value();
            let dyn_env_ptr = self.builder.build_load(self.context.ptr_type(inkwell::AddressSpace::default()), self.builder.build_struct_gep(bundle_ty, ext_env_param, 1, "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?, "dyn_env").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_pointer_value();
            
            // free the bundle
            let free_tmp_fn = self.get_fn("tl_free_tmp")?;
            self.builder.build_call(free_tmp_fn, &[ext_env_param.into()], "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            
            let size = if llvm_ret_ty.is_sized() { llvm_ret_ty.size_of().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("type has no size".to_string())))? } else { self.context.i64_type().const_zero() };
            let size_i64 = if size.get_type() == self.context.i32_type() { self.builder.build_int_z_extend(size, self.context.i64_type(), "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))? } else { size };
            let alloc_tmp_fn = self.get_fn("tl_alloc_tmp")?;
            let call_buf = self.builder.build_call(alloc_tmp_fn, &[size_i64.into()], "buf").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let buf_ptr = match call_buf.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
                _ => return Err(CodegenErrorKind::Internal("tl_alloc_tmp returned void".to_string()).into()),
            };
            let typed_buf = self.builder.build_pointer_cast(buf_ptr, self.context.ptr_type(inkwell::AddressSpace::default()), "typed_buf").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            
            if uses_sret {
                self.builder.build_indirect_call(closure_llvm_ty, dyn_fn_ptr, &[typed_buf.into(), dyn_env_ptr.into()], "call").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            } else {
                let call_ret = self.builder.build_indirect_call(closure_llvm_ty, dyn_fn_ptr, &[dyn_env_ptr.into()], "call").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let ret_val = match call_ret.try_as_basic_value() {
                    inkwell::values::ValueKind::Basic(v) => v,
                    _ => return Err(CodegenErrorKind::Internal("closure returned void unexpectedly".to_string()).into()),
                };
                self.builder.build_store(typed_buf, ret_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            }
            // Return heap buffer
            self.builder.build_return(Some(&buf_ptr)).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            
            // Restore builder
            self.builder.position_at_end(prev_bb);
            
            // Prepare bundle to send
            let bundle_size = bundle_ty.size_of().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("type has no size".to_string())))?;
            let bundle_size_i64 = if bundle_size.get_type() == self.context.i32_type() { self.builder.build_int_z_extend(bundle_size, self.context.i64_type(), "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))? } else { bundle_size };
            let call_bundle = self.builder.build_call(alloc_tmp_fn, &[bundle_size_i64.into()], "bundle").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let bundle_raw = match call_bundle.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
                _ => return Err(CodegenErrorKind::Internal("tl_alloc_tmp returned void".to_string()).into()),
            };
            let bundle_typed = self.builder.build_pointer_cast(bundle_raw, self.context.ptr_type(inkwell::AddressSpace::default()), "bundle_typed").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_store(self.builder.build_struct_gep(bundle_ty, bundle_typed, 0, "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?, fn_ptr).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_store(self.builder.build_struct_gep(bundle_ty, bundle_typed, 1, "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?, env_ptr).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            
            let spawn_fn = self.get_fn("tl_thread_spawn")?;
            let call = self.builder.build_call(spawn_fn, &[trampoline_fn.as_global_value().as_pointer_value().into(), bundle_raw.into()], "spawn").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let thread_id = match call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v,
                _ => return Err(CodegenErrorKind::Internal("tl_thread_spawn returned void".to_string()).into()),
            };
            
            let t_name = if let Type::Struct(name, _) = self.normalize_type(target_type) { name } else { "Thread".to_string() };
            let t_struct_ty = self.context.struct_type(&[self.context.i64_type().into()], false);
            let ret_ptr = self.builder.build_alloca(t_struct_ty, "thread_struct").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_store(self.builder.build_struct_gep(t_struct_ty, ret_ptr, 0, "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?, thread_id).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            
            return Ok((ret_ptr.into(), Type::Struct(t_name, vec![inner_ty])));
        }

        let is_mutex_call = struct_name.starts_with("Mutex") || type_name == "Mutex";
        if is_mutex_call && method == "new" {
            if args.len() != 1 {
                return Err(TlError::from(CodegenErrorKind::Internal("Mutex::new requires 1 argument".to_string())));
            }
            
            let arg_expr = &args[0];
            let (val, ty) = self.compile_expr(arg_expr)?;
            
            let llvm_type = self.get_llvm_type(&ty)?;
            let size = if llvm_type.is_sized() { llvm_type.size_of().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("type has no size".to_string())))? } else { self.context.i64_type().const_zero() };
            // Cast size to i64
            let size_i64 = if size.get_type() == self.context.i32_type() {
                 self.builder.build_int_z_extend(size, self.context.i64_type(), "size_i64").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
            } else {
                 size
            };
            
            let temp_ptr = self.builder.build_alloca(llvm_type, "mutex_val").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let store_val = if matches!(ty, Type::Tensor(_, _) | Type::Struct(_, _) | Type::Tuple(_) | Type::String(_)) {
                self.emit_deep_clone(val, &ty)?
            } else {
                val
            };
            self.builder.build_store(temp_ptr, store_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            
            let data_ptr_cast = self.builder.build_pointer_cast(temp_ptr, self.context.ptr_type(inkwell::AddressSpace::default()), "m_val_cast").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let new_fn = self.get_fn("tl_mutex_new")?;
            let call = self.builder.build_call(new_fn, &[size_i64.into(), data_ptr_cast.into()], "m_id").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let id_val = match call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v,
                _ => return Err(CodegenErrorKind::Internal("tl_mutex_new returned void".to_string()).into()),
            };
            
            // To construct Mutex<T>, we need to dynamically allocate it as a struct.
            let m_name = if let Type::Struct(name, _) = self.normalize_type(target_type) { name } else { "Mutex".to_string() };
            let t_struct_ty = self.context.struct_type(&[self.context.i64_type().into()], false);
            
            let malloc_fn = self.get_fn("malloc")?;
            let m_size = t_struct_ty.size_of().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("type has no size".to_string())))?;
            let m_size_i64 = if m_size.get_type() == self.context.i32_type() {
                 self.builder.build_int_z_extend(m_size, self.context.i64_type(), "size_i64").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
            } else {
                 m_size
            };
            let call_malloc = self.builder.build_call(malloc_fn, &[m_size_i64.into()], "m_malloc").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let raw_ptr = match call_malloc.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
                _ => return Err(CodegenErrorKind::Internal("malloc returned void".to_string()).into()),
            };
            
            // register
            let cast_ptr = self.builder.build_pointer_cast(raw_ptr, self.context.ptr_type(inkwell::AddressSpace::default()), "cast_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let name_global = self.builder.build_global_string_ptr(&m_name, "struct_name").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let register_fn = self.get_fn("tl_mem_register_struct_named")?;
            self.builder.build_call(register_fn, &[cast_ptr.into(), name_global.as_pointer_value().into()], "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            
            // store id
            let m_ptr = self.builder.build_pointer_cast(raw_ptr, self.context.ptr_type(inkwell::AddressSpace::default()), "m_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let id_gep = self.builder.build_struct_gep(t_struct_ty, m_ptr, 0, "id_f").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_store(id_gep, id_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            
            return Ok((m_ptr.into(), Type::Struct(m_name, vec![ty])));
        }

        // 0. Check if this is an Enum Variant initialization (priority check)
        //    This handles cases like Entry_i64_i64::Empty where the type_name is
        //    the mangled enum name and method is a variant name.
        if let Some(mut enum_def) = self.enum_defs.get(struct_name).cloned() {
            if let Some(variant_idx) = enum_def.variants.iter().position(|v| v.name == method) {
                // If enum_def is still generic, monomorphize with default type
                if !enum_def.generics.is_empty() {
                    // Use type args from target_type if available
                    let default_generics = match target_type {
                        Type::Struct(_, args) | Type::Enum(_, args) if !args.is_empty() => args.clone(),
                        _ => return Err(TlError::from(CodegenErrorKind::Internal(format!("Enum {} lacks generic parameters in compile_static_method_call. Implicit fallback is strictly prohibited.", struct_name)))),
                    };
                    let mangled = self.mangle_type_name(struct_name, &default_generics);
                    if let Some(specialized) = self.enum_defs.get(&mangled) {
                        enum_def = specialized.clone();
                    } else {
                        self.monomorphize_enum(struct_name, &default_generics).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        enum_def = self.enum_defs.get(&mangled)
                            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Monomorphization failed for {} -> {}", struct_name, mangled))))?
                            .clone();
                    }
                }
                // This is an enum variant constructor, compile it as EnumInit
                return self.compile_enum_variant_as_static_method_call(
                    &enum_def.name, method, args, variant_idx, &enum_def, target_type
                );
            }
        }

        // 1. Try TypeManager (AST-defined methods) - find matching overload
        if let Some(type_info) = self.type_manager.get_type(type_name) {
            if type_info.has_static_method(method) {
                // Get overloads and try to find a match
                if let Some(overloads) = type_info.get_static_overloads(method) {
                    // For Unevaluated methods, we need to try each overload
                    // For now, try the first unevaluated if any, or match by arg count
                    for overload in overloads {
                        if overload.arg_types.len() == args.len() {
                            match &overload.impl_fn {
                                StaticMethod::Evaluated(func) => {
                                    let func = *func;
                                    let mut compiled_args = Vec::new();
                                    for arg in args {
                                        compiled_args.push(self.compile_expr(arg)?);
                                    }
                                    return func(self, compiled_args, Some(target_type));
                                }
                                StaticMethod::Unevaluated(func) => {
                                    let func = *func;
                                    return func(self, args, Some(target_type));
                                }
                                StaticMethod::SignatureOnly => {
                                    // This method has no implementation, skip this overload
                                    continue;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // 2. Check self.static_methods (registered via register_all_methods)
        // Copy the method to avoid borrow conflict with self.compile_expr
        let method_opt = self.static_methods.get(type_name).and_then(|m| m.get(method).copied());
        if let Some(method_fn) = method_opt {
            match method_fn {
                StaticMethod::Evaluated(func) => {
                    let mut compiled_args = Vec::new();
                    for arg in args {
                        compiled_args.push(self.compile_expr(arg)?);
                    }
                    return func(self, compiled_args, Some(target_type));
                }
                StaticMethod::Unevaluated(func) => {
                    return func(self, args, Some(target_type));
                }
                StaticMethod::SignatureOnly => {
                    // Should not happen for registered methods, fall through
                }
            }
        }

        // 2.5. Built-in Map::load using tl_gguf_load
        if type_name == "Map" && method == "load" {
            if args.len() != 1 {
                return Err(TlError::from(CodegenErrorKind::Internal("Map::load requires 1 argument".to_string())));
            }
            let (path_val, path_ty) = self.compile_expr(&args[0])?;
            
            // TL String is StringStruct { ptr: *c_char, len: i64 }
            // tl_gguf_load expects *mut StringStruct
            let string_struct_ptr = if matches!(path_ty, Type::String(_)) {
                let struct_ptr = path_val.into_pointer_value();
                struct_ptr
            } else {
                return Err(TlError::from(CodegenErrorKind::Internal(format!("Map::load expects String argument, got {:?}", path_ty))));
            };
            
            let fn_val = self
                .module
                .get_function("tl_gguf_load")
                .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_gguf_load not found".to_string())))?;
            
            // Cast to generic pointer if needed (though struct ptr usually works)
            let cast_ptr = self.builder.build_pointer_cast(
                string_struct_ptr,
                self.context.ptr_type(inkwell::AddressSpace::default()), 
                "string_struct_ptr"
            ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            let call = self
                .builder
                .build_call(fn_val, &[cast_ptr.into()], "map_load")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                
            let res = match call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v,
                _ => return Err(CodegenErrorKind::Internal("Invalid return from Map::load".to_string()).into()),
            };
            return Ok((res, Type::Struct("Map".to_string(), vec![])));
        }

        // 3. User Generic Fallback (Monomorphize)
        // Normalize target_type to convert Path to Struct/Enum
        let normalized_target_type = self.normalize_type(target_type);
        let generics = match &normalized_target_type {
            Type::Struct(n, args) if n == type_name => args.clone(),
            Type::Enum(n, args) if n == type_name => args.clone(),
            _ => vec![],
        };
        
        let generic_result = if self.generic_impls.contains_key(type_name) {
             if let Ok(name) = self.monomorphize_method(type_name, method, &generics) {
                 self.module.get_function(&name).map(|f| (f, name))
             } else {
                 None
             }
        } else {
            None
        };
        
        let simple_type_name = &struct_name;
        let mangled_name = format!("tl_{}_{}", simple_type_name, method);
        let stdlib_name = format!("tl_{}_{}", simple_type_name.to_lowercase(), method);

        let (func, actual_name) = if let Some((f, name)) = generic_result {
             (f, name)
        } else if let Some(f) = self.module.get_function(&mangled_name) {
            (f, mangled_name)
        } else if let Some(f) = self.module.get_function(&stdlib_name) {
            (f, stdlib_name)
        } else if let Some(f) = self.module.get_function(method) {
            (f, method.to_string())
        } else {
            // Method not found - enum variant initialization is handled at function start
            return Err(TlError::from(CodegenErrorKind::Internal(format!(
                "Static method {}::{} not found (checked {}, {}, and {})",
                struct_name, method, mangled_name, stdlib_name, method
            ))));
        };

        // 3. Generic Fallback: Compile Args & Handle SRET
        // Get return type - first check registered method_return_types, then fall back to signature inference
        // 3. Generic Fallback: Compile Args & Handle SRET
        // Get return type - first check registered method_return_types, then fall back to signature inference
        let ret_ty = if let Some(ret) = self.method_return_types.get(&actual_name) {
            ret.clone()
        } else {
            self.get_return_type_from_signature(func)
        };

        // SRET Logic
        let uses_sret = matches!(&ret_ty, Type::Struct(name, _) if name != "Tensor" && name != "String");
        let mut sret_ptr = None;

        if uses_sret {
             // NEW: Heap Allocation (malloc + register) -> Correct for RefCounted Structs/SRET
             let (struct_name, generics) = match &ret_ty {
                 Type::Struct(n, g) | Type::Enum(n, g) => (n, g),
                 _ => return Err(CodegenErrorKind::Internal("SRET used on non-aggregate type".to_string()).into()),
             };
             
             let mangled_name = if generics.is_empty() {
                 struct_name.to_string()
             } else {
                 // Use base name to avoid double-mangling (e.g. Entry[i64][i64] -> Entry[i64][i64][i64][i64])
                 let base = mangle_base_name(struct_name);
                 self.mangle_type_name(base, generics)
             };
             
             // Simple name lookup (as done in compile_struct_init)
             let simple_lookup_name = mangled_name.clone();

             // Ensure type is monomorphized and registered
             let _ = self.get_or_monomorphize_type(&ret_ty).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

             let struct_type = self.struct_types.get(&simple_lookup_name)
                 .or_else(|| self.enum_types.get(&simple_lookup_name))
                 .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Struct type {} not found for SRET allocation", simple_lookup_name))))?;
             
             let size = struct_type.size_of().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("Cannot determine size for SRET struct".to_string())))?;
             
             // 2. Malloc
             let malloc_fn = self.module.get_function("malloc").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("malloc not found".to_string())))?;
             let size_i64 = self.builder.build_int_z_extend(size, self.context.i64_type(), "size_i64").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
             let call_malloc = self.builder.build_call(malloc_fn, &[size_i64.into()], "sret_malloc").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
             
             let raw_ptr = match call_malloc.try_as_basic_value() {
                 inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
                 _ => return Err(CodegenErrorKind::Internal("malloc returned void".to_string()).into()),
             };
             
             // 3. Register (Initialize RefCount=1, Header)
             let struct_name_str = match &ret_ty {
                 Type::Struct(n, _) => n.as_str(),
                 _ => "AnonymousStruct",
             };
             let name_global = self.builder.build_global_string_ptr(struct_name_str, "struct_name").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
             let register_fn = self.module.get_function("tl_mem_register_struct_named").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_mem_register_struct_named not found".to_string())))?;
             
             let cast_ptr = self.builder.build_pointer_cast(raw_ptr, self.context.ptr_type(inkwell::AddressSpace::default()), "cast_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
             let _ = self.builder.build_call(register_fn, &[cast_ptr.into(), name_global.as_pointer_value().into()], "");

             sret_ptr = Some(cast_ptr);
        }

        let mut compiled_args = Vec::with_capacity(args.len());
        let mut compiled_args_types = Vec::with_capacity(args.len());
        for arg in args {
            let (val, ty) = self.compile_expr(arg)?;

            let (val, ty) = if type_name == "VarBuilder" {
                 if let Type::String(_) = ty {
                     // VarBuilder methods expect char* (i8*) not String struct
                     let ptr_to_struct = val.into_pointer_value();
                     let i64_ptr_ty = self.context.i64_type().ptr_type(inkwell::AddressSpace::default());
                     let ptr_to_first_field = self.builder.build_pointer_cast(ptr_to_struct, i64_ptr_ty, "str_ptr_cast").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                     let str_addr_i64 = self.builder.build_load(self.context.i64_type(), ptr_to_first_field, "str_addr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value();
                     let i8_ptr_ty = self.context.i8_type().ptr_type(inkwell::AddressSpace::default());
                     let char_ptr = self.builder.build_int_to_ptr(str_addr_i64, i8_ptr_ty, "cstr_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                     (char_ptr.into(), ty)
                 } else {
                     (val, ty)
                 }
            } else {
                (val, ty)
            };
            compiled_args.push(val.into());
            compiled_args_types.push((val, ty));
        }

        // 4. Call
        if let Some(ptr) = sret_ptr {
             compiled_args.insert(0, ptr.into());
        }

        let call = self
            .builder
            .build_call(func, &compiled_args, "static_call")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        if let Some(ptr) = sret_ptr {
             let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
             let loaded = self.builder.build_load(ptr_ty, ptr, "sret_static_loaded").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
             Ok((loaded, ret_ty))
        } else {
             match call.try_as_basic_value() {
                 inkwell::values::ValueKind::Basic(_) => {
                     let v = self.check_tensor_result(call, "static_call_error")?;
                     Ok((v, ret_ty))
                 }
                 _ => Ok((
                     self.context.i64_type().const_int(0, false).into(),
                     Type::Void,
                 )),
             }
        }
    }

    fn compile_tensor_literal(
        &mut self,
        elements: &[Expr],
    ) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
        // 1. Calculate shape and total elements
        fn count_elements(exprs: &[Expr]) -> (usize, Vec<usize>) {
            if exprs.is_empty() {
                return (0, vec![0]);
            }

            let is_nested = matches!(exprs[0].inner, ExprKind::TensorLiteral(_));
            if is_nested {
                let mut total = 0;
                let mut first_shape = None;
                for e in exprs {
                    if let ExprKind::TensorLiteral(children) = &e.inner {
                        let (count, shape) = count_elements(children);
                        total += count;
                        if first_shape.is_none() {
                            first_shape = Some(shape);
                        }
                    }
                }
                let mut shape = vec![exprs.len()];
                if let Some(s) = first_shape {
                    shape.extend(s);
                }
                (total, shape)
            } else {
                (exprs.len(), vec![exprs.len()])
            }
        }

        let (total_elements, shape) = count_elements(elements);
        let rank = shape.len();

        // 2. Flatten elements
        fn flatten_exprs(exprs: &[Expr], result: &mut Vec<Expr>) {
            for e in exprs {
                if let ExprKind::TensorLiteral(children) = &e.inner {
                    flatten_exprs(children, result);
                } else {
                    result.push(e.clone());
                }
            }
        }

        let mut flat_exprs = Vec::new();
        flatten_exprs(elements, &mut flat_exprs);

        // 3. Compile all elements and determine target type
        let mut compiled_vals = Vec::with_capacity(flat_exprs.len());
        let mut has_float = false;

        for expr in &flat_exprs {
            let (val, ty) = self.compile_expr(expr)?;
            compiled_vals.push((val, ty.clone()));
            if matches!(ty, Type::F32 | Type::F64) {
                has_float = true;
            }
        }

        // Determine target type: F32 if any float present, else I64 (if all are numeric ints)
        let i64_type = self.context.i64_type();
        let f32_type = self.context.f32_type();

        // 4. Allocate temporary buffer for data
        let alloc_tmp_fn = self
            .module
            .get_function("tl_alloc_tmp")
            .expect("tl_alloc_tmp not found");

        let element_size = if has_float { 4 } else { 8 };
        let size = self
            .builder
            .build_int_mul(
                i64_type.const_int(total_elements as u64, false),
                i64_type.const_int(element_size, false),
                "alloc_size",
            )
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        let call_idx = self
            .builder
            .build_call(alloc_tmp_fn, &[size.into()], "buf_void")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        let data_alloca = match call_idx.try_as_basic_value() {
            inkwell::values::ValueKind::Basic(inkwell::values::BasicValueEnum::PointerValue(v)) => {
                v
            }
            _ => return Err(CodegenErrorKind::Internal("Invalid tl_alloc_tmp return".to_string()).into()),
        };

        // 5. Store elements into buffer
        for (i, (val, ty)) in compiled_vals.iter().enumerate() {
            if has_float {
                let float_val = match ty {
                    Type::F32 => val.into_float_value(),
                    Type::F64 => self
                        .builder
                        .build_float_cast(val.into_float_value(), f32_type, "f64_to_f32")
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?,
                    Type::I64 => self
                        .builder
                        .build_signed_int_to_float(val.into_int_value(), f32_type, "i64_to_f32")
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?,
                    Type::I32 | Type::Bool => self
                        .builder
                        .build_signed_int_to_float(
                            self.builder
                                .build_int_z_extend(val.into_int_value(), i64_type, "zext")
                                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?,
                            f32_type,
                            "i_to_f32",
                        )
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?,
                    _ => return Err(TlError::from(CodegenErrorKind::TypeError(format!("Cannot cast {:?} to F32", ty)))),
                };

                let ptr = unsafe {
                    self.builder
                        .build_in_bounds_gep(
                            f32_type,
                            data_alloca,
                            &[i64_type.const_int(i as u64, false)],
                            "elem_ptr",
                        )
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                };
                self.builder.build_store(ptr, float_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            } else {
                let int_val = match ty {
                    Type::I64 => val.into_int_value(),
                    Type::I32 => self
                        .builder
                        .build_int_z_extend(val.into_int_value(), i64_type, "zext")
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?,
                    Type::Bool => self
                        .builder
                        .build_int_z_extend(val.into_int_value(), i64_type, "zext")
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?,
                    _ => return Err(TlError::from(CodegenErrorKind::TypeError(format!("Cannot cast {:?} to I64", ty)))),
                };

                let ptr = unsafe {
                    self.builder
                        .build_in_bounds_gep(
                            i64_type,
                            data_alloca,
                            &[i64_type.const_int(i as u64, false)],
                            "elem_ptr",
                        )
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                };
                self.builder.build_store(ptr, int_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            }
        }

        // 6. Allocate and fill shape buffer
        let shape_size = rank as u64 * 8; // i64 size
        let shape_alloc_call = self
            .builder
            .build_call(
                alloc_tmp_fn,
                &[i64_type.const_int(shape_size, false).into()],
                "shape_alloc",
            )
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        let shape_alloca = match shape_alloc_call.try_as_basic_value() {
            ValueKind::Basic(v) => v.into_pointer_value(),
            _ => return Err(CodegenErrorKind::Internal("tl_alloc_tmp failed".to_string()).into()),
        };

        for (i, dim) in shape.iter().enumerate() {
            let ptr = unsafe {
                self.builder
                    .build_in_bounds_gep(
                        i64_type,
                        shape_alloca,
                        &[i64_type.const_int(i as u64, false)],
                        "shape_ptr",
                    )
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
            };
            self.builder
                .build_store(ptr, i64_type.const_int(*dim as u64, false))
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        }

        // 7. Call appropriate runtime tensor constructor
        let (new_fn_name, result_inner_type) = if has_float {
            ("tl_tensor_new", Type::F32)
        } else {
            ("tl_tensor_new_i64", Type::I64)
        };

        let new_fn = self
            .module
            .get_function(new_fn_name)
            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("{} not found", new_fn_name))))?;

        let shape_ptr_cast = self
            .builder
            .build_pointer_cast(
                shape_alloca,
                self.context.ptr_type(inkwell::AddressSpace::default()),
                "shap_ptr_cast",
            )
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        let data_ptr_cast = self
            .builder
            .build_pointer_cast(
                data_alloca,
                self.context.ptr_type(inkwell::AddressSpace::default()),
                "data_ptr_cast",
            )
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        let call = self
            .builder
            .build_call(
                new_fn,
                &[
                    data_ptr_cast.into(),
                    i64_type.const_int(rank as u64, false).into(),
                    shape_ptr_cast.into(),
                ],
                "new_tensor",
            )
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        let res = self.check_tensor_result(call, "new_tensor_error")?;

        // 8. Free temporary buffers
        let free_tmp_fn = self
            .module
            .get_function("tl_free_tmp")
            .expect("tl_free_tmp not found");

        self.builder
            .build_call(free_tmp_fn, &[data_alloca.into()], "")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        self.builder
            .build_call(free_tmp_fn, &[shape_alloca.into()], "")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        let result_ty = Type::Tensor(Box::new(result_inner_type), rank);


        Ok((res, result_ty))
    }

    fn compile_tensor_const_literal(
        &mut self,
        elements: &[Expr],
    ) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
        // Optimized path for constant tensor literals - static extraction
        fn flatten_const(exprs: &[Expr]) -> Result<(Vec<f64>, Vec<usize>, bool), TlError> {
            if exprs.is_empty() {
                return Ok((vec![], vec![0], false));
            }

            let is_nested = matches!(
                &exprs[0].inner,
                ExprKind::TensorConstLiteral(_) | ExprKind::TensorLiteral(_)
            );
            if is_nested {
                let mut flat_data = Vec::new();
                let mut first_shape = None;
                let mut all_ints = true;

                for e in exprs {
                    let (children, shape, ints) = match &e.inner {
                        ExprKind::TensorConstLiteral(c) | ExprKind::TensorLiteral(c) => {
                            flatten_const(c)?
                        }
                        _ => return Err(CodegenErrorKind::Internal("Mixed types in const tensor".to_string()).into()),
                    };
                    if let Some(ref s) = first_shape {
                        if s != &shape {
                            return Err(TlError::from(CodegenErrorKind::UnsupportedOperation("Ragged tensors not supported".to_string())));
                        }
                    } else {
                        first_shape = Some(shape.clone());
                    }
                    flat_data.extend(children);
                    all_ints &= ints;
                }

                let mut shape = vec![exprs.len()];
                if let Some(s) = first_shape {
                    shape.extend(s);
                }
                Ok((flat_data, shape, all_ints))
            } else {
                let mut data = Vec::new();
                let mut all_ints = true;
                for e in exprs {
                    match &e.inner {
                        ExprKind::Float(f) => {
                            data.push(*f);
                            all_ints = false;
                        }
                        ExprKind::Int(i) => data.push(*i as f64),
                        _ => return Err(CodegenErrorKind::Internal("Const tensor must contain only literals".to_string()).into()),
                    }
                }
                Ok((data, vec![exprs.len()], all_ints))
            }
        }

        let (flat_data, shape, all_ints) = flatten_const(elements)?;
        let rank = shape.len();
        let len = flat_data.len();

        // Fall back to standard tensor creation for larger tensors (and now all tensors)
        let len = len as u64;
        let f32_type = self.context.f32_type();
        let i64_type = self.context.i64_type();

        // Use tl_alloc_tmp instead of malloc
        let alloc_tmp_fn = self
            .module
            .get_function("tl_alloc_tmp")
            .expect("tl_alloc_tmp not found");
        let free_tmp_fn = self
            .module
            .get_function("tl_free_tmp")
            .expect("tl_free_tmp not found");

        if all_ints {
            // I64 TENSOR Creation
            let data_size_bytes = len * 8; // i64 = 8 bytes
            let alloc_call = self
                .builder
                .build_call(
                    alloc_tmp_fn,
                    &[i64_type.const_int(data_size_bytes, false).into()],
                    "temp_data_alloc_i64",
                )
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let data_ptr = match alloc_call.try_as_basic_value() {
                ValueKind::Basic(v) => v.into_pointer_value(),
                _ => return Err(CodegenErrorKind::Internal("tl_alloc_tmp returned non-pointer".to_string()).into()),
            };

            // Populate data buffer (i64)
            for (i, val) in flat_data.iter().enumerate() {
                let int_val = i64_type.const_int(*val as u64, false); // val is f64, safe cast for ints
                let elem_ptr = unsafe {
                    self.builder
                        .build_in_bounds_gep(
                            i64_type,
                            data_ptr, // treated as i64*
                            &[i64_type.const_int(i as u64, false)],
                            "data_elem",
                        )
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                };
                // Cast pointer if needed, but here we treat malloc'd void* as i64* implicitly via GEP type
                // Actually build_in_bounds_gep on opaque ptr might require explicit type.
                // LLVM 15+ uses opaque pointers, so GEP type matches element type.
                self.builder
                    .build_store(elem_ptr, int_val)
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            }

            // Allocate shape buffer
            let shape_size_bytes = rank as u64 * 8;
            let shape_alloc_call = self
                .builder
                .build_call(
                    alloc_tmp_fn,
                    &[i64_type.const_int(shape_size_bytes, false).into()],
                    "temp_shape_alloc",
                )
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let shape_ptr = match shape_alloc_call.try_as_basic_value() {
                ValueKind::Basic(v) => v.into_pointer_value(),
                _ => return Err(CodegenErrorKind::Internal("tl_alloc_tmp returned non-pointer".to_string()).into()),
            };

            // Populate shape (same for both)
            for (i, dim) in shape.iter().enumerate() {
                let elem_ptr = unsafe {
                    self.builder
                        .build_in_bounds_gep(
                            i64_type,
                            shape_ptr,
                            &[i64_type.const_int(i as u64, false)],
                            "shape_elem",
                        )
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                };
                self.builder
                    .build_store(elem_ptr, i64_type.const_int(*dim as u64, false))
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            }

            // Call tl_tensor_new_i64
            let new_fn = self
                .module
                .get_function("tl_tensor_new_i64")
                .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_tensor_new_i64 not found".to_string())))?;

            let call = self
                .builder
                .build_call(
                    new_fn,
                    &[
                        data_ptr.into(),
                        i64_type.const_int(rank as u64, false).into(),
                        shape_ptr.into(),
                    ],
                    "new_const_tensor_i64",
                )
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            let res = self.check_tensor_result(call, "new_const_tensor_i64_error")?;

            // FREE temps
            self.builder
                .build_call(free_tmp_fn, &[data_ptr.into()], "")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder
                .build_call(free_tmp_fn, &[shape_ptr.into()], "")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            Ok((res, Type::Tensor(Box::new(Type::I64), rank)))
        } else {
            // F32 TENSOR Creation
            let data_size_bytes = len * 4; // f32 = 4 bytes
            let alloc_call = self
                .builder
                .build_call(
                    alloc_tmp_fn,
                    &[i64_type.const_int(data_size_bytes, false).into()],
                    "temp_data_alloc",
                )
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let data_ptr = match alloc_call.try_as_basic_value() {
                ValueKind::Basic(v) => v.into_pointer_value(),
                _ => return Err(CodegenErrorKind::Internal("tl_alloc_tmp returned non-pointer".to_string()).into()),
            };

            // Populate data buffer (f32)
            for (i, val) in flat_data.iter().enumerate() {
                let float_val = f32_type.const_float(*val);
                let elem_ptr = unsafe {
                    self.builder
                        .build_in_bounds_gep(
                            f32_type,
                            data_ptr,
                            &[i64_type.const_int(i as u64, false)],
                            "data_elem",
                        )
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                };
                self.builder
                    .build_store(elem_ptr, float_val)
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            }

            // Allocate shape buffer
            let shape_size_bytes = rank as u64 * 8; // i64 = 8 bytes
            let shape_alloc_call = self
                .builder
                .build_call(
                    alloc_tmp_fn,
                    &[i64_type.const_int(shape_size_bytes, false).into()],
                    "temp_shape_alloc",
                )
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let shape_ptr = match shape_alloc_call.try_as_basic_value() {
                ValueKind::Basic(v) => v.into_pointer_value(),
                _ => return Err(CodegenErrorKind::Internal("tl_alloc_tmp returned non-pointer".to_string()).into()),
            };

            // Populate shape buffer
            for (i, dim) in shape.iter().enumerate() {
                let elem_ptr = unsafe {
                    self.builder
                        .build_in_bounds_gep(
                            i64_type,
                            shape_ptr,
                            &[i64_type.const_int(i as u64, false)],
                            "shape_elem",
                        )
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                };
                self.builder
                    .build_store(elem_ptr, i64_type.const_int(*dim as u64, false))
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            }

            // Call tl_tensor_new
            let new_fn = self
                .module
                .get_function("tl_tensor_new")
                .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_tensor_new not found".to_string())))?;

            let call = self
                .builder
                .build_call(
                    new_fn,
                    &[
                        data_ptr.into(),
                        i64_type.const_int(rank as u64, false).into(),
                        shape_ptr.into(),
                    ],
                    "new_const_tensor",
                )
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            let res = self.check_tensor_result(call, "new_const_tensor_error")?;

            // FREE temps
            self.builder
                .build_call(free_tmp_fn, &[data_ptr.into()], "")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder
                .build_call(free_tmp_fn, &[shape_ptr.into()], "")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            Ok((res, Type::Tensor(Box::new(Type::F32), rank)))
        }
    }

    fn compile_match_like(
        &mut self,
        subject_expr: &Expr,
        arms: &[(Pattern, Expr)],
    ) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
        let (subject_val, raw_subject_ty) = self.compile_expr(subject_expr)?;
        let subject_ty = raw_subject_ty.flatten_specialized();
        let (enum_name, raw_generic_args) = match &subject_ty {
            Type::Enum(n, args) | Type::Struct(n, args) => (n, args.clone()),
            Type::SpecializedType { .. } => return Err(CodegenErrorKind::Internal("Match on SpecializedType not fully supported yet".to_string()).into()),
            Type::Path(segments, args) => {
                if let Some(n) = segments.last() {
                    (n, args.clone())
                } else {
                    return Err(TlError::from(CodegenErrorKind::Internal("Match on empty path".to_string())));
                }
            }
            _ => return Err(TlError::from(CodegenErrorKind::Internal(format!("Match on non-enum: {:?}", subject_ty)))),
        };
        
        let generic_args: Vec<Type> = raw_generic_args;


        // Ensure enum layout is generated
        // Check if enum is already specialized (or concrete) to avoid double-monomorphization
        // Check if enum is already specialized (or concrete) to avoid double-monomorphization
        let actual_generic_args = if let Some(def) = self.enum_defs.get(enum_name) {
             if def.generics.is_empty() {
                 &[]
             } else {
                 generic_args.as_slice()
             }
        } else {
             // Def not found yet? Try blindly.
             generic_args.as_slice()
        };
        let mangled_name = self.monomorphize_enum(enum_name, actual_generic_args)
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(format!("Failed to monomorphize enum {}: {}", enum_name, e))))?;
        
        let enum_ty = self.enum_types.get(&mangled_name).cloned()
            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Enum LLVM type {} not found", mangled_name))))?;

        let enum_def = self.enum_defs.get(&mangled_name)
            .or_else(|| self.enum_defs.get(enum_name))
            .or_else(|| self.enum_defs.get(mangle_base_name(enum_name)))
            .cloned()
            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Enum def not found (tried {}, {}, {})", mangled_name, enum_name, mangle_base_name(enum_name)))))?;


        let ptr = subject_val.into_pointer_value();

        // Load Tag
        let tag_ptr = self
            .builder
            .build_struct_gep(enum_ty, ptr, 0, "tag_ptr")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        let tag_val = self
            .builder
            .build_load(self.context.i32_type(), tag_ptr, "tag")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
            .into_int_value();

        let current_func = self
            .builder
            .get_insert_block()
            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("unexpected None".to_string())))?
            .get_parent()
            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("unexpected None".to_string())))?;
        let merge_block = self.context.append_basic_block(current_func, "match_merge");

        let mut arm_blocks = Vec::with_capacity(arms.len());
        for i in 0..arms.len() {
            arm_blocks.push(
                self.context
                    .append_basic_block(current_func, &format!("arm_{}", i)),
            );
        }

        let mut switch_cases = vec![];
        let mut used_variants = HashSet::new();
        for (arm_idx, (pat, _)) in arms.iter().enumerate() {
            if let Pattern::EnumPattern { variant_name, .. } = pat {
                // Fetch def - try mangled_name first, then enum_name, then base name extraction
                let enum_def = if let Some(def) = self.enum_defs.get(&mangled_name) {
                    def
                } else if let Some(def) = self.enum_defs.get(enum_name) {
                    def
                } else {
                    // Try to convert underscore format to angle-bracket format
                    let base_name = if mangle_has_args(enum_name) && !enum_name.contains('<') {
                        mangle_base_name(enum_name)
                    } else {
                        enum_name
                    };
                    // Try angle-bracket version from monomorphize result
                    self.enum_defs.get(&mangled_name)
                        .or_else(|| self.enum_defs.get(base_name))
                        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Enum def '{}' not found in pattern. Tried: {}, {}. Available: {:?}", enum_name, mangled_name, base_name, self.enum_defs.keys().collect::<Vec<_>>()))))?
                };
                
                let simple_variant_name = variant_name.as_str();

                let idx = enum_def
                    .variants
                    .iter()
                    .position(|v| v.name == simple_variant_name)
                    .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Enum variant '{}' not found in {}. Available: {:?}", simple_variant_name, enum_name, enum_def.variants.iter().map(|v| &v.name).collect::<Vec<_>>()))))?;
                if used_variants.insert(idx) {
                    switch_cases.push((
                        self.context.i32_type().const_int(idx as u64, false),
                        arm_blocks[arm_idx],
                    ));
                }
            }
        }

        let wildcard_block = arms
            .iter()
            .position(|(p, _)| matches!(p, Pattern::Wildcard))
            .map(|i| arm_blocks[i]);

        let default_block = if let Some(wb) = wildcard_block {
            wb
        } else {
            let func = self
                .builder
                .get_insert_block()
                .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("unexpected None".to_string())))?
                .get_parent()
                .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("unexpected None".to_string())))?;
            let unreachable_bb = self.context.append_basic_block(func, "match_unreachable");
            let current_block = self.current_block()?;
            self.builder.position_at_end(unreachable_bb);
            let _ = self.builder.build_unreachable();
            self.builder.position_at_end(current_block);
            unreachable_bb
        };

        self.builder
            .build_switch(tag_val, default_block, &switch_cases)
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        let mut incoming_vals = vec![];
        let mut result_type = Type::Void;

        for (i, (pat, body)) in arms.iter().enumerate() {
            let block = arm_blocks[i];
            self.builder.position_at_end(block);

            self.enter_scope();

            if let Pattern::EnumPattern {
                variant_name,
                bindings,
                ..
            } = pat
            {
                let simple_variant_name = variant_name.as_str();

                let variant_idx = enum_def
                    .variants
                    .iter()
                    .position(|v| v.name == simple_variant_name)
                    .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("Enum variant not found in bindings".to_string())))?;
                self.bind_enum_pattern_fields(
                    current_func,
                    enum_ty,
                    ptr,
                    &enum_def,
                    variant_idx,
                    bindings,
                    &generic_args,
                )?;
            }

            let (val, ty) = self.compile_match_arm_body(body)?;
            // Use the first non-Never/non-Void type as result_type
            // This handles cases like unwrap_err where panic arm (Never) comes first
            if result_type == Type::Void || result_type == Type::Never {
                if ty != Type::Never {
                    result_type = ty.clone();
                }
            }

            self.exit_scope();

            let current_insert_block = self.current_block()?;
            if current_insert_block.get_terminator().is_none() {
                self.builder
                    .build_unconditional_branch(merge_block)
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                incoming_vals.push((val, current_insert_block));
            }
        }

        self.builder.position_at_end(merge_block);

        if result_type == Type::Void || incoming_vals.is_empty() {
            Ok((
                self.context.i64_type().const_int(0, false).into(),
                Type::Void,
            ))
        } else {
            // Determine phi type from the actual LLVM type of the first incoming value
            // This is more reliable than using result_type, which may be Undefined
            // from monomorphization issues
            let phi_type: inkwell::types::BasicTypeEnum = incoming_vals[0].0.get_type();
            let phi = self.builder.build_phi(phi_type, "match_res").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let incomings: Vec<(
                &dyn inkwell::values::BasicValue,
                inkwell::basic_block::BasicBlock,
            )> = incoming_vals
                .iter()
                .map(|(v, b)| (v as &dyn inkwell::values::BasicValue, *b))
                .collect();
            phi.add_incoming(&incomings);

            Ok((phi.as_basic_value(), result_type))
        }
    }

    fn compile_match_arm_body(
        &mut self,
        body: &Expr,
    ) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
        if let ExprKind::Block(stmts) = &body.inner {
            self.enter_scope();
            let mut last_val = None;
            let mut last_is_lvalue = false;
            for (i, stmt) in stmts.iter().enumerate() {
                if i == stmts.len() - 1 {
                    if let StmtKind::Expr(e) = &stmt.inner {
                        last_is_lvalue = Self::is_lvalue_expr(e);
                        last_val = Some(self.compile_expr(e)?);
                    } else {
                        self.compile_stmt(stmt)?;
                    }
                } else {
                    self.compile_stmt(stmt)?;
                }
            }

            let final_res = last_val.unwrap_or((
                self.context.i64_type().const_int(0, false).into(),
                Type::Void,
            ));
            let promoted = self.promote_match_result(final_res.0, &final_res.1, last_is_lvalue)?;
            self.exit_scope();
            Ok((promoted, final_res.1))
        } else {
            let (val, ty) = self.compile_expr(body)?;
            let promoted = self.promote_match_result(val, &ty, Self::is_lvalue_expr(body))?;
            Ok((promoted, ty))
        }
    }

    fn promote_match_result(
        &mut self,
        val: BasicValueEnum<'ctx>,
        ty: &Type,
        is_lvalue: bool,
    ) -> Result<BasicValueEnum<'ctx>, TlError> {

        let is_ref_type = matches!(
            ty,
            Type::Tensor(_, _)
                | Type::Struct(_, _)
                | Type::Enum(_, _)
                | Type::String(_)
                | Type::Tuple(_)
        );
        if !is_ref_type {
            return Ok(val);
        }

        if is_lvalue {
            self.emit_deep_clone(val, ty)
        } else {
            if let Some(unreg_fn) = self.module.get_function("tl_mem_unregister") {
                let ptr = val.into_pointer_value();
                let cast_ptr = self
                    .builder
                    .build_pointer_cast(
                        ptr,
                        self.context.ptr_type(inkwell::AddressSpace::default()),
                        "cast",
                    )
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder
                    .build_call(unreg_fn, &[cast_ptr.into()], "")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            }
            Ok(val)
        }
    }

    fn bind_enum_pattern_fields(
        &mut self,
        current_func: inkwell::values::FunctionValue<'ctx>,
        enum_ty: inkwell::types::StructType<'ctx>,
        enum_ptr: inkwell::values::PointerValue<'ctx>,
        enum_def: &EnumDef,
        variant_idx: usize,
        bindings: &crate::compiler::ast::EnumPatternBindings,
        generic_args: &[Type],
    ) -> Result<(), TlError> {
        let variant_def = &enum_def.variants[variant_idx];

        // Build substitution map for concrete types
        // If enum_def.generics is empty (monomorphized enum like Entry_i64_i64),
        // try to find the original generic enum definition (e.g., Entry) to get generic param names
        let mut subst: HashMap<String, Type> = HashMap::new();
        
        if enum_def.generics.is_empty() && !generic_args.is_empty() {
            // Try to find the original generic enum definition
            let base_name = mangle_base_name(&enum_def.name);
            if let Some(generic_def) = self.enum_defs.get(base_name) {
                for (i, param_name) in generic_def.generics.iter().enumerate() {
                    if let Some(arg) = generic_args.get(i) {
                        subst.insert(param_name.clone(), arg.clone());
                    }
                }
            }
        } else {
            for (i, param_name) in enum_def.generics.iter().enumerate() {
                if let Some(arg) = generic_args.get(i) {
                    subst.insert(param_name.clone(), arg.clone());
                }
            }
        }

        // Get Payload Pointer (Index 1)
        // Opaque Layout: { i32, [u8...]}
        // We cast this pointer to the Variant Struct Pointer
        let payload_ptr_raw = self.builder
            .build_struct_gep(enum_ty, enum_ptr, 1, "payload_ptr_raw")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        // internal helper to get concrete types for this variant
        let mut field_types = Vec::new();
        match &variant_def.kind {
            crate::compiler::ast::VariantKind::Unit => {},
            crate::compiler::ast::VariantKind::Tuple(types) => {
                for ty in types {
                    let concrete_ty = self.substitute_type_simple_bind(ty, &subst);
                    if concrete_ty != Type::Void {
                        field_types.push(concrete_ty);
                    }
                }
            },
            crate::compiler::ast::VariantKind::Struct(fields) => {
                for (_, ty) in fields {
                    let concrete_ty = self.substitute_type_simple_bind(ty, &subst);
                    if concrete_ty != Type::Void {
                        field_types.push(concrete_ty);
                    }
                }
            }
            crate::compiler::ast::VariantKind::Array(ty, size) => {
                for _ in 0..*size {
                    let concrete_ty = self.substitute_type_simple_bind(ty, &subst);
                    if concrete_ty != Type::Void {
                        field_types.push(concrete_ty);
                    }
                }
            }
        }

        // Construct Variant LLVM Struct Type
        let mut llvm_field_types = Vec::new();
        for f_ty in &field_types {
            llvm_field_types.push(self.get_llvm_type(f_ty)?);
        }
        let variant_struct_ty = self.context.struct_type(&llvm_field_types, false);

        // Cast payload ptr to variant struct ptr
        // Use pointer_cast (BitCast)
        // Since payload_ptr_raw is [u8]* (inside struct), and we want { Fields... }*
        // We cast to PointerType(variant_struct_ty)
        let variant_ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
        // Wait, build_pointer_cast works on values.
        // We handle opaque pointers in recent LLVM, so just casting the value is enough?
        // But `build_struct_gep` needs the Type info if we didn't pass it.
        // `builder.build_struct_gep(variant_struct_ty, payload_ptr, idx, ...)`

        let payload_ptr = self.builder.build_pointer_cast(
            payload_ptr_raw,
            variant_ptr_type, // Actually just mapped to ptr
            "payload_cast"
        ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;


        // Bind fields
        let mut current_field_idx = 0; // Relative to Variant Struct

        match (&variant_def.kind, bindings) {
            (crate::compiler::ast::VariantKind::Unit, crate::compiler::ast::EnumPatternBindings::Unit) => {
                Ok(())
            },
            (crate::compiler::ast::VariantKind::Tuple(types), crate::compiler::ast::EnumPatternBindings::Tuple(vars)) => {
                if types.len() != vars.len() {
                    return Err(TlError::from(CodegenErrorKind::Internal(format!("Tuple pattern length mismatch: expected {}, found {}", types.len(), vars.len()))));
                }
                
                for (i, bind_name) in vars.iter().enumerate() {
                    let field_ty = &types[i];
                    let concrete_ty = self.substitute_type_simple_bind(field_ty, &subst);

                    if concrete_ty == Type::Void {
                         continue;
                    }
                    
                    if bind_name == "_" {
                         current_field_idx += 1;
                         continue;
                    }
                    
                    // Access field from cast payload
                    let f_ptr = self
                        .builder
                        .build_struct_gep(variant_struct_ty, payload_ptr, current_field_idx, "field_ptr")
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    current_field_idx += 1;

                    let llvm_ty = self.get_llvm_type(&concrete_ty)?;
                    let f_val = self.builder.build_load(llvm_ty, f_ptr, "bind_val").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                    let alloca = self.create_entry_block_alloca(current_func, bind_name, &concrete_ty)?;
                    let stored_val = match &concrete_ty {
                        Type::Tuple(ts) if ts.is_empty() => f_val,
                        Type::Tensor(_, _)
                        | Type::Struct(_, _)
                        | Type::Enum(_, _)
                        | Type::Tuple(_)
                        | Type::String(_) => {
                             self.emit_deep_clone(f_val, &concrete_ty)?
                        }
                        _ => f_val,
                    };
                    self.builder.build_store(alloca, stored_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                    self.variables
                        .last_mut()
                        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("unexpected None".to_string())))?
                        .insert(bind_name.clone(), (alloca.into(), concrete_ty.clone(), super::CLEANUP_FULL));
                }
                Ok(())
            },
            (crate::compiler::ast::VariantKind::Struct(fields), crate::compiler::ast::EnumPatternBindings::Struct(bindings_list)) => {
                 for (def_field_name, def_ty) in fields {
                     let concrete_ty = self.substitute_type_simple_bind(def_ty, &subst);
                     if concrete_ty == Type::Void { continue; }
                     
                     let binding = bindings_list.iter().find(|(n, _)| n == def_field_name);
                     
                     if let Some((_, bind_var_name)) = binding {
                         if bind_var_name != "_" {
                             // Access field
                             let f_ptr = self
                                 .builder
                                 .build_struct_gep(variant_struct_ty, payload_ptr, current_field_idx, "field_ptr")
                                 .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                             
                             let llvm_ty = self.get_llvm_type(&concrete_ty)?;
                             let f_val = self.builder.build_load(llvm_ty, f_ptr, "bind_val").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                             
                             let alloca = self.create_entry_block_alloca(current_func, bind_var_name, &concrete_ty)?;
                             let stored_val = match &concrete_ty {
                                 Type::Tuple(ts) if ts.is_empty() => f_val,
                                 Type::Tensor(_, _)
                                 | Type::Struct(_, _)
                                 | Type::Enum(_, _)
                                 | Type::Tuple(_)
                                 | Type::String(_) => {
                                      self.emit_deep_clone(f_val, &concrete_ty)?
                                 }
                                 _ => f_val,
                             }; 
                             self.builder.build_store(alloca, stored_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                             
                             self.variables.last_mut().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("no variable scope".to_string())))?.insert(bind_var_name.clone(), (alloca.into(), concrete_ty.clone(), super::CLEANUP_FULL));
                         }
                     }
                     // Always increment standard layout index
                     current_field_idx += 1;
                 }
                 Ok(())
            }
             _ => Err(CodegenErrorKind::Internal("Invalid pattern binding for variant".to_string()).into())
        }
    }
    
    fn substitute_type_simple_bind(&self, ty: &Type, subst: &HashMap<String, Type>) -> Type {
        let substitutor = crate::compiler::ast_subst::TypeSubstitutor::new(subst.clone());
        substitutor.substitute_type(ty)
    }

    #[allow(dead_code)]
    pub(crate) fn substitute_current_generics(&self, ty: &Type) -> Type {
        if let Some(subst) = &self.current_method_generics {
            self.substitute_type_simple_bind(ty, subst)
        } else {
            ty.clone()
        }
    }

/*
             crate::compiler::ast::VariantKind::Unit => Box::new(std::iter::empty()),
             crate::compiler::ast::VariantKind::Tuple(types) => Box::new(types.iter()),
             crate::compiler::ast::VariantKind::Struct(fields) => Box::new(fields.iter().map(|(_, t)| t)),
        };
        
        for ty in fields_iter {
            field_types_llvm.push(self.get_llvm_type(ty)?);
        }
        let variant_struct_ty = self.context.struct_type(&field_types_llvm, false);

        // Get Payload Pointer
        // Always GEP to index 1 (payload) if not unit?
        // Wait, if Unit, layout is just Tag?
        // Enum layout: { i32 tag, Union payload }? Or { i32 tag, [MaxPayloadSize x i8] }?
        // `builtin_ast.rs` defines Enum layout.
        // Assuming Enum is { tag, payload }.
        let payload_ptr_raw = self
            .builder
            .build_struct_gep(enum_ty, enum_ptr, 1, "payload_ptr_raw")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        let payload_ptr = self
            .builder
            .build_pointer_cast(
                payload_ptr_raw,
                self.context.ptr_type(inkwell::AddressSpace::default()),
                "payload_cast",
            )
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        match (&variant_def.kind, bindings) {
            (crate::compiler::ast::VariantKind::Unit, crate::compiler::ast::EnumPatternBindings::Unit) => {
                Ok(())
            },
            (crate::compiler::ast::VariantKind::Tuple(types), crate::compiler::ast::EnumPatternBindings::Tuple(vars)) => {
                if types.len() != vars.len() {
                    return Err(TlError::from(CodegenErrorKind::Internal(format!("Tuple pattern length mismatch: expected {}, found {}", types.len(), vars.len()))));
                }
                
                for (i, bind_name) in vars.iter().enumerate() {
                    let f_ty = &types[i];
                    
                    let f_ptr = self
                        .builder
                        .build_struct_gep(variant_struct_ty, payload_ptr, i as u32, "field_ptr")
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                    let llvm_ty = self.get_llvm_type(f_ty)?;
                    let f_val = self.builder.build_load(llvm_ty, f_ptr, "bind_val").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                    let alloca = self.create_entry_block_alloca(current_func, bind_name, f_ty)?;
                    let stored_val = if matches!(
                        f_ty,
                        Type::Tensor(_, _)
                            | Type::Struct(_, _)
                            | Type::Enum(_, _)
                            
                            | Type::Tuple(_)
                    ) {
                        self.emit_deep_clone(f_val, f_ty)?
                    } else {
                        f_val
                    };
                    self.builder.build_store(alloca, stored_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                    self.variables
                        .last_mut()
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                        .insert(bind_name.clone(), (alloca.into(), f_ty.clone(), super::CLEANUP_FULL));
                }
                Ok(())
            },
            (crate::compiler::ast::VariantKind::Struct(fields), crate::compiler::ast::EnumPatternBindings::Struct(bindings_list)) => {
                for (field_name, bind_name) in bindings_list {
                    let f_idx = fields
                        .iter()
                        .position(|(n, _)| n == field_name)
                        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Field {} not found in variant {}", field_name, variant_def.name))))?;
                    
                    let (_, f_ty) = &fields[f_idx];

                    let f_ptr = self
                        .builder
                        .build_struct_gep(variant_struct_ty, payload_ptr, f_idx as u32, "field_ptr")
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                    let llvm_ty = self.get_llvm_type(f_ty)?;
                    let f_val = self.builder.build_load(llvm_ty, f_ptr, "bind_val").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                    let alloca = self.create_entry_block_alloca(current_func, bind_name, f_ty)?;
                    let stored_val = if matches!(
                        f_ty,
                        Type::Tensor(_, _)
                            | Type::Struct(_, _)
                            | Type::Enum(_, _)
                            
                            | Type::Tuple(_)
                    ) {
                        self.emit_deep_clone(f_val, f_ty)?
                    } else {
                        f_val
                    };
                    self.builder.build_store(alloca, stored_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                    self.variables
                        .last_mut()
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                        .insert(bind_name.clone(), (alloca.into(), f_ty.clone(), super::CLEANUP_FULL));
                }
                Ok(())
            },
             _ => Ok(()), 
        }
*/

    fn is_lvalue_expr(expr: &Expr) -> bool {
        matches!(
            &expr.inner,
            ExprKind::Variable(_)
                | ExprKind::FieldAccess(_, _)
                | ExprKind::IndexAccess(_, _)
                | ExprKind::TupleAccess(_, _)
        )
    }


    pub(crate) fn create_entry_block_alloca_manual(
        &self,
        function: FunctionValue<'ctx>,
        name: &str,
        llvm_ty: &inkwell::types::BasicTypeEnum<'ctx>,
    ) -> Result<inkwell::values::PointerValue<'ctx>, TlError> {
        let builder = self.context.create_builder();
        let entry = function.get_first_basic_block().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("function has no entry block".to_string())))?;
        match entry.get_first_instruction() {
            Some(first_instr) => builder.position_before(&first_instr),
            None => builder.position_at_end(entry),
        }
        Ok(builder.build_alloca(*llvm_ty, name).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?)
    }

    pub(crate) fn create_entry_block_alloca(
        &self,
        function: FunctionValue<'ctx>,
        name: &str,
        ty: &Type,
    ) -> Result<inkwell::values::PointerValue<'ctx>, TlError> {
        let builder = self.context.create_builder();
        if ty == &Type::Void {
        }
        let entry = function.get_first_basic_block().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("function has no entry block".to_string())))?;
        match entry.get_first_instruction() {
            Some(first_instr) => builder.position_before(&first_instr),
            None => builder.position_at_end(entry),
        }

        let llvm_type: inkwell::types::BasicTypeEnum = self
            .get_llvm_type(ty)
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(format!("Failed to get LLVM type for alloca: {}", e))))?;
        let alloca = builder.build_alloca(llvm_type, name).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        if let Some(instr) = alloca.as_instruction_value() {
            // Force 16-byte alignment to satisfy SIMD/slice requirements
            instr.set_alignment(16).ok();
        }
        Ok(alloca)
    }

    // Debug method to print IR
    pub fn dump_llvm_ir(&self) {
        self.module.print_to_stderr();
    }

    pub(crate) fn extract_index_bounds(
        &mut self,
        expr: &Expr,
        bounds: &mut HashMap<String, inkwell::values::IntValue<'ctx>>,
    ) -> Result<(), TlError> {
        match &expr.inner {
            ExprKind::IndexAccess(target, indices) => {
                // Target should be ExprKind::Ident for variable access
                // Instead of compiling, look up the variable directly
                let tensor_ptr = match &target.inner {
                    ExprKind::Variable(name) => {
                        let (val, ty) = self
                            .lookup_variable(name)
                            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Variable {} not found", name))))?;
                        // Load pointer if needed
                        match ty {
                            Type::Tensor(_, _) => {
                                // val is a pointer to the tensor pointer
                                self.builder
                                    .build_load(
                                        self.context.ptr_type(inkwell::AddressSpace::default()),
                                        val.into_pointer_value(),
                                        name,
                                    )
                                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                                    .into_pointer_value()
                            }
                            _ => return Err(CodegenErrorKind::Internal("Expected tensor variable".to_string()).into()),
                        }
                    }
                    _ => {
                        return Err(TlError::from(CodegenErrorKind::UnsupportedOperation("Complex index target not supported in bounds extraction".to_string())))
                    }
                };

                let dim_fn = self.get_fn("tl_tensor_dim")?;
                for (i, idx_expr) in indices.iter().enumerate() {
                    match &idx_expr.inner {
                        ExprKind::Int(_) | ExprKind::Float(_) => continue,
                        ExprKind::Variable(name) => {
                            if !bounds.contains_key(name) {
                                let dim_idx_val =
                                    self.context.i64_type().const_int(i as u64, false);
                                let call_result = self
                                    .builder
                                    .build_call(
                                        dim_fn,
                                        &[tensor_ptr.into(), dim_idx_val.into()],
                                        "dim_size",
                                    )
                                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                                let dim_size = match call_result.try_as_basic_value() {
                                    ValueKind::Basic(v) => v.into_int_value(),
                                    _ => return Err(CodegenErrorKind::Internal("Invalid dim return".to_string()).into()),
                                };
                                bounds.insert(name.clone(), dim_size);
                            }
                        }
                        _ => continue,
                    }
                }
            }
            ExprKind::BinOp(lhs, _, rhs) => {
                self.extract_index_bounds(lhs, bounds)?;
                self.extract_index_bounds(rhs, bounds)?;
            }
            ExprKind::UnOp(_, inner) => self.extract_index_bounds(inner, bounds)?,
            ExprKind::Block(stmts) => {
                for stmt in stmts {
                    if let StmtKind::Expr(e) = &stmt.inner {
                        self.extract_index_bounds(e, bounds)?;
                    } else if let StmtKind::Let { value, .. } = &stmt.inner {
                        self.extract_index_bounds(value, bounds)?;
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
    fn compile_method_call(
        &mut self,
        obj: &Expr,
        method: &str,
        args: &[Expr],
    ) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
        let (val, ty) = self.compile_method_call_inner(obj, method, args)?;
        self.add_temp(val, ty.clone());
        Ok((val, ty))
    }

    fn compile_method_call_inner(
        &mut self,
        obj: &Expr,
        method: &str,
        args: &[Expr],
    ) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
        // === Vec/Option/Result closure methods ===
        // Detect BEFORE compile_expr(obj) so we have full AST access to the closure.
        // The element type comes from the closure argument's type annotation,
        // NOT from parsing mangled type names.
        let closure_methods = ["map", "filter", "any", "all", "and_then", "unwrap_or_else", "map_err", "reduce", "read", "modify"];
        if closure_methods.contains(&method) && args.len() == 1 {
            if let ExprKind::Closure { args: closure_args, body, .. } = &args[0].inner {
                // Now compile the object expression to get obj_val
                let (obj_val, raw_obj_ty) = self.compile_expr(obj)?;
                let obj_ty = raw_obj_ty.flatten_specialized();

                // Extract elem_ty from obj_ty's type_args (first generic parameter, except for map_err which uses the second), or fallback to closure argument type
                let mut parsed_args_storage = Vec::new(); // to store parsed args safely
                let elem_ty_opt = obj_ty.as_named_type().and_then(|(name, args)| {
                    let mut actual_args = args;
                    if actual_args.is_empty() && name.contains('[') {
                         let parsed_ty = crate::compiler::mangler::MANGLER.parse_type_str(name);
                         match parsed_ty {
                             Type::Struct(_, parsed_args) | Type::Enum(_, parsed_args) => {
                                 parsed_args_storage = parsed_args;
                                 actual_args = &parsed_args_storage;
                             }
                             _ => {}
                         }
                    }
                    if method == "map_err" {
                        actual_args.get(1).cloned()
                    } else {
                        actual_args.first().cloned()
                    }
                }).or_else(|| closure_args.first().and_then(|(_, ty_opt)| ty_opt.clone()));
                
                
                let elem_ty = elem_ty_opt.ok_or_else(|| TlError::from(CodegenErrorKind::Internal("Could not determine element type for dynamic collection method".to_string())))?;

                // Dispatch: Option vs Result vs Vec vs Mutex
                let (is_option, is_result, is_mutex) = match &obj_ty {
                    Type::Enum(name, _) => {
                        let base = crate::compiler::ast::mangle_base_name(name);
                        (base == "Option", base == "Result", false)
                    }
                    Type::Struct(name, _) => {
                        let base = crate::compiler::ast::mangle_base_name(name);
                        (false, false, base == "Mutex")
                    }
                    Type::SpecializedType { gen_type, .. } => {
                        if gen_type.is_enum_type() {
                            let base = gen_type.get_base_name();
                            (base == "Option", base == "Result", false)
                        } else {
                            let base = gen_type.get_base_name();
                            (false, false, base == "Mutex")
                        }
                    }
                    _ => (false, false, false),
                };

                if is_option && (method == "map" || method == "and_then" || method == "unwrap_or_else") {
                    return self.compile_option_closure_method(
                        obj_val, &obj_ty, &elem_ty, method, closure_args, body,
                    );
                }

                if is_result && (method == "map" || method == "map_err" || method == "and_then" || method == "unwrap_or_else") {
                    return self.compile_result_closure_method(
                        obj_val, &obj_ty, &elem_ty, method, closure_args, body,
                    );
                }
                
                if is_mutex && (method == "read" || method == "modify") {
                    return self.compile_mutex_closure_method(
                        obj_val, &obj_ty, &elem_ty, method, &args[0]
                    );
                }

                return self.compile_vec_closure_method(
                    obj_val, &obj_ty, &elem_ty, method,
                    closure_args, body,
                );
            }
        }

        let (obj_val, raw_obj_ty) = self.compile_expr(obj)?;
        // Normalize SpecializedType to Struct/Enum with mangled_name and type_args preserved.
        // This ensures all downstream pattern matches (e.g., Type::Struct(name, args)) work correctly
        // while type_args remain accessible.
        let obj_ty = self.normalize_type(&raw_obj_ty);

        // Trait Object Dynamic Dispatch
        if let Type::TraitObject(trait_name) = &obj_ty {
            let fat_ptr = obj_val.into_struct_value();
            let data_ptr = self.builder.build_extract_value(fat_ptr, 0, "dyn_data_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_pointer_value();
            let vtable_ptr = self.builder.build_extract_value(fat_ptr, 1, "dyn_vtable_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_pointer_value();
            
            let trait_def = self.trait_defs.get(trait_name).ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Trait {} not found", trait_name))))?.clone();
            let method_idx = trait_def.methods.iter().position(|m| m.name == method).ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Method {} not found in trait {}", method, trait_name))))?;
            let method_sig = &trait_def.methods[method_idx];

            let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
            let fn_ptr_ptr = unsafe {
                self.builder.build_gep(ptr_type, vtable_ptr, &[self.context.i32_type().const_int(method_idx as u64, false)], "fn_ptr_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
            };
            let fn_ptr = self.builder.build_load(ptr_type, fn_ptr_ptr, "fn_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_pointer_value();

            let mut compiled_args: Vec<inkwell::values::BasicMetadataValueEnum<'ctx>> = Vec::new();
            let mut param_types: Vec<inkwell::types::BasicTypeEnum> = Vec::new();
            
            compiled_args.push(data_ptr.into());
            param_types.push(ptr_type.into());
            for arg_expr in args {
                let (arg_val, arg_ty) = self.compile_expr(arg_expr)?;
                compiled_args.push(arg_val.into());
                let arg_llvm_ty = self.get_llvm_type(&arg_ty).unwrap_or(self.context.i64_type().into());
                param_types.push(arg_llvm_ty);
            }

            let ret_llvm_ty = self.get_llvm_type(&method_sig.return_type).unwrap_or_else(|_| self.context.i8_type().into());
            let fn_type = ret_llvm_ty.fn_type(&param_types.iter().map(|&t| t.into()).collect::<Vec<_>>(), false);
            let call_site = self.builder.build_indirect_call(fn_type, fn_ptr, &compiled_args, "dyn_call").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let ret_val = match call_site.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v,
                _ => self.context.i8_type().const_zero().into(),
            };
            return Ok((ret_val, method_sig.return_type.clone()));
        }

        // === Vec.join(sep: String) -> String ===
        if method == "join" && args.len() == 1 {
            let is_vec = match &obj_ty {
                Type::Struct(name, _) => {
                    let base = mangle_base_name(name);
                    base == "Vec"
                }
                _ => false,
            };
            if is_vec {
                return self.compile_vec_join(obj_val, &obj_ty, args);
            }
        }
        

        // === Channel.send(val: T) -> bool ===
        if method == "send" && args.len() == 1 {
            if let Type::Struct(name, type_args) = &obj_ty {
                if mangle_base_name(name) == "Channel" {
                    let struct_name = if type_args.is_empty() { name.clone() } else { self.mangle_type_name("Channel", type_args) };
                    let llvm_struct_ty = self.context.get_struct_type(&struct_name).ok_or_else(|| TlError::from(CodegenErrorKind::Internal("unexpected None".to_string())))?;
                    let id_val = if obj_val.is_pointer_value() {
                        let id_ptr = self.builder.build_struct_gep(llvm_struct_ty, obj_val.into_pointer_value(), 0, "id_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        self.builder.build_load(self.context.i64_type(), id_ptr, "id").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                    } else if obj_val.is_struct_value() {
                        self.builder.build_extract_value(obj_val.into_struct_value(), 0, "id").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value().into()
                    } else {
                        return Err(TlError::from(CodegenErrorKind::Internal("Channel obj_val is neither pointer nor struct".to_string())));
                    };

                    let (arg_val, _) = self.compile_expr(&args[0])?;
                    
                    let arg_i64 = match arg_val {
                        inkwell::values::BasicValueEnum::IntValue(v) => {
                            if v.get_type().get_bit_width() == 64 { v }
                            else { self.builder.build_int_z_extend(v, self.context.i64_type(), "zext").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))? }
                        },
                        inkwell::values::BasicValueEnum::FloatValue(v) => {
                            if v.get_type() == self.context.f64_type() {
                                self.builder.build_bit_cast(v, self.context.i64_type(), "bitcast_f64").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value()
                            } else {
                                let ext = self.builder.build_float_ext(v, self.context.f64_type(), "fext").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                                self.builder.build_bit_cast(ext, self.context.i64_type(), "bitcast").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value()
                            }
                        },
                        inkwell::values::BasicValueEnum::PointerValue(v) => {
                            self.builder.build_ptr_to_int(v, self.context.i64_type(), "ptr_to_int").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                        },
                        _ => return Err(TlError::from(CodegenErrorKind::UnsupportedOperation("Unsupported Channel generic type".to_string()))),
                    };

                    let send_fn = self.module.get_function("tl_channel_send").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_channel_send not found".to_string())))?;
                    let call = self.builder.build_call(send_fn, &[id_val.into(), arg_i64.into()], "send_ret").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    let res_val = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err(CodegenErrorKind::Internal("tl_channel_send returned void".to_string()).into()),
                    };
                    
                    // tl_channel_send returns u64 boolean (wait, it returns bool! we declared it as returning bool?)
                    // FFI says `fn tl_channel_send(...) -> bool`. In C ABI, Rust `bool` is usually i8.
                    let bool_val = self.builder.build_int_truncate(res_val.into_int_value(), self.context.bool_type(), "trunc").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    
                    return Ok((bool_val.into(), Type::Bool));
                }
            }
        }

        // === Channel.recv() -> T ===
        if method == "recv" && args.len() == 0 {
            if let Type::Struct(name, _type_args) = &obj_ty {
                if mangle_base_name(name) == "Channel" {
                    let inner_ty = self.extract_inner_ty(&obj_ty);
                    let struct_name = if inner_ty != Type::I64 { self.mangle_type_name("Channel", &[inner_ty.clone()]) } else { name.clone() };
                    let llvm_struct_ty = self.context.get_struct_type(&struct_name).ok_or_else(|| TlError::from(CodegenErrorKind::Internal("unexpected None".to_string())))?;
                    let id_val = if obj_val.is_pointer_value() {
                        let id_ptr = self.builder.build_struct_gep(llvm_struct_ty, obj_val.into_pointer_value(), 0, "id_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        self.builder.build_load(self.context.i64_type(), id_ptr, "id").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                    } else if obj_val.is_struct_value() {
                        self.builder.build_extract_value(obj_val.into_struct_value(), 0, "id").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value().into()
                    } else {
                        return Err(TlError::from(CodegenErrorKind::Internal("Channel obj_val is neither pointer nor struct".to_string())));
                    };

                    let recv_fn = self.module.get_function("tl_channel_recv").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_channel_recv not found".to_string())))?;
                    let call = self.builder.build_call(recv_fn, &[id_val.into()], "recv_ret").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    let raw_i64 = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v.into_int_value(),
                        _ => return Err(CodegenErrorKind::Internal("tl_channel_recv returned void".to_string()).into()),
                    };
                    
                    let target_ty = self.get_llvm_type(&inner_ty)?;
                    let final_val: inkwell::values::BasicValueEnum<'ctx> = if target_ty.is_pointer_type() {
                        self.builder.build_int_to_ptr(raw_i64, target_ty.into_pointer_type(), "int_to_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into()
                    } else if target_ty.is_float_type() {
                        if target_ty.into_float_type() == self.context.f64_type() {
                            self.builder.build_bit_cast(raw_i64, target_ty, "bitcast").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into()
                        } else {
                            let f64_val = self.builder.build_bit_cast(raw_i64, self.context.f64_type(), "bitcast").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_float_value();
                            self.builder.build_float_trunc(f64_val, target_ty.into_float_type(), "ftrunc").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into()
                        }
                    } else if target_ty.is_int_type() {
                        let width = target_ty.into_int_type().get_bit_width();
                        if width < 64 {
                            self.builder.build_int_truncate(raw_i64, target_ty.into_int_type(), "trunc").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into()
                        } else {
                            raw_i64.into()
                        }
                    } else {
                        return Err(TlError::from(CodegenErrorKind::UnsupportedOperation("Unsupported Channel output type".to_string())));
                    };

                    return Ok((final_val, inner_ty.clone()));
                }
            }
        }
        
        // === Channel.try_recv() -> Result<T, TlError> ===
        if method == "try_recv" && args.len() == 0 {
            if let Type::Struct(name, type_args) = &obj_ty {
                if mangle_base_name(name) == "Channel" {
                    let inner_ty = if type_args.len() == 1 { type_args[0].clone() } else { Type::I64 };
                    let struct_name = if type_args.is_empty() { name.clone() } else { self.mangle_type_name("Channel", type_args) };
                    let llvm_struct_ty = self.context.get_struct_type(&struct_name).ok_or_else(|| TlError::from(CodegenErrorKind::Internal("unexpected None".to_string())))?;
                    let id_val = if obj_val.is_pointer_value() {
                        let id_ptr = self.builder.build_struct_gep(llvm_struct_ty, obj_val.into_pointer_value(), 0, "id_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        self.builder.build_load(self.context.i64_type(), id_ptr, "id").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                    } else if obj_val.is_struct_value() {
                        self.builder.build_extract_value(obj_val.into_struct_value(), 0, "id").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value().into()
                    } else {
                        return Err(TlError::from(CodegenErrorKind::Internal("Channel obj_val is neither pointer nor struct".to_string())));
                    };

                    let success_alloc = self.builder.build_alloca(self.context.bool_type(), "success_out").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    let recv_fn = self.module.get_function("tl_channel_try_recv").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_channel_try_recv not found".to_string())))?;
                    let call = self.builder.build_call(recv_fn, &[id_val.into(), success_alloc.into()], "recv_ret").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    let raw_i64 = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v.into_int_value(),
                        _ => return Err(CodegenErrorKind::Internal("tl_channel_try_recv returned void".to_string()).into()),
                    };
                    
                    let target_ty = self.get_llvm_type(&inner_ty)?;
                    let final_val: inkwell::values::BasicValueEnum<'ctx> = if target_ty.is_pointer_type() {
                        self.builder.build_int_to_ptr(raw_i64, target_ty.into_pointer_type(), "int_to_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into()
                    } else if target_ty.is_float_type() {
                        if target_ty.into_float_type() == self.context.f64_type() {
                            self.builder.build_bit_cast(raw_i64, target_ty, "bitcast").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into()
                        } else {
                            let f64_val = self.builder.build_bit_cast(raw_i64, self.context.f64_type(), "bitcast").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_float_value();
                            self.builder.build_float_trunc(f64_val, target_ty.into_float_type(), "ftrunc").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into()
                        }
                    } else if target_ty.is_int_type() {
                        let width = target_ty.into_int_type().get_bit_width();
                        if width < 64 {
                            self.builder.build_int_truncate(raw_i64, target_ty.into_int_type(), "trunc").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into()
                        } else {
                            raw_i64.into()
                        }
                    } else {
                        return Err(TlError::from(CodegenErrorKind::UnsupportedOperation("Unsupported Channel output type".to_string())));
                    };

                    let is_success = self.builder.build_load(self.context.bool_type(), success_alloc, "is_success").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value();
                    
                    let res_mangled = self.mangle_type_name("Option", &[inner_ty.clone()]);
                    let res_ty = Type::Enum(res_mangled.clone(), vec![]);
                    let llvm_res_ty = self.get_llvm_type(&res_ty)?;
                    
                    // Ensure the enum is monomorphized and compiled
                    let enum_struct_ty = if let Some(ty) = self.enum_types.get(&res_mangled) {
                        *ty
                    } else {
                        self.monomorphize_enum("Option", &[inner_ty.clone()]).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        *self.enum_types.get(&res_mangled).ok_or_else(|| TlError::from(CodegenErrorKind::Internal("Failed to monomorphize Option".to_string())))?
                    };

                    let res_alloc = self.builder.build_alloca(llvm_res_ty, "res_alloc").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    
                    let ok_bb = self.context.append_basic_block(self.current_function()?, "recv_ok");
                    let err_bb = self.context.append_basic_block(self.current_function()?, "recv_err");
                    let cont_bb = self.context.append_basic_block(self.current_function()?, "recv_cont");
                    
                    self.builder.build_conditional_branch(is_success, ok_bb, err_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    
                    let malloc_fn = self.module.get_function("malloc").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("malloc not found".to_string())))?;
                    let reg_fn = self.module.get_function("tl_mem_register_struct_named").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_mem_register_struct_named not found".to_string())))?;
                    let unreg_fn = self.module.get_function("tl_mem_unregister").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_mem_unregister not found".to_string())))?;
                    
                    let size_ptr = unsafe { self.builder.build_gep(enum_struct_ty, self.context.ptr_type(inkwell::AddressSpace::default()).const_null(), &[self.context.i64_type().const_int(1, false)], "size_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))? };
                    let size = self.builder.build_ptr_to_int(size_ptr, self.context.i64_type(), "enum_size").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    let name_ptr = self.builder.build_global_string_ptr(&res_mangled, "enum_name").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.as_pointer_value();

                    // OK block
                    self.builder.position_at_end(ok_bb);
                    let call_ok = self.builder.build_call(malloc_fn, &[size.into()], "ok_malloc").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    let ok_malloc = match call_ok.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
                        _ => return Err(CodegenErrorKind::Internal("malloc returned void".to_string()).into()),
                    };
                    self.builder.build_call(reg_fn, &[ok_malloc.into(), name_ptr.into()], "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    
                    let tag_ptr = self.builder.build_struct_gep(enum_struct_ty, ok_malloc, 0, "tag_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    self.builder.build_store(tag_ptr, self.context.i32_type().const_int(0, false)).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?; // Some(0)
                    let payload_ptr = self.builder.build_struct_gep(enum_struct_ty, ok_malloc, 1, "payload_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    let typed_payload_ptr = self.builder.build_pointer_cast(payload_ptr, self.context.ptr_type(inkwell::AddressSpace::default()), "typed_payload").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    self.builder.build_store(typed_payload_ptr, final_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    self.builder.build_store(res_alloc, ok_malloc).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    self.builder.build_unconditional_branch(cont_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    
                    // ERR block
                    self.builder.position_at_end(err_bb);
                    let call_err = self.builder.build_call(malloc_fn, &[size.into()], "err_malloc").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    let err_malloc = match call_err.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
                        _ => return Err(CodegenErrorKind::Internal("malloc returned void".to_string()).into()),
                    };
                    self.builder.build_call(reg_fn, &[err_malloc.into(), name_ptr.into()], "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    
                    let tag_ptr = self.builder.build_struct_gep(enum_struct_ty, err_malloc, 0, "tag_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    self.builder.build_store(tag_ptr, self.context.i32_type().const_int(1, false)).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?; // None(1)
                    self.builder.build_store(res_alloc, err_malloc).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    self.builder.build_unconditional_branch(cont_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    
                    self.builder.position_at_end(cont_bb);
                    let final_res = self.builder.build_load(llvm_res_ty, res_alloc, "res").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    
                    // Shallow Unregister: Transfer ownership to caller
                    self.builder.build_call(unreg_fn, &[final_res.into()], "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    
                    return Ok((final_res, res_ty));
                }
            }
        }

                    // We must allocate a new String! But wait, 'tl_string_new' was used earlier!
                    // I will just use a null pointer for String and assume it's just meant as "Empty". Actually, we can return Option<T> instead of Result!
                    // Wait, `try_recv` usually returns Option<T> in TL?
                    // Oh, TL standard Option has tag 0 for Some, 1 for None. Let's make it Option<T> !!

        // === Thread.join() -> Result<T, TlError> ===
        if method == "join" && args.len() == 0 {
            let is_thread = match &obj_ty {
                Type::Struct(name, _) => {
                    let base = mangle_base_name(name);
                    base == "Thread"
                }
                _ => false,
            };
            if is_thread {
                let inner_ty = self.extract_inner_ty(&obj_ty);
                
                let t_struct_ty = self.context.struct_type(&[self.context.i64_type().into()], false);
                let ptr = if obj_val.is_pointer_value() {
                    obj_val.into_pointer_value()
                } else {
                    return Err(TlError::from(CodegenErrorKind::Internal("Thread obj is not pointer".to_string())));
                };
                let id_val = self.builder.build_load(self.context.i64_type(), self.builder.build_struct_gep(t_struct_ty, ptr, 0, "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?, "id").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                
                let join_fn = self.get_fn("tl_thread_join")?;
                let call = self.builder.build_call(join_fn, &[id_val.into()], "join_ret").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let raw_ptr = match call.try_as_basic_value() {
                    inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
                    _ => return Err(CodegenErrorKind::Internal("tl_thread_join returned void".to_string()).into()),
                };
                
                // Check if raw_ptr is Null
                let is_null = self.builder.build_is_null(raw_ptr, "is_null").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                
                let res_ty = Type::Enum(self.mangle_type_name("Result", &[inner_ty.clone(), Type::String("String".to_string())]), vec![]);
                let llvm_res_ty = self.get_llvm_type(&res_ty)?;
                let res_alloc = self.builder.build_alloca(llvm_res_ty, "res_alloc").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                
                let ok_bb = self.context.append_basic_block(self.current_function()?, "join_ok");
                let err_bb = self.context.append_basic_block(self.current_function()?, "join_err");
                let cont_bb = self.context.append_basic_block(self.current_function()?, "join_cont");
                
                self.builder.build_conditional_branch(is_null, err_bb, ok_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                
                // OK block
                self.builder.position_at_end(ok_bb);
                let typed_ptr = self.builder.build_pointer_cast(raw_ptr, self.context.ptr_type(inkwell::AddressSpace::default()), "typed").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let llvm_inner_ty = self.get_llvm_type(&inner_ty)?;
                // if it uses sret, or if it is directly loaded:
                let loaded = if matches!(inner_ty, Type::Tensor(_, _) | Type::Struct(_, _) | Type::Tuple(_) | Type::String(_)) {
                    self.builder.build_load(llvm_inner_ty, typed_ptr, "inner_val").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                } else {
                    self.builder.build_load(llvm_inner_ty, typed_ptr, "inner_val").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                };
                
                let res_mangled = self.mangle_type_name("Result", &[inner_ty.clone(), Type::String("String".to_string())]);
                // Recompile enum definition just in case it's not present
                let enum_ty = if let Some(ty) = self.enum_types.get(&res_mangled) { 
                    *ty 
                } else { 
                    self.monomorphize_enum("Result", &[inner_ty.clone(), Type::String("String".to_string())]).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    let specialized_def = self.enum_defs.get(&res_mangled).ok_or_else(|| TlError::from(CodegenErrorKind::Internal("index out of bounds".to_string())))?.clone();
                    self.compile_enum_defs(&[specialized_def]).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    *self.enum_types.get(&res_mangled).ok_or_else(|| TlError::from(CodegenErrorKind::Internal("index out of bounds".to_string())))?
                };
                let malloc_fn = self.get_fn("malloc")?;
                let size_ptr = unsafe { self.builder.build_gep(enum_ty, self.context.ptr_type(inkwell::AddressSpace::default()).const_null(), &[self.context.i64_type().const_int(1, false)], "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))? };
                let size = self.builder.build_ptr_to_int(size_ptr, self.context.i64_type(), "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let call_ok = self.builder.build_call(malloc_fn, &[size.into()], "res_ok").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let ok_ptr = match call_ok.try_as_basic_value() { inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(), _ => return Err(CodegenErrorKind::Internal("malloc void".to_string()).into()), };
                self.builder.build_store(self.builder.build_struct_gep(enum_ty, ok_ptr, 0, "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?, self.context.i32_type().const_int(0, false)).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let payload_ok = self.builder.build_pointer_cast(self.builder.build_struct_gep(enum_ty, ok_ptr, 1, "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?, self.context.ptr_type(inkwell::AddressSpace::default()), "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let variant_ok_ty = self.context.struct_type(&[llvm_inner_ty], false);
                self.builder.build_store(self.builder.build_struct_gep(variant_ok_ty, payload_ok, 0, "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?, loaded).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                
                self.builder.build_store(res_alloc, ok_ptr).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                
                let free_tmp_fn = self.get_fn("tl_free_tmp")?;
                self.builder.build_call(free_tmp_fn, &[raw_ptr.into()], "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_unconditional_branch(cont_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                
                // ERR block
                self.builder.position_at_end(err_bb);
                let err_msg = "Thread panicked: joined null pointer".to_string();
                let msg_str = self.compile_string_literal(&err_msg).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                
                let call_err = self.builder.build_call(malloc_fn, &[size.into()], "res_err").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let err_ptr = match call_err.try_as_basic_value() { inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(), _ => return Err(CodegenErrorKind::Internal("malloc void".to_string()).into()), };
                self.builder.build_store(self.builder.build_struct_gep(enum_ty, err_ptr, 0, "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?, self.context.i32_type().const_int(1, false)).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let payload_err = self.builder.build_pointer_cast(self.builder.build_struct_gep(enum_ty, err_ptr, 1, "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?, self.context.ptr_type(inkwell::AddressSpace::default()), "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let variant_err_ty = self.context.struct_type(&[self.get_llvm_type(&Type::String("String".to_string()))?], false);
                self.builder.build_store(self.builder.build_struct_gep(variant_err_ty, payload_err, 0, "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?, msg_str.0).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                
                self.builder.build_store(res_alloc, err_ptr).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_unconditional_branch(cont_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                
                // CONT block
                self.builder.position_at_end(cont_bb);
                let final_res = self.builder.build_load(llvm_res_ty, res_alloc, "final_res").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                return Ok((final_res, res_ty));
            }
        }

        // === Vec: enumerate, flatten, zip ===
        if method == "enumerate" || method == "flatten" || method == "zip" {
            let is_vec = match &obj_ty {
                Type::Struct(name, _) => {
                    let base = mangle_base_name(name);
                    base == "Vec"
                }
                _ => false,
            };
            if is_vec {
                return self.compile_vec_builtin_method(obj_val, &obj_ty, method, args);
            }
        }

        // === Mutex: release ===
        if method == "release" && args.len() == 0 {
            let is_mutex = match &obj_ty {
                Type::Struct(name, _) => {
                    let base = mangle_base_name(name);
                    base == "Mutex"
                }
                _ => false,
            };
            if is_mutex {
                let m_struct_ty = self.context.struct_type(&[self.context.i64_type().into()], false);
                let ptr = if obj_val.is_pointer_value() {
                    obj_val.into_pointer_value()
                } else {
                    return Err(TlError::from(CodegenErrorKind::Internal("Mutex value is not pointer".to_string())));
                };
                let id_gep = self.builder.build_struct_gep(m_struct_ty, ptr, 0, "id_field").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let id_val = self.builder.build_load(self.context.i64_type(), id_gep, "id_val").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let fn_val = self.module.get_function("tl_mutex_release").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_mutex_release not found".to_string())))?;
                self.builder.build_call(fn_val, &[id_val.into()], "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                return Ok((self.context.i64_type().const_zero().into(), Type::Void));
            }
        }

        self.emit_method_call(obj, obj_val, obj_ty, method, args)
    }

    /// IndexAccess で使うメソッド名を解決する。
    /// ユーザー定義の `index` メソッドがあれば "index"、なければ "get" を返す。
    pub(crate) fn resolve_index_method(&self, ty: &Type) -> String {
        let type_name = ty.get_base_name();
        let base_name = mangle_base_name(&type_name).to_string();

        // 1. generic_impls にある impl ブロックから index メソッドを探す
        if let Some(impls) = self.generic_impls.get(&base_name) {
            for impl_block in impls {
                for method in &impl_block.methods {
                    if method.name == "index" {
                        return "index".to_string();
                    }
                }
            }
        }

        // 2. 既にモノモーフ化された関数名で探す（tl_{Type}_index）
        let mangled_fn = format!("tl_{}_index", type_name);
        if self.module.get_function(&mangled_fn).is_some() {
            return "index".to_string();
        }
        let mangled_fn_lower = format!("tl_{}_index", type_name.to_lowercase());
        if self.module.get_function(&mangled_fn_lower).is_some() {
            return "index".to_string();
        }

        // 3. フォールバック: get メソッド
        "get".to_string()
    }

    /// IndexMut の代入 (expr[i] = v) で使うメソッド名を解決する。
    /// ユーザー定義の `index_mut` メソッドがあれば "index_mut"、なければ "set" を返す。
    pub(crate) fn resolve_index_mut_method(&self, ty: &Type) -> String {
        let type_name = ty.get_base_name();
        let base_name = mangle_base_name(&type_name).to_string();

        // 1. generic_impls から index_mut メソッドを探す
        if let Some(impls) = self.generic_impls.get(&base_name) {
            for impl_block in impls {
                for method in &impl_block.methods {
                    if method.name == "index_mut" {
                        return "index_mut".to_string();
                    }
                }
            }
        }

        // 2. 既にモノモーフ化された関数名で探す
        let mangled_fn = format!("tl_{}_index_mut", type_name);
        if self.module.get_function(&mangled_fn).is_some() {
            return "index_mut".to_string();
        }

        // 3. フォールバック: set メソッド
        "set".to_string()
    }

    fn emit_method_call(
        &mut self,
        obj_expr_context: &Expr, // Passed for fallback Unevaluated methods which need AST access
        obj_val: BasicValueEnum<'ctx>,
        obj_ty: Type,
        method: &str,
        args: &[Expr],
    ) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {


        // Try TypeManager first
        let type_name_opt = match &obj_ty {

            Type::Struct(n, _) => Some(n.clone()),
            Type::Enum(n, _) => Some(n.clone()),
            Type::Tensor(_, _) => Some("Tensor".to_string()),
            Type::GradTensor(_, _) => Some("Tensor".to_string()),
            Type::String(_) => Some("String".to_string()),
            Type::I64 => Some("I64".to_string()),
            Type::I32 => Some("I32".to_string()),
            Type::F32 => Some("F32".to_string()),
            Type::F64 => Some("F64".to_string()),
            Type::Bool => Some("Bool".to_string()),
            _ => None,
        };

        if let Some(name) = type_name_opt {
             // Handle special logical types if we want to support UserDefined alias canonicalization? 
             // Logic above handles it if UserDefined("Custom") returns "Custom".
             if let Some(type_info) = self.type_manager.get_type(&name) {
                 if type_info.has_instance_method(method) {
                     if let Some(overloads) = type_info.get_instance_overloads(method) {
                         // Try to find matching overload by arg count
                         for overload in overloads {
                             if overload.arg_types.len() == args.len() {
                                 match &overload.impl_fn {
                                     InstanceMethod::Evaluated(func) => {
                                         let func = *func;
                                         let mut compiled_args = Vec::new();
                                         for arg in args {
                                             compiled_args.push(self.compile_expr(arg)?);
                                         }
                                         return func(self, obj_val, obj_ty, compiled_args);
                                     }
                                     InstanceMethod::Unevaluated(func) => {
                                         let func = *func;
                                         return func(self, obj_expr_context, method, args);
                                     }
                                     InstanceMethod::SignatureOnly => {
                                         // No implementation, skip this overload
                                         continue;
                                     }
                                 }
                             }
                         }
                     }
                 }
             }
        }


        // 2. Resolve Type Name to check Manager
        let type_name = match &obj_ty {
            Type::Struct(name, _) => name.clone(),
            Type::Enum(name, _) => name.clone(),
            Type::Tensor(_, _) => "Tensor".to_string(),
            Type::GradTensor(_, _) => "Tensor".to_string(),
            Type::F32 => "F32".to_string(),
            Type::F64 => "F64".to_string(),
            Type::I64 => "I64".to_string(),
            Type::I32 => "I32".to_string(),
            Type::String(_) => "String".to_string(),
            _ => "".to_string(),
        };

        // 3. Manager Lookup
        // Use copied() to drop borrow of self.instance_methods
        let method_opt = if !type_name.is_empty() {
            self.instance_methods
                .get(&type_name)
                .and_then(|m| m.get(method))
                .copied()
        } else {
            None
        };

        if let Some(method_fn) = method_opt {
            match method_fn {
                InstanceMethod::Evaluated(func) => {
                    // Compile args
                    let mut compiled_args = Vec::with_capacity(args.len());
                    let mut compiled_args_types = Vec::with_capacity(args.len());
                    for arg in args {
                        let (val, ty) = self.compile_expr(arg)?;
                        compiled_args.push((val, ty.clone()));
                        compiled_args_types.push((val, ty));
                    }

                    // Call function
                    let res_result = func(self, obj_val, obj_ty.clone(), compiled_args);


                    return res_result;

                }
                InstanceMethod::Unevaluated(func) => {
                    // Unevaluated methods handle their own arg compilation and cleanup
                    return func(self, obj_expr_context, method, args);
                }
                InstanceMethod::SignatureOnly => {
                    // Should not happen, fall through to lookup elsewhere
                }
            }
        }


        // Map and String methods are now handled by TypeManager (Evaluated).
        // See map_methods.rs and string_methods.rs for implementations.


        // Special Handling for Tensor methods was removed in favor of TypeManager registration.
        // See builtin_types/non_generic/tensor.rs for method implementations.

        // 4. Generic Fallback (Struct Methods / Mangled Names)
        let struct_name = match &obj_ty {

            Type::Struct(name, _) | Type::Enum(name, _) => name.clone(),
            Type::Path(segments, _) => if let Some(n) = segments.last() { n.clone() } else { return Err(CodegenErrorKind::Internal("Empty path".to_string()).into()) },
            Type::Tensor(_, _) => "Tensor".to_string(),
            Type::String(_) => "String".to_string(),
            Type::SpecializedType { mangled_name, .. } => mangled_name.clone(),
            _ => panic!("PANIC_METHOD_NOT_FOUND: obj_ty={:?} method={}", obj_ty, method),
        };

        let type_name = struct_name.clone();


        // Extract simple name from module path for mangling
        let simple_struct_name = struct_name.clone();

        // Try exact mangling first: tl_{Struct}_{Method}


        // Check for generic impls first
        // Extract base name from mangled type name (e.g., "Option<i64>" -> "Option", "Result_i32" -> "Result")
        let (base_type_name, inferred_generics) = if type_name.contains('<') {
            let base = type_name.split('<').next().unwrap_or(&type_name).to_string();
            // Extract generics from e.g., "Option<i64>" -> [I64]
            let generics_str = type_name.trim_start_matches(&base).trim_start_matches('<').trim_end_matches('>');
            let parsed_generics: Vec<Type> = generics_str.split(',')
                .filter_map(|s| {
                    let s = s.trim();
                    match s.to_lowercase().as_str() {
                        "i64" => Some(Type::I64),
                        "i32" => Some(Type::I32),
                        "f32" => Some(Type::F32),
                        "f64" => Some(Type::F64),
                        "bool" => Some(Type::Bool),
                        "u8" => Some(Type::U8),
                        "string" => Some(Type::String("String".to_string())),
                        "" => None,
                        _ => Some(Type::Struct(s.to_string(), vec![])),
                    }
                })
                .collect();
            (base, parsed_generics)
        } else if mangle_has_args(&type_name) {
            let base = mangle_base_name(&type_name).to_string();
            let parsed_generics = match &obj_ty {
                Type::Struct(_, args) | Type::Enum(_, args) => args.clone(),
                _ => vec![],
            };
            (base, parsed_generics)
        } else {
            (type_name.clone(), vec![])
        };

        let mangled_name = format!("tl_{}_{}", simple_struct_name, method);
        // Fallback to lowercase
        let stdlib_name = format!("tl_{}_{}", simple_struct_name.to_lowercase(), method);

        let (func_val, final_name) = if let Some(f) = self.module.get_function(&mangled_name) {
            (f, mangled_name)
        } else if let Some(f) = self.module.get_function(&stdlib_name) {
            (f, stdlib_name)
        } else {
            return Err(TlError::from(CodegenErrorKind::Internal(format!(
                "Method {} not found in struct {} (checked {} and {})",
                method, struct_name, mangled_name, stdlib_name
            ))));
        };

        // Get return type (for SRET check)
        let mut ret_ty = if let Some(ret) = self.method_return_types.get(&final_name) {
             ret.clone()
        } else {
             self.get_return_type_from_signature(func_val)
        };
        
        // FIX: For generic runtime functions (like tl_vec_ptr_get), the return type may have 
        // generic placeholders (e.g., Option<T>) that need to be substituted with actual generics.
        // This is necessary because runtime functions like tl_vec_ptr_get are shared across all
        // Vec<T> types, but the actual T in Option<T> depends on the call site.
        if !inferred_generics.is_empty() {
            match &ret_ty {
                Type::Enum(name, args) if (name == "Option" || name == "Result") && args.len() == 1 => {
                    // For Vec.get -> Option<T>, replace T with the actual element type
                    if method == "get" && base_type_name == "Vec" {
                        ret_ty = Type::Enum(name.clone(), inferred_generics.clone());
                    }
                }
                Type::Enum(name, args) if name == "Result" && args.len() == 2 => {
                    // Handle Result<T, E> if needed
                    if method == "get" && base_type_name == "Vec" {
                        let new_ok = inferred_generics.get(0).cloned().unwrap_or(args[0].clone());
                        let new_err = args[1].clone();
                        ret_ty = Type::Enum(name.clone(), vec![new_ok, new_err]);
                    }
                }
                _ => {}
            }
        }
        
        // For Option.unwrap/unwrap_or, the return type T should be replaced with generics[0]
        // Note: Get generics from obj_ty which has the correct type from the call site
        if base_type_name == "Option" && (method == "unwrap" || method == "unwrap_or") {
            let args = match &obj_ty {
                Type::Enum(_, args) => Some(args.clone()),
                Type::SpecializedType { type_args, .. } => Some(type_args.clone()),
                _ => None,
            };
            if let Some(args) = args {
                if let Some(inner_ty) = args.get(0) {
                    ret_ty = inner_ty.clone();
                }
            }
        }
        
        // For Result.unwrap, the return type T should be replaced with generics[0]
        if base_type_name == "Result" && method == "unwrap" {
            let args = match &obj_ty {
                Type::Enum(_, args) => Some(args.clone()),
                Type::SpecializedType { type_args, .. } => Some(type_args.clone()),
                _ => None,
            };
            if let Some(args) = args {
                if let Some(ok_ty) = args.get(0) {
                    ret_ty = ok_ty.clone();
                }
            }
        }

        // Normalize ret_ty so SpecializedType is resolved for SRET allocation
        ret_ty = self.normalize_type(&ret_ty);

        // SRET Check: Use the actual LLVM function signature for consistency.
        // compile_fn_proto names the first parameter "sret" when SRET is used.
        let uses_sret = func_val.count_params() > 0
            && func_val.get_first_param()
                .map(|p| p.get_name().to_str().unwrap_or("") == "sret")
                .unwrap_or(false);
        let mut sret_ptr = None;

        if uses_sret {
             // OLD: Stack Allocation (alloca) -> Causes Stack Corruption
             // NEW: Heap Allocation (malloc + register) -> Correct for RefCounted Structs/SRET

             // 1. Get Struct/Enum Type and Size from CodeGen struct_types map
             let (struct_name, generics) = match &ret_ty {
                 Type::Struct(n, g) | Type::Enum(n, g) => (n, g),
                 _ => return Err(TlError::from(CodegenErrorKind::Internal(format!("SRET used on non-aggregate type: {:?}", ret_ty)))),
             };
             
             let mangled_name = if generics.is_empty() {
                 struct_name.to_string()
             } else {
                 // Use base name to avoid double-mangling (e.g. Entry[i64][i64] -> Entry[i64][i64][i64][i64])
                 let base = mangle_base_name(struct_name);
                 self.mangle_type_name(base, generics)
             };
             
             // Simple name lookup (as done in compile_struct_init)
             let simple_lookup_name = mangled_name.clone();

             // Ensure type is monomorphized and registered (handles both struct and enum generics)
             let _ = self.get_or_monomorphize_type(&ret_ty).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

             let _ = self.struct_types.get(&simple_lookup_name)
                 .or_else(|| self.enum_types.get(&simple_lookup_name))
                 .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Struct type {} not found for SRET allocation (tried {})", simple_lookup_name, simple_lookup_name))))?;
             
             let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
             let alloca = self.create_entry_block_alloca_manual(self.current_function()?, "sret_ptr", &ptr_type.into()).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
             sret_ptr = Some(alloca);
        }

        let mut compiled_args_vals = Vec::with_capacity(args.len() + 1);
        let mut compiled_args_types = Vec::with_capacity(args.len());

        // Push SRET Ptr if needed
        if let Some(ptr) = sret_ptr {
            compiled_args_vals.push(ptr.into());
        }

        // Push Receiver
        compiled_args_vals.push(obj_val.into());

        for arg in args {
            let (val, ty) = self.compile_expr(arg)?;
            
            // ARGUMENT PASSING FIX: Retain ownership because Callee takes ownership (and releases at end)
            // If we don't retain, the Callee's release will free the memory while Caller still holds it (if var).
            // We skip String because it uses manual management not compatible with refcounts map yet.
            // FIX (2026-02-16): Runtime functions (and TL functions) generally BORROW arguments.
            // They do NOT release them at the end.
            // Therefore, we should NOT retain here.
            
            let should_retain = match &ty {
                Type::Struct(n, _) if n == "String" => false, // String manual management
                // Type::Struct(_, _) | Type::Enum(_, _) | Type::Tensor(_, _) | Type::Tuple(_) | Type::SpecializedType { .. } => true,
                _ => false, 
            };
            
            if should_retain {
                 self.emit_retain(val, &ty)?;
            }

            compiled_args_vals.push(val.into());
            compiled_args_types.push((val, ty));
        }

        // Call
        let call = self
            .builder
            .build_call(func_val, &compiled_args_vals, "method_call")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        // Return handling
        // Return handling
        if let Some(ptr) = sret_ptr {
             let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
             let loaded = self.builder.build_load(ptr_ty, ptr, "sret_method_loaded").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
             Ok((loaded, ret_ty))
        } else {
            match call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(_) => {
                    let v = self.check_tensor_result(call, "method_call_error")?;
                    Ok((v, ret_ty))
                }
                _ => Ok((
                    self.context.i64_type().const_int(0, false).into(),
                    Type::Void,
                )),
            }
        }
    }

    fn compile_relation_call(
        &mut self,
        name: &str,
        args: &[Expr],
    ) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
        let function = self
            .module
            .get_function(name)
            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Relation wrapper {} not found", name))))?;

        let mut mask: i64 = 0;
        let mut compiled_args = Vec::new();
        let mut tags = Vec::new();
        let i64_type = self.context.i64_type();
        let i8_type = self.context.i8_type();

        // Semantics phase no longer injects metadata. Start index is always 0.
        let start_index = 0;

        for (i, arg) in args.iter().skip(start_index).enumerate() {
            match &arg.inner {
                ExprKind::LogicVar(_) => {
                    // Free variable — mask bit NOT set (0 = unbound)
                    // Pass 0 placeholder
                    compiled_args.push(i64_type.const_int(0, false).into());
                    tags.push(3); // Default to Entity
                }
                ExprKind::Symbol(sym_name) => {
                    // Check if variable exists in scope
                    let mut found = false;
                    for scope in self.variables.iter().rev() {
                        if scope.contains_key(sym_name) {
                            found = true;
                            break;
                        }
                    }

                    if found {
                        // Bound variable — set mask bit (1 = bound)
                        mask |= 1 << i;
                        let var_expr = Expr {
                            inner: ExprKind::Variable(sym_name.clone()),
                            span: arg.span.clone(),
                        };
                        let (val, ty) = self.compile_expr(&var_expr)?;
                        compiled_args.push(val);
                        let tag = match ty {
                            Type::I64 | Type::I32 | Type::I8 | Type::U8 | Type::U16 | Type::U32 | Type::Usize => 0, // Int
                            Type::F32 | Type::F64 | Type::F16 | Type::BF16 => 1,             // Float
                            Type::Bool => 2,                        // Bool
                            Type::Entity => 3,                      // Entity
                            _ => 3,
                        };
                        tags.push(tag);
                    } else {
                        // Assume constant Entity Name
                        let add_entity_fn = self.get_fn("tl_kb_add_entity")?;
                        let name_ptr = self
                            .builder
                            .build_global_string_ptr(sym_name, "ent_name")
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        let call = self
                            .builder
                            .build_call(
                                add_entity_fn,
                                &[name_ptr.as_pointer_value().into()],
                                "ent_id",
                            )
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                        let val = match call.try_as_basic_value() {
                            inkwell::values::ValueKind::Basic(v) => v,
                            _ => return Err(CodegenErrorKind::Internal("Invalid return from add_entity".to_string()).into()),
                        };
                        // Bound constant entity — set mask bit
                        mask |= 1 << i;
                        compiled_args.push(val);
                        tags.push(3); // ConstantTag::Entity
                    }
                }
                ExprKind::StringLiteral(s) => {
                    let add_entity_fn = self.get_fn("tl_kb_add_entity")?;
                    let name_ptr = self
                        .builder
                        .build_global_string_ptr(s, "ent_name")
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    let call = self
                        .builder
                        .build_call(
                            add_entity_fn,
                            &[name_ptr.as_pointer_value().into()],
                            "ent_id",
                        )
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    let val = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err(CodegenErrorKind::Internal("Invalid return from add_entity".to_string()).into()),
                    };
                    // Bound string constant — set mask bit
                    mask |= 1 << i;
                    compiled_args.push(val);
                    tags.push(4); // ConstantTag::String
                }
                _ => {
                    let (val, ty) = self.compile_expr(arg)?;
                     // Force cast to i64
                    let i64_type = self.context.i64_type();
                    let val_i64 = if val.is_int_value() {
                        self.builder.build_int_cast(val.into_int_value(), i64_type, "arg_cast").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into()
                    } else {
                        // If float, cast to int? Or bitcast?
                        // For now assume logic args are ints or entities (i64)
                        // If boolean?
                         if val.is_int_value() { // Bool is int(1)
                             self.builder.build_int_cast(val.into_int_value(), i64_type, "bool_cast").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into()
                         } else {
                             val // Cannot cast pointer to int easily here without ptrtoint
                         }
                    };
                    // Bound expression — set mask bit
                    mask |= 1 << i;
                    compiled_args.push(val_i64);
                    let tag = match ty {
                        Type::I64 | Type::I32 | Type::I8 | Type::U8 | Type::U16 | Type::U32 | Type::Usize => 0, // Int
                        Type::F32 | Type::F64 | Type::F16 | Type::BF16 => 1,             // Float
                        Type::Bool => 2,                        // Bool
                        Type::Entity => 3,                      // Entity
                        _ => 0,
                    };
                    tags.push(tag);
                }
            }
        }

        // Generate tags pointer
        let tags_ptr = if tags.is_empty() {
            self.context.ptr_type(inkwell::AddressSpace::default()).const_null()
        } else {
            let tags_arr_type = i8_type.array_type(tags.len() as u32);
            let tags_alloca = self.builder.build_alloca(tags_arr_type, "query_tags").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            for (i, &tag) in tags.iter().enumerate() {
                let ptr = unsafe {
                    self.builder
                        .build_gep(
                            tags_arr_type,
                            tags_alloca,
                            &[
                                i64_type.const_int(0, false),
                                i64_type.const_int(i as u64, false),
                            ],
                            "",
                        )
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                };
                self.builder
                    .build_store(ptr, i8_type.const_int(tag as u64, false))
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            }
            unsafe {
                self.builder
                    .build_gep(
                        tags_arr_type,
                        tags_alloca,
                        &[i64_type.const_int(0, false), i64_type.const_int(0, false)],
                        "tags_decayed",
                    )
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
            }
        };

        // Insert Mask at beginning
        let mask_val = i64_type.const_int(mask as u64, false);
        compiled_args.insert(0, mask_val.into());
        // Append tags pointer at end (as i64)
        // let null_tags = i8_type.ptr_type(inkwell::AddressSpace::default()).const_null();
        let tags_int = self.builder.build_ptr_to_int(tags_ptr, i64_type, "tags_int").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        compiled_args.push(tags_int.into());
        // compiled_args.push(null_tags.into()); // FORCE NULL TAGS
        // compiled_args.push(tags_ptr.into());

        let final_args: Vec<inkwell::values::BasicMetadataValueEnum> =
            compiled_args.iter().map(|&val| val.into()).collect();

        let call = self
            .builder
            .build_call(function, &final_args, "query_res")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        // Debug signature mismatch
        

        let res = match call.try_as_basic_value() {
            inkwell::values::ValueKind::Basic(v) => v,
            _ => return Err(CodegenErrorKind::Internal("Relation wrapper returned void".to_string()).into()),
        };

        Ok((res, Type::Tensor(Box::new(Type::F32), 1)))
    }

    // Helper to derive return type from LLVM Function Signature
    pub(crate) fn get_return_type_from_signature(&self, func: inkwell::values::FunctionValue<'ctx>) -> Type {
        let name = func.get_name().to_str().unwrap_or("");

        // 1. Check if we have an explicit mapping for this function
        if let Some(ty) = self.method_return_types.get(name) {
             return ty.clone();
        }

        let ret = func.get_type().get_return_type();
        match ret {
            None => Type::Void,
            Some(inkwell::types::BasicTypeEnum::IntType(i)) => {
                let width = i.get_bit_width();
                if width == 1 {
                    Type::Bool
                } else if width == 64 {
                    Type::I64
                } else {
                    Type::I32 // Fallback
                }
            }
            Some(inkwell::types::BasicTypeEnum::FloatType(_f)) => {
                 Type::F32 
            }
            Some(inkwell::types::BasicTypeEnum::PointerType(_)) => {
                // method_return_types (add_fn_typed 経由) で登録済みのはず。
                // ここに到達するのは未登録のFFI関数 → tensor系のデフォルトか、登録漏れバグ。
                if name.contains("alloc") || name.contains("init") {
                    Type::I64
                } else if name.contains("tensor") {
                    Type::Tensor(Box::new(Type::F32), 0)
                } else {
                    // ポインタを返すFFI関数が method_return_types に未登録 → 登録漏れの可能性
                    // builtins.rs の add_fn_typed で登録すること
                    debug_assert!(
                        false,
                        "FFI function '{}' returns a pointer but has no entry in method_return_types. \
                         Use add_fn_typed in builtins.rs to register both LLVM declaration and return type.",
                        name
                    );
                    Type::I64
                }
            }
            _ => Type::Void, 
        }
    }

    /// Compile Vec closure methods (map, filter, any, all) via inline expansion.
    /// Instead of calling a function pointer, we expand the closure body inline in a loop.
    fn compile_vec_closure_method(
        &mut self,
        vec_val: BasicValueEnum<'ctx>,
        vec_ty: &Type,
        elem_ty: &Type,
        method: &str,
        closure_args: &[(String, Option<Type>)],
        closure_body: &[Stmt],
    ) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
        let i64_type = self.context.i64_type();
        let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());

        // Get Vec struct type for GEP access — must look up from struct_types registry,
        // not get_llvm_type() which returns a pointer.
        let vec_type_name = vec_ty.codegen_name()
            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("Cannot get Vec codegen name".to_string())))?;
        let vec_struct_ty = *self.struct_types.get(&vec_type_name)
            .or_else(|| self.struct_types.get(&vec_ty.get_base_name()))
            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Vec struct type {} not found in struct_types", vec_type_name))))?;
        let vec_ptr = vec_val.into_pointer_value();

        // Load Vec fields: ptr, len
        let data_ptr_field = self.builder.build_struct_gep(
            vec_struct_ty, vec_ptr, 0, "data_ptr_field",
        ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        let data_ptr = self.builder.build_load(ptr_type, data_ptr_field, "data_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        let len_field = self.builder.build_struct_gep(
            vec_struct_ty, vec_ptr, 2, "len_field",
        ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        let len = self.builder.build_load(i64_type, len_field, "len").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value();

        let elem_llvm_ty = self.get_llvm_type(elem_ty)?;

        let current_fn = self.current_function()?;
        let zero = i64_type.const_zero();

        match method {
            "map" => {
                // map(|x| expr) -> Vec<U> where U is the return type of the closure
                // Allocate result Vec directly: { ptr, cap, len }
                let result_ptr = self.builder.build_alloca(vec_struct_ty, "result_vec").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // Allocate data buffer: malloc(len * sizeof(elem))
                let elem_size_val = elem_llvm_ty.size_of().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("type has no size".to_string())))?;
                let buf_size = self.builder.build_int_mul(len, elem_size_val, "buf_size").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let malloc_fn = self.module.get_function("malloc")
                    .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("malloc not found".to_string())))?;
                let raw_buf = self.builder.build_call(malloc_fn, &[buf_size.into()], "raw_buf")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let buf_ptr = match raw_buf.try_as_basic_value() {
                    inkwell::values::ValueKind::Basic(v) => v,
                    _ => return Err(CodegenErrorKind::Internal("malloc returned no value".to_string()).into()),
                };

                // Set fields: ptr, cap, len
                let f0 = self.builder.build_struct_gep(vec_struct_ty, result_ptr, 0, "f0").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(f0, buf_ptr).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let f1 = self.builder.build_struct_gep(vec_struct_ty, result_ptr, 1, "f1").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(f1, len).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?; // cap = len
                let f2 = self.builder.build_struct_gep(vec_struct_ty, result_ptr, 2, "f2").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(f2, zero).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?; // len = 0 (will be set at end)

                let result_data = buf_ptr;

                // Loop: for i in 0..len
                let loop_bb = self.context.append_basic_block(current_fn, "map_loop");
                let body_bb = self.context.append_basic_block(current_fn, "map_body");
                let done_bb = self.context.append_basic_block(current_fn, "map_done");

                // i = 0
                let i_alloca = self.builder.build_alloca(i64_type, "i").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(i_alloca, zero).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_unconditional_branch(loop_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // Loop check: i < len
                self.builder.position_at_end(loop_bb);
                let i_val = self.builder.build_load(i64_type, i_alloca, "i").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value();
                let cond = self.builder.build_int_compare(
                    inkwell::IntPredicate::SLT, i_val, len, "cond",
                ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_conditional_branch(cond, body_bb, done_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // Body: load element, apply closure, store result
                self.builder.position_at_end(body_bb);
                let elem_addr = unsafe {
                    self.builder.build_gep(elem_llvm_ty, data_ptr.into_pointer_value(), &[i_val], "elem_addr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                };
                let elem_val = self.builder.build_load(elem_llvm_ty, elem_addr, "elem").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // Bind closure arg to element value
                self.enter_scope();
                let arg_name = closure_args.first().map(|(n, _)| n.as_str()).unwrap_or("x");
                let arg_alloca = self.builder.build_alloca(elem_llvm_ty, arg_name).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(arg_alloca, elem_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.variables.last_mut().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("no variable scope".to_string())))?.insert(
                    arg_name.to_string(),
                    (arg_alloca.into(), elem_ty.clone(), crate::compiler::codegen::CLEANUP_NONE),
                );

                // Compile closure body
                let mut result_val = None;
                let body_len = closure_body.len();
                for (idx, stmt) in closure_body.iter().enumerate() {
                    if idx == body_len - 1 {
                        if let crate::compiler::ast::StmtKind::Expr(e) = &stmt.inner {
                            result_val = Some(self.compile_expr(e)?);
                        } else {
                            self.compile_stmt(stmt)?;
                        }
                    } else {
                        self.compile_stmt(stmt)?;
                    }
                }
                self.exit_scope();

                // Store result in output vec
                let (mapped_val, mapped_ty) = result_val.unwrap_or((zero.into(), Type::I64));
                let mapped_llvm_ty = self.get_llvm_type(&mapped_ty)?;
                let result_elem_addr = unsafe {
                    self.builder.build_gep(mapped_llvm_ty, result_data.into_pointer_value(), &[i_val], "result_addr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                };
                self.builder.build_store(result_elem_addr, mapped_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // i += 1
                let next_i = self.builder.build_int_add(i_val, i64_type.const_int(1, false), "next_i").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(i_alloca, next_i).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_unconditional_branch(loop_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // Done: set result len
                self.builder.position_at_end(done_bb);
                let result_len_field = self.builder.build_struct_gep(
                    vec_struct_ty, result_ptr, 2, "result_len_field",
                ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(result_len_field, len).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // Return type: Vec<mapped_ty> (use same vec type for now)
                Ok((result_ptr.into(), vec_ty.clone()))
            }
            "filter" => {
                // filter(|x| bool_expr) -> Vec<T>
                // Allocate result Vec directly
                let result_ptr = self.builder.build_alloca(vec_struct_ty, "result_vec").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                let elem_size_val = elem_llvm_ty.size_of().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("type has no size".to_string())))?;
                let buf_size = self.builder.build_int_mul(len, elem_size_val, "buf_size").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let malloc_fn = self.module.get_function("malloc")
                    .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("malloc not found".to_string())))?;
                let raw_buf = self.builder.build_call(malloc_fn, &[buf_size.into()], "raw_buf")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let buf_ptr = match raw_buf.try_as_basic_value() {
                    inkwell::values::ValueKind::Basic(v) => v,
                    _ => return Err(CodegenErrorKind::Internal("malloc returned no value".to_string()).into()),
                };

                let f0 = self.builder.build_struct_gep(vec_struct_ty, result_ptr, 0, "f0").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(f0, buf_ptr).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let f1 = self.builder.build_struct_gep(vec_struct_ty, result_ptr, 1, "f1").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(f1, len).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let f2 = self.builder.build_struct_gep(vec_struct_ty, result_ptr, 2, "f2").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(f2, zero).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                let result_data = buf_ptr;

                // Count for result length
                let count_alloca = self.builder.build_alloca(i64_type, "count").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(count_alloca, zero).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                let loop_bb = self.context.append_basic_block(current_fn, "filter_loop");
                let body_bb = self.context.append_basic_block(current_fn, "filter_body");
                let store_bb = self.context.append_basic_block(current_fn, "filter_store");
                let skip_bb = self.context.append_basic_block(current_fn, "filter_skip");
                let done_bb = self.context.append_basic_block(current_fn, "filter_done");

                let i_alloca = self.builder.build_alloca(i64_type, "i").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(i_alloca, zero).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_unconditional_branch(loop_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                self.builder.position_at_end(loop_bb);
                let i_val = self.builder.build_load(i64_type, i_alloca, "i").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value();
                let cond = self.builder.build_int_compare(
                    inkwell::IntPredicate::SLT, i_val, len, "cond",
                ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_conditional_branch(cond, body_bb, done_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                self.builder.position_at_end(body_bb);
                let elem_addr = unsafe {
                    self.builder.build_gep(elem_llvm_ty, data_ptr.into_pointer_value(), &[i_val], "elem_addr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                };
                let elem_val = self.builder.build_load(elem_llvm_ty, elem_addr, "elem").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // Bind closure arg
                self.enter_scope();
                let arg_name = closure_args.first().map(|(n, _)| n.as_str()).unwrap_or("x");
                let arg_alloca = self.builder.build_alloca(elem_llvm_ty, arg_name).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(arg_alloca, elem_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.variables.last_mut().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("no variable scope".to_string())))?.insert(
                    arg_name.to_string(),
                    (arg_alloca.into(), elem_ty.clone(), crate::compiler::codegen::CLEANUP_NONE),
                );

                let mut predicate_val = None;
                let body_len = closure_body.len();
                for (idx, stmt) in closure_body.iter().enumerate() {
                    if idx == body_len - 1 {
                        if let crate::compiler::ast::StmtKind::Expr(e) = &stmt.inner {
                            predicate_val = Some(self.compile_expr(e)?);
                        }
                    } else {
                        self.compile_stmt(stmt)?;
                    }
                }
                self.exit_scope();

                let pred = predicate_val.map(|(v, _)| v.into_int_value())
                    .unwrap_or(self.context.bool_type().const_zero());
                self.builder.build_conditional_branch(pred, store_bb, skip_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // Store element in result
                self.builder.position_at_end(store_bb);
                let count_val = self.builder.build_load(i64_type, count_alloca, "count").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value();
                let result_elem_addr = unsafe {
                    self.builder.build_gep(elem_llvm_ty, result_data.into_pointer_value(), &[count_val], "result_addr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                };
                self.builder.build_store(result_elem_addr, elem_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let next_count = self.builder.build_int_add(count_val, i64_type.const_int(1, false), "next_count").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(count_alloca, next_count).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_unconditional_branch(skip_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // Skip / increment
                self.builder.position_at_end(skip_bb);
                let i_val2 = self.builder.build_load(i64_type, i_alloca, "i2").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value();
                let next_i = self.builder.build_int_add(i_val2, i64_type.const_int(1, false), "next_i").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(i_alloca, next_i).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_unconditional_branch(loop_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // Done: set result len
                self.builder.position_at_end(done_bb);
                let final_count = self.builder.build_load(i64_type, count_alloca, "final_count").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let result_len_field = self.builder.build_struct_gep(
                    vec_struct_ty, result_ptr, 2, "result_len_field",
                ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(result_len_field, final_count).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                Ok((result_ptr.into(), vec_ty.clone()))
            }
            "any" | "all" => {
                // any(|x| bool_expr) -> bool / all(|x| bool_expr) -> bool
                let is_any = method == "any";
                let result_alloca = self.builder.build_alloca(self.context.bool_type(), "result").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let init_val = if is_any {
                    self.context.bool_type().const_zero() // any starts false
                } else {
                    self.context.bool_type().const_all_ones() // all starts true
                };
                self.builder.build_store(result_alloca, init_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                let loop_bb = self.context.append_basic_block(current_fn, "anyall_loop");
                let body_bb = self.context.append_basic_block(current_fn, "anyall_body");
                let early_bb = self.context.append_basic_block(current_fn, "anyall_early");
                let cont_bb = self.context.append_basic_block(current_fn, "anyall_cont");
                let done_bb = self.context.append_basic_block(current_fn, "anyall_done");

                let i_alloca = self.builder.build_alloca(i64_type, "i").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(i_alloca, zero).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_unconditional_branch(loop_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                self.builder.position_at_end(loop_bb);
                let i_val = self.builder.build_load(i64_type, i_alloca, "i").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value();
                let cond = self.builder.build_int_compare(
                    inkwell::IntPredicate::SLT, i_val, len, "cond",
                ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_conditional_branch(cond, body_bb, done_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                self.builder.position_at_end(body_bb);
                let elem_addr = unsafe {
                    self.builder.build_gep(elem_llvm_ty, data_ptr.into_pointer_value(), &[i_val], "elem_addr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                };
                let elem_val = self.builder.build_load(elem_llvm_ty, elem_addr, "elem").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                self.enter_scope();
                let arg_name = closure_args.first().map(|(n, _)| n.as_str()).unwrap_or("x");
                let arg_alloca = self.builder.build_alloca(elem_llvm_ty, arg_name).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(arg_alloca, elem_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.variables.last_mut().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("no variable scope".to_string())))?.insert(
                    arg_name.to_string(),
                    (arg_alloca.into(), elem_ty.clone(), crate::compiler::codegen::CLEANUP_NONE),
                );

                let mut pred_val = None;
                let body_len = closure_body.len();
                for (idx, stmt) in closure_body.iter().enumerate() {
                    if idx == body_len - 1 {
                        if let crate::compiler::ast::StmtKind::Expr(e) = &stmt.inner {
                            pred_val = Some(self.compile_expr(e)?);
                        }
                    } else {
                        self.compile_stmt(stmt)?;
                    }
                }
                self.exit_scope();

                let pred = pred_val.map(|(v, _)| v.into_int_value())
                    .unwrap_or(self.context.bool_type().const_zero());

                if is_any {
                    // any: if true, set result=true and break
                    self.builder.build_conditional_branch(pred, early_bb, cont_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                } else {
                    // all: if false, set result=false and break
                    let not_pred = self.builder.build_not(pred, "not_pred").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    self.builder.build_conditional_branch(not_pred, early_bb, cont_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                }

                // Early exit
                self.builder.position_at_end(early_bb);
                let early_val = if is_any {
                    self.context.bool_type().const_all_ones()
                } else {
                    self.context.bool_type().const_zero()
                };
                self.builder.build_store(result_alloca, early_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_unconditional_branch(done_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // Continue loop
                self.builder.position_at_end(cont_bb);
                let next_i = self.builder.build_int_add(i_val, i64_type.const_int(1, false), "next_i").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(i_alloca, next_i).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_unconditional_branch(loop_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // Done
                self.builder.position_at_end(done_bb);
                let result = self.builder.build_load(self.context.bool_type(), result_alloca, "result").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                Ok((result, Type::Bool))
            }
            "reduce" => {
                // reduce(|acc, x| expr) -> T
                // acc starts with vec[0], loop from i=1 to len
                let acc_alloca = self.builder.build_alloca(elem_llvm_ty, "acc").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // Handle empty vec: return zero
                let has_elements = self.builder.build_int_compare(
                    inkwell::IntPredicate::SGT, len, zero, "has_elements",
                ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                let init_bb = self.context.append_basic_block(current_fn, "reduce_init");
                let loop_bb = self.context.append_basic_block(current_fn, "reduce_loop");
                let body_bb = self.context.append_basic_block(current_fn, "reduce_body");
                let done_bb = self.context.append_basic_block(current_fn, "reduce_done");

                self.builder.build_conditional_branch(has_elements, init_bb, done_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // Init: acc = vec[0]
                self.builder.position_at_end(init_bb);
                let first_addr = unsafe {
                    self.builder.build_gep(elem_llvm_ty, data_ptr.into_pointer_value(), &[zero], "first_addr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                };
                let first_val = self.builder.build_load(elem_llvm_ty, first_addr, "first").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(acc_alloca, first_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // i = 1
                let i_alloca = self.builder.build_alloca(i64_type, "i").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(i_alloca, i64_type.const_int(1, false)).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_unconditional_branch(loop_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // Loop check: i < len
                self.builder.position_at_end(loop_bb);
                let i_val = self.builder.build_load(i64_type, i_alloca, "i").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value();
                let cond = self.builder.build_int_compare(
                    inkwell::IntPredicate::SLT, i_val, len, "cond",
                ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_conditional_branch(cond, body_bb, done_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // Body: acc = closure(acc, elem)
                self.builder.position_at_end(body_bb);
                let elem_addr = unsafe {
                    self.builder.build_gep(elem_llvm_ty, data_ptr.into_pointer_value(), &[i_val], "elem_addr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                };
                let elem_val = self.builder.build_load(elem_llvm_ty, elem_addr, "elem").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let acc_val = self.builder.build_load(elem_llvm_ty, acc_alloca, "acc_val").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // Bind closure args (acc, x)
                self.enter_scope();
                let acc_name = closure_args.first().map(|(n, _)| n.as_str()).unwrap_or("acc");
                let x_name = closure_args.get(1).map(|(n, _)| n.as_str()).unwrap_or("x");

                let acc_arg = self.builder.build_alloca(elem_llvm_ty, acc_name).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(acc_arg, acc_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.variables.last_mut().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("no variable scope".to_string())))?.insert(
                    acc_name.to_string(),
                    (acc_arg.into(), elem_ty.clone(), crate::compiler::codegen::CLEANUP_NONE),
                );

                let x_arg = self.builder.build_alloca(elem_llvm_ty, x_name).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(x_arg, elem_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.variables.last_mut().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("no variable scope".to_string())))?.insert(
                    x_name.to_string(),
                    (x_arg.into(), elem_ty.clone(), crate::compiler::codegen::CLEANUP_NONE),
                );

                // Compile closure body
                let mut result_val = None;
                let body_len = closure_body.len();
                for (idx, stmt) in closure_body.iter().enumerate() {
                    if idx == body_len - 1 {
                        if let crate::compiler::ast::StmtKind::Expr(e) = &stmt.inner {
                            result_val = Some(self.compile_expr(e)?);
                        } else {
                            self.compile_stmt(stmt)?;
                        }
                    } else {
                        self.compile_stmt(stmt)?;
                    }
                }
                self.exit_scope();

                let (new_acc, _) = result_val.unwrap_or((elem_llvm_ty.const_zero().into(), elem_ty.clone()));
                self.builder.build_store(acc_alloca, new_acc).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // i += 1
                let next_i = self.builder.build_int_add(i_val, i64_type.const_int(1, false), "next_i").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(i_alloca, next_i).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_unconditional_branch(loop_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // Done
                self.builder.position_at_end(done_bb);
                let final_acc = self.builder.build_load(elem_llvm_ty, acc_alloca, "final_acc").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                Ok((final_acc, elem_ty.clone()))
            }
            _ => Err(TlError::from(CodegenErrorKind::Internal(format!("Unknown Vec closure method: {}", method)))),
        }
    }

    /// Vec.join(sep: String) -> String のインラインIR生成
    fn compile_vec_join(
        &mut self,
        vec_val: BasicValueEnum<'ctx>,
        vec_ty: &Type,
        args: &[Expr],
    ) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
        let i64_type = self.context.i64_type();
        let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
        let string_type_tl = Type::String("String".to_string());

        // Compile separator argument
        let (sep_val, _sep_ty) = self.compile_expr(&args[0])?;

        // Get Vec struct layout
        let vec_type_name = vec_ty.codegen_name()
            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("Cannot get Vec codegen name".to_string())))?;
        let vec_struct_ty = *self.struct_types.get(&vec_type_name)
            .or_else(|| self.struct_types.get(&vec_ty.get_base_name()))
            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Vec struct type {} not found", vec_type_name))))?;
        let vec_ptr = vec_val.into_pointer_value();

        // Load ptr, len
        let data_ptr_field = self.builder.build_struct_gep(vec_struct_ty, vec_ptr, 0, "data_ptr_field")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        let data_ptr = self.builder.build_load(ptr_type, data_ptr_field, "data_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        let len_field = self.builder.build_struct_gep(vec_struct_ty, vec_ptr, 2, "len_field")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        let len = self.builder.build_load(i64_type, len_field, "len").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value();

        // Determine element type from Vec type params
        let elem_ty = match vec_ty {
            Type::Struct(_name, params) if !params.is_empty() => params[0].clone(),
            Type::SpecializedType { type_args, .. } if !type_args.is_empty() => type_args[0].clone(),
            Type::Struct(_name, _) => Type::I64,
            _ => Type::I64,
        };
        let elem_llvm_ty = self.get_llvm_type(&elem_ty)?;

        // Determine the conversion function name based on element type
        let to_string_fn_name: Option<&str> = match &elem_ty {
            Type::I64 | Type::I32 => Some("tl_string_from_int"),
            Type::F64 | Type::F32 => Some("tl_string_from_f64"),
            Type::Bool => Some("tl_string_from_bool"),
            Type::String(_) => None, // already a string
            Type::Struct(name, _) if name == "String" => None, // String as Struct variant
            _ => Some("tl_string_from_int"), // fallback
        };

        let concat_fn = self.module.get_function("tl_string_concat")
            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_string_concat not found".to_string())))?;

        // Create empty string as initial result
        let empty_str_fn = self.module.get_function("tl_string_new")
            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_string_new not found".to_string())))?;
        let empty_c_str = self.builder.build_global_string_ptr("", "empty_str").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        let init_str = self.builder.build_call(empty_str_fn, &[empty_c_str.as_pointer_value().into()], "init_str")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        let init_str_val = match init_str.try_as_basic_value() {
            inkwell::values::ValueKind::Basic(v) => v,
            _ => return Err(CodegenErrorKind::Internal("tl_string_new returned void".to_string()).into()),
        };

        let result_alloca = self.builder.build_alloca(ptr_type, "join_result").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        self.builder.build_store(result_alloca, init_str_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        let current_fn = self.current_function()?;
        let zero = i64_type.const_zero();

        let loop_bb = self.context.append_basic_block(current_fn, "join_loop");
        let body_bb = self.context.append_basic_block(current_fn, "join_body");
        let done_bb = self.context.append_basic_block(current_fn, "join_done");

        let i_alloca = self.builder.build_alloca(i64_type, "i").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        self.builder.build_store(i_alloca, zero).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        self.builder.build_unconditional_branch(loop_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        // Loop check
        self.builder.position_at_end(loop_bb);
        let i_val = self.builder.build_load(i64_type, i_alloca, "i").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value();
        let cond = self.builder.build_int_compare(
            inkwell::IntPredicate::SLT, i_val, len, "cond",
        ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        self.builder.build_conditional_branch(cond, body_bb, done_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        // Body
        self.builder.position_at_end(body_bb);

        // If i > 0, append separator first
        let is_not_first = self.builder.build_int_compare(
            inkwell::IntPredicate::SGT, i_val, zero, "is_not_first",
        ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        let sep_bb = self.context.append_basic_block(current_fn, "join_sep");
        let elem_bb = self.context.append_basic_block(current_fn, "join_elem");

        self.builder.build_conditional_branch(is_not_first, sep_bb, elem_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        // Append separator
        self.builder.position_at_end(sep_bb);
        let cur_str = self.builder.build_load(ptr_type, result_alloca, "cur_str").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        let with_sep = self.builder.build_call(concat_fn, &[cur_str.into(), sep_val.into()], "with_sep")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        let with_sep_val = match with_sep.try_as_basic_value() {
            inkwell::values::ValueKind::Basic(v) => v,
            _ => return Err(CodegenErrorKind::Internal("tl_string_concat returned void".to_string()).into()),
        };
        self.builder.build_store(result_alloca, with_sep_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        self.builder.build_unconditional_branch(elem_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        // Append element
        self.builder.position_at_end(elem_bb);
        let elem_addr = unsafe {
            self.builder.build_gep(elem_llvm_ty, data_ptr.into_pointer_value(), &[i_val], "elem_addr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
        };
        let elem_val = self.builder.build_load(elem_llvm_ty, elem_addr, "elem").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        // Convert element to string if needed
        let elem_str = if let Some(fn_name) = to_string_fn_name {
            let conv_fn = self.module.get_function(fn_name)
                .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("{} not found", fn_name))))?;
            let conv_val = match &elem_ty {
                Type::I32 => {
                    let ext = self.builder.build_int_s_extend(elem_val.into_int_value(), i64_type, "ext").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    ext.into()
                }
                Type::F32 => {
                    let ext = self.builder.build_float_ext(elem_val.into_float_value(), self.context.f64_type(), "ext").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    ext.into()
                }
                _ => elem_val,
            };
            let str_result = self.builder.build_call(conv_fn, &[conv_val.into()], "elem_str")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            match str_result.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v,
                _ => return Err(TlError::from(CodegenErrorKind::Internal(format!("{} returned void", fn_name)))),
            }
        } else {
            elem_val
        };

        let cur_str2 = self.builder.build_load(ptr_type, result_alloca, "cur_str2").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        let appended = self.builder.build_call(concat_fn, &[cur_str2.into(), elem_str.into()], "appended")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        let appended_val = match appended.try_as_basic_value() {
            inkwell::values::ValueKind::Basic(v) => v,
            _ => return Err(CodegenErrorKind::Internal("tl_string_concat returned void".to_string()).into()),
        };
        self.builder.build_store(result_alloca, appended_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        // i += 1
        let next_i = self.builder.build_int_add(i_val, i64_type.const_int(1, false), "next_i").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        self.builder.build_store(i_alloca, next_i).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        self.builder.build_unconditional_branch(loop_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        // Done
        self.builder.position_at_end(done_bb);
        let final_result = self.builder.build_load(ptr_type, result_alloca, "join_result").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        Ok((final_result, string_type_tl))
    }

    fn compile_vec_builtin_method(
        &mut self,
        vec_val: BasicValueEnum<'ctx>,
        vec_ty: &Type,
        method: &str,
        args: &[Expr],
    ) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
        let i64_type = self.context.i64_type();
        let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());

        let vec_type_name = vec_ty.codegen_name()
            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("Cannot get Vec codegen name".to_string())))?;
        let vec_struct_ty = *self.struct_types.get(&vec_type_name)
            .or_else(|| self.struct_types.get(&vec_ty.get_base_name()))
            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Vec struct type {} not found", vec_type_name))))?;
        let vec_ptr = vec_val.into_pointer_value();

        let data_ptr_field = self.builder.build_struct_gep(vec_struct_ty, vec_ptr, 0, "d_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        let data_ptr = self.builder.build_load(ptr_type, data_ptr_field, "d").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        let len_field = self.builder.build_struct_gep(vec_struct_ty, vec_ptr, 2, "l_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        let len = self.builder.build_load(i64_type, len_field, "l").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value();

        let elem_ty = match vec_ty {
            Type::Struct(_, params) if !params.is_empty() => params[0].clone(),
            _ => Type::I64,
        };
        let elem_llvm_ty = self.get_llvm_type(&elem_ty)?;

        let current_fn = self.current_function()?;
        let zero = i64_type.const_zero();

        if method == "enumerate" {
            let out_elem_ty = Type::Tuple(vec![Type::I64, elem_ty.clone()]);
            let out_elem_llvm_ty = self.get_llvm_type(&out_elem_ty)?;
            let out_vec_ty = Type::Struct("Vec".to_string(), vec![out_elem_ty.clone()]);
            let _ = self.get_or_monomorphize_type(&out_vec_ty)?;

            let res_ptr = self.builder.build_alloca(vec_struct_ty, "res").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            let malloc_fn = self.get_fn("malloc")?;
            let buf_size = self.builder.build_int_mul(len, out_elem_llvm_ty.size_of().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("type has no size".to_string())))?, "bsz").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let raw_buf_call = self.builder.build_call(malloc_fn, &[buf_size.into()], "buf").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let raw_buf = match raw_buf_call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v,
                _ => return Err(CodegenErrorKind::Internal("malloc failed".to_string()).into()),
            };

            let f0 = self.builder.build_struct_gep(vec_struct_ty, res_ptr, 0, "f0").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_store(f0, raw_buf).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let f1 = self.builder.build_struct_gep(vec_struct_ty, res_ptr, 1, "f1").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_store(f1, len).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let f2 = self.builder.build_struct_gep(vec_struct_ty, res_ptr, 2, "f2").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_store(f2, len).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            let loop_bb = self.context.append_basic_block(current_fn, "lp");
            let body_bb = self.context.append_basic_block(current_fn, "bd");
            let done_bb = self.context.append_basic_block(current_fn, "dn");

            let i_var = self.builder.build_alloca(i64_type, "i").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_store(i_var, zero).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_unconditional_branch(loop_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            self.builder.position_at_end(loop_bb);
            let i_val = self.builder.build_load(i64_type, i_var, "ival").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value();
            let cond = self.builder.build_int_compare(inkwell::IntPredicate::SLT, i_val, len, "cnd").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_conditional_branch(cond, body_bb, done_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            self.builder.position_at_end(body_bb);
            let in_addr = unsafe { self.builder.build_gep(elem_llvm_ty, data_ptr.into_pointer_value(), &[i_val], "ia").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))? };
            let in_val = self.builder.build_load(elem_llvm_ty, in_addr, "iv").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            // Malloc tuple
            let tuple_struct_ty = self.context.struct_type(&[i64_type.into(), elem_llvm_ty.into()], false);
            let tuple_sz = self.builder.build_int_z_extend(tuple_struct_ty.size_of().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("type has no size".to_string())))?, i64_type, "tsz").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let tuple_call = self.builder.build_call(malloc_fn, &[tuple_sz.into()], "mtup").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let tuple_ptr = match tuple_call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
                _ => return Err(CodegenErrorKind::Internal("malloc failed".to_string()).into()),
            };

            let t0 = self.builder.build_struct_gep(tuple_struct_ty, tuple_ptr, 0, "t0").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_store(t0, i_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let t1 = self.builder.build_struct_gep(tuple_struct_ty, tuple_ptr, 1, "t1").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_store(t1, in_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            let out_addr = unsafe { self.builder.build_gep(out_elem_llvm_ty, raw_buf.into_pointer_value(), &[i_val], "oa").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))? };
            self.builder.build_store(out_addr, tuple_ptr).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            self.emit_retain(in_val, &elem_ty)?;

            let nxt = self.builder.build_int_add(i_val, i64_type.const_int(1, false), "nxt").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_store(i_var, nxt).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_unconditional_branch(loop_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            self.builder.position_at_end(done_bb);
            return Ok((res_ptr.into(), out_vec_ty));
        }

        if method == "zip" {
            let (other_val, other_ty) = self.compile_expr(&args[0])?;
            let other_elem_ty = match &other_ty {
                Type::Struct(_, p) if !p.is_empty() => p[0].clone(),
                _ => Type::I64,
            };
            let other_elem_llvm_ty = self.get_llvm_type(&other_elem_ty)?;
            let other_vec_name = other_ty.codegen_name().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("unexpected None".to_string())))?;
            let other_vec_struct_ty = *self.struct_types.get(&other_vec_name).ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("struct type not found"))))?;
            
            let o_ptr = other_val.into_pointer_value();
            let o_data_field = self.builder.build_struct_gep(other_vec_struct_ty, o_ptr, 0, "od").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let o_data = self.builder.build_load(ptr_type, o_data_field, "odt").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let o_len_field = self.builder.build_struct_gep(other_vec_struct_ty, o_ptr, 2, "ol").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let o_len = self.builder.build_load(i64_type, o_len_field, "oln").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value();

            let out_elem_ty = Type::Tuple(vec![elem_ty.clone(), other_elem_ty.clone()]);
            let out_elem_llvm_ty = self.get_llvm_type(&out_elem_ty)?;
            let out_vec_ty = Type::Struct("Vec".to_string(), vec![out_elem_ty.clone()]);
            let _ = self.get_or_monomorphize_type(&out_vec_ty)?;

            let min_len = self.builder.build_select(
                self.builder.build_int_compare(inkwell::IntPredicate::SLT, len, o_len, "cmp").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?,
                len, o_len, "min"
            ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value();

            let res_ptr = self.builder.build_alloca(vec_struct_ty, "res").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let malloc_fn = self.get_fn("malloc")?;
            let buf_sz = self.builder.build_int_mul(min_len, out_elem_llvm_ty.size_of().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("type has no size".to_string())))?, "bsz").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let raw_buf_call = self.builder.build_call(malloc_fn, &[buf_sz.into()], "buf").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let raw_buf = match raw_buf_call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v,
                _ => return Err(CodegenErrorKind::Internal("malloc failed".to_string()).into()),
            };

            let f0 = self.builder.build_struct_gep(vec_struct_ty, res_ptr, 0, "f0").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_store(f0, raw_buf).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let f1 = self.builder.build_struct_gep(vec_struct_ty, res_ptr, 1, "f1").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_store(f1, min_len).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let f2 = self.builder.build_struct_gep(vec_struct_ty, res_ptr, 2, "f2").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_store(f2, min_len).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            let loop_bb = self.context.append_basic_block(current_fn, "lp");
            let body_bb = self.context.append_basic_block(current_fn, "bd");
            let done_bb = self.context.append_basic_block(current_fn, "dn");

            let i_var = self.builder.build_alloca(i64_type, "i").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_store(i_var, zero).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_unconditional_branch(loop_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            self.builder.position_at_end(loop_bb);
            let i_val = self.builder.build_load(i64_type, i_var, "iv").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value();
            let cond = self.builder.build_int_compare(inkwell::IntPredicate::SLT, i_val, min_len, "cnd").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_conditional_branch(cond, body_bb, done_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            self.builder.position_at_end(body_bb);
            let in1 = self.builder.build_load(elem_llvm_ty, unsafe { self.builder.build_gep(elem_llvm_ty, data_ptr.into_pointer_value(), &[i_val], "ia1").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))? }, "v1").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let in2 = self.builder.build_load(other_elem_llvm_ty, unsafe { self.builder.build_gep(other_elem_llvm_ty, o_data.into_pointer_value(), &[i_val], "ia2").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))? }, "v2").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            // Malloc tuple
            let tuple_struct_ty = self.context.struct_type(&[elem_llvm_ty.into(), other_elem_llvm_ty.into()], false);
            let tuple_sz = self.builder.build_int_z_extend(tuple_struct_ty.size_of().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("type has no size".to_string())))?, i64_type, "tsz").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let tuple_call = self.builder.build_call(malloc_fn, &[tuple_sz.into()], "mtup").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let tuple_ptr = match tuple_call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
                _ => return Err(CodegenErrorKind::Internal("malloc failed".to_string()).into()),
            };

            let t0 = self.builder.build_struct_gep(tuple_struct_ty, tuple_ptr, 0, "t0").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_store(t0, in1).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let t1 = self.builder.build_struct_gep(tuple_struct_ty, tuple_ptr, 1, "t1").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_store(t1, in2).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            let out_addr = unsafe { self.builder.build_gep(out_elem_llvm_ty, raw_buf.into_pointer_value(), &[i_val], "oa").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))? };
            self.builder.build_store(out_addr, tuple_ptr).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            self.emit_retain(in1, &elem_ty)?;
            self.emit_retain(in2, &other_elem_ty)?;

            let nxt = self.builder.build_int_add(i_val, i64_type.const_int(1, false), "nxt").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_store(i_var, nxt).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_unconditional_branch(loop_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            self.builder.position_at_end(done_bb);
            return Ok((res_ptr.into(), out_vec_ty));
        }

        if method == "flatten" {
            let inner_elem_ty = match &elem_ty {
                Type::Struct(_, p) if !p.is_empty() => p[0].clone(),
                _ => Type::I64,
            };
            let inner_elem_llvm_ty = self.get_llvm_type(&inner_elem_ty)?;
            let out_vec_ty = Type::Struct("Vec".to_string(), vec![inner_elem_ty.clone()]);
            let _ = self.get_or_monomorphize_type(&out_vec_ty)?;

            let elem_vec_name = elem_ty.codegen_name().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("unexpected None".to_string())))?;
            let elem_vec_struct_ty = *self.struct_types.get(&elem_vec_name).ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("struct type not found"))))?;

            let tot_var = self.builder.build_alloca(i64_type, "tot").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_store(tot_var, zero).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            // Pass 1
            let p1_lp = self.context.append_basic_block(current_fn, "p1lp");
            let p1_bd = self.context.append_basic_block(current_fn, "p1bd");
            let p1_dn = self.context.append_basic_block(current_fn, "p1dn");

            let i_var = self.builder.build_alloca(i64_type, "i").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_store(i_var, zero).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_unconditional_branch(p1_lp).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            self.builder.position_at_end(p1_lp);
            let i_val = self.builder.build_load(i64_type, i_var, "iv").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value();
            let cond = self.builder.build_int_compare(inkwell::IntPredicate::SLT, i_val, len, "cnd").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_conditional_branch(cond, p1_bd, p1_dn).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            self.builder.position_at_end(p1_bd);
            let sub_vec = self.builder.build_load(elem_llvm_ty, unsafe { self.builder.build_gep(elem_llvm_ty, data_ptr.into_pointer_value(), &[i_val], "sa").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))? }, "sv").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_pointer_value();
            let sub_len = self.builder.build_load(i64_type, self.builder.build_struct_gep(elem_vec_struct_ty, sub_vec, 2, "slf").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?, "sln").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value();
            let ctot = self.builder.build_load(i64_type, tot_var, "ct").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value();
            self.builder.build_store(tot_var, self.builder.build_int_add(ctot, sub_len, "nt").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            
            self.builder.build_store(i_var, self.builder.build_int_add(i_val, i64_type.const_int(1, false), "nxt").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_unconditional_branch(p1_lp).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            // Pass 2
            self.builder.position_at_end(p1_dn);
            let tot_val = self.builder.build_load(i64_type, tot_var, "tv").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value();
            let res_ptr = self.builder.build_alloca(vec_struct_ty, "res").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let malloc_fn = self.get_fn("malloc")?;
            let buf_sz = self.builder.build_int_mul(tot_val, inner_elem_llvm_ty.size_of().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("type has no size".to_string())))?, "bsz").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let raw_buf_call = self.builder.build_call(malloc_fn, &[buf_sz.into()], "buf").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let raw_buf = match raw_buf_call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v,
                _ => return Err(CodegenErrorKind::Internal("malloc failed".to_string()).into()),
            };

            self.builder.build_store(self.builder.build_struct_gep(vec_struct_ty, res_ptr, 0, "f0").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?, raw_buf).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_store(self.builder.build_struct_gep(vec_struct_ty, res_ptr, 1, "f1").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?, tot_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_store(self.builder.build_struct_gep(vec_struct_ty, res_ptr, 2, "f2").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?, tot_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            let p2_lp = self.context.append_basic_block(current_fn, "p2lp");
            let p2_bd = self.context.append_basic_block(current_fn, "p2bd");
            let p2_dn = self.context.append_basic_block(current_fn, "p2dn");
            
            self.builder.build_store(i_var, zero).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let out_idx = self.builder.build_alloca(i64_type, "oidx").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_store(out_idx, zero).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_unconditional_branch(p2_lp).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            self.builder.position_at_end(p2_lp);
            let i_val2 = self.builder.build_load(i64_type, i_var, "iv2").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value();
            let cond2 = self.builder.build_int_compare(inkwell::IntPredicate::SLT, i_val2, len, "cnd2").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_conditional_branch(cond2, p2_bd, p2_dn).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            self.builder.position_at_end(p2_bd);
            let sub_vec2 = self.builder.build_load(elem_llvm_ty, unsafe { self.builder.build_gep(elem_llvm_ty, data_ptr.into_pointer_value(), &[i_val2], "sa2").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))? }, "sv2").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_pointer_value();
            let sub_dat = self.builder.build_load(ptr_type, self.builder.build_struct_gep(elem_vec_struct_ty, sub_vec2, 0, "sdf").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?, "sd").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let sub_ln = self.builder.build_load(i64_type, self.builder.build_struct_gep(elem_vec_struct_ty, sub_vec2, 2, "slf2").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?, "sln2").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value();

            let in_lp = self.context.append_basic_block(current_fn, "ilp");
            let in_bd = self.context.append_basic_block(current_fn, "ibd");
            let in_dn = self.context.append_basic_block(current_fn, "idn");

            let j_var = self.builder.build_alloca(i64_type, "j").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_store(j_var, zero).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_unconditional_branch(in_lp).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            self.builder.position_at_end(in_lp);
            let j_val = self.builder.build_load(i64_type, j_var, "jv").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value();
            let cond3 = self.builder.build_int_compare(inkwell::IntPredicate::SLT, j_val, sub_ln, "cnd3").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_conditional_branch(cond3, in_bd, in_dn).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            self.builder.position_at_end(in_bd);
            let e_val = self.builder.build_load(inner_elem_llvm_ty, unsafe { self.builder.build_gep(inner_elem_llvm_ty, sub_dat.into_pointer_value(), &[j_val], "ea").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))? }, "ev").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let cur_oidx = self.builder.build_load(i64_type, out_idx, "co").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value();
            self.builder.build_store(unsafe { self.builder.build_gep(inner_elem_llvm_ty, raw_buf.into_pointer_value(), &[cur_oidx], "ofa").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))? }, e_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            self.emit_retain(e_val, &inner_elem_ty)?;

            self.builder.build_store(out_idx, self.builder.build_int_add(cur_oidx, i64_type.const_int(1, false), "no").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_store(j_var, self.builder.build_int_add(j_val, i64_type.const_int(1, false), "nj").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_unconditional_branch(in_lp).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            self.builder.position_at_end(in_dn);
            self.builder.build_store(i_var, self.builder.build_int_add(i_val2, i64_type.const_int(1, false), "ni2").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            self.builder.build_unconditional_branch(p2_lp).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            self.builder.position_at_end(p2_dn);
            return Ok((res_ptr.into(), out_vec_ty));
        }

        Err(TlError::from(CodegenErrorKind::Internal(format!("Unknown built-in method {}", method))))
    }


    /// Option.map/and_then/unwrap_or_else のインラインIR生成
    /// Option は Enum { i32 tag, [i64 x N] payload } レイアウト。
    /// Some = tag 0, None = tag 1。
    fn compile_option_closure_method(
        &mut self,
        opt_val: BasicValueEnum<'ctx>,
        opt_ty: &Type,
        elem_ty: &Type,
        method: &str,
        closure_args: &[(String, Option<Type>)],
        closure_body: &[Stmt],
    ) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
        let i32_type = self.context.i32_type();
        let i64_type = self.context.i64_type();
        let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());

        // Get the Option enum type from enum_types registry
        let opt_type_name = match opt_ty {
            Type::Enum(name, args) if !args.is_empty() => self.mangle_type_name(name, &args),
            _ => opt_ty.codegen_name().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("Cannot get Option codegen name".to_string())))?
        };
        let enum_ty = *self.enum_types.get(&opt_type_name)
            .or_else(|| self.enum_types.get(&opt_ty.get_base_name()))
            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Option enum type {} not found", opt_type_name))))?;
        let opt_ptr = opt_val.into_pointer_value();

        // Read tag: GEP to field 0, load i32
        let tag_ptr = self.builder.build_struct_gep(enum_ty, opt_ptr, 0, "tag_ptr")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        let tag = self.builder.build_load(i32_type, tag_ptr, "tag")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value();

        // Check: tag == 0 (Some)
        let is_some = self.builder.build_int_compare(
            inkwell::IntPredicate::EQ, tag, i32_type.const_zero(), "is_some",
        ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        let current_fn = self.current_function()?;
        let some_bb = self.context.append_basic_block(current_fn, "opt_some");
        let none_bb = self.context.append_basic_block(current_fn, "opt_none");
        let merge_bb = self.context.append_basic_block(current_fn, "opt_merge");

        self.builder.build_conditional_branch(is_some, some_bb, none_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        // === Some branch ===
        self.builder.position_at_end(some_bb);

        // Read payload from Option (field 1 = payload area)
        let payload_ptr = self.builder.build_struct_gep(enum_ty, opt_ptr, 1, "payload_ptr")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        let payload_cast = self.builder.build_pointer_cast(payload_ptr, ptr_type, "payload_cast").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        let elem_llvm_ty = self.get_llvm_type(elem_ty)?;
        let payload_val = self.builder.build_load(elem_llvm_ty, payload_cast, "payload_val").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        // Bind closure argument
        self.enter_scope();
        let arg_name = closure_args.first().map(|(n, _)| n.as_str()).unwrap_or("x");
        let arg_alloca = self.builder.build_alloca(elem_llvm_ty, arg_name).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        self.builder.build_store(arg_alloca, payload_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        self.variables.last_mut().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("no variable scope".to_string())))?.insert(
            arg_name.to_string(),
            (arg_alloca.into(), elem_ty.clone(), crate::compiler::codegen::CLEANUP_NONE),
        );

        // Compile closure body
        let mut result_val = None;
        let body_len = closure_body.len();
        for (idx, stmt) in closure_body.iter().enumerate() {
            if idx == body_len - 1 {
                if let crate::compiler::ast::StmtKind::Expr(e) = &stmt.inner {
                    result_val = Some(self.compile_expr(e)?);
                } else {
                    self.compile_stmt(stmt)?;
                }
            } else {
                self.compile_stmt(stmt)?;
            }
        }
        self.exit_scope();

        let (mapped_val, mapped_ty) = result_val.unwrap_or((i64_type.const_zero().into(), Type::I64));

        match method {
            "unwrap_or_else" => {
                // Some: return the payload directly (no closure call needed for Some)
                // Actually, unwrap_or_else: Some(x) -> x, None -> closure()
                // So for Some branch, just return payload_val
                self.builder.build_unconditional_branch(merge_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let some_end_bb = self.current_block()?;

                // None branch: call the closure (with no arg?) to get default
                self.builder.position_at_end(none_bb);
                // For unwrap_or_else, the closure was already compiled above (in Some branch)
                // We need to compile again in None branch. Actually, the closure should be called
                // only in the None path. Let's re-structure:
                // The closure was bound and compiled in Some path above, but we should not have done that.
                // For unwrap_or_else, the closure returns T (not Option), and is called only when None.
                // Since we already compiled the closure in the Some branch, the value is available.
                // Actually in the IR the closure was compiled inline, so the result is only valid in some_bb.
                // We need a different approach: compile closure in none_bb.
                
                // Re-read: unwrap_or_else: Some(x) → x, None → f()
                // We should NOT compile the closure in Some branch.
                // But the dispatch logic already compiled it... 
                // Actually, `mapped_val` was compiled in some_bb, so it's only valid there.
                // For the None branch, we need to compile closure body again.
                self.enter_scope();
                let none_arg_alloca = self.builder.build_alloca(elem_llvm_ty, arg_name).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(none_arg_alloca, elem_llvm_ty.const_zero()).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.variables.last_mut().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("no variable scope".to_string())))?.insert(
                    arg_name.to_string(),
                    (none_arg_alloca.into(), elem_ty.clone(), crate::compiler::codegen::CLEANUP_NONE),
                );
                let mut none_result = None;
                for (idx, stmt) in closure_body.iter().enumerate() {
                    if idx == body_len - 1 {
                        if let crate::compiler::ast::StmtKind::Expr(e) = &stmt.inner {
                            none_result = Some(self.compile_expr(e)?);
                        } else {
                            self.compile_stmt(stmt)?;
                        }
                    } else {
                        self.compile_stmt(stmt)?;
                    }
                }
                self.exit_scope();
                let (none_val, _) = none_result.unwrap_or((elem_llvm_ty.const_zero().into(), elem_ty.clone()));

                self.builder.build_unconditional_branch(merge_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let none_end_bb = self.current_block()?;

                // Merge: phi between payload_val (from Some) and none_val (from closure in None)
                self.builder.position_at_end(merge_bb);
                let phi = self.builder.build_phi(elem_llvm_ty, "unwrap_or_else_result").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                phi.add_incoming(&[(&payload_val, some_end_bb), (&none_val, none_end_bb)]);

                Ok((phi.as_basic_value(), elem_ty.clone()))
            }
            "and_then" => {
                // and_then: Some(x) -> f(x) [returns Option<U>], None -> None
                // mapped_val is the result of f(x) which should be an Option
                let some_result_ptr = mapped_val;
                self.builder.build_unconditional_branch(merge_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let some_end_bb = self.current_block()?;

                // None branch: return None as-is
                self.builder.position_at_end(none_bb);
                let malloc_fn = self.module.get_function("malloc").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("malloc not found".to_string())))?;
                let target_data = self.execution_engine.get_target_data();
                let enum_size = target_data.get_store_size(&enum_ty);
                let size_val = i64_type.const_int(enum_size, false);
                let none_raw = self.builder.build_call(malloc_fn, &[size_val.into()], "none_raw")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let none_ptr = match none_raw.try_as_basic_value() {
                    inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
                    _ => return Err(CodegenErrorKind::Internal("malloc returned void".to_string()).into()),
                };
                let none_tag_ptr = self.builder.build_struct_gep(enum_ty, none_ptr, 0, "none_tag")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(none_tag_ptr, i32_type.const_int(1, false)).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                self.builder.build_unconditional_branch(merge_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let none_end_bb = self.current_block()?;

                // Merge
                self.builder.position_at_end(merge_bb);
                let phi = self.builder.build_phi(ptr_type, "and_then_result").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                phi.add_incoming(&[(&some_result_ptr.into_pointer_value(), some_end_bb), (&none_ptr, none_end_bb)]);

                Ok((phi.as_basic_value(), mapped_ty))
            }
            _ => {
                // "map" — original behavior
                // Allocate new Option enum and set tag=0 (Some), payload=mapped_val
                let target_data = self.execution_engine.get_target_data();
                let enum_size = target_data.get_store_size(&enum_ty);
                let malloc_fn = self.module.get_function("malloc").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("malloc not found".to_string())))?;
                let size_val = i64_type.const_int(enum_size, false);
                let some_raw = self.builder.build_call(malloc_fn, &[size_val.into()], "some_raw")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let some_ptr = match some_raw.try_as_basic_value() {
                    inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
                    _ => return Err(CodegenErrorKind::Internal("malloc returned void".to_string()).into()),
                };

                // Set tag = 0 (Some)
                let some_tag_ptr = self.builder.build_struct_gep(enum_ty, some_ptr, 0, "some_tag")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(some_tag_ptr, i32_type.const_zero()).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // Set payload
                let some_payload_ptr = self.builder.build_struct_gep(enum_ty, some_ptr, 1, "some_payload")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let some_payload_cast = self.builder.build_pointer_cast(some_payload_ptr, ptr_type, "payload_cast").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let store_ptr = self.builder.build_pointer_cast(some_payload_cast,
                    self.context.ptr_type(inkwell::AddressSpace::default()), "store_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(store_ptr, mapped_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // Register with memory manager
                if let Some(reg_fn) = self.module.get_function("tl_mem_register") {
                    let file_str = self.builder.build_global_string_ptr("option_map", "opt_map_file").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    let tag_str = self.builder.build_global_string_ptr(&opt_type_name, "opt_map_tag").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    self.builder.build_call(
                        reg_fn,
                        &[some_ptr.into(), file_str.as_pointer_value().into(),
                          i32_type.const_zero().into(), i32_type.const_zero().into(),
                          tag_str.as_pointer_value().into()],
                        "",
                    ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                }

                self.builder.build_unconditional_branch(merge_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let some_end_bb = self.current_block()?;

                // === None branch ===
                self.builder.position_at_end(none_bb);

                // Allocate new None: same enum type, tag=1, no payload
                let none_raw = self.builder.build_call(malloc_fn, &[size_val.into()], "none_raw")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let none_ptr = match none_raw.try_as_basic_value() {
                    inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
                    _ => return Err(CodegenErrorKind::Internal("malloc returned void".to_string()).into()),
                };
                let none_tag_ptr = self.builder.build_struct_gep(enum_ty, none_ptr, 0, "none_tag")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(none_tag_ptr, i32_type.const_int(1, false)).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                if let Some(reg_fn) = self.module.get_function("tl_mem_register") {
                    let file_str = self.builder.build_global_string_ptr("option_map", "opt_map_file2").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    let tag_str = self.builder.build_global_string_ptr(&opt_type_name, "opt_map_tag2").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    self.builder.build_call(
                        reg_fn,
                        &[none_ptr.into(), file_str.as_pointer_value().into(),
                          i32_type.const_zero().into(), i32_type.const_zero().into(),
                          tag_str.as_pointer_value().into()],
                        "",
                    ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                }

                self.builder.build_unconditional_branch(merge_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let none_end_bb = self.current_block()?;

                // === Merge ===
                self.builder.position_at_end(merge_bb);
                let phi = self.builder.build_phi(ptr_type, "opt_mapped").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                phi.add_incoming(&[(&some_ptr, some_end_bb), (&none_ptr, none_end_bb)]);

                Ok((phi.as_basic_value(), opt_ty.clone()))
            }
        }
    }

    /// Result.map/map_err/and_then/unwrap_or_else のインラインIR生成
    /// Result は Enum { i32 tag, [payload] } レイアウト。 Ok = tag 0, Err = tag 1。
    fn compile_result_closure_method(
        &mut self,
        result_val: BasicValueEnum<'ctx>,
        result_ty: &Type,
        elem_ty: &Type,
        method: &str,
        closure_args: &[(String, Option<Type>)],
        closure_body: &[Stmt],
    ) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
        let i32_type = self.context.i32_type();
        let i64_type = self.context.i64_type();
        let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());

        let result_type_name = match result_ty {
            Type::Enum(name, args) if !args.is_empty() => self.mangle_type_name(name, &args),
            _ => result_ty.codegen_name().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("Cannot get Result codegen name".to_string())))?
        };
        let enum_ty = *self.enum_types.get(&result_type_name)
            .or_else(|| self.enum_types.get(&result_ty.get_base_name()))
            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Result enum type {} not found", result_type_name))))?;
        let result_ptr = result_val.into_pointer_value();

        let tag_ptr = self.builder.build_struct_gep(enum_ty, result_ptr, 0, "tag_ptr")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        let tag = self.builder.build_load(i32_type, tag_ptr, "tag")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value();

        // tag == 0 means Ok
        let is_ok = self.builder.build_int_compare(
            inkwell::IntPredicate::EQ, tag, i32_type.const_zero(), "is_ok",
        ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        let current_fn = self.current_function()?;
        let ok_bb = self.context.append_basic_block(current_fn, "result_ok");
        let err_bb = self.context.append_basic_block(current_fn, "result_err");
        let merge_bb = self.context.append_basic_block(current_fn, "result_merge");

        let payload_ptr = self.builder.build_struct_gep(enum_ty, result_ptr, 1, "payload_ptr")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        let payload_cast = self.builder.build_pointer_cast(payload_ptr, ptr_type, "payload_cast").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        let elem_llvm_ty = self.get_llvm_type(elem_ty)?;

        self.builder.build_conditional_branch(is_ok, ok_bb, err_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        let target_data = self.execution_engine.get_target_data();
        let enum_size = target_data.get_store_size(&enum_ty);
        let malloc_fn = self.module.get_function("malloc").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("malloc not found".to_string())))?;
        let size_val = i64_type.const_int(enum_size, false);

        match method {
            "map" => {
                // map: Ok(x) -> Ok(f(x)), Err(e) -> Err(e)
                self.builder.position_at_end(ok_bb);
                let payload_val = self.builder.build_load(elem_llvm_ty, payload_cast, "ok_val").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                self.enter_scope();
                let arg_name = closure_args.first().map(|(n, _)| n.as_str()).unwrap_or("x");
                let arg_alloca = self.builder.build_alloca(elem_llvm_ty, arg_name).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(arg_alloca, payload_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.variables.last_mut().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("no variable scope".to_string())))?.insert(
                    arg_name.to_string(),
                    (arg_alloca.into(), elem_ty.clone(), crate::compiler::codegen::CLEANUP_NONE),
                );
                let mut result_val_inner = None;
                let body_len = closure_body.len();
                for (idx, stmt) in closure_body.iter().enumerate() {
                    if idx == body_len - 1 {
                        if let crate::compiler::ast::StmtKind::Expr(e) = &stmt.inner {
                            result_val_inner = Some(self.compile_expr(e)?);
                        } else { self.compile_stmt(stmt)?; }
                    } else { self.compile_stmt(stmt)?; }
                }
                self.exit_scope();

                let (mapped_val, _mapped_ty) = result_val_inner.unwrap_or((i64_type.const_zero().into(), Type::I64));

                // Allocate new Ok result
                let ok_raw = self.builder.build_call(malloc_fn, &[size_val.into()], "ok_raw")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let ok_ptr = match ok_raw.try_as_basic_value() {
                    inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
                    _ => return Err(CodegenErrorKind::Internal("malloc returned void".to_string()).into()),
                };
                let ok_tag_ptr = self.builder.build_struct_gep(enum_ty, ok_ptr, 0, "ok_tag")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(ok_tag_ptr, i32_type.const_zero()).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let ok_payload_ptr = self.builder.build_struct_gep(enum_ty, ok_ptr, 1, "ok_payload")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let ok_payload_cast = self.builder.build_pointer_cast(ok_payload_ptr, ptr_type, "ok_pc").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(ok_payload_cast, mapped_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                self.builder.build_unconditional_branch(merge_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let ok_end_bb = self.current_block()?;

                // Err branch: pass through
                self.builder.position_at_end(err_bb);
                self.builder.build_unconditional_branch(merge_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let err_end_bb = self.current_block()?;

                self.builder.position_at_end(merge_bb);
                let phi = self.builder.build_phi(ptr_type, "result_mapped").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                phi.add_incoming(&[(&ok_ptr, ok_end_bb), (&result_ptr, err_end_bb)]);

                Ok((phi.as_basic_value(), result_ty.clone()))
            }
            "map_err" => {
                // map_err: Ok(x) -> Ok(x), Err(e) -> Err(f(e))
                self.builder.position_at_end(ok_bb);
                self.builder.build_unconditional_branch(merge_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let ok_end_bb = self.current_block()?;

                self.builder.position_at_end(err_bb);
                let err_payload = self.builder.build_load(elem_llvm_ty, payload_cast, "err_val").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                self.enter_scope();
                let arg_name = closure_args.first().map(|(n, _)| n.as_str()).unwrap_or("e");
                let arg_alloca = self.builder.build_alloca(elem_llvm_ty, arg_name).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(arg_alloca, err_payload).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.variables.last_mut().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("no variable scope".to_string())))?.insert(
                    arg_name.to_string(),
                    (arg_alloca.into(), elem_ty.clone(), crate::compiler::codegen::CLEANUP_NONE),
                );
                let mut result_val_inner = None;
                let body_len = closure_body.len();
                for (idx, stmt) in closure_body.iter().enumerate() {
                    if idx == body_len - 1 {
                        if let crate::compiler::ast::StmtKind::Expr(e) = &stmt.inner {
                            result_val_inner = Some(self.compile_expr(e)?);
                        } else { self.compile_stmt(stmt)?; }
                    } else { self.compile_stmt(stmt)?; }
                }
                self.exit_scope();
                let (mapped_err, _) = result_val_inner.unwrap_or((i64_type.const_zero().into(), Type::I64));

                let err_new = self.builder.build_call(malloc_fn, &[size_val.into()], "err_raw")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let err_ptr = match err_new.try_as_basic_value() {
                    inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
                    _ => return Err(CodegenErrorKind::Internal("malloc returned void".to_string()).into()),
                };
                let err_tag = self.builder.build_struct_gep(enum_ty, err_ptr, 0, "err_tag")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(err_tag, i32_type.const_int(1, false)).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let err_payload_ptr2 = self.builder.build_struct_gep(enum_ty, err_ptr, 1, "err_payload2")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let err_pc = self.builder.build_pointer_cast(err_payload_ptr2, ptr_type, "err_pc").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(err_pc, mapped_err).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                self.builder.build_unconditional_branch(merge_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let err_end_bb = self.current_block()?;

                self.builder.position_at_end(merge_bb);
                let phi = self.builder.build_phi(ptr_type, "result_map_err").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                phi.add_incoming(&[(&result_ptr, ok_end_bb), (&err_ptr, err_end_bb)]);

                Ok((phi.as_basic_value(), result_ty.clone()))
            }
            "and_then" => {
                // and_then: Ok(x) -> f(x) [returns Result], Err(e) -> Err(e)
                self.builder.position_at_end(ok_bb);
                let payload_val = self.builder.build_load(elem_llvm_ty, payload_cast, "ok_val").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                self.enter_scope();
                let arg_name = closure_args.first().map(|(n, _)| n.as_str()).unwrap_or("x");
                let arg_alloca = self.builder.build_alloca(elem_llvm_ty, arg_name).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(arg_alloca, payload_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.variables.last_mut().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("no variable scope".to_string())))?.insert(
                    arg_name.to_string(),
                    (arg_alloca.into(), elem_ty.clone(), crate::compiler::codegen::CLEANUP_NONE),
                );
                let mut result_val_inner = None;
                let body_len = closure_body.len();
                for (idx, stmt) in closure_body.iter().enumerate() {
                    if idx == body_len - 1 {
                        if let crate::compiler::ast::StmtKind::Expr(e) = &stmt.inner {
                            result_val_inner = Some(self.compile_expr(e)?);
                        } else { self.compile_stmt(stmt)?; }
                    } else { self.compile_stmt(stmt)?; }
                }
                self.exit_scope();
                let (chained_val, chained_ty) = result_val_inner.unwrap_or((result_ptr.into(), result_ty.clone()));
                let chained_ptr = chained_val.into_pointer_value();

                self.builder.build_unconditional_branch(merge_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let ok_end_bb = self.current_block()?;

                // Err branch: pass through
                self.builder.position_at_end(err_bb);
                self.builder.build_unconditional_branch(merge_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let err_end_bb = self.current_block()?;

                self.builder.position_at_end(merge_bb);
                let phi = self.builder.build_phi(ptr_type, "result_and_then").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                phi.add_incoming(&[(&chained_ptr, ok_end_bb), (&result_ptr, err_end_bb)]);

                Ok((phi.as_basic_value(), chained_ty))
            }
            "unwrap_or_else" => {
                // unwrap_or_else: Ok(x) -> x, Err(e) -> f(e)
                self.builder.position_at_end(ok_bb);
                let ok_payload = self.builder.build_load(elem_llvm_ty, payload_cast, "ok_val").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_unconditional_branch(merge_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let ok_end_bb = self.current_block()?;

                // Err branch: call closure with err payload
                self.builder.position_at_end(err_bb);
                let err_payload = self.builder.build_load(elem_llvm_ty, payload_cast, "err_val").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.enter_scope();
                let arg_name = closure_args.first().map(|(n, _)| n.as_str()).unwrap_or("e");
                let arg_alloca = self.builder.build_alloca(elem_llvm_ty, arg_name).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(arg_alloca, err_payload).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.variables.last_mut().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("no variable scope".to_string())))?.insert(
                    arg_name.to_string(),
                    (arg_alloca.into(), elem_ty.clone(), crate::compiler::codegen::CLEANUP_NONE),
                );
                let mut result_val_inner = None;
                let body_len = closure_body.len();
                for (idx, stmt) in closure_body.iter().enumerate() {
                    if idx == body_len - 1 {
                        if let crate::compiler::ast::StmtKind::Expr(e) = &stmt.inner {
                            result_val_inner = Some(self.compile_expr(e)?);
                        } else { self.compile_stmt(stmt)?; }
                    } else { self.compile_stmt(stmt)?; }
                }
                self.exit_scope();
                let (fallback_val, _) = result_val_inner.unwrap_or((elem_llvm_ty.const_zero().into(), elem_ty.clone()));

                self.builder.build_unconditional_branch(merge_bb).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let err_end_bb = self.current_block()?;

                self.builder.position_at_end(merge_bb);
                let phi = self.builder.build_phi(elem_llvm_ty, "unwrap_or_else").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                phi.add_incoming(&[(&ok_payload, ok_end_bb), (&fallback_val, err_end_bb)]);

                Ok((phi.as_basic_value(), elem_ty.clone()))
            }
            _ => Err(TlError::from(CodegenErrorKind::Internal(format!("Unknown Result closure method: {}", method)))),
        }
    }


    pub(crate) fn compile_fn_call(
        &mut self,
        name: &str,
        args: &[Expr],
    ) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
        let (val, ty) = self.compile_fn_call_dps(name, args, None)?;
        
        let mode = match &ty {
             Type::Struct(_, _) | Type::String(_) | Type::Tensor(_, _) => super::CLEANUP_FULL,
             _ => super::CLEANUP_STACK,
        };
        
        self.add_temp_with_mode(val, ty.clone(), mode);
        Ok((val, ty))
    }

    pub(crate) fn compile_fn_call_dps(
        &mut self,
        name: &str,
        args: &[Expr],
        dest: Option<BasicValueEnum<'ctx>>,
    ) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
        // 0. Check if it's a relation query
        let simple_name = name;

        if self.relations.contains(name) {
            return self.compile_relation_call(name, args);
        } else if self.relations.contains(simple_name) {
            return self.compile_relation_call(simple_name, args);
        }

        // 0.5. Check if name is a closure variable (Type::Fn)
        {
            let mut closure_info = None;
            for scope in self.variables.iter().rev() {
                if let Some((val, ty, _)) = scope.get(name) {
                    if let Type::Fn(_, _) = ty {
                        closure_info = Some((*val, ty.clone()));
                    }
                    break;
                }
            }
            if let Some((fn_ptr_alloca, var_ty)) = closure_info {
              if let Type::Fn(param_types, ret_ty) = var_ty {
                let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());

                // The closure variable stores a fat pointer: {fn_ptr, env_ptr} struct
                let closure_struct_ty = self.context.struct_type(&[ptr_type.into(), ptr_type.into()], false);
                let fat_ptr = self.builder.build_load(
                    closure_struct_ty,
                    fn_ptr_alloca.into_pointer_value(),
                    "load_fat_ptr",
                ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // Extract fn_ptr and env_ptr from the fat pointer struct
                let fn_ptr = self.builder.build_extract_value(
                    fat_ptr.into_struct_value(), 0, "extract_fn_ptr"
                ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let env_ptr = self.builder.build_extract_value(
                    fat_ptr.into_struct_value(), 1, "extract_env_ptr"
                ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                // Build LLVM function type (ALL closures have env_ptr as first param)
                let mut llvm_param_types: Vec<inkwell::types::BasicMetadataTypeEnum<'ctx>> = Vec::new();
                llvm_param_types.push(ptr_type.into()); // env_ptr always first
                for pt in &param_types {
                    let llvm_ty = self.get_llvm_type(pt)?;
                    llvm_param_types.push(llvm_ty.into());
                }

                let fn_type = if matches!(*ret_ty, Type::Void) {
                    self.context.void_type().fn_type(&llvm_param_types, false)
                } else {
                    let ret_llvm_ty = self.get_llvm_type(&ret_ty)?;
                    ret_llvm_ty.fn_type(&llvm_param_types, false)
                };

                // Compile arguments: env_ptr first, then user args
                let mut compiled_args: Vec<inkwell::values::BasicMetadataValueEnum<'ctx>> = Vec::new();
                compiled_args.push(env_ptr.into()); // env_ptr always first
                for arg in args {
                    let (val, _) = self.compile_expr(arg)?;
                    compiled_args.push(val.into());
                }

                // Build indirect call
                let call_val = self.builder.build_indirect_call(
                    fn_type,
                    fn_ptr.into_pointer_value(),
                    &compiled_args,
                    "closure_call",
                ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                if matches!(*ret_ty, Type::Void) {
                    return Ok((self.context.i64_type().const_zero().into(), *ret_ty));
                } else {
                    let result = match call_val.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err(CodegenErrorKind::Internal("Closure call returned no value".to_string()).into()),
                    };
                    return Ok((result, *ret_ty));
                }
              }
            }
        }

        // 1. Builtins
        let builtin_opt = self.builtin_manager.get(name).copied();
        if let Some(builtin) = builtin_opt {
            match builtin {
                BuiltinFn::Evaluated(func) => {
                    let mut compiled_args = Vec::with_capacity(args.len());
                    for arg in args {
                        let (val, ty) = self.compile_expr(arg)?;
                        compiled_args.push((val, ty));
                    }
                    return func(self, compiled_args);
                }
                BuiltinFn::Unevaluated(func) => {
                    return func(self, args);
                }
            }
        }

        // Intrinsic: tl_core_hash<T>(val: T) -> i64
        if name.starts_with("tl_core_hash") {
            if args.len() != 1 {
                return Err(TlError::from(CodegenErrorKind::Internal("tl_core_hash expects exactly 1 argument".to_string())));
            }
            let (val, ty) = self.compile_expr(&args[0])?;
            let i64_type = self.context.i64_type();

            let res: inkwell::values::IntValue = match &ty {
                Type::I64 => val.into_int_value(),
                Type::I32 | Type::Char(_) => self.builder.build_int_z_extend(val.into_int_value(), i64_type, "zext").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?,
                Type::Bool => self.builder.build_int_z_extend(val.into_int_value(), i64_type, "zext").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?,
                Type::F32 => {
                    let i32_val = self.builder.build_bit_cast(val.into_float_value(), self.context.i32_type(), "f32_cast").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value();
                    self.builder.build_int_z_extend(i32_val, i64_type, "zext").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                },
                Type::F64 => {
                     self.builder.build_bit_cast(val.into_float_value(), i64_type, "f64_cast").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value()
                },
                Type::String(_) => {
                     let fn_val = self.module.get_function("tl_hash_string")
                         .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_hash_string runtime function not found".to_string())))?;
                     let call = self.builder.build_call(fn_val, &[val.into()], "hash_call").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                     match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v.into_int_value(),
                        _ => return Err(CodegenErrorKind::Internal("tl_hash_string did not return a value".to_string()).into()),
                     }
                },
                Type::Struct(_, _) | Type::Enum(_, _) | Type::Tensor(_, _) | Type::Tuple(_) | Type::SpecializedType { .. } => {
                    if val.is_pointer_value() {
                         self.builder.build_ptr_to_int(val.into_pointer_value(), i64_type, "ptr_int").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                    } else {
                         // Fallback for immediate structs (very small ones potentially? not standard in TL currently)
                         // Return 0 to be safe/lazy, or error?
                         return Err(TlError::from(CodegenErrorKind::UnsupportedOperation(format!("Hashing immediate struct/value type not supported: {:?}", ty))));
                    }
                },
                _ => return Err(TlError::from(CodegenErrorKind::UnsupportedOperation(format!("Unsupported type for hash: {:?}", ty)))),
            };
            
            return Ok((res.into(), Type::I64));
        }

        // Intrinsic: __builtin_unsafe_to_i64
        if name.starts_with("__builtin_unsafe_to_i64") {
            if args.len() != 1 {
                return Err(TlError::from(CodegenErrorKind::Internal("__builtin_unsafe_to_i64 expects exactly 1 argument".to_string())));
            }
            let (val, ty) = self.compile_expr(&args[0])?;
            
            let i64_type = self.context.i64_type();
            let res = match val {
                inkwell::values::BasicValueEnum::IntValue(i) => {
                    if i.get_type().get_bit_width() == 64 {
                        i.into()
                    } else {
                        // Extend (zext)
                        self.builder.build_int_z_extend(i, i64_type, "zext").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into()
                    }
                }
                inkwell::values::BasicValueEnum::PointerValue(p) => {
                    self.builder.build_ptr_to_int(p, i64_type, "ptr2int").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into()
                }
                inkwell::values::BasicValueEnum::FloatValue(f) => {
                    // Bitcast to i32 then zext to i64
                    let i32_type = self.context.i32_type();
                    let as_i32 = self.builder.build_bit_cast(f, i32_type, "f32cast").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_int_value();
                    self.builder.build_int_z_extend(as_i32, i64_type, "zext").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into()
                }
                _ => return Err(TlError::from(CodegenErrorKind::UnsupportedOperation(format!("Unsupported type for unsafe_to_i64: {:?}", ty)))),
            };
            return Ok((res, Type::I64));
        }

        // Intrinsic: __builtin_unsafe_from_i64(val: i64, marker: PhantomData<T>) -> T
        if name.starts_with("__builtin_unsafe_from_i64") {
            if args.len() != 2 {
                return Err(TlError::from(CodegenErrorKind::Internal("__builtin_unsafe_from_i64 expects 2 arguments (val, marker)".to_string())));
            }
            let (val, _val_ty) = self.compile_expr(&args[0])?;
            let (_, marker_ty) = self.compile_expr(&args[1])?;
            
            // Extract T from PhantomData<T>
            let target_type = if let Type::Struct(name, generics) = &marker_ty {
                if name.contains("PhantomData") {
                    if !generics.is_empty() {
                        generics[0].clone()
                    } else if let Some(suffix) = name.strip_prefix("PhantomData_") {
                        // Specialized
                        match suffix {
                            "i64" => Type::I64,
                            "i32" => Type::I32,
                            "f64" => Type::F64,
                            "f32" => Type::F32,
                            "bool" => Type::Bool,
                            "String" => Type::String("String".to_string()),
                            "Char" => Type::Char("Char".to_string()),
                            _ => Type::Struct(suffix.to_string(), vec![]),
                        }
                    } else {
                         return Err(TlError::from(CodegenErrorKind::Internal(format!("Arg 2 must be PhantomData<T> (specialized or generic), got {:?}", marker_ty))));
                    }
                } else {
                    return Err(TlError::from(CodegenErrorKind::Internal(format!("Arg 2 must be PhantomData<T>, got {:?}", marker_ty))));
                }
            } else {
                return Err(TlError::from(CodegenErrorKind::Internal(format!("Arg 2 must be Struct PhantomData, got {:?}", marker_ty))));
            };

            // Cast i64 to T
            let res: inkwell::values::BasicValueEnum = match &target_type {
                Type::I64 => val, // Identity
                Type::Struct(_, _) | Type::Tensor(_, _) => {
                    // i64 -> ptr
                    let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
                    if val.is_int_value() {
                        self.builder.build_int_to_ptr(val.into_int_value(), ptr_type, "int2ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into()
                    } else if val.is_pointer_value() {
                        val // Identity if already ptr? But arg 0 should be i64.
                    } else {
                         return Err(TlError::from(CodegenErrorKind::Internal("Input must be int or ptr".to_string())));
                    }
                },
                Type::Bool => {
                     // i64 -> bool (trunc)
                     let i1_type = self.context.bool_type();
                     self.builder.build_int_truncate(val.into_int_value(), i1_type, "trunc_bool").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into()
                },
                Type::String(_) => {
                     // i64 -> ptr
                     let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
                     self.builder.build_int_to_ptr(val.into_int_value(), ptr_type, "int2ptr_str").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into()
                },
                // Add more types if needed (u8, f32)
                Type::F32 => {
                     // i64 -> i32 -> float
                     let i32_val = self.builder.build_int_truncate(val.into_int_value(), self.context.i32_type(), "trunc_f32").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                     self.builder.build_bit_cast(i32_val, self.context.f32_type(), "bitcast").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into()
                },
                _ => return Err(TlError::from(CodegenErrorKind::UnsupportedOperation(format!("Unsupported target type for from_i64: {:?}", target_type)))),
            };
            
            return Ok((res, target_type));
        }

        // Intrinsic: __builtin_is_ref(marker: PhantomData<T>) -> bool
        if name.starts_with("__builtin_is_ref") {
            if args.len() != 1 {
                return Err(TlError::from(CodegenErrorKind::Internal("__builtin_is_ref expects 1 argument (marker)".to_string())));
            }
            let (_, marker_ty) = self.compile_expr(&args[0])?;
            
            // Extract T
            let target_type = if let Type::Struct(name, generics) = &marker_ty {
                if name.contains("PhantomData") {
                    if !generics.is_empty() {
                        generics[0].clone()
                    } else if let Some(suffix) = name.strip_prefix("PhantomData_") {
                        // Specialized: PhantomData_i64, PhantomData_String, etc.
                        match suffix {
                            "i64" => Type::I64,
                            "i32" => Type::I32,
                            "f64" => Type::F64,
                            "f32" => Type::F32,
                            "bool" => Type::Bool,
                            "String" => Type::String("String".to_string()),
                            "Char" => Type::Char("Char".to_string()),
                            _ => Type::Struct(suffix.to_string(), vec![]), // Assume struct/ref
                        }
                    } else {
                         // PhantomData generic unspecialized with no args? Should not happen in monomorphized code
                         return Err(TlError::from(CodegenErrorKind::Internal(format!("Arg 1 must be PhantomData<T> (specialized or generic), got {:?}", marker_ty))));
                    }
                } else {
                    return Err(TlError::from(CodegenErrorKind::Internal(format!("Arg 1 must be PhantomData<T>, got {:?}", marker_ty))));
                }
            } else {
                return Err(TlError::from(CodegenErrorKind::Internal(format!("Arg 1 must be Struct PhantomData, got {:?}", marker_ty))));
            };

            let is_ref = match target_type {
                Type::Struct(name, _) if name == "String" => false, // String literals crash if treated as RefCounted
                Type::String(_) => false,
                Type::Struct(_, _) | Type::Tensor(_, _) | Type::Enum(_, _) | Type::Tuple(_) | Type::SpecializedType { .. } => true,
                _ => false,
            };
            
            let bool_val = self.context.bool_type().const_int(if is_ref { 1 } else { 0 }, false);
            return Ok((bool_val.into(), Type::Bool));
        }

        // 2. Generic Function Call / Struct Init
        let llvm_func_name = match name {
            "slice" => "tl_tensor_slice",
            "sum" => "tl_tensor_sum", // Fallback for global sum if not caught by builtin (redundant but safe)
            "enable_grad" => "tl_tensor_enable_grad",
            _ => name,
        };

        // Lookup return type FIRST to handle sret
        // Simplified: FnCall handles simple names. StaticMethodCall handles :: names.
        // We only keep simple name resolution for legacy builtins or simple function lookups.
        let resolved_name = llvm_func_name.to_string();

        // Monomorphization Logic: Check if generic
        let mut final_resolved_name = resolved_name.clone();
        let mut precompiled_args = None;
        
        if self.module.get_function(&final_resolved_name).is_none() && self.generic_fn_defs.contains_key(&final_resolved_name) {
             let mut args_vec = Vec::new();
             let mut arg_types = Vec::new();
             for arg in args {
                  let (val, ty) = self.compile_expr(arg)?;

                  args_vec.push((val, ty.clone()));
                  arg_types.push(ty);
             }
             
             final_resolved_name = self.monomorphize_generic_function(&resolved_name, &arg_types)?;
             precompiled_args = Some(args_vec);
        }

        let func_opt = self.module.get_function(&final_resolved_name);

        let func = if let Some(f) = func_opt {
            f
        } else {
            // Fallback to Struct Initialization
            let simple_name = name;

            if self.struct_defs.contains_key(simple_name) {
                return self.compile_tuple_struct_init(simple_name, args);
            }

            return Err(TlError::from(CodegenErrorKind::FunctionNotFound(format!(
                "Function {} not found (resolved: {})",
                name, resolved_name
            ))));
        };

        let ret_type = if let Some(ret) = self.method_return_types.get(&resolved_name) {
            ret.clone()
        } else {
            self.get_return_type_from_signature(func)
        };

        let mut compiled_args_vals = Vec::with_capacity(args.len() + 1);
        let mut compiled_args_types = Vec::with_capacity(args.len());

        // DPS: If return type is Struct (and SRET enabled), handle hidden dest argument
        // Tensors are returned by pointer directly, so exclude them.
        let mut dest_val = None;
        let uses_sret = match ret_type {
             Type::Struct(ref name, _) => name != "Tensor" && name != "String",
             _ => false
        };
        if uses_sret {
             if let Some(d) = dest {
                 dest_val = Some(d);
             } else {
                 // NEW: Heap Allocation (malloc + register) -> Correct for RefCounted Structs/SRET
                 let (struct_name, generics) = match &ret_type {
                     Type::Struct(n, g) | Type::Enum(n, g) => (n, g),
                     _ => return Err(CodegenErrorKind::Internal("SRET used on non-aggregate type".to_string()).into()),
                 };
                 
                 let mangled_name = if generics.is_empty() {
                     struct_name.to_string()
                 } else {
                     let base = mangle_base_name(struct_name);
                     self.mangle_type_name(base, generics)
                 };
                 
                 let simple_lookup_name = mangled_name.clone();
                 let _ = self.get_or_monomorphize_type(&ret_type).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    
                 let struct_type = self.struct_types.get(&simple_lookup_name)
                     .or_else(|| self.enum_types.get(&simple_lookup_name))
                     .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Struct type {} not found for SRET allocation", simple_lookup_name))))?;
                 
                 let size = struct_type.size_of().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("Cannot determine size for SRET struct".to_string())))?;
                 
                 let malloc_fn = self.module.get_function("malloc").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("malloc not found".to_string())))?;
                 let size_i64 = self.builder.build_int_z_extend(size, self.context.i64_type(), "size_i64").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                 let call_malloc = self.builder.build_call(malloc_fn, &[size_i64.into()], "sret_malloc").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                 
                 let raw_ptr = match call_malloc.try_as_basic_value() {
                     inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
                     _ => return Err(CodegenErrorKind::Internal("malloc returned void".to_string()).into()),
                 };
                 
                 let struct_name_str = match &ret_type {
                     Type::Struct(n, _) => n.as_str(),
                     _ => "AnonymousStruct",
                 };
                 let name_global = self.builder.build_global_string_ptr(struct_name_str, "struct_name").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                 let register_fn = self.module.get_function("tl_mem_register_struct_named").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_mem_register_struct_named not found".to_string())))?;
                 
                 let cast_ptr = self.builder.build_pointer_cast(raw_ptr, self.context.ptr_type(inkwell::AddressSpace::default()), "cast_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                 let _ = self.builder.build_call(register_fn, &[cast_ptr.into(), name_global.as_pointer_value().into()], "");
    
                 dest_val = Some(cast_ptr.into());
             }
             
             if let Some(d) = dest_val {
                 compiled_args_vals.push(d.into());
             }
        }


        if let Some(pre_values) = precompiled_args {
             for (val, ty) in pre_values {
                  compiled_args_vals.push(val.into());
                  compiled_args_types.push((val, ty));
             }
        } else {
            for arg in args {
                let (val, ty) = self.compile_expr(arg)?;

                // NOTE: 引数は Callee に「借用」として渡される (CLEANUP_NONE)。
                // Callee は関数終了時に引数を release しない。
                // したがって caller 側で retain (RC+1) する必要はない。
                // 以前ここに emit_retain があったが、対応する release が存在せずリークの原因だった。

                compiled_args_vals.push(val.into());
                compiled_args_types.push((val, ty));
            }
        }
        if let Some(_block) = self.builder.get_insert_block() {
        } else {
        }
        if self.builder.get_insert_block().is_none() {
            return Err(TlError::from(CodegenErrorKind::Internal(format!("INTERNAL ERROR: Builder has no insert block when calling {}", final_resolved_name))));
        }

        for (_i, _arg) in compiled_args_vals.iter().enumerate() {
        }
        let call_name = if ret_type == Type::Void { "" } else { "call_tmp" };

        let call = self
            .builder
            .build_call(func, &compiled_args_vals, call_name)
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        // FIX: Free temporary arguments
        for (i, (_, ty)) in compiled_args_types.into_iter().enumerate() {
            let arg_expr = &args[i];
            if self.is_safe_to_free(arg_expr, &ty) {
                // self.emit_recursive_free(val, &ty, super::CLEANUP_FULL)?;
            }
        }

        if let Some(d) = dest_val {
             let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
             let loaded = self.builder.build_load(ptr_ty, d.into_pointer_value(), "sret_loaded").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
             return Ok((loaded, ret_type));
        }

        let res = match call.try_as_basic_value() {
            ValueKind::Basic(_) => self.check_tensor_result(call, "call_error")?,
            _ => {
                if ret_type == Type::Void {
                    self.context.i64_type().const_int(0, false).into()
                } else {
                    // For debug
                    return Ok((
                        self.context.i64_type().const_int(0, false).into(),
                        Type::Void,
                    ));
                }
            }
        };

        // V4.5: Caller-Side Registration
        // When a function returns a tensor, it is "promoted" (floating).
        // The caller must register it to its own scope immediately.
        // We only do this for USER functions (which have basic blocks). 
        // Builtins/FFI functions (blocks=0) register internally to the current scope.
        if matches!(ret_type, Type::Tensor(_, _)) {
            if func.count_basic_blocks() > 0 {
                self.emit_register_tensor(res, &ret_type)?;
            }
        }

        // Normalize: If return type is Struct("Tensor"), treat it as Type::Tensor
        // (Parser may resolve `-> Tensor` as Struct("Tensor"))
        let ret_type = if let Type::Struct(ref name, _) = ret_type {
            if name == "Tensor" {
                Type::Tensor(Box::new(Type::F32), 0)
            } else {
                ret_type
            }
        } else {
            ret_type
        };

        // For Struct returns (SRET deprecated logic, assuming by-value or pointer return if not SRET)
        Ok((res, ret_type))
    }
}

// Param::zero_grad() — clear all gradients
// --- Reshape Unevaluated Implementation ---

impl<'ctx> CodeGenerator<'ctx> {
    /// Compile an enum variant constructor that was parsed as StaticMethodCall.
    /// This handles cases like Entry_i64_i64::Empty where the mangled enum name
    /// and variant name come through as a static method call.
    fn compile_enum_variant_as_static_method_call(
        &mut self,
        enum_name: &str,
        variant_name: &str,
        args: &[crate::compiler::ast::Expr],
        variant_idx: usize,
        enum_def: &crate::compiler::ast::EnumDef,
        target_type: &crate::compiler::ast::Type,
    ) -> Result<(inkwell::values::BasicValueEnum<'ctx>, crate::compiler::ast::Type), TlError> {
        use crate::compiler::ast::{Type, VariantKind};
        
        let variant_def = &enum_def.variants[variant_idx];
        let field_count = match &variant_def.kind {
            VariantKind::Unit => 0,
            VariantKind::Tuple(t) => t.len(),
            VariantKind::Struct(f) => f.len(),
            VariantKind::Array(_, size) => *size,
        };
        if args.len() != field_count {
            return Err(TlError::from(CodegenErrorKind::Internal(format!("Enum variant {}::{} expects {} args, got {}", enum_name, variant_name, field_count, args.len()))));
        }
        
        // Use enum_def.name (monomorphized name like "Option<i64>") instead of enum_name (might be base name "Option")
        let actual_enum_name = &enum_def.name;
        let enum_ty = if let Some(ty) = self.enum_types.get(actual_enum_name) {
            *ty
        } else if let Some(ty) = self.enum_types.get(enum_name) {
            *ty
        } else {
            // Try to compile on-demand
            self.compile_enum_defs(&[enum_def.clone()])?;
            *self.enum_types.get(actual_enum_name)
                .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Enum type {} not found (tried {} and {})", enum_name, actual_enum_name, enum_name))))?
        };
        
        // Allocate memory for enum
        let size_ptr = unsafe {
            self.builder.build_gep(
                enum_ty,
                self.context.ptr_type(inkwell::AddressSpace::default()).const_null(),
                &[self.context.i64_type().const_int(1, false)],
                "size_ptr",
            ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
        };
        let size = self.builder
            .build_ptr_to_int(size_ptr, self.context.i64_type(), "size")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        let malloc_fn = self.module.get_function("malloc").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("malloc not found".to_string())))?;
        let alloca = match self.builder
            .build_call(malloc_fn, &[size.into()], &format!("enum_{}", enum_name))
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
            .try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v.into_pointer_value(),
                _ => return Err(CodegenErrorKind::Internal("malloc returned void".to_string()).into()),
            };
        
        // Store Tag
        let tag_ptr = self.builder.build_struct_gep(enum_ty, alloca, 0, "tag_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        let tag_val = variant_idx as u64;
        self.builder.build_store(tag_ptr, self.context.i32_type().const_int(tag_val, false)).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        
        // Store Payload if any
        if !args.is_empty() {
            match &variant_def.kind {
                VariantKind::Tuple(types) => {
                    let payload_ptr_raw = self.builder
                        .build_struct_gep(enum_ty, alloca, 1, "payload_ptr_raw")
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    
                    let mut field_types_llvm: Vec<inkwell::types::BasicTypeEnum> = vec![];
                    for ty in types {
                        let llvm_ty = self.get_llvm_type(ty)?;
                        field_types_llvm.push(llvm_ty);
                    }
                    let variant_struct_ty = self.context.struct_type(&field_types_llvm, false);
                    
                    let payload_ptr = self.builder.build_pointer_cast(
                        payload_ptr_raw,
                        self.context.ptr_type(inkwell::AddressSpace::default()),
                        "payload_cast"
                    ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    
                    for (idx, (arg, _f_ty)) in args.iter().zip(types.iter()).enumerate() {
                        let (val, _) = self.compile_expr(arg)?;
                        let f_ptr = self.builder.build_struct_gep(variant_struct_ty, payload_ptr, idx as u32, "field_ptr")
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        self.builder.build_store(f_ptr, val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    }
                }
                VariantKind::Struct(fields) => {
                    let payload_ptr_raw = self.builder
                        .build_struct_gep(enum_ty, alloca, 1, "payload_ptr_raw")
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    
                    let mut field_types_llvm: Vec<inkwell::types::BasicTypeEnum> = vec![];
                    for (_, ty) in fields {
                        let llvm_ty = self.get_llvm_type(ty)?;
                        field_types_llvm.push(llvm_ty);
                    }
                    let variant_struct_ty = self.context.struct_type(&field_types_llvm, false);
                    
                    let payload_ptr = self.builder.build_pointer_cast(
                        payload_ptr_raw,
                        self.context.ptr_type(inkwell::AddressSpace::default()),
                        "payload_cast"
                    ).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    
                    for (idx, arg) in args.iter().enumerate() {
                        let (val, _) = self.compile_expr(arg)?;
                        let f_ptr = self.builder.build_struct_gep(variant_struct_ty, payload_ptr, idx as u32, "field_ptr")
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        self.builder.build_store(f_ptr, val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    }
                }
                _ => {}
            }
        }
        
        let type_args = match target_type {
            Type::Struct(_, a) | Type::Enum(_, a) => a.clone(),
            _ => vec![],
        };
        Ok((alloca.into(), Type::Enum(enum_name.to_string(), type_args)))
    }

    fn compile_mutex_closure_method(
        &mut self,
        obj_val: inkwell::values::BasicValueEnum<'ctx>,
        _obj_ty: &Type,
        _elem_ty: &Type,
        method: &str,
        closure_expr: &crate::compiler::ast::Expr,
    ) -> Result<(inkwell::values::BasicValueEnum<'ctx>, Type), TlError> {
        // Compile closure to {fn_ptr, env_ptr} using compile_expr
        let (closure_val, closure_ty) = self.compile_expr(closure_expr)?;

        let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
        
        // closure_val is a StructValue { fn_ptr, env_ptr }
        let closure_struct = closure_val.into_struct_value();
        let actual_fn_ptr = self.builder.build_extract_value(closure_struct, 0, "fn_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_pointer_value();
        let actual_env_ptr = self.builder.build_extract_value(closure_struct, 1, "env_ptr").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_pointer_value();
        
        // Generate a Trampoline function: void trampoline(env_ptr, arg_ptr, out_ptr)
        let void_ty = self.context.void_type();
        let trampoline_fn_type = void_ty.fn_type(&[ptr_type.into(), ptr_type.into(), ptr_type.into()], false);
        static TRAMPOLINE_ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let tid = TRAMPOLINE_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let trampoline_name = format!("__tl_mutex_trampoline_{}", tid);
        let trampoline_fn = self.module.add_function(&trampoline_name, trampoline_fn_type, None);
        
        // Build Trampoline body
        let saved_block = self.builder.get_insert_block();
        let tramp_bb = self.context.append_basic_block(trampoline_fn, "entry");
        self.builder.position_at_end(tramp_bb);
        
        let t_env_ptr = trampoline_fn.get_nth_param(0).ok_or_else(|| TlError::from(CodegenErrorKind::Internal("missing function parameter".to_string())))?.into_pointer_value();
        let t_arg_ptr = trampoline_fn.get_nth_param(1).ok_or_else(|| TlError::from(CodegenErrorKind::Internal("missing function parameter".to_string())))?.into_pointer_value();
        let t_out_ptr = trampoline_fn.get_nth_param(2).ok_or_else(|| TlError::from(CodegenErrorKind::Internal("missing function parameter".to_string())))?.into_pointer_value();
        
        // Get the arg according to elem_ty
        let llvm_elem_ty = self.get_llvm_type(_elem_ty)?;
        let t_arg_val = if llvm_elem_ty.is_pointer_type() {
            // For structs/tensors, arg is a pointer, but in FFI we receive a pointer to the pointer?
            // Actually, `guard.data_ptr` IS the pointer to the heap object (the struct itself).
            // So if `_elem_ty` is Struct, the closure expects `i8*`.
            // The FFI passes `guard.data_ptr` which is `i8*`! So we just use `t_arg_ptr` directly!
            // Wait, t_arg_ptr IS the `data_ptr` value inside Mutex state. 
            // If T is I64, `data_ptr` points to an 8-byte allocation holding the I64.
            // So we always LOAD from t_arg_ptr for primitive types, and pass directly for pointer types?
            if matches!(_elem_ty, Type::Tensor(_,_) | Type::Struct(_,_) | Type::Tuple(_) | Type::String(_)) {
                t_arg_ptr.into()
            } else {
                self.builder.build_load(llvm_elem_ty, t_arg_ptr, "arg_loaded").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
            }
        } else {
            self.builder.build_load(llvm_elem_ty, t_arg_ptr, "arg_loaded").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
        };
        
        // Ret type
        let ret_ty = if let Type::Fn(_, ref ret) = closure_ty { (**ret).clone() } else { Type::Void };
        let llvm_ret_ty = self.get_llvm_type(&ret_ty)?;
        
        // Call actual closure
        // Re-construct the FnType of the closure to call it indirectly
        let actual_fn_type = if ret_ty == Type::Void {
            void_ty.fn_type(&[ptr_type.into(), llvm_elem_ty.into()], false)
        } else {
            llvm_ret_ty.fn_type(&[ptr_type.into(), llvm_elem_ty.into()], false)
        };
        
        let call_res = self.builder.build_indirect_call(actual_fn_type, actual_fn_ptr, &[t_env_ptr.into(), t_arg_val.into()], "call_res").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        
        if let inkwell::values::ValueKind::Basic(res_val) = call_res.try_as_basic_value() {
            // Store res_val into t_out_ptr
            if llvm_ret_ty.is_pointer_type() {
                if matches!(ret_ty, Type::Tensor(_,_) | Type::Struct(_,_) | Type::Tuple(_) | Type::String(_)) {
                    let out_ptr_cast = self.builder.build_pointer_cast(t_out_ptr, ptr_type, "out_ptr_cast").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    self.builder.build_store(out_ptr_cast, res_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                } else {
                    let out_ptr_cast = self.builder.build_pointer_cast(t_out_ptr, ptr_type, "out_ptr_cast").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    self.builder.build_store(out_ptr_cast, res_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                }
            } else {
                let out_ptr_cast = self.builder.build_pointer_cast(t_out_ptr, ptr_type, "out_ptr_cast").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                self.builder.build_store(out_ptr_cast, res_val).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            }
        }
        
        self.builder.build_return(None).map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        
        // Restore builder
        if let Some(sb) = saved_block {
             self.builder.position_at_end(sb);
        }
        
        let fn_ptr = trampoline_fn.as_global_value().as_pointer_value();
        let env_ptr = actual_env_ptr;
        
        let m_struct_ty = self.context.struct_type(&[self.context.i64_type().into()], false);
        let id_gep = self.builder.build_struct_gep(m_struct_ty, obj_val.into_pointer_value(), 0, "id_gep").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        let id_val = self.builder.build_load(self.context.i64_type(), id_gep, "id_val").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        
        if method == "modify" {
            let fn_val = self.module.get_function("tl_mutex_modify").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_mutex_modify not found".to_string())))?;
            self.builder.build_call(fn_val, &[id_val.into(), fn_ptr.into(), env_ptr.into()], "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            Ok((self.context.i64_type().const_zero().into(), Type::Void))
        } else {
            let fn_val = self.module.get_function("tl_mutex_read").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_mutex_read not found".to_string())))?;
            
            let ret_ty = if let Type::Fn(_, ref ret) = closure_ty {
                (**ret).clone()
            } else {
                Type::Void
            };
            
            let llvm_ret_ty = self.get_llvm_type(&ret_ty)?;
            let _size = if llvm_ret_ty.is_sized() { llvm_ret_ty.size_of().ok_or_else(|| TlError::from(CodegenErrorKind::Internal("type has no size".to_string())))? } else { self.context.i64_type().const_zero() };
            // Wait, we need to pass a pointer to output. The FFI copies `out_size` bytes into `out_ptr`.
            // Wait, does our FFI require `out_size` in `read`?
            // Signature of tl_mutex_read: void tl_mutex_read(int64_t id, MutexAccessorFn accessor, void* env_ptr, void* out_data);
            // It does not take out_size. It just takes `out_data` pointer to cast back.
            let ret_ptr = self.builder.build_alloca(llvm_ret_ty, "ret_val").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let ret_ptr_cast = self.builder.build_pointer_cast(ret_ptr, ptr_type, "ret_ptr_cast").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            
            self.builder.build_call(fn_val, &[id_val.into(), fn_ptr.into(), env_ptr.into(), ret_ptr_cast.into()], "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            
            let res = self.builder.build_load(llvm_ret_ty, ret_ptr, "res").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            Ok((res, ret_ty))
        }
    }
}
