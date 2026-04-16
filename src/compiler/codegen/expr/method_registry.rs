//! codegen/expr/method_registry.rs
//!
//! メソッドレジストリ: register_all_methods, emit_trait_object_upcast 等。
//! CodeGenerator にインスタンスメソッド・静的メソッドを登録する。
use crate::compiler::error::{TlError, CodegenErrorKind};

use super::tensor_methods::*;
use super::builtin_fns::*;
use super::types::*;
use crate::compiler::ast::*;
use crate::compiler::codegen::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    pub(crate) fn emit_trait_object_upcast(
        &mut self,
        val: inkwell::values::BasicValueEnum<'ctx>,
        struct_name: &str,
        trait_name: &str,
    ) -> Result<inkwell::values::BasicValueEnum<'ctx>, TlError> {
        let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
        let fat_ptr_type = self.context.struct_type(&[ptr_type.into(), ptr_type.into()], false);

        let data_ptr = if val.is_pointer_value() {
            self.builder.build_pointer_cast(val.into_pointer_value(), ptr_type, "trait_data_cast").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
        } else {
            return Err(TlError::from(CodegenErrorKind::Internal("Expected pointer value for upcast".to_string())));
        };

        let vtable_name = format!("vtable_{}_for_{}", trait_name, struct_name);
        let vtable_global = if let Some(global) = self.module.get_global(&vtable_name) {
            global
        } else {
            let trait_def = self.trait_defs.get(trait_name).ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Trait {} not found in registry", trait_name))))?.clone();
            let vtable_ty = ptr_type.array_type(trait_def.methods.len() as u32);
            let global = self.module.add_global(vtable_ty, Some(inkwell::AddressSpace::default()), &vtable_name);
            global.set_linkage(inkwell::module::Linkage::Internal);
            global.set_constant(true);

            let mut fn_ptrs = Vec::new();
            for m in &trait_def.methods {
                let mangled_name = format!("tl_{}_{}", struct_name, m.name);
                let fn_val = self.module.get_function(&mangled_name).ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Missing implementation of {} for trait {} in struct {}: looking for {}", m.name, trait_name, struct_name, mangled_name))))?;
                fn_ptrs.push(fn_val.as_global_value().as_pointer_value());
            }
            global.set_initializer(&ptr_type.const_array(&fn_ptrs));
            global
        };

        let vtable_ptr = vtable_global.as_pointer_value();
        let mut fat_ptr_val = fat_ptr_type.const_zero();
        fat_ptr_val = self.builder.build_insert_value(fat_ptr_val, data_ptr, 0, "fat_d").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_struct_value();
        fat_ptr_val = self.builder.build_insert_value(fat_ptr_val, vtable_ptr, 1, "fat_v").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into_struct_value();
        Ok(fat_ptr_val.into())
    }
    pub(super) fn substitute_type_generic(&self, ty: &Type, generics: &[String], args: &[Type]) -> Type {
        match ty {
            Type::Struct(name, inner_args) => {
                 if let Some(idx) = generics.iter().position(|g| g == name) {
                     return args[idx].clone();
                 }
                // If the struct is Generic, we must substitute
                 Type::Struct(name.clone(), inner_args.iter().map(|t| self.substitute_type_generic(t, generics, args)).collect())
            }
            Type::Enum(name, inner_args) => {
                 if let Some(idx) = generics.iter().position(|g| g == name) {
                     return args[idx].clone();
                 }
                 Type::Enum(name.clone(), inner_args.iter().map(|t| self.substitute_type_generic(t, generics, args)).collect())
            }
            Type::Path(segments, inner_args) => {
                 // Check if single-segment path is a type parameter
                 if segments.len() == 1 {
                     if let Some(idx) = generics.iter().position(|g| g == &segments[0]) {
                         return args[idx].clone();
                     }
                 }
                 Type::Path(segments.clone(), inner_args.iter().map(|t| self.substitute_type_generic(t, generics, args)).collect())
            }
            Type::Tensor(inner, rank) => Type::Tensor(Box::new(self.substitute_type_generic(inner, generics, args)), *rank),
            Type::Tuple(types) => Type::Tuple(types.iter().map(|t| self.substitute_type_generic(t, generics, args)).collect()),
            Type::Ptr(inner) => Type::Ptr(Box::new(self.substitute_type_generic(inner, generics, args))),

             _ => ty.clone(),
        }
    }


    pub(crate) fn register_all_methods(&mut self) {
        // --- Tensor Instance Methods ---
        let mut tensor_methods = InstanceMethodManager::new();
        tensor_methods.register_eval("get", compile_tensor_get);
        tensor_methods.register_eval("backward", compile_tensor_backward);
        tensor_methods.register_eval("clone", compile_tensor_clone);
        tensor_methods.register_eval("detach", compile_tensor_detach);
        tensor_methods.register_eval("grad", compile_tensor_grad);
        tensor_methods.register_eval("contiguous", compile_tensor_contiguous);
        tensor_methods.register_eval("save", compile_tensor_save);
        tensor_methods.register_uneval("reshape", compile_tensor_reshape_uneval);
        tensor_methods.register_eval("sum", compile_tensor_sum);
        tensor_methods.register_eval("slice", compile_tensor_slice2);
        tensor_methods.register_eval("to", compile_tensor_to);
        tensor_methods.register_eval("to_device", compile_tensor_to);
        tensor_methods.register_eval("add_assign", compile_tensor_add_assign);
        tensor_methods.register_eval("sub_assign", compile_tensor_sub_assign);
        tensor_methods.register_eval("mul_assign", compile_tensor_mul_assign);
        tensor_methods.register_eval("div_assign", compile_tensor_div_assign);
        tensor_methods.register_eval("transpose", compile_tensor_transpose);
        tensor_methods.register_eval("permute", compile_tensor_transpose); // permute aliases transpose logic for now
        tensor_methods.register_eval("pow", compile_tensor_pow);
        tensor_methods.register_eval("get", compile_tensor_get);

        self.instance_methods
            .insert("Tensor".to_string(), tensor_methods);

        // --- F32 Instance Methods ---
        let mut f32_methods = InstanceMethodManager::new();
        f32_methods.register_eval("abs", compile_f32_abs);
        f32_methods.register_eval("acos", compile_f32_acos);
        f32_methods.register_eval("acosh", compile_f32_acosh);
        f32_methods.register_eval("asin", compile_f32_asin);
        f32_methods.register_eval("asinh", compile_f32_asinh);
        f32_methods.register_eval("atan", compile_f32_atan);
        f32_methods.register_eval("atan2", compile_f32_atan2);
        f32_methods.register_eval("atanh", compile_f32_atanh);
        f32_methods.register_eval("cbrt", compile_f32_cbrt);
        f32_methods.register_eval("ceil", compile_f32_ceil);
        f32_methods.register_eval("copysign", compile_f32_copysign);
        f32_methods.register_eval("cos", compile_f32_cos);
        f32_methods.register_eval("cosh", compile_f32_cosh);
        f32_methods.register_eval("exp", compile_f32_exp);
        f32_methods.register_eval("exp2", compile_f32_exp2);
        f32_methods.register_eval("exp_m1", compile_f32_exp_m1);
        f32_methods.register_eval("floor", compile_f32_floor);
        f32_methods.register_eval("fract", compile_f32_fract);
        f32_methods.register_eval("hypot", compile_f32_hypot);
        f32_methods.register_eval("ln", compile_f32_ln);
        f32_methods.register_eval("ln_1p", compile_f32_ln_1p);
        f32_methods.register_eval("log", compile_f32_log);
        f32_methods.register_eval("log10", compile_f32_log10);
        f32_methods.register_eval("log2", compile_f32_log2);
        f32_methods.register_eval("powf", compile_f32_powf);
        f32_methods.register_eval("pow", compile_f32_pow); // Alias
        f32_methods.register_eval("powi", compile_f32_powi);
        f32_methods.register_eval("recip", compile_f32_recip);
        f32_methods.register_eval("round", compile_f32_round);
        f32_methods.register_eval("signum", compile_f32_signum);
        f32_methods.register_eval("sin", compile_f32_sin);
        f32_methods.register_eval("sinh", compile_f32_sinh);
        f32_methods.register_eval("sqrt", compile_f32_sqrt);
        f32_methods.register_eval("tan", compile_f32_tan);
        f32_methods.register_eval("tanh", compile_f32_tanh);
        f32_methods.register_eval("to_degrees", compile_f32_to_degrees);
        f32_methods.register_eval("to_radians", compile_f32_to_radians);
        f32_methods.register_eval("trunc", compile_f32_trunc);
        self.instance_methods.insert("F32".to_string(), f32_methods);

        // --- F64 Instance Methods ---
        let mut f64_methods = InstanceMethodManager::new();
        f64_methods.register_eval("abs", compile_f64_abs);
        f64_methods.register_eval("acos", compile_f64_acos);
        f64_methods.register_eval("acosh", compile_f64_acosh);
        f64_methods.register_eval("asin", compile_f64_asin);
        f64_methods.register_eval("asinh", compile_f64_asinh);
        f64_methods.register_eval("atan", compile_f64_atan);
        f64_methods.register_eval("atan2", compile_f64_atan2);
        f64_methods.register_eval("atanh", compile_f64_atanh);
        f64_methods.register_eval("cbrt", compile_f64_cbrt);
        f64_methods.register_eval("ceil", compile_f64_ceil);
        f64_methods.register_eval("copysign", compile_f64_copysign);
        f64_methods.register_eval("cos", compile_f64_cos);
        f64_methods.register_eval("cosh", compile_f64_cosh);
        f64_methods.register_eval("exp", compile_f64_exp);
        f64_methods.register_eval("exp2", compile_f64_exp2);
        f64_methods.register_eval("exp_m1", compile_f64_exp_m1);
        f64_methods.register_eval("floor", compile_f64_floor);
        f64_methods.register_eval("fract", compile_f64_fract);
        f64_methods.register_eval("hypot", compile_f64_hypot);
        f64_methods.register_eval("ln", compile_f64_ln);
        f64_methods.register_eval("ln_1p", compile_f64_ln_1p);
        f64_methods.register_eval("log", compile_f64_log);
        f64_methods.register_eval("log10", compile_f64_log10);
        f64_methods.register_eval("log2", compile_f64_log2);
        f64_methods.register_eval("powf", compile_f64_powf);
        f64_methods.register_eval("pow", compile_f64_pow); // Alias
        f64_methods.register_eval("powi", compile_f64_powi);
        f64_methods.register_eval("recip", compile_f64_recip);
        f64_methods.register_eval("round", compile_f64_round);
        f64_methods.register_eval("signum", compile_f64_signum);
        f64_methods.register_eval("sin", compile_f64_sin);
        f64_methods.register_eval("sinh", compile_f64_sinh);
        f64_methods.register_eval("sqrt", compile_f64_sqrt);
        f64_methods.register_eval("tan", compile_f64_tan);
        f64_methods.register_eval("tanh", compile_f64_tanh);
        f64_methods.register_eval("to_degrees", compile_f64_to_degrees);
        f64_methods.register_eval("to_radians", compile_f64_to_radians);
        f64_methods.register_eval("trunc", compile_f64_trunc);
        self.instance_methods.insert("F64".to_string(), f64_methods);

        // --- I64 Instance Methods ---
        let mut i64_methods = InstanceMethodManager::new();
        i64_methods.register_eval("abs", compile_i64_abs);
        i64_methods.register_eval("signum", compile_i64_signum);
        i64_methods.register_eval("pow", compile_i64_pow);
        i64_methods.register_eval("div_euclid", compile_i64_div_euclid);
        i64_methods.register_eval("rem_euclid", compile_i64_rem_euclid);
        i64_methods.register_eval("is_positive", compile_i64_is_positive);
        i64_methods.register_eval("is_negative", compile_i64_is_negative);
        self.instance_methods.insert("I64".to_string(), i64_methods);

        // --- I32 Instance Methods ---
        let mut i32_methods = InstanceMethodManager::new();
        i32_methods.register_eval("abs", compile_i32_abs);
        i32_methods.register_eval("signum", compile_i32_signum);
        i32_methods.register_eval("pow", compile_i32_pow);
        i32_methods.register_eval("div_euclid", compile_i32_div_euclid);
        i32_methods.register_eval("rem_euclid", compile_i32_rem_euclid);
        i32_methods.register_eval("is_positive", compile_i32_is_positive);
        i32_methods.register_eval("is_negative", compile_i32_is_negative);
        self.instance_methods.insert("I32".to_string(), i32_methods);



        // --- VarBuilder Static Methods ---
        let mut varbuilder_static = StaticMethodManager::new();
        varbuilder_static.register_uneval("get", compile_varbuilder_get_static);
        self.static_methods
            .insert("VarBuilder".to_string(), varbuilder_static);

        // --- Param Static Methods ---
        let mut param_static = StaticMethodManager::new();
        param_static.register_eval("save_all", compile_save_all_params);
        param_static.register_eval("load_all", compile_load_all_params);
        param_static.register_eval("save", compile_save_weights);
        param_static.register_eval("load", compile_load_weights);
        param_static.register_eval("add", compile_add_parameter);
        param_static.register_eval("register", compile_parameter);
        param_static.register_eval("update_all", compile_update_all_params);
        param_static.register_eval("zero_grad", compile_clear_grads);
        param_static.register_eval("register_modules", compile_register_modules);
        param_static.register_uneval("checkpoint", compile_checkpoint);
        param_static.register_uneval("set_device", compile_set_device);
        self.static_methods
            .insert("Param".to_string(), param_static);

    }

}
