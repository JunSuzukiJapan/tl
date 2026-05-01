//! codegen/expr/builtin_fns.rs
//!
//! 組み込み関数のコンパイル実装。
//! IO (print/println/format), 文字列操作, 引数アクセス,
//! パラメータ管理, VarBuilder, 型変換ヘルパー等。
use crate::compiler::error::{TlError, CodegenErrorKind};

use inkwell::values::*;

use crate::compiler::ast::*;
use crate::compiler::codegen::CodeGenerator;



#[allow(deprecated)]
pub(super) fn compile_varbuilder_get_static<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    // Arg 0: Name (String)
    let (name_val, name_ty) = codegen.compile_expr(&args[0])?;
    let name_ptr = if let Type::String(_) = name_ty {
        // String is { i64 ptr } or similar. extraction needed.
        // Assuming String struct layout: Field 0 is ptr (i64).
        let ptr_to_struct = name_val.into_pointer_value();
        let i64_ptr_ty = codegen.context.i64_type().ptr_type(inkwell::AddressSpace::default());
        let ptr_to_first_field = codegen
            .builder
            .build_pointer_cast(ptr_to_struct, i64_ptr_ty, "str_ptr_cast")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        let str_addr_i64 = codegen
            .builder
            .build_load(codegen.context.i64_type(), ptr_to_first_field, "str_addr")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
            .into_int_value();
        let i8_ptr_ty = codegen.context.i8_type().ptr_type(inkwell::AddressSpace::default());
        codegen
            .builder
            .build_int_to_ptr(str_addr_i64, i8_ptr_ty, "cstr_ptr")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
    } else {
         return Err(TlError::from(CodegenErrorKind::Internal("VarBuilder::get name must be String".to_string())));
    };

    // Arg 1..: Shape
    let i64_type = codegen.context.i64_type();

    let (rank, shape_alloca) = if args.len() == 2 {
        match &args[1].inner {
            ExprKind::TensorLiteral(elements) | ExprKind::TensorConstLiteral(elements) => {
                let rank = elements.len();
                let shape_array_type = i64_type.array_type(rank as u32);
                let shape_alloca = codegen
                    .builder
                    .build_alloca(shape_array_type, "shape_arr")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                for (i, elem) in elements.iter().enumerate() {
                    let (val, ty) = codegen.compile_expr(elem)?;
                    let i_val = match ty {
                        Type::I64 => val.into_int_value(),
                        Type::I32 => codegen
                            .builder
                            .build_int_z_extend(val.into_int_value(), i64_type, "dim_zext")
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?,
                        _ => {
                            return Err(TlError::from(CodegenErrorKind::Internal(format!(
                                "VarBuilder::get expects integer dimensions, got {:?}",
                                ty
                            ))))
                        }
                    };
                    let ptr = unsafe {
                        codegen
                            .builder
                            .build_in_bounds_gep(
                                shape_array_type,
                                shape_alloca,
                                &[
                                    i64_type.const_int(0, false),
                                    i64_type.const_int(i as u64, false),
                                ],
                                "tmp",
                            )
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                    };
                    codegen
                        .builder
                        .build_store(ptr, i_val)
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                }
                (rank, shape_alloca)
            }
            _ => {
                let rank = args.len() - 1;
                let shape_array_type = i64_type.array_type(rank as u32);
                let shape_alloca = codegen
                    .builder
                    .build_alloca(shape_array_type, "shape_arr")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

                for (i, arg) in args[1..].iter().enumerate() {
                    let (val, ty) = codegen.compile_expr(arg)?;
                    let i_val = if ty == Type::I64 {
                        val.into_int_value()
                    } else {
                        return Err(TlError::from(CodegenErrorKind::Internal(format!(
                            "VarBuilder::get expects integer dimensions, got {:?}",
                            ty
                        ))));
                    };
                    let ptr = unsafe {
                        codegen
                            .builder
                            .build_in_bounds_gep(
                                shape_array_type,
                                shape_alloca,
                                &[
                                    i64_type.const_int(0, false),
                                    i64_type.const_int(i as u64, false),
                                ],
                                "tmp",
                            )
                            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
                    };
                    codegen
                        .builder
                        .build_store(ptr, i_val)
                        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                }
                (rank, shape_alloca)
            }
        }
    } else {
        let rank = args.len() - 1;
        let shape_array_type = i64_type.array_type(rank as u32);
        let shape_alloca = codegen
            .builder
            .build_alloca(shape_array_type, "shape_arr")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        for (i, arg) in args[1..].iter().enumerate() {
            let (val, ty) = codegen.compile_expr(arg)?;
            let i_val = if ty == Type::I64 {
                val.into_int_value()
            } else {
                return Err(TlError::from(CodegenErrorKind::Internal(format!(
                    "VarBuilder::get expects integer dimensions, got {:?}",
                    ty
                ))));
            };
            let ptr = unsafe {
                codegen
                    .builder
                    .build_in_bounds_gep(
                                shape_array_type,
                                shape_alloca,
                                &[
                                    i64_type.const_int(0, false),
                                    i64_type.const_int(i as u64, false),
                                ],
                                "tmp",
                            )
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
            };
            codegen
                .builder
                .build_store(ptr, i_val)
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        }
        (rank, shape_alloca)
    };

    let f = codegen
        .module
        .get_function("tl_varbuilder_get")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_varbuilder_get not found".to_string())))?;
    let call = codegen
        .builder
        .build_call(
            f,
            &[
                name_ptr.into(),
                i64_type.const_int(rank as u64, false).into(),
                shape_alloca.into(),
            ],
            "vb_get_res",
        )
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

    let v = codegen.check_tensor_result(call, "vb_get_error")?;
    let result_ty = Type::Tensor(Box::new(Type::F32), rank);
    codegen.emit_register_tensor(v, &result_ty)?;
    Ok((v, result_ty))
}

#[allow(deprecated)]
pub(super) fn compile_varbuilder_grad_static<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 {
        return Err(TlError::from(CodegenErrorKind::Internal(
            "VarBuilder::grad requires 1 argument (name: String)".to_string(),
        )));
    }

    // Arg 0: Name (String) → *const c_char
    let (name_val, name_ty) = codegen.compile_expr(&args[0])?;
    let name_ptr = if let Type::String(_) = name_ty {
        let ptr_to_struct = name_val.into_pointer_value();
        let i64_ptr_ty = codegen.context.i64_type().ptr_type(inkwell::AddressSpace::default());
        let ptr_to_first_field = codegen
            .builder
            .build_pointer_cast(ptr_to_struct, i64_ptr_ty, "str_ptr_cast")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        let str_addr_i64 = codegen
            .builder
            .build_load(codegen.context.i64_type(), ptr_to_first_field, "str_addr")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
            .into_int_value();
        let i8_ptr_ty = codegen.context.i8_type().ptr_type(inkwell::AddressSpace::default());
        codegen
            .builder
            .build_int_to_ptr(str_addr_i64, i8_ptr_ty, "cstr_ptr")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
    } else {
        return Err(TlError::from(CodegenErrorKind::Internal(
            "VarBuilder::grad name must be String".to_string(),
        )));
    };

    let f = codegen
        .module
        .get_function("tl_varbuilder_grad")
        .ok_or_else(|| {
            TlError::from(CodegenErrorKind::Internal(
                "tl_varbuilder_grad not found".to_string(),
            ))
        })?;
    let call = codegen
        .builder
        .build_call(f, &[name_ptr.into()], "vb_grad_res")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

    let v = codegen.check_tensor_result(call, "vb_grad_error")?;
    // 勾配テンソルは元のテンソルと同じ型 (Tensor<f32, N>) だが rank は不明
    // 安全のため rank=0 (dynamic) とする
    let result_ty = Type::Tensor(Box::new(Type::F32), 0);
    codegen.emit_register_tensor(v, &result_ty)?;
    Ok((v, result_ty))
}

pub(super) fn compile_set_device<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 {
        return Err(TlError::from(CodegenErrorKind::Internal("set_device expects 1 argument".to_string())));
    }
    let (arg_val, arg_ty) = codegen.compile_expr(&args[0])?;

    // Expect Device Enum
    let is_device_enum = match &arg_ty {
        Type::Enum(e, _) | Type::Struct(e, _) if e == "Device" => true,
        _ => false,
    };

    if !is_device_enum {
        return Err(TlError::from(CodegenErrorKind::Internal(format!(
            "set_device argument must be a Device enum, found {:?}",
            arg_ty
        ))));
    }

    let fn_val = codegen
        .module
        .get_function("tl_set_device")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_set_device not found".to_string())))?;

    // Argument is pointer to Device enum (which is opaque* in LLVM)
    let arg_ptr = match arg_val {
        BasicValueEnum::PointerValue(p) => p,
        _ => return Err(CodegenErrorKind::Internal("Expected pointer to Device enum".to_string()).into()),
    };

    codegen
        .builder
        .build_call(fn_val, &[arg_ptr.into()], "")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

pub(super) fn compile_checkpoint<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 2 {
        return Err(TlError::from(CodegenErrorKind::Internal("checkpoint requires 2 arguments: (method_ref, input)".to_string())));
    }

    // Inline expansion: Param::checkpoint(obj.method, input) → obj.method(input)
    // Instead of going through the runtime (which has JIT ABI issues),
    // we directly compile the method call using the normal codegen path.
    if let ExprKind::FieldAccess(obj_expr, method_name) = &args[0].inner {
        // Compile input as the method argument
        let input_args = &args[1..2];

        // Use the standard method call compilation path
        let (val, ty) = codegen.compile_method_call(obj_expr, method_name, input_args)?;
        Ok((val, ty))
    } else {
        Err(TlError::from(CodegenErrorKind::Internal("checkpoint first argument must be 'obj.method'".to_string())))
    }
}

pub(super) fn compile_print_common<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    is_newline: bool,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    // Shared name arg for error message (not passed here but can infer)
    if args.len() != 1 {
        return Err(TlError::from(CodegenErrorKind::Internal("print/println requires 1 argument".to_string())));
    }
    // Check type of arg
    let (arg_val, arg_type) = &args[0];

    match arg_type {
        Type::I64 => {
            let fn_name = if is_newline {
                "tl_print_i64"
            } else {
                "tl_display_i64"
            };
            let fn_val = codegen.module.get_function(fn_name)
                .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("{} not found", fn_name))))?;
            codegen
                .builder
                .build_call(fn_val, &[(*arg_val).into()], "print_call")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        }
        Type::Char(_) => {
             let fn_name = if is_newline {
                 "tl_print_char"
             } else {
                 "tl_display_char"
             }; // Char is u8 internally (from char_at)
             // Check if arg is Int(8/32/64) or Ptr? generic is usually Any/U64
             // char_at returns Char which usually maps to i8/u8. 
             // If arg_val is i8, we might need cast if function expects i32/u8?
             // Runtime signatures in lib.rs define what expectations are.
             let fn_val = codegen.module.get_function(fn_name)
                 .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("{} not found", fn_name))))?;

             // If we need a cast:
             let arg_casted = if arg_val.is_int_value() {
                 let int_val = arg_val.into_int_value();
                 let i32_type = codegen.context.i32_type();
                 if int_val.get_type() != i32_type {
                      codegen.builder.build_int_cast(int_val, i32_type, "char_cast").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into()
                 } else {
                      (*arg_val).into()
                 }
             } else {
                 (*arg_val).into()
             };

             codegen
                 .builder
                 .build_call(fn_val, &[arg_casted], "print_call")
                 .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        }
        Type::I32 => {
            let fn_name = if is_newline {
                "tl_print_i32"
            } else {
                "tl_display_i32"
            };
            let fn_val = codegen.module.get_function(fn_name)
                .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("{} not found", fn_name))))?;
            codegen
                .builder
                .build_call(fn_val, &[(*arg_val).into()], "print_call")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        }
        Type::F32 => {
            let fn_name = if is_newline {
                "tl_print_f32"
            } else {
                "tl_display_f32"
            };
            let fn_val = codegen.module.get_function(fn_name)
                .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("{} not found", fn_name))))?;
            codegen
                .builder
                .build_call(fn_val, &[(*arg_val).into()], "print_call")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        }
        Type::F64 => {
            let fn_name = if is_newline {
                "tl_print_f64"
            } else {
                "tl_display_f64"
            };
            let fn_val = codegen.module.get_function(fn_name)
                .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("{} not found", fn_name))))?;
            codegen
                .builder
                .build_call(fn_val, &[(*arg_val).into()], "print_call")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        }
        Type::Bool => {
            let fn_name = if is_newline {
                "tl_print_bool"
            } else {
                "tl_display_bool"
            };
            let fn_val = codegen.module.get_function(fn_name)
                .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("{} not found", fn_name))))?;
            codegen
                .builder
                .build_call(fn_val, &[(*arg_val).into()], "print_call")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        }
        Type::Tuple(elem_types) => {
            // Print tuple as (a, b, c)
            let display_fn = codegen
                .module
                .get_function("tl_display_string")
                .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_display_string not found".to_string())))?;
            let print_fn = codegen
                .module
                .get_function("tl_print_string")
                .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_print_string not found".to_string())))?;

            fn emit_tuple_str<'ctx>(
                codegen: &mut CodeGenerator<'ctx>,
                s: &str,
                newline: bool,
                display_fn: inkwell::values::FunctionValue<'ctx>,
                print_fn: inkwell::values::FunctionValue<'ctx>,
            ) -> Result<(), TlError> {
                let (str_struct_ptr, _) = codegen.compile_string_literal(s)?;
                let ptr = str_struct_ptr.into_pointer_value();

                let fn_val = if newline { print_fn } else { display_fn };
                codegen
                    .builder
                    .build_call(fn_val, &[ptr.into()], "print_tuple_part")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                Ok(())
            }

            emit_tuple_str(codegen, "(", false, display_fn, print_fn)?;

            if !arg_val.is_pointer_value() {
                return Err(TlError::from(CodegenErrorKind::Internal("Tuple value is not a pointer".to_string())));
            }
            let tuple_ptr = arg_val.into_pointer_value();

            let mut llvm_types = Vec::new();
            for ty in elem_types.iter() {
                llvm_types.push(codegen.get_llvm_type(ty)?);
            }
            let tuple_struct_type = codegen.context.struct_type(&llvm_types, false);

            for (idx, ty) in elem_types.iter().enumerate() {
                if idx > 0 {
                    emit_tuple_str(codegen, ", ", false, display_fn, print_fn)?;
                }
                let field_ptr = codegen
                    .builder
                    .build_struct_gep(tuple_struct_type, tuple_ptr, idx as u32, "tuple_elem_ptr")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let llvm_field_ty = codegen.get_llvm_type(ty)?;
                let val = codegen
                    .builder
                    .build_load(llvm_field_ty, field_ptr, "tuple_elem")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                compile_print_common(codegen, vec![(val, ty.clone())], false)?;
            }

            emit_tuple_str(codegen, ")", false, display_fn, print_fn)?;
            if is_newline {
                emit_tuple_str(codegen, "", true, display_fn, print_fn)?;
            }
        }
        Type::Tensor(_, _) => {
            let fn_name = if is_newline {
                "tl_tensor_print"
            } else {
                "tl_tensor_display"
            };
            let fn_val = codegen.module.get_function(fn_name)
                .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("{} not found", fn_name))))?;
            codegen
                .builder
                .build_call(fn_val, &[(*arg_val).into()], "print_call")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        }
        Type::Struct(s, _) | Type::Struct(s, _) if s == "Tensor" => {
            let fn_name = if is_newline {
                "tl_tensor_print"
            } else {
                "tl_tensor_display"
            };
            let fn_val = codegen.module.get_function(fn_name)
                .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("{} not found", fn_name))))?;
            codegen
                .builder
                .build_call(fn_val, &[(*arg_val).into()], "print_call")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        }
        Type::String(_) => {
            let fn_name = if is_newline {
                "tl_print_string"
            } else {
                "tl_display_string"
            };
            let fn_val = codegen.module.get_function(fn_name);
            if let Some(f) = fn_val {
                codegen
                    .builder
                    .build_call(f, &[(*arg_val).into()], "print_call")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            } else {
                return Err(TlError::from(CodegenErrorKind::Internal(format!("{} not found (add to init)", fn_name))));
            }
        },
        _ => {
            let base_name = arg_type.get_base_name();
            let type_args = match arg_type {
                Type::Struct(_, args) => args.clone(),
                _ => vec![],
            };
            
            if let Ok(mangled) = codegen.monomorphize_method(&base_name, "to_string", &type_args) {
                if let Some(to_str_fn) = codegen.module.get_function(&mangled) {
                    let call = codegen.builder.build_call(to_str_fn, &[(*arg_val).into()], "to_string_call").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                    if let inkwell::values::ValueKind::Basic(str_val) = call.try_as_basic_value() {
                        let print_fn_name = if is_newline { "tl_print_string" } else { "tl_display_string" };
                        let print_fn = codegen.module.get_function(print_fn_name).ok_or_else(|| TlError::from(CodegenErrorKind::Internal("print string not found".to_string())))?;
                        codegen.builder.build_call(print_fn, &[str_val.into()], "print_call").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                        
                        return Ok((
                            codegen.context.i64_type().const_int(0, false).into(),
                            Type::Void,
                        ));
                    }
                }
            }
            
            return Err(TlError::from(CodegenErrorKind::Internal(format!("Cannot print type {:?} (does not implement Display)", arg_type))))
        }
    }
    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

pub(super) fn compile_print_uneval<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    compile_print_formatted(codegen, args, false)
}

pub(super) fn compile_println_uneval<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    compile_print_formatted(codegen, args, true)
}

/// format("pattern {}", args...) -> String
/// println と同じフォーマット文字列解析を行い、結果を String として返す
pub(super) fn compile_format_uneval<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.is_empty() {
        // format() = empty string
        return codegen.compile_string_literal("");
    }

    let concat_fn = codegen.module.get_function("tl_string_concat")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_string_concat not found".to_string())))?;

    // Check for format string
    let fmt_str_opt = if let ExprKind::StringLiteral(s) = &args[0].inner {
        if s.contains("{}") { Some(s.clone()) } else { None }
    } else {
        None
    };

    if let Some(fmt_str) = fmt_str_opt {
        let parts: Vec<&str> = fmt_str.split("{}").collect();
        let arg_count = args.len() - 1;
        let placeholder_count = parts.len() - 1;

        if arg_count != placeholder_count {
            return Err(TlError::from(CodegenErrorKind::Internal(format!(
                "Format string has {} placeholders but {} arguments were provided",
                placeholder_count, arg_count
            ))));
        }

        // Start with first literal part
        let (mut result, _) = codegen.compile_string_literal(parts[0])?;

        for (i, part) in parts.iter().enumerate().skip(1) {
            // Convert argument to string
            let (arg_val, arg_ty) = codegen.compile_expr(&args[i])?;
            let arg_str = compile_value_to_string(codegen, arg_val, &arg_ty)?;

            // Concat result + arg_str
            let call = codegen.builder.build_call(concat_fn, &[result.into(), arg_str.into()], "fmt_concat")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            result = match call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v,
                _ => return Err(CodegenErrorKind::Internal("concat returned void".to_string()).into()),
            };

            // Concat literal part
            if !part.is_empty() {
                let (lit_str, _) = codegen.compile_string_literal(part)?;
                let call = codegen.builder.build_call(concat_fn, &[result.into(), lit_str.into()], "fmt_concat")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                result = match call.try_as_basic_value() {
                    inkwell::values::ValueKind::Basic(v) => v,
                    _ => return Err(CodegenErrorKind::Internal("concat returned void".to_string()).into()),
                };
            }
        }

        Ok((result, Type::String("String".to_string())))
    } else {
        // No format string: format(value) = value.to_string()
        if args.len() != 1 {
            return Err(TlError::from(CodegenErrorKind::Internal("format requires format string or 1 argument".to_string())));
        }
        let (val, ty) = codegen.compile_expr(&args[0])?;
        let str_val = compile_value_to_string(codegen, val, &ty)?;
        Ok((str_val, Type::String("String".to_string())))
    }
}

/// 値を String に変換するヘルパー (format() 用)
pub(super) fn compile_value_to_string<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    val: BasicValueEnum<'ctx>,
    ty: &Type,
) -> Result<BasicValueEnum<'ctx>, TlError> {
    match ty {
        Type::String(_) => Ok(val), // already a string
        Type::I64 => {
            let fn_val = codegen.module.get_function("tl_string_from_int")
                .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_string_from_int not found".to_string())))?;
            let call = codegen.builder.build_call(fn_val, &[val.into()], "i64_to_str")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            match call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => Ok(v),
                _ => Err(CodegenErrorKind::Internal("tl_string_from_int returned void".to_string()).into()),
            }
        }
        Type::F64 => {
            let fn_val = codegen.module.get_function("tl_string_from_f64")
                .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_string_from_f64 not found".to_string())))?;
            let call = codegen.builder.build_call(fn_val, &[val.into()], "f64_to_str")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            match call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => Ok(v),
                _ => Err(CodegenErrorKind::Internal("tl_string_from_f64 returned void".to_string()).into()),
            }
        }
        Type::Bool => {
            let fn_val = codegen.module.get_function("tl_string_from_bool")
                .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_string_from_bool not found".to_string())))?;
            let call = codegen.builder.build_call(fn_val, &[val.into()], "bool_to_str")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            match call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => Ok(v),
                _ => Err(CodegenErrorKind::Internal("tl_string_from_bool returned void".to_string()).into()),
            }
        }
        Type::I32 | Type::F32 => {
            // Cast to i64/f64 first, then convert
            let i64_type = codegen.context.i64_type();
            let casted = if matches!(ty, Type::I32) {
                codegen.builder.build_int_s_extend(val.into_int_value(), i64_type, "i32_ext").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into()
            } else {
                let f64_type = codegen.context.f64_type();
                codegen.builder.build_float_ext(val.into_float_value(), f64_type, "f32_ext").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into()
            };
            let fn_name = if matches!(ty, Type::I32) { "tl_string_from_int" } else { "tl_string_from_f64" };
            let fn_val = codegen.module.get_function(fn_name)
                .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("{} not found", fn_name))))?;
            let call = codegen.builder.build_call(fn_val, &[casted], "cast_to_str")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            match call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => Ok(v),
                _ => Err(CodegenErrorKind::Internal("conversion returned void".to_string()).into()),
            }
        }
        _ => {
            // Fallback: try tl_string_from_int (for Char, U8 etc.)
            let fn_val = codegen.module.get_function("tl_string_from_int")
                .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_string_from_int not found".to_string())))?;
            let i64_type = codegen.context.i64_type();
            let int_val = if val.is_int_value() {
                codegen.builder.build_int_s_extend_or_bit_cast(val.into_int_value(), i64_type, "to_i64").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?.into()
            } else {
                val.into()
            };
            let call = codegen.builder.build_call(fn_val, &[int_val], "fallback_to_str")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            match call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => Ok(v),
                _ => Err(CodegenErrorKind::Internal("fallback conversion returned void".to_string()).into()),
            }
        }
    }
}

pub(super) fn compile_read_line_uneval<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 {
        return Err(TlError::from(CodegenErrorKind::Internal("read_line requires 1 argument".to_string())));
    }
    let (prompt_val, _prompt_ty) = codegen.compile_expr(&args[0])?;
    let fn_val = codegen
        .module
        .get_function("tl_read_line")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_read_line not found".to_string())))?;
    let call = codegen
        .builder
        .build_call(fn_val, &[prompt_val.into()], "read_line_res")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err(CodegenErrorKind::Internal("Invalid return from read_line".to_string()).into()),
    };
    Ok((res, Type::String("String".to_string())))
}

/// Compile panic! function - prints error message, calls abort, returns Never type
pub(super) fn compile_panic_uneval<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 {
        return Err(TlError::from(CodegenErrorKind::Internal("panic requires 1 argument (error message)".to_string())));
    }
    
    let (msg_val, _msg_ty) = codegen.compile_expr(&args[0])?;
    
    // Print the panic message using tl_display_string
    let display_fn = codegen
        .module
        .get_function("tl_display_string")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_display_string not found".to_string())))?;
    
    // Print "[PANIC] " prefix using compile_string_literal for proper TL string format
    let (prefix_val, _) = codegen.compile_string_literal("[PANIC] ")?;
    codegen.builder.build_call(display_fn, &[prefix_val.into()], "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

    // Print the actual message
    codegen.builder.build_call(display_fn, &[msg_val.into()], "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

    // Print newline
    let (newline_val, _) = codegen.compile_string_literal("\n")?;
    codegen.builder.build_call(display_fn, &[newline_val.into()], "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

    // Call abort() to terminate the program
    let abort_fn = codegen.module.get_function("abort").ok_or_else(|| TlError::from(CodegenErrorKind::Internal("abort function not found".to_string())))?;
    codegen.builder.build_call(abort_fn, &[], "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

    // Insert LLVM unreachable instruction to indicate control doesn't reach here
    codegen.builder.build_unreachable().map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    
    // Return a dummy value with Never type (code won't actually reach here)
    let dummy = codegen.context.i64_type().const_zero();
    Ok((dummy.into(), Type::Never))
}

pub(super) fn compile_assert_uneval<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 2 {
        return Err(TlError::from(CodegenErrorKind::Internal("assert requires 2 arguments (condition, message)".to_string())));
    }

    let (cond_val, _cond_ty) = codegen.compile_expr(&args[0])?;
    let (msg_val, _msg_ty) = codegen.compile_expr(&args[1])?;

    let fn_val = codegen
        .module
        .get_function("tl_assert")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_assert not found".to_string())))?;

    codegen
        .builder
        .build_call(fn_val, &[cond_val.into(), msg_val.into()], "")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

    Ok((
        codegen.context.i64_type().const_zero().into(),
        Type::Void,
    ))
}

pub(super) fn compile_print_formatted<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
    is_newline: bool,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.is_empty() {
        if is_newline {
            // Print newline only
            let fn_val = codegen
                .module
                .get_function("tl_print_string")
                .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_print_string not found".to_string())))?;

            // Create empty string "" and print it (tl_print_string adds newline)
            let s_val = codegen.context.const_string(b"", true);
            let global = codegen.module.add_global(
                s_val.get_type(),
                Some(inkwell::AddressSpace::default()),
                "empty_str",
            );
            global.set_initializer(&s_val);
            global.set_linkage(inkwell::module::Linkage::Internal);
            global.set_constant(true);

            let ptr = unsafe {
                codegen
                    .builder
                    .build_in_bounds_gep(
                        s_val.get_type(),
                        global.as_pointer_value(),
                        &[
                            codegen.context.i64_type().const_int(0, false),
                            codegen.context.i64_type().const_int(0, false),
                        ],
                        "str_ptr",
                    )
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
            };

            codegen
                .builder
                .build_call(fn_val, &[ptr.into()], "print_newline")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        }
        return Ok((
            codegen.context.i64_type().const_int(0, false).into(),
            Type::Void,
        ));
    }

    // Check for format string
    let fmt_str_opt = if let ExprKind::StringLiteral(s) = &args[0].inner {
        if s.contains("{}") {
            Some(s.clone())
        } else {
            None
        }
    } else {
        None
    };

    if let Some(fmt_str) = fmt_str_opt {
        // Formatted print
        let parts: Vec<&str> = fmt_str.split("{}").collect();
        let arg_count = args.len() - 1;
        let placeholder_count = parts.len() - 1;

        if arg_count != placeholder_count {
            return Err(TlError::from(CodegenErrorKind::Internal(format!(
                "Format string has {} placeholders but {} arguments were provided",
                placeholder_count, arg_count
            ))));
        }

        let display_fn = codegen
            .module
            .get_function("tl_display_string")
            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_display_string not found".to_string())))?;

        for (i, part) in parts.iter().enumerate() {
            // 1. Print literal part
            if !part.is_empty() {
                let (str_val, _) = codegen.compile_string_literal(part)?;
                codegen
                    .builder
                    .build_call(display_fn, &[str_val.into()], "print_part")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            }

            // 2. Print argument
            if i < arg_count {
                let expr = &args[i + 1];
                let (val, ty) = codegen.compile_expr(expr)?;
                // Use existing common logic, force is_newline=false
                compile_print_common(codegen, vec![(val, ty)], false)?;
            }
        }

        if is_newline {
            // Print final newline using tl_print_string("")
            let print_fn = codegen
                .module
                .get_function("tl_print_string")
                .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_print_string not found".to_string())))?;

            // Use compile_string_literal to create StringStruct
            let (str_struct_ptr, _) = codegen.compile_string_literal("")?;

            codegen
                .builder
                .build_call(print_fn, &[str_struct_ptr.into()], "print_newline")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        }
    } else {
        // Normal print
        if args.len() != 1 {
            return Err(TlError::from(CodegenErrorKind::Internal("print/println requires 1 argument (or format string)".to_string())));
        }
        let (val, ty) = codegen.compile_expr(&args[0])?;
        compile_print_common(codegen, vec![(val, ty)], is_newline)?;
    }

    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

pub(super) fn compile_args_count<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if !args.is_empty() {
        return Err(TlError::from(CodegenErrorKind::Internal("args_count takes no arguments".to_string())));
    }
    let fn_val = codegen
        .module
        .get_function("tl_args_count")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_args_count not found".to_string())))?;
    let call = codegen
        .builder
        .build_call(fn_val, &[], "args_count_res")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err(CodegenErrorKind::Internal("Invalid args_count return".to_string()).into()),
    };
    Ok((res, Type::I64))
}

pub(super) fn compile_args_get<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 {
        return Err(TlError::from(CodegenErrorKind::Internal("args_get requires 1 argument (index)".to_string())));
    }
    let (idx_val, _) = args[0].clone();
    let fn_val = codegen
        .module
        .get_function("tl_args_get")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_args_get not found".to_string())))?;
    let call = codegen
        .builder
        .build_call(fn_val, &[idx_val.into()], "args_get_res")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err(CodegenErrorKind::Internal("Invalid args_get return".to_string()).into()),
    };
    Ok((res, Type::String("String".to_string())))
}

pub(super) fn compile_string_char_at<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 2 {
        return Err(TlError::from(CodegenErrorKind::Internal("char_at requires 2 arguments (string, index)".to_string())));
    }
    let (str_val, _) = args[0].clone();
    let (idx_val, idx_ty) = args[1].clone();

    // Convert index to i64 if needed
    let idx_i64 = match idx_ty {
        Type::I64 => idx_val.into_int_value(),
        Type::I32 => codegen
            .builder
            .build_int_z_extend(
                idx_val.into_int_value(),
                codegen.context.i64_type(),
                "idx_ext",
            )
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?,
        _ => return Err(TlError::from(CodegenErrorKind::TypeError("Index must be integer".to_string()))),
    };

    let fn_val = codegen
        .module
        .get_function("tl_string_char_at")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_string_char_at not found".to_string())))?;
    let call = codegen
        .builder
        .build_call(fn_val, &[str_val.into(), idx_i64.into()], "char_at_res")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err(CodegenErrorKind::Internal("Invalid char_at return".to_string()).into()),
    };
    Ok((res, Type::Char("Char".to_string())))
}

pub(super) fn compile_string_len<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 {
        return Err(TlError::from(CodegenErrorKind::Internal("len requires 1 argument (string)".to_string())));
    }
    let (str_val, _) = args[0].clone();

    let fn_val = codegen
        .module
        .get_function("tl_string_len")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_string_len not found".to_string())))?;
    let call = codegen
        .builder
        .build_call(fn_val, &[str_val.into()], "len_res")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        _ => return Err(CodegenErrorKind::Internal("Invalid len return".to_string()).into()),
    };
    Ok((res, Type::I64))
}


pub(super) fn compile_save_weights<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 2 {
        return Err(TlError::from(CodegenErrorKind::Internal("save_weights requires 2 arguments: tensor/struct, path".to_string())));
    }
    let (t_val, t_ty) = &args[0];
    let (path_val, path_ty) = &args[1];

    if !matches!(path_ty, Type::String(_)) {
        return Err(TlError::from(CodegenErrorKind::TypeError("Second argument to save_weights must be a String (path)".to_string())));
    }

    match t_ty {
        Type::Tensor(_, _) => {
            let fn_val = codegen
                .module
                .get_function("tl_tensor_save")
                .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_tensor_save not found".to_string())))?;
            codegen
                .builder
                .build_call(fn_val, &[(*t_val).into(), (*path_val).into()], "")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        }
        Type::Struct(struct_name, _) | Type::Struct(struct_name, _) if struct_name != "String" => {
            // Struct serialization
            let new_fn = codegen
                .module
                .get_function("tl_tensor_map_new")
                .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_tensor_map_new not found".to_string())))?;
            let map_call = codegen
                .builder
                .build_call(new_fn, &[], "map")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let map_val = match map_call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v,
                _ => return Err(CodegenErrorKind::Internal("tl_tensor_map_new returned void".to_string()).into()),
            };

            codegen.gen_save_struct(map_val, *t_val, &struct_name, "".to_string())?;

            let save_fn = codegen
                .module
                .get_function("tl_tensor_map_save")
                .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_tensor_map_save not found".to_string())))?;
            codegen
                .builder
                .build_call(save_fn, &[map_val.into(), (*path_val).into()], "")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            let free_fn = codegen
                .module
                .get_function("tl_tensor_map_free")
                .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_tensor_map_free not found".to_string())))?;
            codegen
                .builder
                .build_call(free_fn, &[map_val.into()], "")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        }
        _ => {
            return Err(TlError::from(CodegenErrorKind::Internal(format!(
                "First argument to save_weights must be a tensor or struct. Found: {:?}",
                t_ty
            ))))
        }
    }

    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

pub(super) fn compile_load_weights<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() == 1 {
        let (path_val, path_ty) = &args[0];
        if !matches!(path_ty, Type::String(_)) {
            return Err(TlError::from(CodegenErrorKind::TypeError("Argument to load_weights must be a String (path)".to_string())));
        }

        let fn_val = codegen
            .module
            .get_function("tl_tensor_load")
            .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_tensor_load not found".to_string())))?;
        let call = codegen
            .builder
            .build_call(fn_val, &[(*path_val).into()], "load_res")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

        let res = match call.try_as_basic_value() {
            inkwell::values::ValueKind::Basic(v) => v,
            _ => return Err(CodegenErrorKind::Internal("Invalid load_weights return".to_string()).into()),
        };
        Ok((res, Type::Tensor(Box::new(Type::F32), 0)))
    } else if args.len() == 2 {
        // Struct load
        let (struct_val, s_ty) = &args[0];
        let (path_val, path_ty) = &args[1];
        if !matches!(path_ty, Type::String(_)) {
            return Err(TlError::from(CodegenErrorKind::TypeError("Second argument to load_weights must be a String (path)".to_string())));
        }

        let struct_name_opt = match &s_ty {
            Type::Struct(s, _) => Some(s.clone()),
            _ => None,
        };

        if let Some(struct_name) = struct_name_opt {
            if struct_name == "String" {
                panic!("Struct(String) found in load_weights");
            }

            let load_fn = codegen
                .module
                .get_function("tl_tensor_map_load")
                .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_tensor_map_load not found".to_string())))?;
            let map_call = codegen
                .builder
                .build_call(load_fn, &[(*path_val).into()], "map")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            let map_val = match map_call.try_as_basic_value() {
                inkwell::values::ValueKind::Basic(v) => v,
                _ => return Err(CodegenErrorKind::Internal("tl_tensor_map_load returned void".to_string()).into()),
            };

            codegen.gen_load_struct(map_val, *struct_val, &struct_name, "".to_string())?;

            let free_fn = codegen
                .module
                .get_function("tl_tensor_map_free")
                .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_tensor_map_free not found".to_string())))?;
            codegen
                .builder
                .build_call(free_fn, &[map_val.into()], "")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;

            Ok((
                codegen.context.i64_type().const_int(0, false).into(),
                Type::Void,
            ))
        } else {
            Err(TlError::from(CodegenErrorKind::Internal(format!(
                "First argument to load_weights (2 args) must be a struct. Found: {:?}",
                s_ty
            ))))
        }
    } else {
        Err(TlError::from(CodegenErrorKind::Internal("load_weights requires 1 or 2 arguments".to_string())))
    }
}

pub(super) fn compile_register_modules<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 {
        return Err(TlError::from(CodegenErrorKind::Internal("register_modules requires 1 argument (struct)".to_string())));
    }
    let (val, ty) = &args[0];
    match ty {
        Type::Struct(sname, _) => {
            codegen.gen_register_params(*val, &sname, "".to_string())?;
            return Ok((codegen.context.i64_type().const_zero().into(), Type::Void));
        }
        _ => return Err(CodegenErrorKind::Internal("register_modules expects a struct argument".to_string()).into()),
    }
}

pub(super) fn compile_update_all_params<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 {
        return Err(TlError::from(CodegenErrorKind::Internal("update_all_params requires 1 argument".to_string())));
    }
    let (lr_val, _) = &args[0];
    let fn_val = codegen.module.get_function("tl_update_all_params")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_update_all_params not found".to_string())))?;
    codegen
        .builder
        .build_call(fn_val, &[(*lr_val).into()], "")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

pub(super) fn compile_clear_grads<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = codegen.module.get_function("tl_clear_grads")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_clear_grads not found".to_string())))?;
    codegen.builder.build_call(fn_val, &[], "").map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((codegen.context.i64_type().const_int(0, false).into(), Type::Void))
}

pub(super) fn compile_add_parameter<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = codegen.module.get_function("tl_add_parameter")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_add_parameter not found".to_string())))?;
    let (name_val, _) = &args[0];
    let (tensor_val, _) = &args[1];
    codegen
        .builder
        .build_call(fn_val, &[(*name_val).into(), (*tensor_val).into()], "")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

pub(super) fn compile_load_all_params<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = codegen.module.get_function("tl_load_all_params")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_load_all_params not found".to_string())))?;
    let path_val = if args.len() == 2 {
        let (struct_val, struct_ty) = &args[0];
        let struct_name = match struct_ty {
            Type::Struct(s, _) => s,
            _ => return Err(CodegenErrorKind::Internal("Expected struct as first arg".to_string()).into()),
        };
        codegen.gen_register_params(*struct_val, &struct_name, "".to_string())?;
        let (path, _) = &args[1];
        path
    } else {
        let (path, _) = &args[0];
        path
    };

    codegen
        .builder
        .build_call(fn_val, &[(*path_val).into()], "load_all_res")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}

pub(super) fn compile_parameter<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 {
        return Err(TlError::from(CodegenErrorKind::Internal("parameter requires 1 argument".to_string())));
    }
    let (arg_val, arg_ty) = &args[0];
    let fn_val = codegen
        .module
        .get_function("tl_register_parameter")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_register_parameter not found".to_string())))?;
    let call = codegen
        .builder
        .build_call(fn_val, &[(*arg_val).into()], "param_reg")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(CodegenErrorKind::Internal("Invalid parameter return".to_string()).into()),
    };
    Ok((res, (*arg_ty).clone()))
}

pub(super) fn compile_save_all_params<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _target: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    let fn_val = codegen.module.get_function("tl_save_all_params")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_save_all_params not found".to_string())))?;
    let path_val = if args.len() == 2 {
        let (struct_val, struct_ty) = &args[0];
        let struct_name = match struct_ty {
            Type::Struct(s, _) => s,
            _ => return Err(CodegenErrorKind::Internal("Expected struct as first arg".to_string()).into()),
        };
        codegen.gen_register_params(*struct_val, &struct_name, "".to_string())?;
        let (path, _) = &args[1];
        path
    } else {
        let (path, _) = &args[0];
        path
    };

    codegen
        .builder
        .build_call(fn_val, &[(*path_val).into()], "save_all_res")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    Ok((
        codegen.context.i64_type().const_int(0, false).into(),
        Type::Void,
    ))
}



pub(super) fn compile_varbuilder_get<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: &[Expr],
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() < 2 {
        return Err(TlError::from(CodegenErrorKind::Internal("varbuilder_get requires at least 2 arguments (name and dimensions)".to_string())));
    }
    let (name_val, name_ty) = codegen.compile_expr(&args[0])?;
    if !matches!(name_ty, Type::String(_)) {
        return Err(TlError::from(CodegenErrorKind::Internal(format!(
            "varbuilder_get expects String as first argument, found {:?}",
            name_ty
        ))));
    }
    let name_ptr = name_val.into_pointer_value();

    let (rank, shape_ptr) = if args.len() == 2
        && matches!(
            codegen.compile_expr(&args[1])?.1,
            Type::Tensor(_, _)
        ) {
        let (shape_val, arg1_ty) = codegen.compile_expr(&args[1])?;
        let (num_elements, shape_vals) = match &arg1_ty {
            Type::Tensor(_, _) => {
                let len_fn = codegen
                    .module
                    .get_function("tl_tensor_len")
                    .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_tensor_len not found".to_string())))?;
                let call = codegen
                    .builder
                    .build_call(len_fn, &[shape_val.into()], "len")
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
                let _len = match call.try_as_basic_value() {
                    ValueKind::Basic(v) => v.into_int_value(),
                    _ => return Err(CodegenErrorKind::Internal("Invalid len return".to_string()).into()),
                };

                match &args[1].inner {
                    ExprKind::TensorLiteral(elements) | ExprKind::TensorConstLiteral(elements) => (
                        elements.len(),
                        elements
                            .iter()
                            .map(|e| {
                                let (val, _) = codegen.compile_expr(e)?;
                                Ok(val)
                            })
                            .collect::<Result<Vec<_>, TlError>>()?,
                    ),
                    _ => return Err(TlError::from(CodegenErrorKind::TypeError("varbuilder_get shape must be a literal array".to_string()))),
                }
            }
            _ => unreachable!(),
        };

        let i64_type = codegen.context.i64_type();
        let shape_alloca = codegen
            .builder
            .build_alloca(i64_type.array_type(num_elements as u32), "shape_arr")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        for (i, val) in shape_vals.iter().enumerate() {
            let idx = codegen.context.i64_type().const_int(i as u64, false);
            let ptr = unsafe {
                codegen
                    .builder
                    .build_in_bounds_gep(
                        i64_type.array_type(num_elements as u32),
                        shape_alloca,
                        &[codegen.context.i64_type().const_zero(), idx],
                        "shptr",
                    )
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
            };
            codegen
                .builder
                .build_store(ptr, val.into_int_value())
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        }
        (num_elements, shape_alloca)
    } else {
        let num_dims = args.len() - 1;
        let i64_type = codegen.context.i64_type();
        let shape_alloca = codegen
            .builder
            .build_alloca(i64_type.array_type(num_dims as u32), "shape_arr")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        for (i, arg) in args[1..].iter().enumerate() {
            let (val, _) = codegen.compile_expr(arg)?;
            let idx = i64_type.const_int(i as u64, false);
            let ptr = unsafe {
                codegen
                    .builder
                    .build_in_bounds_gep(
                        i64_type.array_type(num_dims as u32),
                        shape_alloca,
                        &[i64_type.const_zero(), idx],
                        "shptr",
                    )
                    .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?
            };
            codegen
                .builder
                .build_store(ptr, val.into_int_value())
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
        }
        (num_dims, shape_alloca)
    };

    let fn_val = codegen.module.get_function("tl_varbuilder_get")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("tl_varbuilder_get not found".to_string())))?;
    let call = codegen
        .builder
        .build_call(
            fn_val,
            &[
                name_ptr.into(),
                codegen
                    .context
                    .i64_type()
                    .const_int(rank as u64, false)
                    .into(),
                shape_ptr.into(),
            ],
            "varbuilder_get_result",
        )
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = codegen.check_tensor_result(call, "varbuilder_get_error")?;
    let res_ty = Type::Tensor(Box::new(Type::F32), 0);
    codegen.emit_register_tensor(res, &res_ty)?;
    Ok((res, res_ty))
}

pub(super) fn cast_value_to_f32<'ctx>(
    codegen: &CodeGenerator<'ctx>,
    val: BasicValueEnum<'ctx>,
    ty: &Type,
) -> Result<FloatValue<'ctx>, TlError> {
    let f32_type = codegen.context.f32_type();
    match ty {
        Type::F32 => Ok(val.into_float_value()),
        Type::F64 => Ok(codegen
            .builder
            .build_float_cast(val.into_float_value(), f32_type, "f64_to_f32")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?),
        Type::I64 => Ok(codegen
            .builder
            .build_signed_int_to_float(val.into_int_value(), f32_type, "i64_to_f32")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?),
        Type::I32 => Ok(codegen
            .builder
            .build_signed_int_to_float(val.into_int_value(), f32_type, "i32_to_f32")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?),
        Type::Bool => {
            let i64_type = codegen.context.i64_type();
            let i64_val = codegen
                .builder
                .build_int_z_extend(val.into_int_value(), i64_type, "bool_to_i64")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            Ok(codegen
                .builder
                .build_signed_int_to_float(i64_val, f32_type, "bool_to_f32")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?)
        }
        _ => Err(TlError::from(CodegenErrorKind::TypeError(format!("Cannot cast {:?} to F32", ty)))),
    }
}

pub(super) fn cast_value_to_f64<'ctx>(
    codegen: &CodeGenerator<'ctx>,
    val: BasicValueEnum<'ctx>,
    ty: &Type,
) -> Result<FloatValue<'ctx>, TlError> {
    let f64_type = codegen.context.f64_type();
    match ty {
        Type::F64 => Ok(val.into_float_value()),
        Type::F32 => Ok(codegen
            .builder
            .build_float_ext(val.into_float_value(), f64_type, "f32_to_f64")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?),
        Type::I64 => Ok(codegen
            .builder
            .build_signed_int_to_float(val.into_int_value(), f64_type, "i64_to_f64")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?),
        Type::I32 => Ok(codegen
            .builder
            .build_signed_int_to_float(val.into_int_value(), f64_type, "i32_to_f64")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?),
        Type::Bool => {
            let i64_type = codegen.context.i64_type();
            let i64_val = codegen
                .builder
                .build_int_z_extend(val.into_int_value(), i64_type, "bool_to_i64")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
            Ok(codegen
                .builder
                .build_signed_int_to_float(i64_val, f64_type, "bool_to_f64")
                .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?)
        }
        _ => Err(TlError::from(CodegenErrorKind::TypeError(format!("Cannot cast {:?} to F64", ty)))),
    }
}

pub(crate) fn cast_value_to_i64<'ctx>(
    codegen: &CodeGenerator<'ctx>,
    val: BasicValueEnum<'ctx>,
    ty: &Type,
) -> Result<IntValue<'ctx>, TlError> {
    let i64_type = codegen.context.i64_type();
    match ty {
        Type::I64 => Ok(val.into_int_value()),
        Type::I32 => Ok(codegen
            .builder
            .build_int_s_extend(val.into_int_value(), i64_type, "i32_to_i64")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?),
        _ => Err(TlError::from(CodegenErrorKind::TypeError(format!("Cannot cast {:?} to I64", ty)))),
    }
}

pub(super) fn cast_value_to_i32<'ctx>(
    codegen: &CodeGenerator<'ctx>,
    val: BasicValueEnum<'ctx>,
    ty: &Type,
) -> Result<IntValue<'ctx>, TlError> {
    let i32_type = codegen.context.i32_type();
    match ty {
        Type::I32 => Ok(val.into_int_value()),
        Type::I64 => Ok(codegen
            .builder
            .build_int_cast(val.into_int_value(), i32_type, "i64_to_i32")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?),
        _ => Err(TlError::from(CodegenErrorKind::TypeError(format!("Cannot cast {:?} to I32", ty)))),
    }
}

pub(super) fn compile_f32_unary_math<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    op_name: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if !args.is_empty() {
        return Err(TlError::from(CodegenErrorKind::Internal(format!("{} requires 0 arguments", op_name))));
    }
    let obj_f32 = cast_value_to_f32(codegen, obj_val, &obj_ty)?;
    let fn_name = format!("tl_f32_{}", op_name);
    let fn_val = codegen
        .module
        .get_function(&fn_name)
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Function {} not found", fn_name))))?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_f32.into()], "f32_unary")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(TlError::from(CodegenErrorKind::Internal(format!("Invalid {} return", op_name)))),
    };
    Ok((res, Type::F32))
}

pub(super) fn compile_f32_binary_math<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    op_name: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 {
        return Err(TlError::from(CodegenErrorKind::Internal(format!("{} requires 1 argument", op_name))));
    }
    let obj_f32 = cast_value_to_f32(codegen, obj_val, &obj_ty)?;
    let (arg_val, arg_ty) = &args[0];
    let arg_f32 = cast_value_to_f32(codegen, *arg_val, arg_ty)?;
    let fn_name = format!("tl_f32_{}", op_name);
    let fn_val = codegen
        .module
        .get_function(&fn_name)
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Function {} not found", fn_name))))?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_f32.into(), arg_f32.into()], "f32_binary")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(TlError::from(CodegenErrorKind::Internal(format!("Invalid {} return", op_name)))),
    };
    Ok((res, Type::F32))
}

pub(super) fn compile_f32_powi<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 {
        return Err(TlError::from(CodegenErrorKind::Internal("powi requires 1 argument".to_string())));
    }
    let obj_f32 = cast_value_to_f32(codegen, obj_val, &obj_ty)?;
    let (arg_val, arg_ty) = &args[0];
    let i64_type = codegen.context.i64_type();
    let arg_i64 = match arg_ty {
        Type::I64 => arg_val.into_int_value(),
        Type::I32 | Type::Bool => codegen
            .builder
            .build_int_z_extend(arg_val.into_int_value(), i64_type, "powi_i64")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?,
        _ => return Err(TlError::from(CodegenErrorKind::TypeError(format!("powi requires integer argument, got {:?}", arg_ty)))),
    };
    let fn_val = codegen
        .module
        .get_function("tl_f32_powi")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("Function tl_f32_powi not found".to_string())))?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_f32.into(), arg_i64.into()], "f32_powi")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(CodegenErrorKind::Internal("Invalid powi return".to_string()).into()),
    };
    Ok((res, Type::F32))
}

pub(super) fn compile_f64_unary_math<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    op_name: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if !args.is_empty() {
        return Err(TlError::from(CodegenErrorKind::Internal(format!("{} requires 0 arguments", op_name))));
    }
    let obj_f64 = cast_value_to_f64(codegen, obj_val, &obj_ty)?;
    let fn_name = format!("tl_f64_{}", op_name);
    let fn_val = codegen
        .module
        .get_function(&fn_name)
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Function {} not found", fn_name))))?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_f64.into()], "f64_unary")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(TlError::from(CodegenErrorKind::Internal(format!("Invalid {} return", op_name)))),
    };
    Ok((res, Type::F64))
}

pub(super) fn compile_f64_binary_math<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    op_name: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 {
        return Err(TlError::from(CodegenErrorKind::Internal(format!("{} requires 1 argument", op_name))));
    }
    let obj_f64 = cast_value_to_f64(codegen, obj_val, &obj_ty)?;
    let (arg_val, arg_ty) = &args[0];
    let arg_f64 = cast_value_to_f64(codegen, *arg_val, arg_ty)?;
    let fn_name = format!("tl_f64_{}", op_name);
    let fn_val = codegen
        .module
        .get_function(&fn_name)
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Function {} not found", fn_name))))?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_f64.into(), arg_f64.into()], "f64_binary")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(TlError::from(CodegenErrorKind::Internal(format!("Invalid {} return", op_name)))),
    };
    Ok((res, Type::F64))
}

pub(super) fn compile_f64_powi<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 {
        return Err(TlError::from(CodegenErrorKind::Internal("powi requires 1 argument".to_string())));
    }
    let obj_f64 = cast_value_to_f64(codegen, obj_val, &obj_ty)?;
    let (arg_val, arg_ty) = &args[0];
    let i64_type = codegen.context.i64_type();
    let arg_i64 = match arg_ty {
        Type::I64 => arg_val.into_int_value(),
        Type::I32 | Type::Bool => codegen
            .builder
            .build_int_z_extend(arg_val.into_int_value(), i64_type, "powi_i64")
            .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?,
        _ => return Err(TlError::from(CodegenErrorKind::TypeError(format!("powi requires integer argument, got {:?}", arg_ty)))),
    };
    let fn_val = codegen
        .module
        .get_function("tl_f64_powi")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("Function tl_f64_powi not found".to_string())))?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_f64.into(), arg_i64.into()], "f64_powi")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(CodegenErrorKind::Internal("Invalid powi return".to_string()).into()),
    };
    Ok((res, Type::F64))
}

pub(super) fn compile_i64_unary_math<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    op_name: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if !args.is_empty() {
        return Err(TlError::from(CodegenErrorKind::Internal(format!("{} requires 0 arguments", op_name))));
    }
    let obj_i64 = cast_value_to_i64(codegen, obj_val, &obj_ty)?;
    let fn_name = format!("tl_i64_{}", op_name);
    let fn_val = codegen
        .module
        .get_function(&fn_name)
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Function {} not found", fn_name))))?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_i64.into()], "i64_unary")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(TlError::from(CodegenErrorKind::Internal(format!("Invalid {} return", op_name)))),
    };
    Ok((res, Type::I64))
}

pub(super) fn compile_i64_binary_math<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    op_name: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 {
        return Err(TlError::from(CodegenErrorKind::Internal(format!("{} requires 1 argument", op_name))));
    }
    let obj_i64 = cast_value_to_i64(codegen, obj_val, &obj_ty)?;
    let (arg_val, arg_ty) = &args[0];
    let arg_i64 = cast_value_to_i64(codegen, *arg_val, arg_ty)?;
    let fn_name = format!("tl_i64_{}", op_name);
    let fn_val = codegen
        .module
        .get_function(&fn_name)
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Function {} not found", fn_name))))?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_i64.into(), arg_i64.into()], "i64_binary")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(TlError::from(CodegenErrorKind::Internal(format!("Invalid {} return", op_name)))),
    };
    Ok((res, Type::I64))
}

pub(super) fn compile_i64_pow<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 {
        return Err(TlError::from(CodegenErrorKind::Internal("pow requires 1 argument".to_string())));
    }
    let obj_i64 = cast_value_to_i64(codegen, obj_val, &obj_ty)?;
    let (arg_val, arg_ty) = &args[0];
    let exp_i64 = cast_value_to_i64(codegen, *arg_val, arg_ty)?;
    let fn_val = codegen
        .module
        .get_function("tl_i64_pow")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("Function tl_i64_pow not found".to_string())))?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_i64.into(), exp_i64.into()], "i64_pow")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(CodegenErrorKind::Internal("Invalid pow return".to_string()).into()),
    };
    Ok((res, Type::I64))
}

pub(super) fn compile_i64_is_positive<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if !args.is_empty() {
        return Err(TlError::from(CodegenErrorKind::Internal("is_positive requires 0 arguments".to_string())));
    }
    let obj_i64 = cast_value_to_i64(codegen, obj_val, &obj_ty)?;
    let fn_val = codegen
        .module
        .get_function("tl_i64_is_positive")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("Function tl_i64_is_positive not found".to_string())))?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_i64.into()], "i64_is_positive")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(CodegenErrorKind::Internal("Invalid is_positive return".to_string()).into()),
    };
    Ok((res, Type::Bool))
}

pub(super) fn compile_i64_is_negative<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if !args.is_empty() {
        return Err(TlError::from(CodegenErrorKind::Internal("is_negative requires 0 arguments".to_string())));
    }
    let obj_i64 = cast_value_to_i64(codegen, obj_val, &obj_ty)?;
    let fn_val = codegen
        .module
        .get_function("tl_i64_is_negative")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("Function tl_i64_is_negative not found".to_string())))?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_i64.into()], "i64_is_negative")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(CodegenErrorKind::Internal("Invalid is_negative return".to_string()).into()),
    };
    Ok((res, Type::Bool))
}

pub(super) fn compile_i32_unary_math<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    op_name: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if !args.is_empty() {
        return Err(TlError::from(CodegenErrorKind::Internal(format!("{} requires 0 arguments", op_name))));
    }
    let obj_i32 = cast_value_to_i32(codegen, obj_val, &obj_ty)?;
    let fn_name = format!("tl_i32_{}", op_name);
    let fn_val = codegen
        .module
        .get_function(&fn_name)
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Function {} not found", fn_name))))?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_i32.into()], "i32_unary")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(TlError::from(CodegenErrorKind::Internal(format!("Invalid {} return", op_name)))),
    };
    Ok((res, Type::I32))
}

pub(super) fn compile_i32_binary_math<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    op_name: &str,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 {
        return Err(TlError::from(CodegenErrorKind::Internal(format!("{} requires 1 argument", op_name))));
    }
    let obj_i32 = cast_value_to_i32(codegen, obj_val, &obj_ty)?;
    let (arg_val, arg_ty) = &args[0];
    let arg_i32 = cast_value_to_i32(codegen, *arg_val, arg_ty)?;
    let fn_name = format!("tl_i32_{}", op_name);
    let fn_val = codegen
        .module
        .get_function(&fn_name)
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal(format!("Function {} not found", fn_name))))?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_i32.into(), arg_i32.into()], "i32_binary")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(TlError::from(CodegenErrorKind::Internal(format!("Invalid {} return", op_name)))),
    };
    Ok((res, Type::I32))
}

pub(super) fn compile_i32_pow<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if args.len() != 1 {
        return Err(TlError::from(CodegenErrorKind::Internal("pow requires 1 argument".to_string())));
    }
    let obj_i32 = cast_value_to_i32(codegen, obj_val, &obj_ty)?;
    let (arg_val, arg_ty) = &args[0];
    let exp_i32 = cast_value_to_i32(codegen, *arg_val, arg_ty)?;
    let fn_val = codegen
        .module
        .get_function("tl_i32_pow")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("Function tl_i32_pow not found".to_string())))?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_i32.into(), exp_i32.into()], "i32_pow")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(CodegenErrorKind::Internal("Invalid pow return".to_string()).into()),
    };
    Ok((res, Type::I32))
}

pub(super) fn compile_i32_is_positive<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if !args.is_empty() {
        return Err(TlError::from(CodegenErrorKind::Internal("is_positive requires 0 arguments".to_string())));
    }
    let obj_i32 = cast_value_to_i32(codegen, obj_val, &obj_ty)?;
    let fn_val = codegen
        .module
        .get_function("tl_i32_is_positive")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("Function tl_i32_is_positive not found".to_string())))?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_i32.into()], "i32_is_positive")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(CodegenErrorKind::Internal("Invalid is_positive return".to_string()).into()),
    };
    Ok((res, Type::Bool))
}

pub(super) fn compile_i32_is_negative<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    if !args.is_empty() {
        return Err(TlError::from(CodegenErrorKind::Internal("is_negative requires 0 arguments".to_string())));
    }
    let obj_i32 = cast_value_to_i32(codegen, obj_val, &obj_ty)?;
    let fn_val = codegen
        .module
        .get_function("tl_i32_is_negative")
        .ok_or_else(|| TlError::from(CodegenErrorKind::Internal("Function tl_i32_is_negative not found".to_string())))?;
    let call = codegen
        .builder
        .build_call(fn_val, &[obj_i32.into()], "i32_is_negative")
        .map_err(|e| TlError::from(CodegenErrorKind::Internal(e.to_string())))?;
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err(CodegenErrorKind::Internal("Invalid is_negative return".to_string()).into()),
    };
    Ok((res, Type::Bool))
}

macro_rules! f32_unary_method {
    ($name:ident, $op:expr) => {
        pub(super) fn $name<'ctx>(
            codegen: &mut CodeGenerator<'ctx>,
            obj_val: BasicValueEnum<'ctx>,
            obj_ty: Type,
            args: Vec<(BasicValueEnum<'ctx>, Type)>,
        ) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
            compile_f32_unary_math(codegen, obj_val, obj_ty, args, $op)
        }
    };
}

macro_rules! f32_binary_method {
    ($name:ident, $op:expr) => {
        pub(super) fn $name<'ctx>(
            codegen: &mut CodeGenerator<'ctx>,
            obj_val: BasicValueEnum<'ctx>,
            obj_ty: Type,
            args: Vec<(BasicValueEnum<'ctx>, Type)>,
        ) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
            compile_f32_binary_math(codegen, obj_val, obj_ty, args, $op)
        }
    };
}

macro_rules! f64_unary_method {
    ($name:ident, $op:expr) => {
        pub(super) fn $name<'ctx>(
            codegen: &mut CodeGenerator<'ctx>,
            obj_val: BasicValueEnum<'ctx>,
            obj_ty: Type,
            args: Vec<(BasicValueEnum<'ctx>, Type)>,
        ) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
            compile_f64_unary_math(codegen, obj_val, obj_ty, args, $op)
        }
    };
}

macro_rules! f64_binary_method {
    ($name:ident, $op:expr) => {
        pub(super) fn $name<'ctx>(
            codegen: &mut CodeGenerator<'ctx>,
            obj_val: BasicValueEnum<'ctx>,
            obj_ty: Type,
            args: Vec<(BasicValueEnum<'ctx>, Type)>,
        ) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
            compile_f64_binary_math(codegen, obj_val, obj_ty, args, $op)
        }
    };
}

macro_rules! i64_unary_method {
    ($name:ident, $op:expr) => {
        pub(super) fn $name<'ctx>(
            codegen: &mut CodeGenerator<'ctx>,
            obj_val: BasicValueEnum<'ctx>,
            obj_ty: Type,
            args: Vec<(BasicValueEnum<'ctx>, Type)>,
        ) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
            compile_i64_unary_math(codegen, obj_val, obj_ty, args, $op)
        }
    };
}

macro_rules! i64_binary_method {
    ($name:ident, $op:expr) => {
        pub(super) fn $name<'ctx>(
            codegen: &mut CodeGenerator<'ctx>,
            obj_val: BasicValueEnum<'ctx>,
            obj_ty: Type,
            args: Vec<(BasicValueEnum<'ctx>, Type)>,
        ) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
            compile_i64_binary_math(codegen, obj_val, obj_ty, args, $op)
        }
    };
}

macro_rules! i32_unary_method {
    ($name:ident, $op:expr) => {
        pub(super) fn $name<'ctx>(
            codegen: &mut CodeGenerator<'ctx>,
            obj_val: BasicValueEnum<'ctx>,
            obj_ty: Type,
            args: Vec<(BasicValueEnum<'ctx>, Type)>,
        ) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
            compile_i32_unary_math(codegen, obj_val, obj_ty, args, $op)
        }
    };
}

macro_rules! i32_binary_method {
    ($name:ident, $op:expr) => {
        pub(super) fn $name<'ctx>(
            codegen: &mut CodeGenerator<'ctx>,
            obj_val: BasicValueEnum<'ctx>,
            obj_ty: Type,
            args: Vec<(BasicValueEnum<'ctx>, Type)>,
        ) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
            compile_i32_binary_math(codegen, obj_val, obj_ty, args, $op)
        }
    };
}

f32_unary_method!(compile_f32_abs, "abs");
f32_unary_method!(compile_f32_acos, "acos");
f32_unary_method!(compile_f32_acosh, "acosh");
f32_unary_method!(compile_f32_asin, "asin");
f32_unary_method!(compile_f32_asinh, "asinh");
f32_unary_method!(compile_f32_atan, "atan");
f32_unary_method!(compile_f32_atanh, "atanh");
f32_unary_method!(compile_f32_cbrt, "cbrt");
f32_unary_method!(compile_f32_ceil, "ceil");
f32_unary_method!(compile_f32_cos, "cos");
f32_unary_method!(compile_f32_cosh, "cosh");
f32_unary_method!(compile_f32_exp, "exp");
f32_unary_method!(compile_f32_exp2, "exp2");
f32_unary_method!(compile_f32_exp_m1, "exp_m1");
f32_unary_method!(compile_f32_floor, "floor");
f32_unary_method!(compile_f32_fract, "fract");
f32_unary_method!(compile_f32_ln, "ln");
f32_unary_method!(compile_f32_ln_1p, "ln_1p");
f32_unary_method!(compile_f32_log10, "log10");
f32_unary_method!(compile_f32_log2, "log2");
f32_unary_method!(compile_f32_recip, "recip");
f32_unary_method!(compile_f32_round, "round");
f32_unary_method!(compile_f32_signum, "signum");
f32_unary_method!(compile_f32_sin, "sin");
f32_unary_method!(compile_f32_sinh, "sinh");
f32_unary_method!(compile_f32_sqrt, "sqrt");
f32_unary_method!(compile_f32_tan, "tan");
f32_unary_method!(compile_f32_tanh, "tanh");
f32_unary_method!(compile_f32_to_degrees, "to_degrees");
f32_unary_method!(compile_f32_to_radians, "to_radians");
f32_unary_method!(compile_f32_trunc, "trunc");

f32_binary_method!(compile_f32_atan2, "atan2");
f32_binary_method!(compile_f32_copysign, "copysign");
f32_binary_method!(compile_f32_hypot, "hypot");
f32_binary_method!(compile_f32_log, "log");
f32_binary_method!(compile_f32_powf, "powf");

pub(super) fn compile_f32_pow<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    compile_f32_binary_math(codegen, obj_val, obj_ty, args, "powf")
}

f64_unary_method!(compile_f64_abs, "abs");
f64_unary_method!(compile_f64_acos, "acos");
f64_unary_method!(compile_f64_acosh, "acosh");
f64_unary_method!(compile_f64_asin, "asin");
f64_unary_method!(compile_f64_asinh, "asinh");
f64_unary_method!(compile_f64_atan, "atan");
f64_unary_method!(compile_f64_atanh, "atanh");
f64_unary_method!(compile_f64_cbrt, "cbrt");
f64_unary_method!(compile_f64_ceil, "ceil");
f64_unary_method!(compile_f64_cos, "cos");
f64_unary_method!(compile_f64_cosh, "cosh");
f64_unary_method!(compile_f64_exp, "exp");
f64_unary_method!(compile_f64_exp2, "exp2");
f64_unary_method!(compile_f64_exp_m1, "exp_m1");
f64_unary_method!(compile_f64_floor, "floor");
f64_unary_method!(compile_f64_fract, "fract");
f64_unary_method!(compile_f64_ln, "ln");
f64_unary_method!(compile_f64_ln_1p, "ln_1p");
f64_unary_method!(compile_f64_log10, "log10");
f64_unary_method!(compile_f64_log2, "log2");
f64_unary_method!(compile_f64_recip, "recip");
f64_unary_method!(compile_f64_round, "round");
f64_unary_method!(compile_f64_signum, "signum");
f64_unary_method!(compile_f64_sin, "sin");
f64_unary_method!(compile_f64_sinh, "sinh");
f64_unary_method!(compile_f64_sqrt, "sqrt");
f64_unary_method!(compile_f64_tan, "tan");
f64_unary_method!(compile_f64_tanh, "tanh");
f64_unary_method!(compile_f64_to_degrees, "to_degrees");
f64_unary_method!(compile_f64_to_radians, "to_radians");
f64_unary_method!(compile_f64_trunc, "trunc");

f64_binary_method!(compile_f64_atan2, "atan2");
f64_binary_method!(compile_f64_copysign, "copysign");
f64_binary_method!(compile_f64_hypot, "hypot");
f64_binary_method!(compile_f64_log, "log");
f64_binary_method!(compile_f64_powf, "powf");

pub(super) fn compile_f64_pow<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj_val: BasicValueEnum<'ctx>,
    obj_ty: Type,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), TlError> {
    compile_f64_binary_math(codegen, obj_val, obj_ty, args, "powf")
}

i64_unary_method!(compile_i64_abs, "abs");
i64_unary_method!(compile_i64_signum, "signum");
i64_binary_method!(compile_i64_div_euclid, "div_euclid");
i64_binary_method!(compile_i64_rem_euclid, "rem_euclid");

i32_unary_method!(compile_i32_abs, "abs");
i32_unary_method!(compile_i32_signum, "signum");
i32_binary_method!(compile_i32_div_euclid, "div_euclid");
i32_binary_method!(compile_i32_rem_euclid, "rem_euclid");


