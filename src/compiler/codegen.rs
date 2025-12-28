// src/compiler/codegen.rs
use crate::compiler::ast::*;
use crate::runtime::{tl_tensor_add, tl_tensor_mul, tl_tensor_new, tl_tensor_print}; // Import runtime functions
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::module::Module as InkwellModule;
use inkwell::values::{BasicValueEnum, FunctionValue, ValueKind};
use inkwell::OptimizationLevel;
use std::collections::HashMap;

pub struct CodeGenerator<'ctx> {
    context: &'ctx Context,
    module: InkwellModule<'ctx>,
    builder: Builder<'ctx>,
    variables: HashMap<String, (BasicValueEnum<'ctx>, Type)>,
    execution_engine: ExecutionEngine<'ctx>,
}

impl<'ctx> CodeGenerator<'ctx> {
    pub fn new(context: &'ctx Context, module_name: &str) -> Self {
        let module = context.create_module(module_name);
        let builder = context.create_builder();
        let execution_engine = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .unwrap();
        CodeGenerator {
            context,
            module,
            builder,
            variables: HashMap::new(),
            execution_engine,
        }
    }

    fn declare_runtime_functions(&self) {
        let i64_type = self.context.i64_type(); // usize
        let f32_ptr = self
            .context
            .f32_type()
            .ptr_type(inkwell::AddressSpace::default());
        let usize_ptr = self
            .context
            .i64_type()
            .ptr_type(inkwell::AddressSpace::default());
        let void_ptr = self.context.ptr_type(inkwell::AddressSpace::default()); // OpaqueTensor*
        let void_type = self.context.void_type();

        // tl_tensor_new(data: *const f32, rank: usize, shape: *const usize) -> *mut OpaqueTensor
        let new_type =
            void_ptr.fn_type(&[f32_ptr.into(), i64_type.into(), usize_ptr.into()], false);
        self.module.add_function("tl_tensor_new", new_type, None);

        // tl_tensor_add(a: *mut, b: *mut) -> *mut
        let bin_type = void_ptr.fn_type(&[void_ptr.into(), void_ptr.into()], false);
        self.module.add_function("tl_tensor_add", bin_type, None);
        self.module.add_function("tl_tensor_mul", bin_type, None);

        // tl_tensor_print(t: *mut) -> void
        let print_type = void_type.fn_type(&[void_ptr.into()], false);
        self.module
            .add_function("tl_tensor_print", print_type, None);

        // Map symbols
        unsafe {
            if let Some(f) = self.module.get_function("tl_tensor_new") {
                self.execution_engine
                    .add_global_mapping(&f, tl_tensor_new as usize);
            }
            if let Some(f) = self.module.get_function("tl_tensor_add") {
                self.execution_engine
                    .add_global_mapping(&f, tl_tensor_add as usize);
            }
            if let Some(f) = self.module.get_function("tl_tensor_mul") {
                self.execution_engine
                    .add_global_mapping(&f, tl_tensor_mul as usize);
            }
            if let Some(f) = self.module.get_function("tl_tensor_print") {
                self.execution_engine
                    .add_global_mapping(&f, tl_tensor_print as usize);
            }
        }
    }

    pub fn jit_execute(&self, function_name: &str) -> Result<u64, String> {
        unsafe {
            let function = self
                .execution_engine
                .get_function::<unsafe extern "C" fn() -> u64>(function_name)
                .map_err(|e| format!("JIT compile error: {}", e))?;
            Ok(function.call())
        }
    }

    pub fn compile_module(&mut self, ast_module: &Module) -> Result<(), String> {
        // 0. Declare runtime functions
        self.declare_runtime_functions();

        // 1. Declare structs (types) - Placeholder

        // 2. Declare functions (prototypes)
        for func in &ast_module.functions {
            self.compile_fn_proto(func)?;
        }

        // 3. Compile function bodies
        for func in &ast_module.functions {
            self.compile_fn(func)?;
        }

        Ok(())
    }

    fn compile_fn_proto(&self, func: &FunctionDef) -> Result<FunctionValue<'ctx>, String> {
        // TODO: Handle types properly. Defaults to void() for now for testing main.
        let void_type = self.context.void_type();
        let fn_type = void_type.fn_type(&[], false); // No args, void return
        let val = self.module.add_function(&func.name, fn_type, None);
        Ok(val)
    }

    fn compile_fn(&mut self, func: &FunctionDef) -> Result<(), String> {
        let function = self
            .module
            .get_function(&func.name)
            .ok_or("Function not found")?;
        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);

        // Clear variables for new function scope
        self.variables.clear();

        // Body
        for stmt in &func.body {
            self.compile_stmt(stmt)?;
        }

        // Implicit return void if not terminated
        if function.get_type().get_return_type().is_none() {
            self.builder.build_return(None).map_err(|e| e.to_string())?;
        }

        Ok(())
    }

    fn compile_stmt(&mut self, stmt: &Stmt) -> Result<(), String> {
        match stmt {
            Stmt::Let { name, value, .. } => {
                let val = self.compile_expr(value)?;
                // Verify stack allocation needed? For now simple register mapping
                // But for mutable variables we need alloca.
                // Let's use alloca for everything for simplicity.
                let alloca = self.create_entry_block_alloca(name, &val.1);
                self.builder
                    .build_store(alloca, val.0)
                    .map_err(|e| e.to_string())?;
                self.variables.insert(name.clone(), (alloca.into(), val.1)); // Store pointer and type
            }
            Stmt::Return(expr) => {
                let val = self.compile_expr(expr)?;
                self.builder
                    .build_return(Some(&val.0))
                    .map_err(|e| e.to_string())?;
            }
            Stmt::Expr(expr) => {
                self.compile_expr(expr)?;
            }
            _ => {}
        }
        Ok(())
    }

    fn compile_expr(&self, expr: &Expr) -> Result<(BasicValueEnum<'ctx>, Type), String> {
        match expr {
            Expr::Int(i) => {
                let i64_type = self.context.i64_type();
                Ok((i64_type.const_int(*i as u64, true).into(), Type::I64))
            }
            Expr::Float(f) => {
                let f32_type = self.context.f32_type();
                Ok((f32_type.const_float(*f).into(), Type::F32))
            }
            Expr::Variable(name) => {
                match self.variables.get(name) {
                    Some((val, ty)) => {
                        // Validate if it is a pointer (alloca) or value.
                        // We stored alloca pointer. So load it.
                        if val.is_pointer_value() {
                            let ptr: inkwell::values::PointerValue<'ctx> = val.into_pointer_value();
                            // LLVM Type to load
                            let llvm_ty: inkwell::types::BasicTypeEnum = match ty {
                                Type::I64 => self.context.i64_type().into(),
                                Type::F32 => self.context.f32_type().into(),
                                // For Tensor, the value itself is a pointer to OpaqueTensor (ptr)
                                // But 'ptr' in variables is *alloca*, which is pointer to pointer.
                                Type::Tensor(_, _) => self
                                    .context
                                    .ptr_type(inkwell::AddressSpace::default())
                                    .into(), // Load the pointer
                                _ => self.context.i64_type().into(), // Fallback
                            };

                            let loaded = self
                                .builder
                                .build_load(llvm_ty, ptr, name)
                                .map_err(|e| e.to_string())?;
                            Ok((loaded, ty.clone()))
                        } else {
                            Ok((*val, ty.clone()))
                        }
                    }
                    None => Err(format!("Variable not found: {}", name)),
                }
            }
            Expr::BinOp(lhs, op, rhs) => {
                let left = self.compile_expr(lhs)?;
                let right = self.compile_expr(rhs)?;

                // Check types
                match (&left.1, &right.1) {
                    (Type::I64, Type::I64) => {
                        let l = left.0.into_int_value();
                        let r = right.0.into_int_value();
                        let res = match op {
                            BinOp::Add => self.builder.build_int_add(l, r, "addtmp"),
                            BinOp::Sub => self.builder.build_int_sub(l, r, "subtmp"),
                            BinOp::Mul => self.builder.build_int_mul(l, r, "multmp"),
                            BinOp::Div => self.builder.build_int_signed_div(l, r, "divtmp"),
                            _ => return Err("Unsupported int op".into()),
                        }
                        .map_err(|e| e.to_string())?;
                        Ok((res.into(), Type::I64))
                    }
                    (Type::F32, Type::F32) => {
                        let l = left.0.into_float_value();
                        let r = right.0.into_float_value();
                        let res = match op {
                            BinOp::Add => self.builder.build_float_add(l, r, "faddtmp"),
                            BinOp::Sub => self.builder.build_float_sub(l, r, "fsubtmp"),
                            BinOp::Mul => self.builder.build_float_mul(l, r, "fmultmp"),
                            BinOp::Div => self.builder.build_float_div(l, r, "fdivtmp"),
                            _ => return Err("Unsupported float op".into()),
                        }
                        .map_err(|e| e.to_string())?;
                        Ok((res.into(), Type::F32))
                    }
                    (Type::Tensor(_, _), Type::Tensor(_, _)) => {
                        // Call Runtime
                        // Assume Tensors are pointers (i8*)
                        let l = left.0.into_pointer_value();
                        let r = right.0.into_pointer_value();

                        let fn_name = match op {
                            BinOp::Add => "tl_tensor_add",
                            BinOp::Mul => "tl_tensor_mul",
                            _ => return Err("Unsupported tensor op".into()),
                        };

                        let runtime_fn = self
                            .module
                            .get_function(fn_name)
                            .ok_or(format!("Runtime function {} not found", fn_name))?;
                        let call = self
                            .builder
                            .build_call(runtime_fn, &[l.into(), r.into()], "tensor_op")
                            .map_err(|e| e.to_string())?;
                        let res = match call.try_as_basic_value() {
                            ValueKind::Basic(v) => v,
                            ValueKind::Instruction(_) => {
                                return Err("Invalid call return: Void".into())
                            }
                        };
                        // Result type is same as operands for now
                        Ok((res, left.1.clone()))
                    }
                    _ => Err(format!("Type mismatch: {:?} vs {:?}", left.1, right.1)),
                }
            }
            Expr::TensorLiteral(elements) => {
                let rank = 1; // Simplify: rank 1 for flat list
                let len = elements.len() as u64;

                let f32_type = self.context.f32_type();
                let i64_type = self.context.i64_type();
                let f32_ptr_type = f32_type.ptr_type(inkwell::AddressSpace::default());

                // 1. Alloca for data
                let data_alloca = self
                    .builder
                    .build_array_alloca(f32_type, i64_type.const_int(len, false), "tensor_data")
                    .unwrap();

                // 2. Store elements
                for (i, elem) in elements.iter().enumerate() {
                    let (val, _) = self.compile_expr(elem)?;
                    // Assume float
                    let float_val = if val.is_float_value() {
                        val.into_float_value()
                    } else if val.is_int_value() {
                        // Cast int to float
                        self.builder
                            .build_signed_int_to_float(val.into_int_value(), f32_type, "cast")
                            .unwrap()
                    } else {
                        return Err("Tensor elements must be number".into());
                    };

                    unsafe {
                        // GEP to index
                        let ptr = self
                            .builder
                            .build_in_bounds_gep(
                                f32_type,
                                data_alloca,
                                &[i64_type.const_int(i as u64, false)],
                                "elem_ptr",
                            )
                            .unwrap();
                        self.builder.build_store(ptr, float_val).unwrap();
                    }
                }

                // 3. Alloca for shape
                let shape_alloca = self
                    .builder
                    .build_array_alloca(i64_type, i64_type.const_int(1, false), "tensor_shape")
                    .unwrap();
                unsafe {
                    let ptr = self
                        .builder
                        .build_in_bounds_gep(
                            i64_type,
                            shape_alloca,
                            &[i64_type.const_int(0, false)],
                            "shape_0",
                        )
                        .unwrap();
                    self.builder
                        .build_store(ptr, i64_type.const_int(len, false))
                        .unwrap();
                }

                // 4. Call Runtime
                let fn_name = "tl_tensor_new";
                let f = self
                    .module
                    .get_function(fn_name)
                    .ok_or("Runtime fn not found")?;
                let call = self
                    .builder
                    .build_call(
                        f,
                        &[
                            data_alloca.into(),
                            i64_type.const_int(rank, false).into(),
                            shape_alloca.into(),
                        ],
                        "new_tensor",
                    )
                    .map_err(|e| e.to_string())?;

                let res = match call.try_as_basic_value() {
                    ValueKind::Basic(v) => v,
                    _ => return Err("Invalid call return".into()),
                };

                // Type is Tensor<f32, 1>
                Ok((res, Type::Tensor(Box::new(Type::F32), 1)))
            }
            Expr::FnCall(name, args) => {
                if name == "print" {
                    // Assume print calls tl_tensor_print for tensor args
                    // For now only support single arg print(tensor)
                    if args.len() != 1 {
                        return Err("print takes exactly 1 argument".into());
                    }
                    let (val, ty) = self.compile_expr(&args[0])?;

                    match ty {
                        Type::Tensor(_, _) => {
                            let runtime_fn = self.module.get_function("tl_tensor_print").unwrap();
                            let arg = val.into_pointer_value();
                            self.builder
                                .build_call(runtime_fn, &[arg.into()], "")
                                .map_err(|e| e.to_string())?;
                            Ok((
                                self.context.i64_type().const_int(0, false).into(),
                                Type::Void,
                            )) // Void return
                        }
                        _ => Err("Print only supports tensors for now".into()),
                    }
                } else {
                    // Generic function call - placeholder
                    // Assume defined in module
                    let func = self
                        .module
                        .get_function(name)
                        .ok_or(format!("Function {} not found", name))?;
                    // TODO: compile args
                    self.builder
                        .build_call(func, &[], "call")
                        .map_err(|e| e.to_string())?;
                    Ok((
                        self.context.i64_type().const_int(0, false).into(),
                        Type::Void,
                    ))
                }
            }
            _ => Err("Unsupported expression".into()),
        }
    }

    fn create_entry_block_alloca(
        &self,
        name: &str,
        ty: &Type,
    ) -> inkwell::values::PointerValue<'ctx> {
        let builder = self.context.create_builder();
        let entry = self
            .builder
            .get_insert_block()
            .unwrap()
            .get_parent()
            .unwrap()
            .get_first_basic_block()
            .unwrap();
        match entry.get_first_instruction() {
            Some(first_instr) => builder.position_before(&first_instr),
            None => builder.position_at_end(entry),
        }

        let llvm_type: inkwell::types::BasicTypeEnum = match ty {
            Type::I64 => self.context.i64_type().into(),
            Type::F32 => self.context.f32_type().into(),
            // Tensor is a pointer to OpaqueTensor struct.
            // We represent it as a generic pointer (ptr) in LLVM 15+, or i8* in older.
            // Inkwell Context has ptr_type
            Type::Tensor(_, _) => self
                .context
                .ptr_type(inkwell::AddressSpace::default())
                .into(),
            _ => self.context.i64_type().into(),
        };

        builder.build_alloca(llvm_type, name).unwrap()
    }

    // Debug method to print IR
    pub fn dump_llvm_ir(&self) {
        self.module.print_to_stderr();
    }
}
