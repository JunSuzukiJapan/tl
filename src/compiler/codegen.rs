// src/compiler/codegen.rs
use crate::compiler::ast::*;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module as InkwellModule;
use inkwell::values::{BasicValueEnum, FunctionValue};
use std::collections::HashMap;

pub struct CodeGenerator<'ctx> {
    context: &'ctx Context,
    module: InkwellModule<'ctx>,
    builder: Builder<'ctx>,
    variables: HashMap<String, BasicValueEnum<'ctx>>,
}

impl<'ctx> CodeGenerator<'ctx> {
    pub fn new(context: &'ctx Context, module_name: &str) -> Self {
        let module = context.create_module(module_name);
        let builder = context.create_builder();
        CodeGenerator {
            context,
            module,
            builder,
            variables: HashMap::new(),
        }
    }

    pub fn compile_module(&mut self, ast_module: &Module) -> Result<(), String> {
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
        // Placeholder stmt compiler
        match stmt {
            Stmt::Let { name, value, .. } => {
                // Compile value
                // Store in stack
            }
            Stmt::Return(expr) => {
                // Compile expr
                // Build return
            }
            _ => {}
        }
        Ok(())
    }

    // Debug method to print IR
    pub fn dump_llvm_ir(&self) {
        self.module.print_to_stderr();
    }
}
