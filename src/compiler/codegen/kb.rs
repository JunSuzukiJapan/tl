use crate::compiler::ast::*;
use crate::compiler::codegen::CodeGenerator;
use inkwell::values::{BasicValueEnum, ValueKind};
use inkwell::AddressSpace;
use std::collections::HashSet;

impl<'ctx> CodeGenerator<'ctx> {
    pub fn compile_kb_init_function(
        &mut self,
        module: &Module,
        module_name: &str,
    ) -> Result<Option<inkwell::values::FunctionValue<'ctx>>, String> {
        // Generate unique function name
        let fn_name = if module_name.is_empty() || module_name == "main" {
            "_tl_init_kb".to_string()
        } else {
            format!("_tl_init_kb_{}", module_name)
        };

        let void_type = self.context.void_type();
        let fn_type = void_type.fn_type(&[], false);

        // Get or add function
        let function = self
            .module
            .get_function(&fn_name)
            .unwrap_or_else(|| self.module.add_function(&fn_name, fn_type, None));

        // If function already has body (e.g. visited twice?), skip?
        // But we are compiling it now.
        if function.count_basic_blocks() > 0 {
            return Ok(Some(function));
        }

        let entry_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry_block);

        // 0. Call submodule inits
        for (sub_name, _) in &module.submodules {
            let sub_init_name = format!("_tl_init_kb_{}", sub_name);
            let sub_fn = self
                .module
                .get_function(&sub_init_name)
                .unwrap_or_else(|| self.module.add_function(&sub_init_name, fn_type, None));
            self.builder
                .build_call(sub_fn, &[], "")
                .map_err(|e| e.to_string())?;
        }

        // 1. Collect entities from Facts (Rules with empty body)
        // We use this to distinguish Constants vs Variables in Rules
        let mut known_entities = HashSet::new();
        for rule in &module.rules {
            if rule.body.is_empty() {
                for arg in &rule.head.args {
                    self.collect_entities(&arg.inner, &mut known_entities);
                }
            }
        }

        // 2. Iterate Rules
        for rule in &module.rules {
            if rule.body.is_empty() {
                self.compile_fact(rule)?;
            } else {
                self.compile_rule(rule, &known_entities)?;
            }
        }

        // 3. Call infer
        if let Some(f) = self.module.get_function("tl_kb_infer") {
            self.builder
                .build_call(f, &[], "")
                .map_err(|e| e.to_string())?;
        }

        self.builder.build_return(None).map_err(|e| e.to_string())?;

        if function.verify(true) {
            Ok(Some(function))
        } else {
            Err("Invalid generated KB init function".into())
        }
    }

    fn collect_entities(&self, expr_kind: &ExprKind, entities: &mut HashSet<String>) {
        match expr_kind {
            ExprKind::Symbol(name) => {
                entities.insert(name.clone());
            }
            // Literal Strings could also be entities in some interpretations?
            // For now sticking to Symbols/Identifiers.
            _ => {}
        }
    }

    fn compile_fact(&self, rule: &Rule) -> Result<(), String> {
        let head = &rule.head;
        let relation_name = &head.predicate;

        // Prepare args array
        let arg_count = head.args.len();
        let i64_type = self.context.i64_type();

        let args_array_type = i64_type.array_type(arg_count as u32);
        let args_alloca = self
            .builder
            .build_alloca(args_array_type, "fact_args")
            .map_err(|e| e.to_string())?;

        for (i, arg) in head.args.iter().enumerate() {
            let val = self.resolve_fact_arg(&arg.inner)?;
            let idx = i64_type.const_int(i as u64, false);
            // GEP into array
            // Try build_struct_gep or plain build_gep. For array it's build_gep.
            let ptr = unsafe {
                self.builder
                    .build_gep(
                        args_array_type,
                        args_alloca,
                        &[i64_type.const_int(0, false).into(), idx.into()],
                        "arg_ptr",
                    )
                    .map_err(|e| e.to_string())?
            };
            self.builder
                .build_store(ptr, val)
                .map_err(|e| e.to_string())?;
        }

        // Call tl_kb_add_fact(rel, args_ptr, arity)
        let add_fact_fn = self.module.get_function("tl_kb_add_fact").unwrap();

        // build_global_string_ptr works but returns pointer inside the module.
        let rel_str = self
            .builder
            .build_global_string_ptr(relation_name, "rel_name")
            .map_err(|e| e.to_string())?;

        let args_ptr = self
            .builder
            .build_pointer_cast(
                args_alloca,
                self.context.ptr_type(AddressSpace::default()),
                "args_ptr_cast",
            )
            .unwrap();

        self.builder
            .build_call(
                add_fact_fn,
                &[
                    rel_str.as_pointer_value().into(),
                    args_ptr.into(),
                    i64_type.const_int(arg_count as u64, false).into(),
                ],
                "",
            )
            .map_err(|e| e.to_string())?;

        Ok(())
    }

    fn resolve_fact_arg(&self, expr_kind: &ExprKind) -> Result<BasicValueEnum<'ctx>, String> {
        match expr_kind {
            ExprKind::Symbol(name) => {
                // It's an entity, call tl_kb_add_entity(name)
                let add_entity_fn = self.module.get_function("tl_kb_add_entity").unwrap();
                let name_ptr = self
                    .builder
                    .build_global_string_ptr(name, "entity_name")
                    .map_err(|e| e.to_string())?;
                let call = self
                    .builder
                    .build_call(
                        add_entity_fn,
                        &[name_ptr.as_pointer_value().into()],
                        "entity_id",
                    )
                    .map_err(|e| e.to_string())?;

                let val_basic = match call.try_as_basic_value() {
                    ValueKind::Basic(v) => v,
                    _ => return Err("Expected basic value from tl_kb_add_entity".to_string()),
                };
                Ok(val_basic)
            }
            ExprKind::Int(val) => Ok(self.context.i64_type().const_int(*val as u64, true).into()),
            _ => Err(format!(
                "Unsupported expression in fact arg: {:?}",
                expr_kind
            )),
        }
    }

    fn compile_rule(&self, rule: &Rule, known_entities: &HashSet<String>) -> Result<(), String> {
        // Collect variables in this rule to map them to indices
        let mut local_vars: std::collections::HashMap<String, i64> =
            std::collections::HashMap::new();
        let mut next_var_idx = 0;

        // Start Rule
        let rule_start_fn = self.module.get_function("tl_kb_rule_start").unwrap();
        let head_rel_str = self
            .builder
            .build_global_string_ptr(&rule.head.predicate, "head_rel")
            .map_err(|e| e.to_string())?;
        self.builder
            .build_call(rule_start_fn, &[head_rel_str.as_pointer_value().into()], "")
            .map_err(|e| e.to_string())?;

        // Head Args
        for arg in &rule.head.args {
            self.emit_rule_arg(
                &arg.inner,
                &mut local_vars,
                &mut next_var_idx,
                known_entities,
                true,
            )?;
        }

        // Body Atoms
        let add_body_atom_fn = self
            .module
            .get_function("tl_kb_rule_add_body_atom")
            .unwrap();
        for atom in &rule.body {
            let rel_str = self
                .builder
                .build_global_string_ptr(&atom.predicate, "body_rel")
                .map_err(|e| e.to_string())?;
            self.builder
                .build_call(add_body_atom_fn, &[rel_str.as_pointer_value().into()], "")
                .map_err(|e| e.to_string())?;

            for arg in &atom.args {
                self.emit_rule_arg(
                    &arg.inner,
                    &mut local_vars,
                    &mut next_var_idx,
                    known_entities,
                    false,
                )?;
            }
        }

        // Finish Rule
        let finish_fn = self.module.get_function("tl_kb_rule_finish").unwrap();
        self.builder
            .build_call(finish_fn, &[], "")
            .map_err(|e| e.to_string())?;

        Ok(())
    }

    fn emit_rule_arg(
        &self,
        expr_kind: &ExprKind,
        local_vars: &mut std::collections::HashMap<String, i64>,
        next_var_idx: &mut i64,
        known_entities: &HashSet<String>,
        is_head: bool,
    ) -> Result<(), String> {
        let (is_var, val) = match expr_kind {
            ExprKind::Symbol(name) => {
                if known_entities.contains(name) {
                    // Constant Entity
                    let add_entity_fn = self.module.get_function("tl_kb_add_entity").unwrap();
                    let name_ptr = self
                        .builder
                        .build_global_string_ptr(name, "entity_name")
                        .map_err(|e| e.to_string())?;
                    let call = self
                        .builder
                        .build_call(
                            add_entity_fn,
                            &[name_ptr.as_pointer_value().into()],
                            "entity_id",
                        )
                        .map_err(|e| e.to_string())?;

                    let val_i64 = match call.try_as_basic_value() {
                        ValueKind::Basic(v) => v,
                        _ => return Err("Expected return value from tl_kb_add_entity".to_string()),
                    };
                    (false, val_i64)
                } else {
                    // Variable
                    let idx = if let Some(&idx) = local_vars.get(name) {
                        idx
                    } else {
                        let idx = *next_var_idx;
                        local_vars.insert(name.clone(), idx);
                        *next_var_idx += 1;
                        idx
                    };
                    (
                        true,
                        self.context.i64_type().const_int(idx as u64, false).into(),
                    )
                }
            }
            ExprKind::LogicVar(name) => {
                // Explicit variable
                let idx = if let Some(&idx) = local_vars.get(name) {
                    idx
                } else {
                    let idx = *next_var_idx;
                    local_vars.insert(name.clone(), idx);
                    *next_var_idx += 1;
                    idx
                };
                (
                    true,
                    self.context.i64_type().const_int(idx as u64, false).into(),
                )
            }
            ExprKind::Int(val) => (
                false,
                self.context.i64_type().const_int(*val as u64, true).into(),
            ),
            _ => {
                return Err(format!(
                    "Unsupported expression in rule arg: {:?}",
                    expr_kind
                ))
            }
        };

        let suffix = if is_head { "head_arg" } else { "body_arg" };
        let type_suffix = if is_var { "var" } else { "const" };
        let fn_name = format!("tl_kb_rule_add_{}_{}", suffix, type_suffix);
        let fn_val = self
            .module
            .get_function(&fn_name)
            .ok_or(format!("Runtime function {} not found", fn_name))?;

        self.builder
            .build_call(fn_val, &[val.into()], "")
            .map_err(|e| e.to_string())?;

        Ok(())
    }
}
