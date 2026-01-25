use crate::compiler::ast::*;
use crate::compiler::codegen::CodeGenerator;
use std::collections::HashSet;

impl<'ctx> CodeGenerator<'ctx> {
    pub fn compile_kb_init_function(
        &mut self,
        module: &Module,
        module_name: &str,
    ) -> Result<Option<inkwell::values::FunctionValue<'ctx>>, String> {
        let fn_name = if module_name.is_empty() || module_name == "main" {
            "_tl_init_kb".to_string()
        } else {
            format!("_tl_init_kb_{}", module_name)
        };

        let void_type = self.context.void_type();
        let fn_type = void_type.fn_type(&[], false);

        let function = self
            .module
            .get_function(&fn_name)
            .unwrap_or_else(|| self.module.add_function(&fn_name, fn_type, None));

        if function.count_basic_blocks() > 0 {
            return Ok(Some(function));
        }

        let entry_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry_block);

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

        let mut known_entities = HashSet::new();
        for rule in &module.rules {
            if rule.body.is_empty() {
                for arg in &rule.head.args {
                    self.collect_entities(&arg.inner, &mut known_entities);
                }
            }
        }

        for rule in &module.rules {
            if rule.body.is_empty() {
                self.compile_fact(rule)?;
            } else {
                self.compile_rule(rule, &known_entities)?;
            }
        }

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
            ExprKind::Symbol(name) | ExprKind::Variable(name) => {
                entities.insert(name.clone());
            }
            _ => {}
        }
    }

    fn compile_fact(&self, rule: &Rule) -> Result<(), String> {
        let head = &rule.head;
        let relation_name = &head.predicate;

        let clear_fn = self.module.get_function("tl_kb_fact_args_clear").unwrap();
        self.builder.build_call(clear_fn, &[], "").map_err(|e| e.to_string())?;

        for arg in &head.args {
            self.emit_fact_arg(&arg.inner)?;
        }

        let add_fact_fn = self.module.get_function("tl_kb_add_fact_serialized").unwrap();
        let rel_str = self
            .builder
            .build_global_string_ptr(relation_name, "rel_name")
            .map_err(|e| e.to_string())?;

        self.builder
            .build_call(add_fact_fn, &[rel_str.as_pointer_value().into()], "")
            .map_err(|e| e.to_string())?;

        Ok(())
    }

    fn emit_fact_arg(&self, expr_kind: &ExprKind) -> Result<(), String> {
        match expr_kind {
            ExprKind::Symbol(name) | ExprKind::Variable(name) => {
                let add_entity_fn = self.module.get_function("tl_kb_add_entity").unwrap();
                let name_ptr = self
                    .builder
                    .build_global_string_ptr(name, "entity_name")
                    .map_err(|e| e.to_string())?;
                let call = self
                    .builder
                    .build_call(add_entity_fn, &[name_ptr.as_pointer_value().into()], "entity_id")
                    .map_err(|e| e.to_string())?;
                
                let entity_id = match call.try_as_basic_value() {
                    inkwell::values::ValueKind::Basic(v) => v,
                    _ => return Err("Failed to get basic value from tl_kb_add_entity".into()),
                };
                
                let add_arg_fn = self.module.get_function("tl_kb_fact_args_add_entity").unwrap();
                self.builder.build_call(add_arg_fn, &[entity_id.into()], "").map_err(|e| e.to_string())?;
            }
            ExprKind::Int(val) => {
                let add_arg_fn = self.module.get_function("tl_kb_fact_args_add_int").unwrap();
                let v = self.context.i64_type().const_int(*val as u64, true);
                self.builder.build_call(add_arg_fn, &[v.into()], "").map_err(|e| e.to_string())?;
            }
            ExprKind::Float(val) => {
                let add_arg_fn = self.module.get_function("tl_kb_fact_args_add_float").unwrap();
                let v = self.context.f64_type().const_float(*val);
                self.builder.build_call(add_arg_fn, &[v.into()], "").map_err(|e| e.to_string())?;
            }
            ExprKind::Bool(val) => {
                let add_arg_fn = self.module.get_function("tl_kb_fact_args_add_bool").unwrap();
                let v = self.context.bool_type().const_int(if *val { 1 } else { 0 }, false);
                self.builder.build_call(add_arg_fn, &[v.into()], "").map_err(|e| e.to_string())?;
            }
            ExprKind::StringLiteral(val) => {
                let add_arg_fn = self.module.get_function("tl_kb_fact_args_add_string").unwrap();
                let s_ptr = self.builder.build_global_string_ptr(val, "fact_str").map_err(|e| e.to_string())?;
                self.builder.build_call(add_arg_fn, &[s_ptr.as_pointer_value().into()], "").map_err(|e| e.to_string())?;
            }
            _ => return Err(format!("Unsupported expression in fact arg: {:?}", expr_kind)),
        }
        Ok(())
    }

    fn compile_rule(&self, rule: &Rule, known_entities: &HashSet<String>) -> Result<(), String> {
        let mut local_vars: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
        let mut next_var_idx = 0;

        let rule_start_fn = self.module.get_function("tl_kb_rule_start").unwrap();
        let head_rel_str = self
            .builder
            .build_global_string_ptr(&rule.head.predicate, "head_rel")
            .map_err(|e| e.to_string())?;
        self.builder
            .build_call(rule_start_fn, &[head_rel_str.as_pointer_value().into()], "")
            .map_err(|e| e.to_string())?;

        for arg in &rule.head.args {
            self.emit_rule_arg(&arg.inner, &mut local_vars, &mut next_var_idx, known_entities, true)?;
        }

        let add_body_atom_fn = self.module.get_function("tl_kb_rule_add_body_atom").unwrap();
        let add_body_atom_neg_fn = self
            .module
            .get_function("tl_kb_rule_add_body_atom_neg")
            .unwrap();

        let mut tmp_idx = 0i64;
        let lowered_body = lower_rule_body(&rule.body, &mut tmp_idx)?;

        for lit in &lowered_body {
            let (atom, negated) = match lit {
                LogicLiteral::Pos(a) => (a, false),
                LogicLiteral::Neg(a) => (a, true),
            };
            let rel_str = self
                .builder
                .build_global_string_ptr(&atom.predicate, "body_rel")
                .map_err(|e| e.to_string())?;
            let add_fn = if negated {
                add_body_atom_neg_fn
            } else {
                add_body_atom_fn
            };
            self.builder
                .build_call(add_fn, &[rel_str.as_pointer_value().into()], "")
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

        let finish_fn = self.module.get_function("tl_kb_rule_finish").unwrap();
        self.builder.build_call(finish_fn, &[], "").map_err(|e| e.to_string())?;

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
        let suffix = if is_head { "head_arg" } else { "body_arg" };

        match expr_kind {
            ExprKind::Symbol(name) | ExprKind::Variable(name) => {
                if known_entities.contains(name) {
                    let add_entity_fn = self.module.get_function("tl_kb_add_entity").unwrap();
                    let name_ptr = self.builder.build_global_string_ptr(name, "entity_name").map_err(|e| e.to_string())?;
                    let call = self.builder.build_call(add_entity_fn, &[name_ptr.as_pointer_value().into()], "entity_id").map_err(|e| e.to_string())?;
                    
                    let entity_id = match call.try_as_basic_value() {
                        inkwell::values::ValueKind::Basic(v) => v,
                        _ => return Err("Failed to get basic value from tl_kb_add_entity".into()),
                    };
                    
                    let fn_name = format!("tl_kb_rule_add_{}_const_entity", suffix);
                    let fn_val = self.module.get_function(&fn_name).unwrap();
                    self.builder.build_call(fn_val, &[entity_id.into()], "").map_err(|e| e.to_string())?;
                } else {
                    let idx = *local_vars.entry(name.clone()).or_insert_with(|| {
                        let i = *next_var_idx;
                        *next_var_idx += 1;
                        i
                    });
                    let fn_name = format!("tl_kb_rule_add_{}_var", suffix);
                    let fn_val = self.module.get_function(&fn_name).unwrap();
                    let v = self.context.i64_type().const_int(idx as u64, false);
                    self.builder.build_call(fn_val, &[v.into()], "").map_err(|e| e.to_string())?;
                }
            }
            ExprKind::LogicVar(name) => {
                let idx = *local_vars.entry(name.clone()).or_insert_with(|| {
                    let i = *next_var_idx;
                    *next_var_idx += 1;
                    i
                });
                let fn_name = format!("tl_kb_rule_add_{}_var", suffix);
                let fn_val = self.module.get_function(&fn_name).unwrap();
                let v = self.context.i64_type().const_int(idx as u64, false);
                self.builder.build_call(fn_val, &[v.into()], "").map_err(|e| e.to_string())?;
            }
            ExprKind::Int(val) => {
                let fn_name = format!("tl_kb_rule_add_{}_const_int", suffix);
                let fn_val = self.module.get_function(&fn_name).unwrap();
                let v = self.context.i64_type().const_int(*val as u64, true);
                self.builder.build_call(fn_val, &[v.into()], "").map_err(|e| e.to_string())?;
            }
            ExprKind::Float(val) => {
                let fn_name = format!("tl_kb_rule_add_{}_const_float", suffix);
                let fn_val = self.module.get_function(&fn_name).unwrap();
                let v = self.context.f64_type().const_float(*val);
                self.builder.build_call(fn_val, &[v.into()], "").map_err(|e| e.to_string())?;
            }
            _ => return Err(format!("Unsupported expression in rule arg: {:?}", expr_kind)),
        }

        Ok(())
    }
}

fn is_builtin_relation(pred: &str) -> bool {
    matches!(
        pred,
        ">"
            | "<"
            | ">="
            | "<="
            | "=="
            | "!="
            | "=:="
            | "=\\="
            | "\\="
            | "\\=="
            | "is"
            | "gt"
            | "lt"
            | "ge"
            | "le"
            | "eq"
            | "ne"
            | "add"
            | "sub"
            | "mul"
            | "div"
            | "mod"
            | "neg"
    )
}

fn is_simple_logic_arg(expr: &Expr) -> bool {
    matches!(
        expr.inner,
        ExprKind::Symbol(_)
            | ExprKind::Variable(_)
            | ExprKind::LogicVar(_)
            | ExprKind::Int(_)
            | ExprKind::Float(_)
    )
}

fn make_tmp_expr(tmp_idx: &mut i64) -> Expr {
    let name = format!("__tl_tmp{}", *tmp_idx);
    *tmp_idx += 1;
    Spanned::dummy(ExprKind::Symbol(name))
}

fn lower_expr_to_simple(
    expr: &Expr,
    tmp_idx: &mut i64,
    out: &mut Vec<LogicLiteral>,
) -> Result<Expr, String> {
    if is_simple_logic_arg(expr) {
        return Ok(expr.clone());
    }

    match &expr.inner {
        ExprKind::BinOp(lhs, op, rhs) => {
            let left = lower_expr_to_simple(lhs, tmp_idx, out)?;
            let right = lower_expr_to_simple(rhs, tmp_idx, out)?;
            let tmp = make_tmp_expr(tmp_idx);
            let pred = match op {
                BinOp::Add => "add",
                BinOp::Sub => "sub",
                BinOp::Mul => "mul",
                BinOp::Div => "div",
                BinOp::Mod => "mod",
                _ => return Err(format!("Unsupported operator in logic expression: {:?}", op)),
            };
            out.push(LogicLiteral::Pos(Atom {
                predicate: pred.to_string(),
                args: vec![left, right, tmp.clone()],
            }));
            Ok(tmp)
        }
        ExprKind::UnOp(UnOp::Neg, inner) => {
            let val = lower_expr_to_simple(inner, tmp_idx, out)?;
            let tmp = make_tmp_expr(tmp_idx);
            out.push(LogicLiteral::Pos(Atom {
                predicate: "neg".to_string(),
                args: vec![val, tmp.clone()],
            }));
            Ok(tmp)
        }
        _ => Err(format!(
            "Unsupported expression in logic arithmetic: {:?}",
            expr.inner
        )),
    }
}

fn lower_builtin_literal(
    atom: &Atom,
    negated: bool,
    tmp_idx: &mut i64,
) -> Result<Vec<LogicLiteral>, String> {
    let mut out = Vec::new();
    if matches!(
        atom.predicate.as_str(),
        "add" | "sub" | "mul" | "div" | "mod"
    ) {
        if atom.args.len() != 3 {
            return Err(format!(
                "Builtin predicate {} requires three arguments",
                atom.predicate
            ));
        }
        out.push(if negated {
            LogicLiteral::Neg(atom.clone())
        } else {
            LogicLiteral::Pos(atom.clone())
        });
        return Ok(out);
    }

    if atom.predicate == "neg" {
        if atom.args.len() != 2 {
            return Err("Builtin predicate neg requires two arguments".to_string());
        }
        out.push(if negated {
            LogicLiteral::Neg(atom.clone())
        } else {
            LogicLiteral::Pos(atom.clone())
        });
        return Ok(out);
    }

    if atom.predicate == "is" {
        if atom.args.len() != 2 {
            return Err("is/2 requires two arguments".to_string());
        }
        if !is_simple_logic_arg(&atom.args[0]) {
            return Err("Left side of is/2 must be a simple term".to_string());
        }
        let right = lower_expr_to_simple(&atom.args[1], tmp_idx, &mut out)?;
        out.push(if negated {
            LogicLiteral::Neg(Atom {
                predicate: "is".to_string(),
                args: vec![atom.args[0].clone(), right],
            })
        } else {
            LogicLiteral::Pos(Atom {
                predicate: "is".to_string(),
                args: vec![atom.args[0].clone(), right],
            })
        });
        return Ok(out);
    }

    if atom.args.len() != 2 {
        return Err(format!(
            "Builtin predicate {} requires two arguments",
            atom.predicate
        ));
    }
    let left = lower_expr_to_simple(&atom.args[0], tmp_idx, &mut out)?;
    let right = lower_expr_to_simple(&atom.args[1], tmp_idx, &mut out)?;
    out.push(if negated {
        LogicLiteral::Neg(Atom {
            predicate: atom.predicate.clone(),
            args: vec![left, right],
        })
    } else {
        LogicLiteral::Pos(Atom {
            predicate: atom.predicate.clone(),
            args: vec![left, right],
        })
    });
    Ok(out)
}

fn lower_rule_body(
    body: &[LogicLiteral],
    tmp_idx: &mut i64,
) -> Result<Vec<LogicLiteral>, String> {
    let mut out = Vec::new();
    for lit in body {
        match lit {
            LogicLiteral::Pos(atom) => {
                if is_builtin_relation(&atom.predicate) {
                    out.extend(lower_builtin_literal(atom, false, tmp_idx)?);
                } else {
                    let mut lowered_args = Vec::new();
                    for arg in &atom.args {
                        if is_simple_logic_arg(arg) {
                            lowered_args.push(arg.clone());
                        } else {
                            let tmp = lower_expr_to_simple(arg, tmp_idx, &mut out)?;
                            lowered_args.push(tmp);
                        }
                    }
                    out.push(LogicLiteral::Pos(Atom {
                        predicate: atom.predicate.clone(),
                        args: lowered_args,
                    }));
                }
            }
            LogicLiteral::Neg(atom) => {
                if is_builtin_relation(&atom.predicate) {
                    out.extend(lower_builtin_literal(atom, true, tmp_idx)?);
                } else {
                    let mut lowered_args = Vec::new();
                    for arg in &atom.args {
                        if is_simple_logic_arg(arg) {
                            lowered_args.push(arg.clone());
                        } else {
                            let tmp = lower_expr_to_simple(arg, tmp_idx, &mut out)?;
                            lowered_args.push(tmp);
                        }
                    }
                    out.push(LogicLiteral::Neg(Atom {
                        predicate: atom.predicate.clone(),
                        args: lowered_args,
                    }));
                }
            }
        }
    }
    Ok(out)
}
