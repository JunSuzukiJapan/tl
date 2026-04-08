use crate::compiler::ast::*;
use std::collections::HashMap;

pub struct TypeSubstitutor {
    pub subst: HashMap<String, Type>,
}

impl TypeSubstitutor {
    pub fn new(subst: HashMap<String, Type>) -> Self {
        Self { subst }
    }

    pub fn substitute_type(&self, ty: &Type) -> Type {
        match ty {
            // UserDefined removed


            Type::Tensor(inner, rank) => Type::Tensor(Box::new(self.substitute_type(inner)), *rank),
            Type::TensorShaped(inner, dims) => Type::TensorShaped(Box::new(self.substitute_type(inner)), dims.clone()),
            Type::Struct(name, args) => {
                if let Some(s) = self.subst.get(name) {
                     if args.is_empty() {
                         return s.clone();
                     }
                }
                let new_args = args.iter().map(|a| self.substitute_type(a)).collect();
                Type::Struct(name.clone(), new_args)
            }
            Type::Enum(name, args) => {
                let new_args = args.iter().map(|a| self.substitute_type(a)).collect();
                Type::Enum(name.clone(), new_args)
            }
            Type::Tuple(types) => {
                let new_types = types.iter().map(|t| self.substitute_type(t)).collect();
                Type::Tuple(new_types)
            }
            Type::Path(segments, generics) => {
                 if segments.len() == 1 {
                     if let Some(s) = self.subst.get(&segments[0]) {
                         // If generic args exist on the param reference (e.g. T<U>), 
                         // we might need to apply them to the substituted type?
                         // But generic params T usually don't take args unless T is higher-kinded (not supported).
                         // Or if s is subst to generic Struct?
                         // If generics is empty, return s.
                         if generics.is_empty() {
                             return s.clone();
                         }
                         // If generics not empty, we are instantiating s with new generics?
                         // e.g. T<int> where T=Vec. -> Vec<int>.
                         if let Type::Struct(name, _) = s {
                              let new_generics = generics.iter().map(|g| self.substitute_type(g)).collect();
                              return Type::Struct(name.clone(), new_generics);
                         }
                         if let Type::Enum(name, _) = s {
                              let new_generics = generics.iter().map(|g| self.substitute_type(g)).collect();
                              return Type::Enum(name.clone(), new_generics);
                         }
                     }
                 }
                 let new_generics: Vec<Type> = generics.iter().map(|g| self.substitute_type(g)).collect();
                 
                 // Return Type::Path as-is - let codegen determine Struct vs Enum
                 // at compile time where it has access to struct_defs and enum_defs
                 Type::Path(segments.clone(), new_generics)
            }
            Type::SpecializedType { gen_type, type_args, type_map: _, mangled_name } => {
                let new_args: Vec<Type> = type_args.iter().map(|a| self.substitute_type(a)).collect();
                Type::SpecializedType {
                    gen_type: gen_type.clone(),
                    type_args: new_args,
                    type_map: vec![], // In ast_subst we may not have the full map preservation yet, or we can just drop it
                    mangled_name: mangled_name.clone(),
                }
            }
            Type::Ptr(inner) => Type::Ptr(Box::new(self.substitute_type(inner))),
            Type::Array(inner, size) => Type::Array(Box::new(self.substitute_type(inner)), *size),
            Type::Fn(args, ret) => {
                let new_args = args.iter().map(|a| self.substitute_type(a)).collect();
                let new_ret = Box::new(self.substitute_type(ret));
                Type::Fn(new_args, new_ret)
            }
            // Type::Ref(inner) => Type::Ref(Box::new(self.substitute_type(inner))), // REMOVED

            _ => ty.clone(),
        }
    }

    /// Convert a Type to a string representation for mangling.
    fn type_to_suffix(&self, ty: &Type) -> String {
        match ty {
            Type::I64 => "i64".to_string(),
            Type::I32 => "i32".to_string(),
            Type::U8 => "u8".to_string(),
            Type::F32 => "f32".to_string(),
            Type::F64 => "f64".to_string(),
            Type::Bool => "bool".to_string(),
            Type::Usize => "usize".to_string(),
            Type::Void => "void".to_string(),
            Type::String(_) => "String".to_string(),
            Type::Char(_) => "Char".to_string(),
            Type::Struct(name, args) => {
                if args.is_empty() || name.contains('[') {
                    name.clone()
                } else {
                    let args_str: Vec<String> = args.iter().map(|t| self.type_to_suffix(t)).collect();
                    mangle_wrap_args(name, &args_str)
                }
            }
            Type::Enum(name, args) => {
                if args.is_empty() || name.contains('[') {
                    name.clone()
                } else {
                    let args_str: Vec<String> = args.iter().map(|t| self.type_to_suffix(t)).collect();
                    mangle_wrap_args(name, &args_str)
                }
            }
            Type::Tensor(inner, rank) => {
                let args = vec![self.type_to_suffix(inner), rank.to_string()];
                mangle_wrap_args("Tensor", &args)
            }
            Type::Tuple(types) => {
                let parts: Vec<String> = types.iter().map(|t| self.type_to_suffix(t)).collect();
                mangle_wrap_args("Tuple", &parts)
            }
            Type::Path(segments, _) => segments.join("_"),
            _ => "unknown".to_string(),
        }
    }

    pub fn substitute_expr(&self, expr: &Expr) -> Expr {
        let new_kind = match &expr.inner {
            ExprKind::Variable(_) => expr.inner.clone(), // Variables are runtime values, type doesn't change name
            ExprKind::Float(_) | ExprKind::Int(_) | ExprKind::Bool(_) | ExprKind::StringLiteral(_) => expr.inner.clone(),
            
            ExprKind::Tuple(exprs) => {
                let new_exprs = exprs.iter().map(|e| self.substitute_expr(e)).collect();
                ExprKind::Tuple(new_exprs)
            }
            
            ExprKind::BinOp(l, op, r) => {
                ExprKind::BinOp(Box::new(self.substitute_expr(l)), op.clone(), Box::new(self.substitute_expr(r)))
            }
            ExprKind::UnOp(op, e) => {
                ExprKind::UnOp(op.clone(), Box::new(self.substitute_expr(e)))
            }
            
            ExprKind::FnCall(name, args) => {
                let new_args = args.iter().map(|a| self.substitute_expr(a)).collect();
                ExprKind::FnCall(name.clone(), new_args)
            }
            
            ExprKind::MethodCall(obj, method, args) => {
                let new_obj = self.substitute_expr(obj);
                let new_args = args.iter().map(|a| self.substitute_expr(a)).collect();
                ExprKind::MethodCall(Box::new(new_obj), method.clone(), new_args)
            }
            
            ExprKind::StaticMethodCall(ty, method, args) => {
                let new_ty = self.substitute_type(ty);
                let new_args = args.iter().map(|a| self.substitute_expr(a)).collect();
                ExprKind::StaticMethodCall(new_ty, method.clone(), new_args)
            }
            
            ExprKind::StructInit(ty, fields) => {
                let new_ty = self.substitute_type(ty);
                let new_fields = fields.iter().map(|(n, e)| (n.clone(), self.substitute_expr(e))).collect();
                ExprKind::StructInit(new_ty, new_fields)
            }
            
            ExprKind::Block(stmts) => {
                let new_stmts = stmts.iter().map(|s| self.substitute_stmt(s)).collect();
                ExprKind::Block(new_stmts)
            }
            
            ExprKind::IfExpr(cond, then_block, else_block) => {
                let new_cond = self.substitute_expr(cond);
                let new_then = then_block.iter().map(|s| self.substitute_stmt(s)).collect();
                let new_else = else_block.as_ref().map(|block| block.iter().map(|s| self.substitute_stmt(s)).collect());
                ExprKind::IfExpr(Box::new(new_cond), new_then, new_else)
            }
            

            
            ExprKind::As(expr, ty) => {
                let new_expr = self.substitute_expr(expr);
                let new_ty = self.substitute_type(ty);
                ExprKind::As(Box::new(new_expr), new_ty)
            }

            ExprKind::FieldAccess(expr, field) => {
                let new_expr = self.substitute_expr(expr);
                ExprKind::FieldAccess(Box::new(new_expr), field.clone())
            }
            
            ExprKind::IndexAccess(expr, indices) => {
                let new_expr = self.substitute_expr(expr);
                let new_indices = indices.iter().map(|e| self.substitute_expr(e)).collect();
                ExprKind::IndexAccess(Box::new(new_expr), new_indices)
            }
            
            ExprKind::EnumInit { enum_name, variant_name, generics, payload } => {
                let new_generics: Vec<Type> = generics.iter().map(|g| self.substitute_type(g)).collect();
                
                // Mangle enum name with concrete generics
                let new_enum_name = if !new_generics.is_empty() {
                    let suffix: Vec<String> = new_generics.iter().map(|t| self.type_to_suffix(t)).collect();
                    mangle_wrap_args(enum_name, &suffix)
                } else {
                    enum_name.clone()
                };
                
                let new_payload = match payload {
                    crate::compiler::ast::EnumVariantInit::Unit => crate::compiler::ast::EnumVariantInit::Unit,
                    crate::compiler::ast::EnumVariantInit::Tuple(exprs) => {
                        crate::compiler::ast::EnumVariantInit::Tuple(exprs.iter().map(|e| self.substitute_expr(e)).collect())
                    }
                    crate::compiler::ast::EnumVariantInit::Struct(fields) => {
                        crate::compiler::ast::EnumVariantInit::Struct(
                            fields.iter().map(|(n, e)| (n.clone(), self.substitute_expr(e))).collect()
                        )
                    }
                };
                ExprKind::EnumInit {
                    enum_name: new_enum_name,
                    variant_name: variant_name.clone(),
                    generics: new_generics, // Preserve type args for codegen
                    payload: new_payload,
                }
            }
            
            ExprKind::Match { expr: subject, arms } => {
                let new_subject = self.substitute_expr(subject);
                let new_arms: Vec<(crate::compiler::ast::Pattern, Expr)> = arms.iter().map(|(pattern, arm_expr)| {
                    (self.substitute_pattern(pattern), self.substitute_expr(arm_expr))
                }).collect();
                ExprKind::Match {
                    expr: Box::new(new_subject),
                    arms: new_arms,
                }
            }
            
            ExprKind::IfLet { pattern, expr, then_block, else_block } => {
                let new_expr = self.substitute_expr(expr);
                let new_then = then_block.iter().map(|s| self.substitute_stmt(s)).collect();
                let new_else = else_block.as_ref().map(|block| block.iter().map(|s| self.substitute_stmt(s)).collect());
                // Also substitute pattern if it has generics
                let new_pattern = self.substitute_pattern(pattern);
                ExprKind::IfLet {
                    pattern: new_pattern,
                    expr: Box::new(new_expr),
                    then_block: new_then,
                    else_block: new_else,
                }
            }
            
            ExprKind::Closure { args, return_type, body, captures } => {
                let new_args = args.iter().map(|(n, t)| (n.clone(), t.as_ref().map(|t| self.substitute_type(t)))).collect();
                let new_rt = return_type.as_ref().map(|t| self.substitute_type(t));
                let new_body = body.iter().map(|s| self.substitute_stmt(s)).collect();
                let new_captures = captures.iter().map(|(n, t, m)| (n.clone(), self.substitute_type(t), *m)).collect();
                ExprKind::Closure {
                    args: new_args,
                    return_type: new_rt,
                    body: new_body,
                    captures: new_captures,
                }
            }
            ExprKind::TensorLiteral(exprs) => {
                ExprKind::TensorLiteral(exprs.iter().map(|e| self.substitute_expr(e)).collect())
            }
            ExprKind::TensorConstLiteral(exprs) => {
                ExprKind::TensorConstLiteral(exprs.iter().map(|e| self.substitute_expr(e)).collect())
            }
            ExprKind::Try(expr) => {
                ExprKind::Try(Box::new(self.substitute_expr(expr)))
            }
            ExprKind::StaticConstAccess(ty, name) => {
                ExprKind::StaticConstAccess(self.substitute_type(ty), name.clone())
            }
            ExprKind::TensorComprehension { indices, clauses, body } => {
                let new_clauses = clauses.iter().map(|c| match c {
                    crate::compiler::ast::ComprehensionClause::Generator { name, range } => {
                        crate::compiler::ast::ComprehensionClause::Generator {
                            name: name.clone(),
                            range: self.substitute_expr(range),
                        }
                    }
                    crate::compiler::ast::ComprehensionClause::Condition(cond) => {
                        crate::compiler::ast::ComprehensionClause::Condition(self.substitute_expr(cond))
                    }
                }).collect();
                let new_body = body.as_ref().map(|e| Box::new(self.substitute_expr(e)));
                ExprKind::TensorComprehension {
                    indices: indices.clone(),
                    clauses: new_clauses,
                    body: new_body,
                }
            }
            // For now, handle commonly used kinds. Extend as needed.
            _ => expr.inner.clone(),
        };
        
        Spanned {
            inner: new_kind,
            span: expr.span.clone(),
        }
    }

     pub fn substitute_lvalue(&self, lvalue: &LValue) -> LValue {
         match lvalue {
              LValue::Variable(name) => LValue::Variable(name.clone()),
              LValue::FieldAccess(inner, field) => LValue::FieldAccess(Box::new(self.substitute_lvalue(inner)), field.clone()),
              LValue::IndexAccess(inner, indices) => {
                   let new_inner = self.substitute_lvalue(inner);
                   let new_indices = indices.iter().map(|e| self.substitute_expr(e)).collect();
                   LValue::IndexAccess(Box::new(new_inner), new_indices)
              }
         }
    }

    pub fn substitute_pattern(&self, pattern: &Pattern) -> Pattern {
        match pattern {
            Pattern::EnumPattern { enum_name, variant_name, bindings } => {
                // Mangle enum name with substituted generics if it's a generic enum
                // Check if enum_name contains generic params that need substitution
                // For now, just return the pattern as-is since the actual type substitution
                // happens through the subject expression's type in compile_match_like
                Pattern::EnumPattern {
                    enum_name: enum_name.clone(),
                    variant_name: variant_name.clone(),
                    bindings: bindings.clone(),
                }
            }
            Pattern::Wildcard => Pattern::Wildcard,
            Pattern::Literal(expr) => Pattern::Literal(Box::new(self.substitute_expr(expr))),
        }
    }

    pub fn substitute_stmt(&self, stmt: &Stmt) -> Stmt {
        let new_kind = match &stmt.inner {
            StmtKind::Let { name, type_annotation, value, mutable } => {
                let new_ty = type_annotation.as_ref().map(|t| self.substitute_type(t));
                let new_value = self.substitute_expr(value);
                StmtKind::Let {
                    name: name.clone(),
                    type_annotation: new_ty,
                    value: new_value,
                    mutable: *mutable,
                }
            }
            StmtKind::Expr(expr) => StmtKind::Expr(self.substitute_expr(expr)),
            StmtKind::Return(opt_expr) => {
                let new_expr = opt_expr.as_ref().map(|e| self.substitute_expr(e));
                StmtKind::Return(new_expr)
            }
            StmtKind::Assign { lhs, op, value } => {
                let new_lhs = self.substitute_lvalue(lhs);
                let new_value = self.substitute_expr(value);
                StmtKind::Assign {
                    lhs: new_lhs,
                    op: op.clone(),
                    value: new_value,
                }
            }
            StmtKind::For { loop_var, iterator, body } => {
                 let new_iter = self.substitute_expr(iterator);
                 let new_body = body.iter().map(|s| self.substitute_stmt(s)).collect();
                 StmtKind::For {
                     loop_var: loop_var.clone(),
                     iterator: new_iter,
                     body: new_body,
                 }
            }
            StmtKind::While { cond, body } => {
                let new_cond = self.substitute_expr(cond);
                let new_body = body.iter().map(|s| self.substitute_stmt(s)).collect();
                StmtKind::While {
                    cond: new_cond,
                    body: new_body,
                }
            }

            StmtKind::TensorDecl { name, type_annotation, init } => {
                let new_ty = self.substitute_type(type_annotation);
                let new_init = init.as_ref().map(|e| self.substitute_expr(e));
                StmtKind::TensorDecl {
                    name: name.clone(),
                    type_annotation: new_ty,
                    init: new_init,
                }
            }
            StmtKind::Loop { body } => {
                let new_body = body.iter().map(|s| self.substitute_stmt(s)).collect();
                StmtKind::Loop { body: new_body }
            }
            _ => stmt.inner.clone(),
        };
        
        Spanned {
            inner: new_kind,
            span: stmt.span.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_subst() -> TypeSubstitutor {
        let mut map = HashMap::new();
        map.insert("K".to_string(), Type::I64);
        map.insert("V".to_string(), Type::String("String".to_string()));
        TypeSubstitutor::new(map)
    }

    #[test]
    fn test_substitute_closure_captures_and_args() {
        let subst = make_subst();
        let closure = Spanned::dummy(ExprKind::Closure {
            args: vec![("x".to_string(), Some(Type::Struct("K".to_string(), vec![])))],
            return_type: Some(Type::Struct("V".to_string(), vec![])),
            body: vec![Spanned::dummy(StmtKind::Expr(Spanned::dummy(ExprKind::Variable("x".to_string()))))],
            captures: vec![("cap".to_string(), Type::Struct("K".to_string(), vec![]), false)],
        });

        let substituted = subst.substitute_expr(&closure);
        match &substituted.inner {
            ExprKind::Closure { args, return_type, captures, .. } => {
                assert_eq!(args[0].1, Some(Type::I64));
                assert_eq!(*return_type, Some(Type::String("String".to_string())));
                assert_eq!(captures[0].1, Type::I64);
            }
            _ => panic!("Expected Closure"),
        }
    }

    #[test]
    fn test_substitute_loop_body() {
        let subst = make_subst();
        let loop_stmt = Spanned::dummy(StmtKind::Loop {
            body: vec![Spanned::dummy(StmtKind::Let {
                name: "var".to_string(),
                type_annotation: Some(Type::Struct("K".to_string(), vec![])),
                value: Spanned::dummy(ExprKind::Variable("x".to_string())),
                mutable: false,
            })],
        });

        let substituted = subst.substitute_stmt(&loop_stmt);
        match &substituted.inner {
            StmtKind::Loop { body } => {
                match &body[0].inner {
                    StmtKind::Let { type_annotation, .. } => {
                        assert_eq!(*type_annotation, Some(Type::I64));
                    }
                    _ => panic!("Expected Let inside Loop"),
                }
            }
            _ => panic!("Expected Loop"),
        }
    }

    #[test]
    fn test_substitute_nested_types() {
        let subst = make_subst();
        
        let array_ty = Type::Array(Box::new(Type::Struct("K".to_string(), vec![])), 10);
        assert_eq!(subst.substitute_type(&array_ty), Type::Array(Box::new(Type::I64), 10));

        let fn_ty = Type::Fn(
            vec![Type::Struct("K".to_string(), vec![])],
            Box::new(Type::Struct("V".to_string(), vec![]))
        );
        let expected_fn = Type::Fn(
            vec![Type::I64],
            Box::new(Type::String("String".to_string()))
        );
        assert_eq!(subst.substitute_type(&fn_ty), expected_fn);
    }
}

