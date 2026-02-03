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
        eprintln!("AST_SUBST: {:?}", ty);
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
                 let new_generics = generics.iter().map(|g| self.substitute_type(g)).collect();
                 Type::Path(segments.clone(), new_generics)
            }
            Type::Ptr(inner) => Type::Ptr(Box::new(self.substitute_type(inner))),
            Type::Ref(inner) => Type::Ref(Box::new(self.substitute_type(inner))),


            _ => ty.clone(),
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
            
            // For now, handle commonly used kinds. Extend as needed.
            _ => expr.inner.clone(),
        };
        
        Spanned {
            inner: new_kind,
            span: expr.span.clone(),
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
            StmtKind::Assign { name, indices, op, value } => {
                let new_indices = indices.as_ref().map(|idxs| idxs.iter().map(|e| self.substitute_expr(e)).collect());
                let new_value = self.substitute_expr(value);
                StmtKind::Assign {
                    name: name.clone(),
                    indices: new_indices,
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
            _ => stmt.inner.clone(),
        };
        
        Spanned {
            inner: new_kind,
            span: stmt.span.clone(),
        }
    }
}
