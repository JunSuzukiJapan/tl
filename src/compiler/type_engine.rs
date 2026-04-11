use std::collections::HashMap;
use crate::compiler::ast::Type;

#[derive(Clone, Debug, Default)]
pub struct TypeEngine {
    bindings: HashMap<u64, Type>,
    next_id: u64,
}

impl TypeEngine {
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
            next_id: 1, // Reserve 0
        }
    }

    pub fn get_next_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn new_undefined(&mut self) -> Type {
        Type::Undefined(self.get_next_id())
    }

    /// Recursively flattens and resolves a type using the current bindings graph.
    pub fn resolve(&self, ty: Type) -> Type {
        match ty {
            Type::Undefined(id) => {
                if let Some(bound_ty) = self.bindings.get(&id) {
                    // Deep resolve to follow chains of constraints
                    self.resolve(bound_ty.clone())
                } else {
                    Type::Undefined(id)
                }
            }
            Type::Struct(name, generics) => {
                let resolved_generics = generics.into_iter().map(|g| self.resolve(g)).collect();
                Type::Struct(name, resolved_generics)
            }
            Type::Enum(name, generics) => {
                let resolved_generics = generics.into_iter().map(|g| self.resolve(g)).collect();
                Type::Enum(name, resolved_generics)
            }
            Type::Array(inner, size) => Type::Array(Box::new(self.resolve(*inner)), size),
            Type::Tensor(inner, rank) => Type::Tensor(Box::new(self.resolve(*inner)), rank),
            Type::Ptr(inner) => Type::Ptr(Box::new(self.resolve(*inner))),
            Type::Fn(args, ret) => {
                let resolved_args = args.into_iter().map(|a| self.resolve(a)).collect();
                Type::Fn(resolved_args, Box::new(self.resolve(*ret)))
            }
            Type::Tuple(types) => {
                let resolved = types.into_iter().map(|t| self.resolve(t)).collect();
                Type::Tuple(resolved)
            }
            _ => ty,
        }
    }

    /// Core Bidirectional Unification Algorithm.
    /// Unifies two types together, returning true if successful.
    pub fn unify(&mut self, t1: &Type, t2: &Type) -> bool {
        let t1 = self.resolve(t1.clone());
        let t2 = self.resolve(t2.clone());

        if t1 == t2 {
            return true;
        }
        

        match (t1, t2) {
            (Type::Undefined(id), other) => {
                if self.occurs(id, &other) {
                    return false; // (e.g. Type = Vec<Type>)
                }
                self.bindings.insert(id, other);
                true
            }
            (other, Type::Undefined(id)) => {
                if self.occurs(id, &other) {
                    return false; // (e.g. Type = Vec<Type>)
                }
                self.bindings.insert(id, other);
                true
            }
            (Type::Struct(n1, g1), Type::Struct(n2, g2)) |
            (Type::Enum(n1, g1), Type::Enum(n2, g2)) => {
                if n1 != n2 || g1.len() != g2.len() {
                    return false;
                }
                for (a1, a2) in g1.iter().zip(g2.iter()) {
                    if !self.unify(a1, a2) {
                        return false;
                    }
                }
                true
            }
            (Type::Array(inner1, s1), Type::Array(inner2, s2)) => {
                let res = s1 == s2 && self.unify(&inner1, &inner2);
                res
            }
            (Type::Tensor(inner1, r1), Type::Tensor(inner2, r2)) => {
                let res = (r1 == r2 || r1 == 0 || r2 == 0) && self.unify(&inner1, &inner2);
                res
            }
            (Type::Ptr(i1), Type::Ptr(i2)) => self.unify(&i1, &i2),
            (Type::Fn(args1, ret1), Type::Fn(args2, ret2)) => {
                if args1.len() != args2.len() {
                    return false;
                }
                for (a1, a2) in args1.iter().zip(args2.iter()) {
                    if !self.unify(a1, a2) {
                        return false;
                    }
                }
                let res = self.unify(&ret1, &ret2);
                res
            }
            (Type::Tuple(t1), Type::Tuple(t2)) => {
                if t1.len() != t2.len() {
                    return false;
                }
                for (a1, a2) in t1.iter().zip(t2.iter()) {
                    if !self.unify(a1, a2) {
                        return false;
                    }
                }
                true
            }
            // Dynamic language feature: Entity works with everything during compile time inference
            (Type::Entity, _) | (_, Type::Entity) => true,
            (_a, _b) => {
                false
            }
        }
    }

    /// Occurs check: ensures we do not bind a variable to a type that contains itself,
    /// avoiding infinite loops in resolution.
    fn occurs(&self, id: u64, ty: &Type) -> bool {
        let ty_res = self.resolve(ty.clone());
        let res = match ty_res {
            Type::Undefined(other_id) => id == other_id,
            Type::Struct(_, ref generics) | Type::Enum(_, ref generics) => {
                generics.iter().any(|g| self.occurs(id, g))
            }
            Type::Array(ref inner, _) | Type::Tensor(ref inner, _) | Type::Ptr(ref inner) => {
                self.occurs(id, inner)
            }
            Type::Fn(ref args, ref ret) => {
                args.iter().any(|a| self.occurs(id, a)) || self.occurs(id, ret)
            }
            Type::Tuple(ref types) => types.iter().any(|t| self.occurs(id, t)),
            _ => false,
        };
        res
    }
}
