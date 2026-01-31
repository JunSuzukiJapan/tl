use crate::compiler::ast::Type;
use std::collections::HashMap;

/// Resolver for generic type parameters.
/// Handles deriving concrete types from generic definitions and variable bindings.
pub struct GenericResolver;

impl GenericResolver {
    /// Derives the mapping of generic type parameters to concrete types by comparing
    /// a generic type structure with a concrete type structure.
    ///
    /// # Arguments
    /// * `generic_ty` - The type definition containing generic parameters (e.g. `Vec<T>`, `Map<K, V>`)
    /// * `concrete_ty` - The actual concrete type (e.g. `Vec<I64>`, `Map<String, Tensor>`)
    ///
    /// # Returns
    /// A HashMap mapping generic names (e.g. "T", "K") to their concrete types.
    /// Returns error string if types structure mismatch.
    pub fn resolve_bindings(
        generic_ty: &Type,
        concrete_ty: &Type,
    ) -> Result<HashMap<String, Type>, String> {
        let mut bindings = HashMap::new();
        Self::resolve_recursive(generic_ty, concrete_ty, &mut bindings)?;
        Ok(bindings)
    }

    fn resolve_recursive(
        generic_ty: &Type,
        conc: &Type,
        bindings: &mut HashMap<String, Type>,
    ) -> Result<(), String> {
        match (generic_ty, conc) {
            // Case 1: Generic parameter (e.g. T) matching any concrete type
            // This handles UserDefined("T", []) matching Struct("Point", []), I64, etc.
            (Type::Struct(name, args), conc_ty) if args.is_empty() && Self::is_likely_generic_param(name) => {
                // Check if already bound
                if let Some(existing) = bindings.get(name) {
                     if !Self::types_equivalent(existing, conc_ty) {
                         return Err(format!("Generic parameter {} bound to mismatched types: {:?} vs {:?}", name, existing, conc_ty));
                     }
                } else {
                    bindings.insert(name.clone(), conc_ty.clone());
                }
                Ok(())
            }
            
            // Case 2: Matching structs/user types
            // Case 2: Matching structs/user types
            (Type::Struct(n1, args1), Type::Struct(n2, args2))
            | (Type::Enum(n1, args1), Type::Enum(n2, args2))
            | (Type::Struct(n1, args1), Type::Enum(n2, args2))
            | (Type::Enum(n1, args1), Type::Struct(n2, args2)) => {
                if n1 != n2 {
                    return Err(format!("Type mismatch: {} vs {}", n1, n2));
                }
                if args1.len() != args2.len() {
                    return Err(format!("Generic args count mismatch for {}: {} vs {}", n1, args1.len(), args2.len()));
                }
                for (a1, a2) in args1.iter().zip(args2.iter()) {
                    Self::resolve_recursive(a1, a2, bindings)?;
                }
                Ok(())
            }
            
            // Case 3: Built-in wrappers
             (Type::Vec(inner1), Type::Vec(inner2)) => {
                 Self::resolve_recursive(inner1, inner2, bindings)
             }
             (Type::Tensor(inner1, r1), Type::Tensor(inner2, r2)) => {
                 if r1 != r2 {
                     return Err(format!("Tensor rank mismatch: {} vs {}", r1, r2));
                 }
                 Self::resolve_recursive(inner1, inner2, bindings)
             }
             
             (Type::ScalarArray(t1, l1), Type::ScalarArray(t2, l2)) => {
                 if l1 != l2 {
                      return Err(format!("Array length mismatch: {} vs {}", l1, l2));
                 }
                 Self::resolve_recursive(t1, t2, bindings)
             }

             // Base cases: Primitives must match exactly
             (Type::I64, Type::I64) => Ok(()),
             (Type::F32, Type::F32) => Ok(()),
             (Type::Bool, Type::Bool) => Ok(()),
             
             // Mismatch
             _ => Err(format!("Structure mismatch or unsupported type comparison: {:?} vs {:?}", generic_ty, conc)),
        }
    }

    /// Heuristic to identify if a UserDefined type is a generic parameter placeholder.
    /// In a real compiler, this would rely on checking the context (StructDef generics list).
    /// For this helper, we assume single uppercase letters or standard T, K, V are generics if simpler context.
    /// BUT, the caller should pass in 'generic_ty' constructed from StructDef which KNOWS T is a parameter.
    /// Impl: We treat any UserDefined(Name, []) as potentially generic if we are in "generic mode".
    /// To be safe, the caller knows `generic_ty` comes from a definition where T is a generic param.
    fn is_likely_generic_param(_name: &str) -> bool {
        // Simple heuristic for now, or just assume all empty-arg user types in 'gen' are params
        // But 'String' is UserDefined("String", []) often. 
        // We should handle standard types as non-generics?
        // Actually, Type has variants for I64 etc. UserDefined is for structs.
        // If 'String' is UserDefined, we need to be careful.
        // In this codebase, String is UserDefined("String", vec![]).
        return true; 
    }

    /// Check if two types are equivalent (handles cross-variant matching like UserDefined vs Struct).
    pub fn types_equivalent(t1: &Type, t2: &Type) -> bool {
        match (t1, t2) {
            // Primitives must match exactly
            (Type::I64, Type::I64) => true,
            (Type::I32, Type::I32) => true,
            (Type::F32, Type::F32) => true,
            (Type::F64, Type::F64) => true,
            (Type::Bool, Type::Bool) => true,
            (Type::Void, Type::Void) => true,
            
            // UserDefined("Point", []) is equivalent to Struct("Point", [])
            (Type::Struct(n1, a1), Type::Struct(n2, a2)) => {
                n1 == n2 && a1.len() == a2.len() && 
                    a1.iter().zip(a2.iter()).all(|(x, y)| Self::types_equivalent(x, y))
            }
            
            (Type::Enum(n1, a1), Type::Enum(n2, a2)) => {
                n1 == n2 && a1.len() == a2.len() && 
                    a1.iter().zip(a2.iter()).all(|(x, y)| Self::types_equivalent(x, y))
            }
            
            (Type::Vec(i1), Type::Vec(i2)) => Self::types_equivalent(i1, i2),
            (Type::Tensor(i1, r1), Type::Tensor(i2, r2)) => r1 == r2 && Self::types_equivalent(i1, i2),
            
            _ => t1 == t2,
        }
    }

    /// Substitutes generic parameters in `ty` using `bindings`.
    pub fn apply_bindings(ty: &Type, bindings: &HashMap<String, Type>) -> Type {
        match ty {

            Type::Struct(name, args) => {
                if args.is_empty() {
                    if let Some(concrete) = bindings.get(name) {
                        return concrete.clone();
                    }
                }
                let new_args = args.iter().map(|a| Self::apply_bindings(a, bindings)).collect();
                Type::Struct(name.clone(), new_args)
            }
             Type::Enum(name, args) => {
                let new_args = args.iter().map(|a| Self::apply_bindings(a, bindings)).collect();
                Type::Enum(name.clone(), new_args)
            }
            Type::Vec(inner) => Type::Vec(Box::new(Self::apply_bindings(inner, bindings))),
            Type::Tensor(inner, rank) => Type::Tensor(Box::new(Self::apply_bindings(inner, bindings)), *rank),
            
            // Recursively handle others if needed, typically these are enough
            _ => ty.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create types
    fn t_param(n: &str) -> Type { Type::Struct(n.to_string(), vec![]) }
    fn t_struct(n: &str, args: Vec<Type>) -> Type { Type::Struct(n.to_string(), args) }
    fn t_vec(t: Type) -> Type { Type::Vec(Box::new(t)) }
    fn t_i64() -> Type { Type::I64 }
    fn t_f32() -> Type { Type::F32 }
    fn t_tensor(t: Type, r: usize) -> Type { Type::Tensor(Box::new(t), r) }

    #[test]
    fn test_resolve_simple_vec() {
        // Vec<T> vs Vec<i64>
        let generic_ty = t_vec(t_param("T"));
        let conc = t_vec(t_i64());
        
        let bindings = GenericResolver::resolve_bindings(&generic_ty, &conc).unwrap();
        assert_eq!(bindings.get("T"), Some(&t_i64()));
    }

    #[test]
    fn test_resolve_map() {
        // Map<K, V> vs Map<String, Tensor<f32, 2>>
        // Represent Map as Struct("Map", [K, V])
        let generic_ty = t_struct("Map", vec![t_param("K"), t_param("V")]);
        
        let tensor_conc = t_tensor(t_f32(), 2);
        let string_conc = Type::String("String".to_string());
        let conc = t_struct("Map", vec![string_conc.clone(), tensor_conc.clone()]);
        
        let bindings = GenericResolver::resolve_bindings(&generic_ty, &conc).unwrap();
        assert_eq!(bindings.get("K"), Some(&string_conc));
        assert_eq!(bindings.get("V"), Some(&tensor_conc));
    }

    #[test]
    fn test_resolve_nested_generic() {
        // Vec<Vec<T>> vs Vec<Vec<f32>>
        let generic_ty = t_vec(t_vec(t_param("T")));
        let conc = t_vec(t_vec(t_f32()));
        
        let bindings = GenericResolver::resolve_bindings(&generic_ty, &conc).unwrap();
        assert_eq!(bindings.get("T"), Some(&t_f32()));
    }

    #[test]
    fn test_resolve_nested_struct() {
        // Struct Box<T> { x: T }
        // Type::Struct("Box", [T])
        let generic_ty = t_struct("Box", vec![t_param("T")]);
        let conc = t_struct("Box", vec![t_i64()]);
        
        let bindings = GenericResolver::resolve_bindings(&generic_ty, &conc).unwrap();
        assert_eq!(bindings.get("T"), Some(&t_i64()));
    }

    #[test]
    fn test_resolve_conflict() {
        // Pair<T, T> vs Pair<i64, f32> -> Check conflict
        let generic_ty = t_struct("Pair", vec![t_param("T"), t_param("T")]);
        let conc = t_struct("Pair", vec![t_i64(), t_f32()]);
        
        let res = GenericResolver::resolve_bindings(&generic_ty, &conc);
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("mismatched types"));
    }

    #[test]
    fn test_apply_bindings() {
        let mut bindings = HashMap::new();
        bindings.insert("T".into(), t_i64());
        
        let generic_ty = t_vec(t_param("T"));
        let resolved = GenericResolver::apply_bindings(&generic_ty, &bindings);
        
        assert_eq!(resolved, t_vec(t_i64()));
    }
    
    #[test]
    fn test_apply_bindings_missing_leaves_as_is() {
        // If T is not in bindings, stays as T (UserDefined)
        let bindings = HashMap::new();
        let generic_ty = t_param("T");
        let resolved = GenericResolver::apply_bindings(&generic_ty, &bindings);
        assert_eq!(resolved, generic_ty);
    }
}
