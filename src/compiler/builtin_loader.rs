use crate::compiler::ast::{
    Module, StructDef, EnumDef, ImplBlock, Type,
    Expr, ExprKind, Stmt, StmtKind, VariantKind, EnumVariantInit,
};
use crate::compiler::parser::parse_from_source;
use crate::compiler::error::TlError;
use std::collections::HashMap;

/// Holds all necessary AST nodes for a builtin type to be registered in TypeManager.
/// Constructed fully from source before registration to avoid lifetime issues.
#[derive(Debug, Clone)]
pub struct BuiltinTypeData {
    pub name: String,
    pub struct_def: Option<StructDef>,
    pub enum_def: Option<EnumDef>,
    pub impl_blocks: Vec<ImplBlock>,
    pub extra_structs: Vec<StructDef>,
    pub destructor: Option<String>,
}

pub struct BuiltinLoader;

impl BuiltinLoader {
    /// Load a module from raw TL source code string.
    pub fn load_from_source(source: &str) -> Result<Module, TlError> {
        parse_from_source(source)
    }
    
    /// Perform lightweight semantic analysis on builtin module.
    /// Converts StaticMethodCall to EnumInit for enum variant constructors.
    fn analyze_builtin_module(module: &mut Module) {
        // 1. Collect enum definitions from this module
        let mut enum_registry: HashMap<String, EnumDef> = HashMap::new();
        for e in &module.enums {
            enum_registry.insert(e.name.clone(), e.clone());
        }
        
        // 2. Transform ImplBlock method bodies
        for impl_block in &mut module.impls {
            for method in &mut impl_block.methods {
                Self::transform_stmts(&mut method.body, &enum_registry);
            }
        }
        
        // 3. Transform top-level functions
        for func in &mut module.functions {
            Self::transform_stmts(&mut func.body, &enum_registry);
        }
    }
    
    fn transform_stmts(stmts: &mut [Stmt], enums: &HashMap<String, EnumDef>) {
        for stmt in stmts {
            Self::transform_stmt(stmt, enums);
        }
    }
    
    fn transform_stmt(stmt: &mut Stmt, enums: &HashMap<String, EnumDef>) {
        match &mut stmt.inner {
            StmtKind::Let { value, .. } => Self::transform_expr(value, enums),
            StmtKind::Expr(e) => Self::transform_expr(e, enums),
            StmtKind::Return(Some(e)) => Self::transform_expr(e, enums),
            StmtKind::While { cond, body } => {
                Self::transform_expr(cond, enums);
                Self::transform_stmts(body, enums);
            }
            StmtKind::For { iterator, body, .. } => {
                Self::transform_expr(iterator, enums);
                Self::transform_stmts(body, enums);
            }
            StmtKind::Loop { body } => {
                Self::transform_stmts(body, enums);
            }
            StmtKind::Assign { value, .. } => {
                Self::transform_expr(value, enums);
            }
            _ => {}
        }
    }
    
    fn transform_expr(expr: &mut Expr, enums: &HashMap<String, EnumDef>) {
        // First, recurse into children
        match &mut expr.inner {
            ExprKind::BinOp(l, _, r) => {
                Self::transform_expr(l, enums);
                Self::transform_expr(r, enums);
            }
            ExprKind::UnOp(_, e) => {
                Self::transform_expr(e, enums);
            }
            ExprKind::MethodCall(obj, _, args) => {
                Self::transform_expr(obj, enums);
                for arg in args {
                    Self::transform_expr(arg, enums);
                }
            }
            ExprKind::FnCall(_, args) => {
                for arg in args {
                    Self::transform_expr(arg, enums);
                }
            }
            ExprKind::IndexAccess(target, indices) => {
                Self::transform_expr(target, enums);
                for idx in indices {
                    Self::transform_expr(idx, enums);
                }
            }
            ExprKind::FieldAccess(obj, _) => {
                Self::transform_expr(obj, enums);
            }
            ExprKind::Match { expr: subject, arms } => {
                Self::transform_expr(subject, enums);
                for (_, arm_expr) in arms {
                    Self::transform_expr(arm_expr, enums);
                }
            }
            ExprKind::Block(stmts) => {
                Self::transform_stmts(stmts, enums);
            }
            ExprKind::IfExpr(cond, then_block, else_block) => {
                Self::transform_expr(cond, enums);
                Self::transform_stmts(then_block, enums);
                if let Some(else_stmts) = else_block {
                    Self::transform_stmts(else_stmts, enums);
                }
            }
            ExprKind::Tuple(exprs) => {
                for e in exprs {
                    Self::transform_expr(e, enums);
                }
            }
            ExprKind::StructInit(_, fields) => {
                for (_, e) in fields {
                    Self::transform_expr(e, enums);
                }
            }
            ExprKind::As(e, _) => {
                Self::transform_expr(e, enums);
            }
            ExprKind::Range(start, end) => {
                Self::transform_expr(start, enums);
                Self::transform_expr(end, enums);
            }
            ExprKind::StaticMethodCall(ty, method, args) => {
                // Transform args first
                for arg in args.iter_mut() {
                    Self::transform_expr(arg, enums);
                }
                
                // Check if this is an enum variant constructor
                let enum_name = Self::get_type_base_name(ty);
                if let Some(enum_def) = enums.get(&enum_name) {
                    if let Some(variant) = enum_def.variants.iter().find(|v| &v.name == method) {
                        // Extract generics from type
                        let generics = match ty {
                            Type::Struct(_, g) | Type::Enum(_, g) | Type::Path(_, g) => g.clone(),
                            _ => vec![],
                        };
                        
                        // Build payload
                        let payload = match &variant.kind {
                            VariantKind::Unit => EnumVariantInit::Unit,
                            VariantKind::Tuple(_) => EnumVariantInit::Tuple(std::mem::take(args)),
                            VariantKind::Struct(_) => EnumVariantInit::Unit, // TODO: struct variant
                            VariantKind::Array(_, _) => EnumVariantInit::Tuple(std::mem::take(args)),
                        };
                        
                        // Replace with EnumInit
                        expr.inner = ExprKind::EnumInit {
                            enum_name,
                            variant_name: method.clone(),
                            generics,
                            payload,
                        };
                        return;
                    }
                }
            }
            _ => {}
        }
    }
    
    fn get_type_base_name(ty: &Type) -> String {
        match ty {
            Type::Struct(name, _) | Type::Enum(name, _) => name.clone(),
            Type::Path(segments, _) => segments.last().cloned().unwrap_or_default(),
            _ => String::new(),
        }
    }

    /// Load a specific builtin type definition from source.
    /// Extracts the Struct/Enum definition matching `type_name` and all relevant Impl blocks.
    pub fn load_builtin_type(source: &str, type_name: &str) -> Result<BuiltinTypeData, TlError> {
        let mut module = Self::load_from_source(source)?;
        
        // Perform lightweight semantic analysis to convert StaticMethodCall to EnumInit
        Self::analyze_builtin_module(&mut module);
        
        // Helper to extract primary struct but keep others
        let mut primary_struct = None;
        let mut extra_structs = Vec::new();
        
        for s in module.structs {
            if s.name == type_name {
                primary_struct = Some(s);
            } else {
                extra_structs.push(s);
            }
        }

        let struct_def = primary_struct;
        
        let enum_def = module.enums.into_iter().find(|e| e.name == type_name);
        
        // Impl blocks: look for target_type matching UserDefined(type_name, ...)
        let impl_blocks: Vec<ImplBlock> = module.impls.into_iter()
            .filter(|i| {
                match &i.target_type {
                    Type::Struct(name, _) | Type::Enum(name, _) => name == type_name,
                    Type::Path(segments, _) => segments.last().map(|s| s == type_name).unwrap_or(false),
                    _ => false,
                }
            })
            .collect();

        if struct_def.is_none() && enum_def.is_none() && impl_blocks.is_empty() {
            // It might be acceptable if we only have impls (extension methods on existing type?), 
            // but for "Builtin Type Definition" we usually expect the type def too.
            // For now, allow it but log? Or return error?
            // "Type not found in source" sounds reasonable.
            return Err(TlError::Parse { 
                kind: crate::compiler::error::ParseErrorKind::Generic(format!("Type '{}' not found in source and no impl blocks found", type_name)),
                span: None
            });
        }

        Ok(BuiltinTypeData {
            name: type_name.to_string(),
            struct_def,
            enum_def,
            impl_blocks,
            extra_structs,
            destructor: None,
        })
    }

    /// Load all definitions from a module source without filtering for specific type.
    pub fn load_module_data(source: &str, module_name: &str) -> Result<BuiltinTypeData, TlError> {
        let mut module = Self::load_from_source(source)?;
        Self::analyze_builtin_module(&mut module);
        
        Ok(BuiltinTypeData {
            name: module_name.to_string(),
            struct_def: None, 
            enum_def: None,
            impl_blocks: module.impls, 
            extra_structs: module.structs, 
            destructor: None,
        })
    }
}
