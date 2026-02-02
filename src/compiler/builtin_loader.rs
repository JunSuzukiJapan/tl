use crate::compiler::ast::{Module, StructDef, EnumDef, ImplBlock, Type};
use crate::compiler::parser::parse_from_source;
use crate::compiler::error::TlError;

/// Holds all necessary AST nodes for a builtin type to be registered in TypeManager.
/// Constructed fully from source before registration to avoid lifetime issues.
#[derive(Debug, Clone)]
pub struct BuiltinTypeData {
    pub name: String,
    pub struct_def: Option<StructDef>,
    pub enum_def: Option<EnumDef>,
    pub impl_blocks: Vec<ImplBlock>,
    pub destructor: Option<String>,
}

pub struct BuiltinLoader;

impl BuiltinLoader {
    /// Load a module from raw TL source code string.
    pub fn load_from_source(source: &str) -> Result<Module, TlError> {
        parse_from_source(source)
    }

    /// Load a specific builtin type definition from source.
    /// Extracts the Struct/Enum definition matching `type_name` and all relevant Impl blocks.
    pub fn load_builtin_type(source: &str, type_name: &str) -> Result<BuiltinTypeData, TlError> {
        let module = Self::load_from_source(source)?;
        
        let struct_def = module.structs.into_iter().find(|s| s.name == type_name);
        
        // Note: module.enums might have been consumed if we iterated it. 
        // Need to re-find if we didn't populate struct_def? 
        // Better to iterate once?
        // But Module fields are Vecs. `module.enums` is accessible.
        
        // Let's re-parse or iterate carefully.
        // Actually `module` is owned here. We can consume fields.
        // But we need to check both structs and enums.
        
        // Workaround: We loaded module. We can clone if needed, but ownership is fine.
        // However, `struct_def` consumes `module.structs` if we use `into_iter`.
        // Let's search inside `module`.
        
        let enum_def = module.enums.into_iter().find(|e| e.name == type_name);
        
        // Impl blocks: look for target_type matching UserDefined(type_name, ...)
        let impl_blocks: Vec<ImplBlock> = module.impls.into_iter()
            .filter(|i| {
                match &i.target_type {
                    Type::Struct(name, _) | Type::Enum(name, _) => name == type_name,
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
            destructor: None,
        })
    }
}
