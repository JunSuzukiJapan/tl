// src/compiler/mod.rs
pub mod ast;
pub mod ast_subst;
pub mod codegen;
pub mod error;
pub mod inference;
pub mod monomorphize;
pub mod parser;
pub mod type_infer;
pub mod generics;
pub mod type_registry;
pub mod liveness;

pub mod semantics;
pub mod shape_analysis; // Phase 2: Static size analysis for Arena
pub mod lexer;
