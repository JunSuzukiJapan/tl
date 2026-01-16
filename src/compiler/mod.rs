// src/compiler/mod.rs
pub mod ast;
pub mod codegen;
pub mod error;
pub mod inference;
pub mod parser;
pub mod type_infer;

pub mod semantics;
pub mod shape_analysis; // Phase 2: Static size analysis for Arena
