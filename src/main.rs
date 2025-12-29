// src/main.rs
mod compiler;
mod runtime;

use crate::compiler::codegen::CodeGenerator;
use crate::compiler::semantics::SemanticAnalyzer;
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use inkwell::context::Context as InkwellContext;
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "tlc")]
#[command(about = "Tensor Logic Compiler", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Check syntax and types
    Check {
        /// Input file
        file: PathBuf,
    },
    /// Compile and run (JIT)
    Run {
        /// Input file
        file: PathBuf,
    },
    /// Compile to executable
    Build {
        /// Input file
        file: PathBuf,
        /// Output file
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Check { file } => {
            println!("Checking file: {:?}", file);
            let content = fs::read_to_string(file)
                .with_context(|| format!("Failed to read file {:?}", file))?;
            match compiler::parser::parse(&content) {
                Ok(ast) => {
                    println!("Syntax OK");
                    // Perform Semantic Analysis
                    let mut analyzer = SemanticAnalyzer::new();
                    match analyzer.check_module(&ast) {
                        Ok(_) => println!("Semantics OK"),
                        Err(e) => {
                            eprintln!("Semantic check failed: {}", e);
                            std::process::exit(1);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Commands::Run { file } => {
            println!("Running file: {:?}", file);
            let content = fs::read_to_string(&file)
                .with_context(|| format!("Failed to read file {:?}", file))?;

            // 1. Parser
            let ast = match compiler::parser::parse(&content) {
                Ok(ast) => ast,
                Err(e) => {
                    eprintln!("Parse error: {}", e);
                    std::process::exit(1);
                }
            };

            // 2. Semantics
            let mut analyzer = SemanticAnalyzer::new();
            if let Err(e) = analyzer.check_module(&ast) {
                eprintln!("Semantic error: {}", e);
                std::process::exit(1);
            }

            // 3. Codegen & JIT
            let context = InkwellContext::create();
            let mut codegen = CodeGenerator::new(&context, "main");
            codegen.compile_module(&ast).unwrap();

            // Debug: dump LLVM IR
            // codegen.dump_llvm_ir();

            println!("Executing main...");
            match codegen.jit_execute("main") {
                Ok(ret) => println!("Program returned: {}", ret),
                Err(e) => println!("Execution failed: {}", e),
            }
        }
        Commands::Build { file, output: _ } => {
            println!("Building file: {:?}", file);
            let content = fs::read_to_string(&file)
                .with_context(|| format!("Failed to read file {:?}", file))?;

            // 1. Parser
            let ast = match compiler::parser::parse(&content) {
                Ok(ast) => ast,
                Err(e) => {
                    eprintln!("Parse error: {}", e);
                    std::process::exit(1);
                }
            };

            // 2. Semantics
            let mut analyzer = SemanticAnalyzer::new();
            if let Err(e) = analyzer.check_module(&ast) {
                eprintln!("Semantic error: {}", e);
                std::process::exit(1);
            }

            // 3. Codegen
            let context = InkwellContext::create();
            let mut codegen = CodeGenerator::new(&context, "main");
            if let Err(e) = codegen.compile_module(&ast) {
                eprintln!("Codegen error: {}", e);
                std::process::exit(1);
            }

            // Debug: print IR
            codegen.dump_llvm_ir();

            // 4. Output (simplified for now to IR dumping or validation)
            // Ideally we'd compile to object file and link.
            // For now, let's just confirm it runs through codegen.
            println!("Codegen finished. IR dumped to stderr.");
        }
    }

    Ok(())
}
