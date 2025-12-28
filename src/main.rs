// src/main.rs
mod compiler;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
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
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Check { file } => {
            println!("Checking file: {:?}", file);
            let content = fs::read_to_string(file)
                .with_context(|| format!("Failed to read file {:?}", file))?;
            let _ast = compiler::parser::parse(&content)?;
            println!("Syntax OK");
        }
        Commands::Run { file } => {
            println!("Running file: {:?}", file);
            let content = fs::read_to_string(file)
                .with_context(|| format!("Failed to read file {:?}", file))?;
            let _ast = compiler::parser::parse(&content)?;
            // TODO: Codegen and JIT
            println!("(JIT execution not implemented yet)");
        }
        Commands::Build { file } => {
            println!("Building file: {:?}", file);
            // TODO: Codegen to binary
        }
    }

    Ok(())
}
