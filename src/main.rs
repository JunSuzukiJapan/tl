// src/main.rs
mod compiler;
mod runtime;

use crate::compiler::codegen::CodeGenerator;
use crate::compiler::inference::{forward_chain, query, GroundAtom, Value};
use crate::compiler::semantics::SemanticAnalyzer;
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use inkwell::context::Context as InkwellContext;
use std::collections::HashSet;
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

            // Check if this is a logic program
            let is_logic_program =
                !ast.relations.is_empty() || !ast.rules.is_empty() || !ast.queries.is_empty();

            if is_logic_program {
                // Execute as logic program
                run_logic_program(&ast);
            } else {
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

/// Execute a logic program using the inference engine.
fn run_logic_program(module: &compiler::ast::Module) {
    use crate::compiler::ast::{Atom, Expr};

    println!("Executing logic program...");

    // 1. Extract facts from rules with empty body or special handling
    // For now, we don't have explicit facts - we'll need to add them.
    // Let's treat rules with body containing a single "true" atom as facts.
    let mut initial_facts: HashSet<GroundAtom> = HashSet::new();

    // 2. Collect actual rules (non-fact rules)
    let mut rules = Vec::new();
    for rule in &module.rules {
        // Check if body is a fact (e.g., edge(1, 2) :- true; or just edge(1, 2).)
        // For simplicity, try to convert head to ground atom if all args are literals
        let head_ground = try_atom_to_ground(&rule.head);
        if let Some(ground) = head_ground {
            // If body is empty or trivially true, treat as fact
            if rule.body.is_empty() || is_trivially_true(&rule.body) {
                initial_facts.insert(ground);
                continue;
            }
        }
        rules.push(rule.clone());
    }

    println!("Initial facts: {}", initial_facts.len());
    println!("Rules: {}", rules.len());

    // 3. Run forward chaining
    let derived_facts = forward_chain(initial_facts, &rules);
    println!("Derived facts: {}", derived_facts.len());

    // 4. Execute queries
    for query_expr in &module.queries {
        // Query expr should be a function call like path(1, 2)
        if let Expr::FnCall(pred, args) = query_expr {
            let query_atom = Atom {
                predicate: pred.clone(),
                args: args.clone(),
            };
            println!("\nQuery: {}({:?})", pred, args);

            let results = query(&query_atom, &derived_facts);
            if results.is_empty() {
                println!("  Result: false (no matches)");
            } else {
                println!("  Result: true ({} matches)", results.len());
                for (i, subst) in results.iter().enumerate() {
                    if !subst.is_empty() {
                        println!("    Match {}: {:?}", i + 1, subst);
                    }
                }
            }
        } else {
            println!("Unsupported query expression: {:?}", query_expr);
        }
    }
}

/// Try to convert an Atom to a GroundAtom (all args must be literals).
fn try_atom_to_ground(atom: &compiler::ast::Atom) -> Option<GroundAtom> {
    use crate::compiler::ast::Expr;

    let mut args = Vec::new();
    for expr in &atom.args {
        match expr {
            Expr::Int(n) => args.push(Value::Int(*n)),
            Expr::Float(f) => args.push(Value::Float(f.to_string())),
            Expr::StringLiteral(s) => args.push(Value::Str(s.clone())),
            _ => return None, // Contains variable or complex expression
        }
    }
    Some(GroundAtom {
        predicate: atom.predicate.clone(),
        args,
    })
}

/// Check if the body is trivially true (empty or contains only "true").
fn is_trivially_true(body: &[compiler::ast::Atom]) -> bool {
    if body.is_empty() {
        return true;
    }
    if body.len() == 1 && body[0].predicate == "true" && body[0].args.is_empty() {
        return true;
    }
    false
}
