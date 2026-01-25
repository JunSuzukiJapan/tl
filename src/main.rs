// mod compiler;
// mod runtime;

use anyhow::{Context, Result};
use clap::Parser;
use inkwell::context::Context as InkwellContext;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tl_lang::compiler::codegen::CodeGenerator;
use tl_lang::compiler::error::{format_error_with_source, TlError};
use tl_lang::compiler::inference::{forward_chain, query, GroundAtom, Value};
use tl_lang::compiler::semantics::SemanticAnalyzer;

#[derive(Parser)]
#[command(name = "tlc")]
#[command(version)]
#[command(about = "Tensor Logic Compiler", long_about = None)]
struct Cli {
    /// Input files
    #[arg(required = true)]
    files: Vec<String>,

    /// Compile to executable
    #[arg(short, long)]
    compile: bool,

    /// Output file
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Emit assembly
    #[arg(short = 'S', long)]
    save_asm: bool,

    /// Device (cpu, metal, cuda, auto)
    #[arg(short, long, default_value = "auto")]
    device: String,

    /// Arguments to pass to the TL program (after --)
    #[arg(last = true)]
    args: Vec<String>,

    /// Enable runtime memory allocation logging
    #[arg(long)]
    mem_log: bool,

    /// Verbose logging (-v info, -vv debug)
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}


fn main() -> Result<()> {
    // env_logger::init(); // Moved to after CLI parse to use verbose flag
    let cli = Cli::parse();

    // Initialize Logger
    let mut builder = env_logger::Builder::new();
    // Default level: WARN
    builder.filter_level(log::LevelFilter::Warn);
    
    // Override with CLI verbose level
    match cli.verbose {
        0 => { 
            // Check RUST_LOG env var if no -v flag
             if std::env::var("RUST_LOG").is_ok() {
                  builder.parse_default_env();
             }
        } 
        1 => { builder.filter_level(log::LevelFilter::Info); }
        2 => { builder.filter_level(log::LevelFilter::Debug); }
        _ => { builder.filter_level(log::LevelFilter::Trace); }
    };

    builder.init();

    // Set device environment variable
    if std::env::var("TL_DEVICE").is_err() {
        std::env::set_var("TL_DEVICE", &cli.device);
    }
    if cli.mem_log {
        std::env::set_var("TL_MEM_LOG", "1");
    }

    let mut source_files = Vec::new();
    let mut object_files = Vec::new();

    for f in &cli.files {
        let p = PathBuf::from(f);
        if let Some(ext) = p.extension() {
            if ext == "tl" {
                source_files.push(p);
            } else if ext == "o" || ext == "s" {
                object_files.push(p);
            } else {
                // Assume source file if unknown
                source_files.push(p);
            }
        } else {
            source_files.push(p);
        }
    }

    // Determine mode
    let is_compile_mode = cli.compile || cli.output.is_some() || cli.save_asm;

    if is_compile_mode {
        // Compile Mode
        let mut generated_objects = Vec::new();

        for file in &source_files {
            log::info!("Compiling file: {:?}", file);
            let (mut ast, source) = match load_module_with_source(file.clone()) {
                Ok((ast, source)) => (ast, source),
                Err(e) => {
                    let source = fs::read_to_string(file).unwrap_or_default();
                    print_tl_error_with_source(
                        &e,
                        &source,
                        Some(file.to_str().unwrap_or("unknown")),
                    );
                    std::process::exit(1);
                }
            };

            // Semantics
            let mut analyzer = SemanticAnalyzer::new(String::new());
            if let Err(e) = analyzer.check_module(&mut ast) {
                let tl_err = e.with_file(file.to_str().unwrap_or("unknown"));
                print_tl_error_with_source(
                    &tl_err,
                    &source,
                    Some(file.to_str().unwrap_or("unknown")),
                );
                std::process::exit(1);
            }

            // Monomorphization
            let mut monomorphizer = tl_lang::compiler::monomorphize::Monomorphizer::new();
            monomorphizer.run(&mut ast);

            // Codegen
            let context = InkwellContext::create();
            // Module name from filename
            let module_name = file.file_stem().unwrap().to_str().unwrap();
            let mut codegen = CodeGenerator::new(&context, module_name);

            if let Err(e) = codegen.compile_module(&ast, "main") {
                let tl_err = TlError::Codegen {
                    kind: tl_lang::compiler::error::CodegenErrorKind::Generic(e),
                    span: None,
                }
                .with_file(file.to_str().unwrap_or("unknown"));
                print_tl_error_with_source(
                    &tl_err,
                    &source,
                    Some(file.to_str().unwrap_or("unknown")),
                );
                std::process::exit(1);
            }

            if std::env::var("TL_DUMP_IR").is_ok() {
                codegen.dump_ir();
            }

            if cli.save_asm {
                if let Err(e) = codegen.emit_assembly_file(&asm_path) {
                    log::error!("Failed to emit assembly for {:?}: {}", file, e);
                    std::process::exit(1);
                }
                log::info!("Generated assembly: {:?}", asm_path);
            } else {

                let obj_path = file.with_extension("o");
                if let Err(e) = codegen.emit_object_file(&obj_path) {
                    log::error!("Failed to emit object file for {:?}: {}", file, e);
                    std::process::exit(1);
                }
                generated_objects.push(obj_path);
            }
        }

        // Check if output file indicates object file (skip linking)
        let output_is_object = cli
            .output
            .as_ref()
            .map(|p| p.extension().map_or(false, |e| e == "o"))
            .unwrap_or(false);

        // Link Step (only if compiling and not just saving asm, and not explicitly outputting object)
        if (cli.compile || cli.output.is_some()) && !cli.save_asm && !output_is_object {
            let mut link_args = Vec::new();
            link_args.extend(
                generated_objects
                    .iter()
                    .map(|p| p.to_str().unwrap().to_string()),
            );
            link_args.extend(object_files.iter().map(|p| p.to_str().unwrap().to_string()));

            // Determine output filename
            let output_exe = if let Some(out) = cli.output {
                out
            } else {
                // Default to first source filename without extension
                if !source_files.is_empty() {
                    let mut p = source_files[0].clone();
                    p.set_extension("");
                    p
                } else {
                    PathBuf::from("a.out")
                }
            };

            log::info!("Linking to {:?}", output_exe);

            // Invoke cc
            // We need to link against runtime libs if needed.
            // For now assume standard linking. If we user uses external tensor library (candle),
            // static linking might be complex. But let's try basic link.
            // WARNING: The JIT engine links candle symbols in memory.
            // A standalone executable needs to link against the Rust static library or dylib containing these symbols?
            // Wait, currently `tl` is self-contained.
            // To produce a standalone executable, we need `libtl_runtime.a` or similar?
            // Or we just produce object files and user has to link?
            // The prompt asks to "create executable".
            // Since we don't have a library distribution yet, linking might fail due to missing symbols (tl_runtime functions).
            // However, implementing the CLI structure is the first step.
            // I will attempt to run `cc` with the objects.

            // Add runtime library path and dependency
            // Try to find target directory
            // heuristic: check target/debug or target/release based on current build profile?
            // Since we are running `tl`, we don't strictly know if `libtl_runtime` is debug or release.
            // But usually we build them together.
            let runtime_path = PathBuf::from("target/debug"); // Default to debug for dev

            link_args.push(format!("-L{}", runtime_path.display()));
            link_args.push("-ltl_runtime".to_string());

            // System libraries and Frameworks (MacOS)
            link_args.push("-lpthread".to_string());
            link_args.push("-ldl".to_string());
            link_args.push("-lm".to_string());
            link_args.push("-lc++".to_string());

            #[cfg(target_os = "macos")]
            {
                link_args.push("-framework".to_string());
                link_args.push("Accelerate".to_string());
                link_args.push("-framework".to_string());
                link_args.push("Metal".to_string());
                link_args.push("-framework".to_string());
                link_args.push("Foundation".to_string());
                link_args.push("-framework".to_string());
                link_args.push("MetalPerformanceShaders".to_string());
                link_args.push("-framework".to_string());
                link_args.push("Security".to_string());
                link_args.push("-framework".to_string());
                link_args.push("CoreFoundation".to_string());
                link_args.push("-framework".to_string());
                link_args.push("SystemConfiguration".to_string());
            }

            let status = Command::new("cc")
                .args(&link_args)
                .arg("-o")
                .arg(&output_exe)
                .status()
                .context("Failed to run linker (cc)")?;

            if !status.success() {
                log::error!("Linking failed");
                std::process::exit(1);
            }
            log::info!("Build successful: {:?}", output_exe);
        }
    } else {
        // Interpreter Mode
        // Initialize args
        tl_runtime::args::init_args(cli.args.clone());
        tl_runtime::force_link();

        let mut combined_module = tl_lang::compiler::ast::Module {
            structs: vec![],
            enums: vec![],
            impls: vec![],
            functions: vec![],
            tensor_decls: vec![],
            relations: vec![],
            rules: vec![],
            queries: vec![],
            imports: vec![],
            submodules: std::collections::HashMap::new(),
        };

        // ソースコードを保持（スニペット表示用）
        let mut combined_source = String::new();

        for file in &source_files {
            // println!("Loading file: {:?}", file);
            match load_module_with_source(file.clone()) {
                Ok((mod_, source)) => {
                    // Merge
                    combined_module.structs.extend(mod_.structs);
                    combined_module.enums.extend(mod_.enums);
                    combined_module.impls.extend(mod_.impls);
                    combined_module.functions.extend(mod_.functions);
                    combined_module.tensor_decls.extend(mod_.tensor_decls);
                    combined_module.relations.extend(mod_.relations);
                    combined_module.rules.extend(mod_.rules);
                    combined_module.queries.extend(mod_.queries);
                    combined_module.imports.extend(mod_.imports);
                    combined_module.submodules.extend(mod_.submodules);
                    // 主要なファイルのソースを保持
                    if combined_source.is_empty() {
                        combined_source = source;
                    }
                }
                Err(e) => {
                    // パースエラーはソースが必要なので、ファイルを再読み込み
                    let source = fs::read_to_string(file).unwrap_or_default();
                    print_tl_error_with_source(
                        &e,
                        &source,
                        Some(file.to_str().unwrap_or("unknown")),
                    );
                    std::process::exit(1);
                }
            }
        }

        // Semantics (Check only)
        let mut analyzer = SemanticAnalyzer::new(String::new());
        if let Err(e) = analyzer.check_module(&mut combined_module) {
            let file_hint = if !source_files.is_empty() {
                source_files[0].to_str()
            } else {
                None
            };

            let tl_err = if let Some(f) = file_hint {
                e.with_file(f)
            } else {
                e
            };

            print_tl_error_with_source(&tl_err, &combined_source, file_hint);
            std::process::exit(1);
        }

        // Monomorphization (JIT)
        let mut monomorphizer = tl_lang::compiler::monomorphize::Monomorphizer::new();
        monomorphizer.run(&mut combined_module);

        // JIT Execution
        use tl_runtime::registry;
        registry::reset_global_context();

        let context = InkwellContext::create();
        let mut codegen = CodeGenerator::new(&context, "main");

        if let Err(e) = codegen.compile_module(&combined_module, "main") {
            // StringエラーをTlErrorに変換
            let tl_err = TlError::Codegen {
                kind: tl_lang::compiler::error::CodegenErrorKind::Generic(e),
                span: None,
            };
            print_tl_error_with_source(&tl_err, &combined_source, None);
            std::process::exit(1);
        }

        if std::env::var("TL_DUMP_IR").is_ok() {
            codegen.dump_ir();
        }

        // println!("Executing...");
        match codegen.jit_execute("main") {
            Ok(ret) => {
                // println!("Program returned: {}", ret)
                let _ = ret; // suppress unused
            }
            Err(e) => {
                println!("Execution failed: {}", e);
                std::process::exit(1);
            }
        }

        // Logic program logic
        let is_logic_program = !combined_module.relations.is_empty()
            || !combined_module.rules.is_empty()
            || !combined_module.queries.is_empty();

        if is_logic_program {
            let tensor_context = registry::get_global_context();
            run_logic_program(&combined_module, &tensor_context);
        }
    }

    Ok(())
}

/// Execute a logic program using the inference engine.
fn run_logic_program(
    module: &tl_lang::compiler::ast::Module,
    ctx: &tl_lang::compiler::inference::TensorContext,
) {
) {
use tl_lang::compiler::ast::{Atom, ExprKind};

    log::info!("Executing logic program...");

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

        rules.push(rule.clone());
    }

    log::info!("Initial facts: {}", initial_facts.len());
    log::info!("Rules: {}", rules.len());

    // 3. Run forward chaining
    let derived_facts = forward_chain(initial_facts, &rules, ctx);
    log::info!("Derived facts: {}", derived_facts.len());

    // 4. Execute queries
    for query_expr in &module.queries {
        // Query expr should be a function call like path(1, 2)
        if let ExprKind::FnCall(pred, args) = &query_expr.inner {
            let query_atom = Atom {
                predicate: pred.clone(),
                args: args.clone(),
            };
            println!("\nQuery: {}({:?})", pred, args);

            let results = query(&query_atom, &derived_facts, ctx);
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
fn try_atom_to_ground(atom: &tl_lang::compiler::ast::Atom) -> Option<GroundAtom> {
    use tl_lang::compiler::ast::ExprKind;

    let mut args = Vec::new();
    for expr in &atom.args {
        match &expr.inner {
            ExprKind::Int(n) => args.push(Value::Int(*n)),
            ExprKind::Float(f) => args.push(Value::Float(*f)),
            ExprKind::Symbol(s) => args.push(Value::Str(s.clone())),
            ExprKind::Bool(b) => args.push(Value::Bool(*b)),
            ExprKind::StringLiteral(s) => args.push(Value::Str(s.clone())),
            _ => return None, // Contains variable or complex expression
        }
    }
    Some(GroundAtom {
        predicate: atom.predicate.clone(),
        args,
    })
}

/// Check if the body is trivially true (empty or contains only "true").
fn is_trivially_true(body: &[tl_lang::compiler::ast::LogicLiteral]) -> bool {
    use tl_lang::compiler::ast::LogicLiteral;
    if body.is_empty() {
        return true;
    }
    if body.len() == 1 {
        if let LogicLiteral::Pos(atom) = &body[0] {
            return atom.predicate == "true" && atom.args.is_empty();
        }
    }
    false
}

/// モジュールをロードし、ソースコードも返す
/// モジュールをロードし、ソースコードも返す
fn load_module_with_source(
    path: PathBuf,
) -> Result<(tl_lang::compiler::ast::Module, String), TlError> {
    let mut visited = HashSet::new();
    load_module_recursive(path, &mut visited)
}

fn load_module_recursive(
    path: PathBuf,
    visited: &mut HashSet<PathBuf>,
) -> Result<(tl_lang::compiler::ast::Module, String), TlError> {
    // Canonicalize path to handle relative paths and symlinks consistently
    let canonical_path = match fs::canonicalize(&path) {
        Ok(p) => p,
        Err(_) => path.clone(), // Fallback if file doesn't exist yet (handled by read_to_string later)
    };

    if visited.contains(&canonical_path) {
        return Err(TlError::Parse {
            kind: tl_lang::compiler::error::ParseErrorKind::Generic(format!(
                "Cyclic dependency detected: {:?}",
                path
            )),
            span: None,
        });
    }
    visited.insert(canonical_path.clone());

    let path_str = path.to_str().unwrap_or("unknown").to_string();

    let content = match fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) => {
            // visited.remove(&canonical_path); // Cleanup not strictly necessary on error return
            return Err(TlError::Io(e));
        }
    };
    let source = content.clone();

    let mut module = match tl_lang::compiler::parser::parse_from_source(&content) {
        Ok(m) => m,
        Err(e) => {
            // visited.remove(&canonical_path);
            return Err(e.with_file(&path_str));
        }
    };

    let parent_dir = path.parent().unwrap_or(Path::new("."));

    // Use index to avoid borrowing module.imports while mutating module
    let imports = module.imports.clone();
    for import_name in &imports {
        let is_wildcard = import_name.ends_with("::*");
        let real_name = if is_wildcard {
            import_name.trim_end_matches("::*")
        } else {
            import_name
        };

        let import_path = parent_dir.join(format!("{}.tl", real_name));

        if !import_path.exists() {
            return Err(TlError::Parse {
                kind: tl_lang::compiler::error::ParseErrorKind::Generic(format!(
                    "Module {} not found at {:?}",
                    real_name, import_path
                )),
                span: None,
            });
        }

        match load_module_recursive(import_path, visited) {
            Ok((submodule, _)) => {
                if is_wildcard {
                     // Merge content into current module
                     module.structs.extend(submodule.structs);
                     module.enums.extend(submodule.enums);
                     module.impls.extend(submodule.impls);
                     module.functions.extend(submodule.functions);
                     module.tensor_decls.extend(submodule.tensor_decls);
                     module.relations.extend(submodule.relations);
                     module.rules.extend(submodule.rules);
                     module.queries.extend(submodule.queries);
                     // module.imports.extend(submodule.imports); // Should we merge imports? Recursive loading handles it?
                     // If submodule had imports, they are already loaded into submodule.submodules.
                     // We need to merge submodules too!
                     module.submodules.extend(submodule.submodules);
                } else {
                     module.submodules.insert(import_name.clone(), submodule);
                }
            }
            Err(e) => return Err(e),
        }
    }

    visited.remove(&canonical_path);
    Ok((module, source))
}

/// ソースコードスニペット付きでエラーを表示
fn print_tl_error_with_source(error: &TlError, source: &str, file_hint: Option<&str>) {
    let output = format_error_with_source(error, source, file_hint);
    eprint!("{}", output);
}
