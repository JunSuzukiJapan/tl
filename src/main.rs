// jemalloc をグローバルアロケータとして設定
// macOS/Linux のシステムアロケータはテンソルの大量 alloc/dealloc で
// メモリフラグメンテーションが激しく、RSS が線形成長する。
// jemalloc は未使用ページを定期的に OS に返却するため、RSS を安定化する。
#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

use anyhow::{Context, Result};
use clap::Parser;
use inkwell::context::Context as InkwellContext;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tl_lang::compiler::codegen::CodeGenerator;
use tl_lang::compiler::error::{TlError, format_error_with_source};
use tl_lang::compiler::inference::{GroundAtom, Value, forward_chain, query};
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

    /// Emit LLVM IR
    #[arg(long = "emit-llvm")]
    emit_llvm: bool,

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
    let cli = Cli::parse();
    init_logger(&cli);
    init_env(&cli);

    let (source_files, object_files) = classify_input_files(&cli.files);

    let builtins = load_builtins().context("Failed to load builtins")?;
    log::info!(
        "Loaded builtins: {} structs, {} impls",
        builtins.structs.len(),
        builtins.impls.len()
    );

    let is_compile_mode = cli.compile || cli.output.is_some() || cli.save_asm || cli.emit_llvm;
    if is_compile_mode {
        run_compile_mode(&cli, &source_files, &object_files, &builtins)?;
    } else {
        run_interpret_mode(&cli, &source_files, &builtins)?;
    }

    Ok(())
}

/// ロガーを CLI フラグに従って初期化する。
fn init_logger(cli: &Cli) {
    let mut builder = env_logger::Builder::new();
    builder.filter_level(log::LevelFilter::Warn);
    // tokenizers クレートの警告を抑制（ID ミスマッチ等）
    builder.filter_module("tokenizers", log::LevelFilter::Error);
    match cli.verbose {
        0 => {
            if std::env::var("RUST_LOG").is_ok() {
                builder.parse_default_env();
            }
        }
        1 => { builder.filter_level(log::LevelFilter::Info); }
        2 => { builder.filter_level(log::LevelFilter::Debug); }
        _ => { builder.filter_level(log::LevelFilter::Trace); }
    }
    builder.init();
}

/// 環境変数を CLI フラグから設定する。
fn init_env(cli: &Cli) {
    if std::env::var("TL_DEVICE").is_err() {
        unsafe { std::env::set_var("TL_DEVICE", &cli.device); }
    }
    if cli.mem_log {
        unsafe { std::env::set_var("TL_MEM_LOG", "1"); }
    }
}

/// 入力ファイルを拡張子でソースファイルとオブジェクトファイルに分類する。
fn classify_input_files(files: &[String]) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let mut source_files = Vec::new();
    let mut object_files = Vec::new();
    for f in files {
        let p = PathBuf::from(f);
        match p.extension().and_then(|e| e.to_str()) {
            Some("o") | Some("s") => object_files.push(p),
            _ => source_files.push(p),
        }
    }
    (source_files, object_files)
}

/// AOT コンパイルモード：ソースファイルをオブジェクトファイルに変換してリンクする。
fn run_compile_mode(
    cli: &Cli,
    source_files: &[PathBuf],
    object_files: &[PathBuf],
    builtins: &tl_lang::compiler::ast::Module,
) -> Result<()> {
    let mut generated_objects = Vec::new();

    for file in source_files {
        log::info!("Compiling file: {:?}", file);
        let (mut ast, source) = match load_module_with_source(file.clone()) {
            Ok((ast, source)) => (ast, source),
            Err(e) => {
                let source = fs::read_to_string(file).unwrap_or_default();
                print_tl_error_with_source(&e, &source, file.to_str());
                std::process::exit(1);
            }
        };

        // ビルトインをインジェクト
        ast.merge(builtins.clone());

        // 意味解析
        let mut analyzer = SemanticAnalyzer::new(String::new());
        if let Err(e) = analyzer.check_module(&mut ast) {
            let tl_err = e.with_file(file.to_str().unwrap_or("unknown"));
            print_tl_error_with_source(&tl_err, &source, file.to_str());
            std::process::exit(1);
        }

        // 単相化
        let mut monomorphizer = tl_lang::compiler::monomorphize::Monomorphizer::new();
        if let Err(e) = monomorphizer.run(&mut ast) {
            let tl_err = e.with_file(file.to_str().unwrap_or("unknown"));
            print_tl_error_with_source(&tl_err, &source, file.to_str());
            std::process::exit(1);
        }

        // コード生成
        let context = InkwellContext::create();
        let module_name = file.file_stem()
            .expect("source file has a stem")
            .to_str()
            .expect("source file stem is valid UTF-8");
        let mut codegen = CodeGenerator::new(&context, module_name);

        if let Err(e) = codegen.compile_module(&ast, "main") {
            let tl_err = e.with_file(file.to_str().unwrap_or("unknown"));
            print_tl_error_with_source(&tl_err, &source, file.to_str());
            std::process::exit(1);
        }

        if std::env::var("TL_DUMP_IR").is_ok() { codegen.dump_ir(); }

        if cli.emit_llvm {
            let ll_path = file.with_extension("ll");
            if let Err(e) = codegen.emit_llvm_file(&ll_path) {
                log::error!("Failed to emit LLVM IR for {:?}: {}", file, e);
                std::process::exit(1);
            }
            log::info!("Generated LLVM IR: {:?}", ll_path);
        } else if cli.save_asm {
            let asm_path = file.with_extension("s");
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

    // リンクステップ
    let output_is_object = cli
        .output
        .as_ref()
        .map(|p| p.extension().map_or(false, |e| e == "o"))
        .unwrap_or(false);

    if (cli.compile || cli.output.is_some()) && !cli.save_asm && !cli.emit_llvm && !output_is_object {
        link_objects(cli, &generated_objects, object_files)?;
    }
    Ok(())
}

/// オブジェクトファイルをリンクして実行ファイルを生成する。
fn link_objects(
    cli: &Cli,
    generated_objects: &[PathBuf],
    extra_objects: &[PathBuf],
) -> Result<()> {
    let mut link_args: Vec<String> = generated_objects
        .iter()
        .chain(extra_objects.iter())
        .map(|p| p.to_str().expect("object path is valid UTF-8").to_string())
        .collect();

    let output_exe = cli.output.clone().unwrap_or_else(|| PathBuf::from("a.out"));
    log::info!("Linking to {:?}", output_exe);

    link_args.push("-Ltarget/debug".to_string());
    link_args.push("-ltl_runtime".to_string());
    link_args.push("-lpthread".to_string());
    link_args.push("-ldl".to_string());
    link_args.push("-lm".to_string());
    link_args.push("-lc++".to_string());

    #[cfg(target_os = "macos")]
    for fw in &["Accelerate", "Metal", "Foundation", "MetalPerformanceShaders",
                "Security", "CoreFoundation", "SystemConfiguration"] {
        link_args.push("-framework".to_string());
        link_args.push(fw.to_string());
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
    Ok(())
}

/// JIT インタープリタモード：ソースファイルをその場で実行する。
fn run_interpret_mode(
    cli: &Cli,
    source_files: &[PathBuf],
    builtins: &tl_lang::compiler::ast::Module,
) -> Result<()> {
    tl_runtime::args::init_args(cli.args.clone());
    tl_runtime::force_link();

    let mut combined_module = tl_lang::compiler::ast::Module::new();
    let mut combined_source = String::new();

    for file in source_files {
        match load_module_with_source(file.clone()) {
            Ok((mod_, source)) => {
                combined_module.merge(mod_);
                if combined_source.is_empty() {
                    combined_source = source;
                }
            }
            Err(e) => {
                let source = fs::read_to_string(file).unwrap_or_default();
                print_tl_error_with_source(&e, &source, file.to_str());
                std::process::exit(1);
            }
        }
    }

    // ビルトインをインジェクト
    combined_module.merge(builtins.clone());

    // 意味解析
    let mut analyzer = SemanticAnalyzer::new(String::new());
    if let Err(e) = analyzer.check_module(&mut combined_module) {
        let file_hint = source_files.first().and_then(|p| p.to_str());
        let tl_err = if let Some(f) = file_hint { e.with_file(f) } else { e };
        print_tl_error_with_source(&tl_err, &combined_source, file_hint);
        std::process::exit(1);
    }

    // 単相化
    let mut monomorphizer = tl_lang::compiler::monomorphize::Monomorphizer::new();
    if let Err(e) = monomorphizer.run(&mut combined_module) {
        print_tl_error_with_source(&e, &combined_source, None);
        std::process::exit(1);
    }

    // JIT 実行
    use tl_runtime::registry;
    registry::reset_global_context();

    let context = InkwellContext::create();
    let mut codegen = CodeGenerator::new(&context, "main");

    if let Err(e) = codegen.compile_module(&combined_module, "main") {
        print_tl_error_with_source(&e, &combined_source, None);
        std::process::exit(1);
    }

    if std::env::var("TL_DUMP_IR").is_ok() { codegen.dump_ir(); }

    match codegen.jit_execute("main") {
        Ok(_) => {}
        Err(e) => {
            println!("Execution failed: {}", e);
            std::process::exit(1);
        }
    }

    // Metal GPU 同期 — プロセス終了前にすべての GPU 処理の完了を保証
    tl_runtime::system::tl_metal_sync();

    // ロジックプログラムの実行
    if !combined_module.relations.is_empty()
        || !combined_module.rules.is_empty()
        || !combined_module.queries.is_empty()
    {
        let tensor_context = registry::get_global_context();
        run_logic_program(&combined_module, &tensor_context);
    }

    Ok(())
}

/// Execute a logic program using the inference engine.
fn run_logic_program(
    module: &tl_lang::compiler::ast::Module,
    ctx: &tl_lang::compiler::inference::TensorContext,
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

    log::info!("Initial facts: {}", initial_facts.len());
    log::info!("Rules: {}", rules.len());

    // 3. Run forward chaining
    let derived_facts = match forward_chain(initial_facts, &rules, ctx) {
        Ok(f) => f,
        Err(e) => {
            log::error!("Inference error: {}", e);
            return;
        }
    };
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
                    // ワイルドカードインポート: サブモジュールの内容を現在のモジュールにマージ。
                    // imports はマージしない（再帰ロードで既に処理済みのため）。
                    module.structs.extend(submodule.structs);
                    module.enums.extend(submodule.enums);
                    module.impls.extend(submodule.impls);
                    module.traits.extend(submodule.traits);
                    module.trait_impls.extend(submodule.trait_impls);
                    module.functions.extend(submodule.functions);
                    module.tensor_decls.extend(submodule.tensor_decls);
                    module.relations.extend(submodule.relations);
                    module.rules.extend(submodule.rules);
                    module.queries.extend(submodule.queries);
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

fn load_builtins() -> Result<tl_lang::compiler::ast::Module> {
    use tl_lang::compiler::codegen::builtin_types;

    let sources = [
        builtin_types::traits::SOURCE,
        builtin_types::vec::SOURCE,
        builtin_types::hashmap::SOURCE,
        builtin_types::hashset::SOURCE,
        builtin_types::vec_deque::SOURCE,
        builtin_types::btreemap::SOURCE,
        builtin_types::string_builder::SOURCE,
        builtin_types::option::SOURCE,
        builtin_types::result::SOURCE,
        builtin_types::llm::SOURCE,
        builtin_types::generic::mutex::SOURCE,
        builtin_types::generic::channel::SOURCE,
        builtin_types::non_generic::type_info::SOURCE,
        builtin_types::non_generic::atomic_types::SOURCE_I64,
        builtin_types::non_generic::atomic_types::SOURCE_I32,
        builtin_types::non_generic::time_types::SOURCE_DURATION,
        builtin_types::non_generic::time_types::SOURCE_INSTANT,
        builtin_types::non_generic::time_types::SOURCE_DATETIME,
        builtin_types::non_generic::net::SOURCE,
    ];

    let mut combined = tl_lang::compiler::ast::Module::new();
    for (i, src) in sources.iter().enumerate() {
        let m = tl_lang::compiler::parser::parse_from_source(src)
            .map_err(|e| anyhow::anyhow!("Failed to parse builtin {}: {:?}", i, e))?;
        combined.merge(m);
    }
    Ok(combined)
}
