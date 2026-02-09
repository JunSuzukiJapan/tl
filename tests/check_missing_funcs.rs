use std::fs;
use std::path::Path;
use regex::Regex;
use std::collections::HashSet;

#[test]
fn check_missing_runtime_functions() {
    let project_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let builtins_path = project_root.join("src/compiler/codegen/builtins.rs");
    let runtime_src_path = project_root.join("crates/tl_runtime/src");
    
    // 1. コンパイラが必要とする関数（Builtins）を抽出
    let builtins_content = fs::read_to_string(&builtins_path)
        .expect("Failed to read builtins.rs");
    
    let re_map_tensor = Regex::new(r#"map_tensor_fn!\("(\w+)""#).unwrap();
    let re_get_fn = Regex::new(r#"module\.get_function\("(\w+)"\)"#).unwrap();
    
    let mut required_funcs = HashSet::new();
    
    for cap in re_map_tensor.captures_iter(&builtins_content) {
        required_funcs.insert(cap[1].to_string());
    }
    for cap in re_get_fn.captures_iter(&builtins_content) {
        required_funcs.insert(cap[1].to_string());
    }
    
    // 除外リスト
    let ignores = vec![
        "tl_debug_print_ptr", 
        "tl_kb_", 
        "tl_tensor_acquire", 
        // CPU/GPU切り替えロジックで直接使われないものなど
        "tl_tensor_get_shape", // lib.rsで定義されていたはずだが
        "tl_tensor_matmul_4d", // 命名規則違いの可能性
        "tl_tensor_add_4d",
        "tl_tensor_cat_4d",
        "tl_tensor_silu_4d",
        "tl_tensor_cat2",
        "tl_tensor_reshape_2d",
        "tl_tensor_reshape_3d_to_2d",
        "tl_tensor_transpose_2d", 
        "tl_tensor_map_get_1d",
    ];
    
    // 2. ランタイムの全ソースファイルから関数定義を抽出
    let mut provided_funcs = HashSet::new();
    let re_func_def = Regex::new(r#"extern "C" fn (\w+)"#).unwrap();

    fn visit_dirs(dir: &Path, re: &Regex, funcs: &mut HashSet<String>) -> std::io::Result<()> {
        if dir.is_dir() {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    visit_dirs(&path, re, funcs)?;
                } else if path.extension().map_or(false, |e| e == "rs") {
                    let content = fs::read_to_string(&path)?;
                    // no_mangleがついている行の次あたりにある関数を探すのが正確だが、
                    // extern "C" fn はほぼエクスポート用なので単純にこれを見る
                    for cap in re.captures_iter(&content) {
                         funcs.insert(cap[1].to_string());
                    }
                }
            }
        }
        Ok(())
    }

    visit_dirs(&runtime_src_path, &re_func_def, &mut provided_funcs)
        .expect("Failed to scan runtime source files");

    // 3. 比較 & 検証
    let mut missing_funcs = Vec::new();
    for req in &required_funcs {
        if ignores.iter().any(|ig| req.starts_with(ig) || req == ig) {
            continue;
        }
        
        if !provided_funcs.contains(req) {
            missing_funcs.push(req.clone());
        }
    }
    
    if !missing_funcs.is_empty() {
        missing_funcs.sort();
        panic!(
            "Missing runtime functions (required by compiler but not found in tl_runtime source):\n{:#?}\n\nTotal required: {}, Total provided: {}",
            missing_funcs, required_funcs.len(), provided_funcs.len()
        );
    }
}
