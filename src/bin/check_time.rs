fn main() {
    let source = std::fs::read_to_string("tests/test_datetime.tl").unwrap();
    let mut module = tl_lang::compiler::parser::parse_from_source(&source).unwrap();
    
    // Load builtins like main.rs
    let mut builtins = tl_lang::compiler::parser::parse_from_source(tl_lang::compiler::codegen::builtin_types::non_generic::time_types::SOURCE_DURATION).unwrap();
    module.structs.extend(builtins.structs);
    module.impls.extend(builtins.impls);
    module.functions.extend(builtins.functions);
    
    let mut sem = tl_lang::compiler::semantics::SemanticAnalyzer::new(source.clone());
    sem.check_module(&mut module).unwrap();
    
    // Print the Let d1 statement
    let main_fn = module.functions.iter().find(|f| f.name == "main").unwrap();
    for stmt in &main_fn.body {
        if let tl_lang::compiler::ast::StmtKind::Let { name, type_annotation, .. } = &stmt.inner {
            if name == "d1" {
                println!("d1 type_annotation = {:?}", type_annotation);
                println!("d1 stmt = {:?}", stmt);
            }
        }
    }
}
