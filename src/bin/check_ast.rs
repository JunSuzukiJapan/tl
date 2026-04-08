fn main() {
    let source = std::fs::read_to_string("tests/test_atomic.tl").unwrap();
    let mut module = tl_lang::compiler::parser::parse_from_source(&source).unwrap();
    let mut sem = tl_lang::compiler::semantics::SemanticAnalyzer::new(source);
    sem.check_module(&mut module).unwrap();
    println!("{:?}", module.functions[0].body[0]);
}
