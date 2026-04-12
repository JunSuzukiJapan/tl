fn main() {
    let src = "fn test<K,V>() { let mut result: Vec<(K, V)> = Vec::new(); }";
    let module = tl_lang::compiler::parser::parse_from_source(src).unwrap();
    println!("{:#?}", module.functions);
}
