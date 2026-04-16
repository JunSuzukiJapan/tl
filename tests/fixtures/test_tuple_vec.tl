fn test() {
    let mut v: Vec<(i64, i64)> = Vec::new();
    v.push((1, 2));
    let t = v.get(0).unwrap();
    println(t.0);
}
fn main() {
    test();
}
