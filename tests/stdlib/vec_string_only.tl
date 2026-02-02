fn main() {
    let v: Vec<String> = Vec::new();
    v.push("Hello");
    v.push("World");
    if v.len() != 2 {
        print("Len mismatch");
    }
}
