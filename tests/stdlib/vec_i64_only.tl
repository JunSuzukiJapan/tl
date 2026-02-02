fn main() {
    let v = Vec::new();
    v.push(1);
    v.push(2);
    if v.len() != 2 {
        print("Len mismatch");
    }
}
