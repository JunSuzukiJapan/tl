fn main() {
    let s: String = "Hello";
    let s2: String = " World";
    print(s + s2);

    let v: Vec<String> = Vec::new();
    v.push("VecTest");
    let s3 = v.get(0);
    print(s3);
}
