fn main() {
    let mut v = Vec::new();
    v.push(1);
    v.push(2);
    v.push(3);
    let sum = v.reduce(|acc: i64, x: i64| acc + x);
    println(sum);

    let mut v2 = Vec::new();
    v2.push("a");
    v2.push("b");
    v2.push("c");
    let s = v2.join("-");
    println(s);
}
