fn main() {
    let mut v = Vec::new();
    v.push(10);
    v.push(20);
    print(v.len());

    let mut v2 = Vec::new();
    let p = Point { x: 1, y: 2 };
    v2.push(p);
    print(v2.len());
}

struct Point {
    x: i64,
    y: i64,
}
