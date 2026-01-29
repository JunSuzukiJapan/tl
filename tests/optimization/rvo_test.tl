struct Point {
    x: i64,
    y: i64,
}

fn create_point(x: i64, y: i64) -> Point {
    let p = Point { x: x, y: y };
    p
}

fn main() {
    let p1 = create_point(10, 20);
    print(p1.x);
    print(p1.y);
}
