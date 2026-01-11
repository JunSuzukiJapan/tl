struct Point {
    x: i64,
    y: i64,
}

fn main() {
    let p = Point { x: 10, y: 20 };
    print(p.x); // STDOUT: 10
    print(p.y); // STDOUT: 20
    
    // Mutable updates not tested yet
}
