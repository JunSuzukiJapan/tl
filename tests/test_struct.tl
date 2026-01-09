// Minimal test with struct

struct Point {
    x: f32,
    y: f32,
}

impl Point {
    fn new(x: f32, y: f32) -> Point {
        return Point { x: x, y: y };
    }
}

fn main() {
    print("Testing struct...");
    let p = Point::new(1.0, 2.0);
    print("Struct created!");
}
