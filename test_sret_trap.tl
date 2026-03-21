struct Point {
    x: f32,
    y: f32
}

fn make_point(x: f32, y: f32) -> Point {
    print("Inside make_point");
    Point(x, y)
}

fn main() {
    print("Start");
    let p = make_point(1.0, 2.0);
    print("End");
}
