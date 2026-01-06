fn foo() {
    print("foo called");
}

struct Point {
    x: i64,
    y: i64,
}

impl Point {
    fn new(x: i64, y: i64) -> Point {
        Point { x: x, y: y }
    }
    
    fn print(self) {
        print(self.x);
        print(self.y);
    }
}
