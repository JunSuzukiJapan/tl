struct Rect {
    w: i64,
    h: i64,
}

impl Rect {
    fn new(w: i64, h: i64) -> Rect {
        Rect { w: w, h: h }
    }

    fn area(self) -> i64 {
        self.w * self.h
    }
}

fn main() {
    let r = Rect::new(10, 5);
    let a = r.area();
    print(a); // STDOUT: 50
    print(r.w); // STDOUT: 10
}
