// Test user-defined generic structs
// This verifies that Rect<T> can be instantiated with different types

struct Rect<T> {
    w: T,
    h: T,
}

impl<T> Rect<T> {
    fn area(self) -> T {
        return self.w * self.h;
    }
}

fn main() {
    test_rect_i64();
    test_rect_f32();
    println("All generic struct tests passed!");
}

fn test_rect_i64() {
    let r: Rect<i64> = Rect { w: 10, h: 20 };
    if r.w != 10 {
        println("Rect<i64> w failed");
    }
    if r.h != 20 {
        println("Rect<i64> h failed");
    }
    println("test_rect_i64 passed");
}

fn test_rect_f32() {
    let r: Rect<f32> = Rect { w: 1.5, h: 2.5 };
    // Note: Float comparison may need tolerance, but for now assume exact match works
    if r.w != 1.5 {
        println("Rect<f32> w failed");
    }
    if r.h != 2.5 {
        println("Rect<f32> h failed");
    }
    println("test_rect_f32 passed");
}
