// mono_comprehensive.tl のテスト関数を段階的に実行

struct Box<T> {
    value: T,
}

impl<T> Box<T> {
    fn new(v: T) -> Box<T> {
        Box { value: v }
    }
    
    fn get(self) -> T {
        self.value
    }
}

fn main() {
    println("=== Test 1: Basic Generics ===");
    
    // 単一型パラメータ
    let b1: Box<i64> = Box::new(42);
    println("Box created");
    let v = b1.get();
    println("Box.get: {}", v);
    
    let b2: Box<String> = Box::new("hello");
    println("Box<String> created");
    
    let b3: Box<f32> = Box::new(3.14);
    println("Box<f32> created");
    
    println("Test 1 PASSED");
}
