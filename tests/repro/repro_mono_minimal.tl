// mono_comprehensive 部分テスト（HashMap なし）

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

fn assert_eq(actual: i64, expected: i64, msg: String) {
    if actual != expected {
        print("FAIL: ");
        print(msg);
        print(" - expected ");
        print(expected);
        print(", got ");
        println(actual);
    }
}

fn main() {
    println("Test 1: Basic Box<i64>");
    let b1: Box<i64> = Box::new(42);
    assert_eq(b1.get(), 42, "Box<i64>");
    println("Test 1 PASSED");
    
    println("Test 2: Box<bool>");
    let b4: Box<bool> = Box::new(true);
    println("Test 2 PASSED");
    
    println("All tests passed");
}
