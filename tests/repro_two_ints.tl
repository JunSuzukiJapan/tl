// Two i64 fields test

struct TwoInts {
    a: i64,
    b: i64,
}

impl TwoInts {
    fn new(x: i64, y: i64) -> TwoInts {
        print("Creating TwoInts...");
        let result = TwoInts(x, y);
        print("TwoInts constructed.");
        return result;
    }
    
    fn get_a(self) -> i64 {
        print("get_a: accessing self.a...");
        let a = self.a;
        print("get_a: self.a =");
        print(a);
        return a;
    }
    
    fn get_b(self) -> i64 {
        print("get_b: accessing self.b...");
        let b = self.b;
        print("get_b: self.b =");
        print(b);
        return b;
    }
}

fn main() {
    print("=== Two Ints Test ===");
    let t = TwoInts::new(42, 99);
    print("TwoInts created, calling get_a...");
    let a = t.get_a();
    print("Final a:");
    print(a);
    print("Calling get_b...");
    let b = t.get_b();
    print("Final b:");
    print(b);
}
