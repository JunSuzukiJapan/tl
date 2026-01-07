// Minimal test for sret: struct method returning struct
struct Counter { value: i64 }

impl Counter {
    fn new(v: i64) -> Counter {
        return Counter(v);
    }
    
    fn increment(self) -> Counter {
        let new_val = self.value + 1;
        return Counter(new_val);
    }
}

fn main() {
    print("Test sret with struct method");
    let c1 = Counter::new(0);
    print("c1.value:");
    print(c1.value);
    
    let c2 = c1.increment();
    print("c2.value (should be 1):");
    print(c2.value);
    
    let c3 = c2.increment();
    print("c3.value (should be 2):");
    print(c3.value);
    
    print("Success!");
}
