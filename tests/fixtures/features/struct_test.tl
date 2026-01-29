struct Counter {
    val: i64,
}

impl Counter {
    fn increment(self, amount: i64) -> Counter {
        Counter { val: self.val + amount }
    }
    
    fn get(self) -> i64 {
        self.val
    }
}

print(100);
let mut c = Counter { val: 0 };
print(200);
c = c.increment(10);
print(c.get());
