struct Counter {
    val: i64,
}

impl Counter {
    fn increment(amount: i64) {
        self.val = self.val + amount;
    }
    
    fn get() -> i64 {
        return self.val;
    }
}

print(100);
let c = Counter(0);
print(200);
c.increment(10);
print(c.get());
