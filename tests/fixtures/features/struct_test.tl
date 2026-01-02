struct Counter {
    val: i64,
}

impl Counter {
    fn increment(self: Counter, amount: i64) {
        self.val = self.val + amount;
    }
    
    fn get(self: Counter) -> i64 {
        return self.val;
    }
}

print(100);
let c = Counter(0);
print(200);
c.increment(10);
print(c.get());
