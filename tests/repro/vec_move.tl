
// Externs
// Externs removed - they are now built-in

struct MyVec<T> {
    handle: i64,
    marker: PhantomData<T>,
}

impl MyVec<T> {
    fn new() -> MyVec<T> {
        let h = tl_vec_unified_new();
        return MyVec { handle: h, marker: PhantomData {} };
    }

    fn push(self, val: i64) {
        tl_vec_unified_push(self.handle, val, false);
    }
    
    fn get(self, idx: i64) -> i64 {
        return tl_vec_unified_get(self.handle, idx, false);
    }

    fn free(self) {
        println("MyVec::free called");
    }
}

fn main() {
    println("Allocating v...");
    let v: MyVec<i64> = MyVec::new();
    println("Allocated v.");
    
    v.push(10);
    println("Pushed 10.");
    
    println("Moving v to v2...");
    let v2 = v;
    println("Moved.");
    
    let x = v2.get(0);
    println("Got value");
}
