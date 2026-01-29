struct S { x: i64 }
impl S { fn new(v: i64) -> S { S(v) } }

fn main() {
    print("Testing memory manager with struct allocation...");
    
    let mem_start = System::memory_mb();
    print("Start memory:");
    print(mem_start);
    
    for i in range(0, 10) {
        let s = S::new(i);
        print("Loop iteration:");
        print(i);
    }
    
    let mem_end = System::memory_mb();
    print("End memory:");
    print(mem_end);
    print("Memory should be similar to start!");
}
