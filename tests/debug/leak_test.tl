
fn get_memory() -> i64 {
    return tl_get_memory_mb();
}

struct Inner { t: Tensor<f32, 2> }
impl Inner {
    fn new() -> Inner {
        // 1000x1000 float tensor = 4MB
        return Inner(Tensor::randn([1000, 1000], false));
    }
}

struct Outer { i: Inner }

fn main() {
    let start_mem = get_memory();
    println("Initial Memory: {}", start_mem);

    let idx = 0;
    while idx < 50 {
         let o = Outer(Inner::new());
         // This re-assignment should free the old Inner and its Tensor
         // If logic is missing, it leaks 4MB per iter
         o.i = Inner::new(); 
         
         if idx - (idx / 10) * 10 == 0 {
             println("Iter: {} Mem: {}", idx, get_memory());
         }
         idx = idx + 1;
    }

    let end_mem = get_memory();
    println("Final Memory: {}", end_mem);
    
    if end_mem > start_mem + 50 {
        print("FAIL: Memory Leaked significantly");
    } else {
        print("PASS: Memory Stable");
    }
}
