fn main() {
    let t = Tensor::randn([100, 100], true);
    let mut x = t; 
    let start_mem = System::memory_bytes();
    
    let mut i = 0;
    while i < 10000 {
        x = t; // L-value assignment leak check
        let y = t; // Let L-value leak check
        let z = Tensor::randn([10,10], true); // Let R-value leak check
        i = i + 1;
    }
    
    System::metal_sync();
    let end_mem = System::memory_bytes();
    
    // Allow small fluctuation but not linear growth (10000 iter * 40KB ~ 400MB if leaking)
    println("Start Mem: {}", start_mem);
    println("End Mem: {}", end_mem);
    let diff = if end_mem > start_mem { end_mem - start_mem } else { 0 };
    if diff > 1024 * 1024 * 5 { 
        println("FAIL: Leaked > 5MB");
        // std::process::exit(1);
    } else {
        println("PASS: Memory Stable");
    }
}
