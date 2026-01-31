
fn main() {
    println("Testing System class...");

    // Time
    println("System::time():");
    let t = System::time();
    println(t);

    println("System::sleep(0.1)...");
    System::sleep(0.1);
    let t2 = System::time();
    println(t2);

    // Memory / Stats
    println("System::memory_mb():");
    println(System::memory_mb());

    println("System::pool_count():");
    println(System::pool_count());

    println("System::refcount_count():");
    println(System::refcount_count());

    println("System::scope_depth():");
    println(System::scope_depth());

    // Metal (even if not on Metal, should return 0 or default)
    println("System::metal_pool_bytes():");
    println(System::metal_pool_bytes());

    println("System::metal_pool_mb():");
    println(System::metal_pool_mb());

    println("System::metal_pool_count():");
    println(System::metal_pool_count());

    println("System::metal_sync()...");
    System::metal_sync();

    println("System tests passed!");
    
    // System::exit(0); // Uncomment to test exit
}
