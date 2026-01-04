fn verify_offset(expected: i64) {
    let actual = tl_arena_get_offset();
    if actual != expected {
        print("Arena offset mismatch! Expected: ");
        print(expected);
        print(", Got: ");
        println(actual);
    }
}

// 1. Function level test with early return
fn test_early_return(cond: bool) -> Tensor<f32, 1> {
    // Explicit arena allocation to test scope reset
    let ptr = tl_arena_alloc(128); // 128 bytes
    
    let large1: Tensor<f32, 1> = Tensor::randn([10], false); 
    
    if cond {
        return large1; 
    }
    
    let ptr2 = tl_arena_alloc(256); // Another allocation
    return large1;
}

fn run_func_test() {
    println("Testing function level early return...");
    let start_offset = tl_arena_get_offset();
    
    // Test early return path
    let r1 = test_early_return(true);
    let end_offset1 = tl_arena_get_offset();
    if end_offset1 != start_offset {
        print("FAILED: Early return did not restore offset. Start: ");
        print(start_offset);
        print(", End: ");
        println(end_offset1);
    } else {
        println("PASSED: Early return restored offset");
    }
    
    // Test normal return path
    let r2 = test_early_return(false);
    let end_offset2 = tl_arena_get_offset();
    if end_offset2 != start_offset {
        print("FAILED: Normal return did not restore offset. Start: ");
        print(start_offset);
        print(", End: ");
        println(end_offset2);
    } else {
        println("PASSED: Normal return restored offset");
    }
}

// 2. Loop test
fn run_loop_test() {
    println("Testing loop allocation stability...");
    let start_offset = tl_arena_get_offset();
    
    for i in range(0, 5) {
        let ptr = tl_arena_alloc(64);
        // Arena should reset after each iteration because the loop body is a scope
    }
    
    let end_offset = tl_arena_get_offset();
    if end_offset != start_offset {
        print("FAILED: Loop iterations did not restore offset. Start: ");
        print(start_offset);
        print(", End: ");
        println(end_offset);
    } else {
        println("PASSED: Loop iterations restored offset");
    }
}

// 3. Nested blocks test
fn run_nested_test() {
    println("Testing nested block scope resets...");
    let start_offset = tl_arena_get_offset();
    
    if true {
        let ptr1 = tl_arena_alloc(128);
        let middle_offset = tl_arena_get_offset();
        
        if middle_offset <= start_offset {
             println("FAILED: Outer block did not allocate in arena");
        }

        if true {
            let ptr2 = tl_arena_alloc(256);
            let inner_offset = tl_arena_get_offset();
            if inner_offset <= middle_offset {
                println("FAILED: Inner block did not allocate in arena");
            }
        }
        // After inner block, offset should be back to middle_offset
        if tl_arena_get_offset() != middle_offset {
             print("FAILED: Inner block did not restore offset. Middle: ");
             print(middle_offset);
             print(", After inner: ");
             println(tl_arena_get_offset());
        } else {
             println("PASSED: Inner block restored offset");
        }
    }
    
    let end_offset = tl_arena_get_offset();
    if end_offset != start_offset {
        print("FAILED: Nested blocks did not restore offset. Start: ");
        print(start_offset);
        print(", End: ");
        println(end_offset);
    } else {
        println("PASSED: Nested blocks restored offset");
    }
}

fn main() {
    println("Starting Arena Allocator Integration Tests");
    
    // Initialize arena with enough space
    // Since main doesn't have many allocations, we might need to be careful
    // but the system should init it on first use or based on profile.
    
    let active = tl_arena_is_active();
    if active {
        println("Arena is active.");
    } else {
        println("Arena is NOT active. Creating a tensor to trigger init...");
        let dummy = Tensor::randn([10], false);
        let active2 = tl_arena_is_active();
        if active2 {
             println("Arena is now active.");
        } else {
             println("Arena still NOT active. Manual init...");
             tl_arena_init(65536); // 64KB
        }
    }

    run_func_test();
    run_loop_test();
    run_nested_test();
    
    println("Arena Integration Tests Completed");
}
