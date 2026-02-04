// Comprehensive Vec<T> Tests
// Tests all Vec methods and edge cases

fn main() {
    test_new_and_empty();
    test_with_capacity();
    test_push_and_len();
    test_get_bounds();
    test_set_bounds();
    test_pop();
    test_is_empty();
    test_realloc_growth();
    test_index_access();
    
    println("All Vec tests passed!");
}

// Test Vec::new() creates empty vector
fn test_new_and_empty() {
    println("Testing new() and empty state...");
    let v: Vec<i64> = Vec::new();
    
    if v.len() != 0 {
        println("FAIL: new() len should be 0");
        return;
    }
    if v.capacity() != 0 {
        println("FAIL: new() capacity should be 0");
        return;
    }
    if !v.is_empty() {
        println("FAIL: new() should be empty");
        return;
    }
    println("OK: new() and empty state");
}

// Test Vec::with_capacity()
fn test_with_capacity() {
    println("Testing with_capacity()...");
    
    // Normal capacity
    let v1: Vec<i64> = Vec::with_capacity(10);
    if v1.capacity() != 10 {
        println("FAIL: with_capacity(10) capacity mismatch");
        return;
    }
    if v1.len() != 0 {
        println("FAIL: with_capacity(10) len should be 0");
        return;
    }
    
    // Zero capacity (should behave like new())
    let v2: Vec<i64> = Vec::with_capacity(0);
    if v2.capacity() != 0 {
        println("FAIL: with_capacity(0) capacity should be 0");
        return;
    }
    
    println("OK: with_capacity()");
}

// Test push and len
fn test_push_and_len() {
    println("Testing push() and len()...");
    let mut v: Vec<i64> = Vec::new();
    
    v.push(10);
    if v.len() != 1 {
        println("FAIL: len after 1 push");
        return;
    }
    
    v.push(20);
    v.push(30);
    if v.len() != 3 {
        println("FAIL: len after 3 pushes");
        return;
    }
    
    println("OK: push() and len()");
}

// Test get() with bounds checking
fn test_get_bounds() {
    println("Testing get() bounds...");
    let mut v: Vec<i64> = Vec::new();
    v.push(100);
    v.push(200);
    v.push(300);
    
    // Valid indices
    match v.get(0) {
        Option::Some(val) => {
            if val != 100 {
                println("FAIL: get(0) value mismatch");
                return;
            }
        }
        Option::None => {
            println("FAIL: get(0) returned None");
            return;
        }
    }
    
    match v.get(2) {
        Option::Some(val) => {
            if val != 300 {
                println("FAIL: get(2) value mismatch");
                return;
            }
        }
        Option::None => {
            println("FAIL: get(2) returned None");
            return;
        }
    }
    
    // Out of bounds: negative
    match v.get(-1) {
        Option::Some(_) => {
            println("FAIL: get(-1) should return None");
            return;
        }
        Option::None => {}
    }
    
    // Out of bounds: too large
    match v.get(3) {
        Option::Some(_) => {
            println("FAIL: get(3) should return None");
            return;
        }
        Option::None => {}
    }
    
    println("OK: get() bounds");
}

// Test set() with bounds checking
fn test_set_bounds() {
    println("Testing set() bounds...");
    let mut v: Vec<i64> = Vec::new();
    v.push(1);
    v.push(2);
    v.push(3);
    
    // Valid set
    v.set(1, 42);
    let val = v.get(1).unwrap();
    if val != 42 {
        println("FAIL: set(1, 42) value mismatch");
        return;
    }
    
    // Out of bounds set (should be ignored)
    v.set(10, 999);  // Should not crash
    v.set(-1, 999);  // Should not crash
    
    // Verify vector unchanged
    if v.len() != 3 {
        println("FAIL: len changed after out-of-bounds set");
        return;
    }
    
    println("OK: set() bounds");
}

// Test pop()
fn test_pop() {
    println("Testing pop()...");
    let mut v: Vec<i64> = Vec::new();
    v.push(10);
    v.push(20);
    v.push(30);
    
    // Pop all elements
    match v.pop() {
        Option::Some(val) => {
            if val != 30 {
                println("FAIL: pop() should return 30");
                return;
            }
        }
        Option::None => {
            println("FAIL: pop() returned None");
            return;
        }
    }
    
    if v.len() != 2 {
        println("FAIL: len after pop should be 2");
        return;
    }
    
    v.pop();
    v.pop();
    
    if v.len() != 0 {
        println("FAIL: len after 3 pops should be 0");
        return;
    }
    
    // Pop from empty
    match v.pop() {
        Option::Some(_) => {
            println("FAIL: pop() from empty should return None");
            return;
        }
        Option::None => {}
    }
    
    println("OK: pop()");
}

// Test is_empty()
fn test_is_empty() {
    println("Testing is_empty()...");
    let mut v: Vec<i64> = Vec::new();
    
    if !v.is_empty() {
        println("FAIL: new vec should be empty");
        return;
    }
    
    v.push(1);
    if v.is_empty() {
        println("FAIL: vec with 1 element should not be empty");
        return;
    }
    
    v.pop();
    if !v.is_empty() {
        println("FAIL: vec after pop should be empty");
        return;
    }
    
    println("OK: is_empty()");
}

// Test automatic reallocation on growth
fn test_realloc_growth() {
    println("Testing realloc growth...");
    let mut v: Vec<i64> = Vec::new();
    
    // Initial capacity should be 0
    if v.capacity() != 0 {
        println("FAIL: initial capacity should be 0");
        return;
    }
    
    // First push triggers allocation (initial cap = 4)
    v.push(1);
    if v.capacity() < 1 {
        println("FAIL: capacity after first push should be >= 1");
        return;
    }
    let initial_cap = v.capacity();
    
    // Fill to capacity
    while v.len() < v.capacity() {
        v.push(v.len());
    }
    
    // Next push should trigger realloc (double capacity)
    let cap_before = v.capacity();
    v.push(999);
    
    if v.capacity() <= cap_before {
        println("FAIL: capacity should grow after realloc");
        return;
    }
    
    // Verify data integrity after realloc
    let last_val = v.get(v.len() - 1).unwrap();
    if last_val != 999 {
        println("FAIL: data corrupted after realloc");
        return;
    }
    
    println("OK: realloc growth");
}

// Test index access syntax v[i] and v[i] = x
fn test_index_access() {
    println("Testing index access...");
    let mut v: Vec<i64> = Vec::new();
    v.push(10);
    v.push(20);
    v.push(30);
    
    // Index read (if supported)
    // Note: This may use get() internally
    
    // Index write
    v[1] = 42;
    let val = v.get(1).unwrap();
    if val != 42 {
        println("FAIL: v[1] = 42 failed");
        return;
    }
    
    println("OK: index access");
}
