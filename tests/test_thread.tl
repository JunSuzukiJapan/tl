fn test_thread_basic() {
    let t = Thread::spawn(|| {
        println("Hello from Thread!");
        42
    });
    
    let res = t.join();
    if res == 42 {
        println("OK: Thread returned 42");
    } else {
        println("FAIL: Thread returned something else");
    }
}

fn test_mutex_basic() {
    let m = Mutex::new(100);
    
    // Read the value
    let v1 = m.read(|x: i64| {
        x
    });
    
    if v1 == 100 {
        println("OK: Mutex read returned 100");
    } else {
        println("FAIL: Mutex read returned something else");
    }
    
    // Modify the value
    m.modify(|x: i64| {
        x + 50
    });
    
    // Read again
    let v2 = m.read(|x: i64| x);
    if v2 == 150 {
        println("OK: Mutex modify updated to 150");
    } else {
        println("FAIL: Mutex modify didn't update to 150");
    }
}

fn main() {
    println("--- Testing Thread ---");
    test_thread_basic();
    
    println("--- Testing Mutex ---");
    test_mutex_basic();
}
