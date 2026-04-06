fn verify_inference(name: String, test_val: i64, expected: i64) {
    if test_val == expected {
        print("OK: " + name);
    } else {
        print("FAIL: " + name);
    }
}

fn test_collections() {
    print("--- Collection Inference ---");
    // 1. Array/Vec mapping with implicit args and return types
    let arr: Vec<i64> = Vec::new();
    arr.push(1);
    arr.push(2);
    arr.push(3);
    let mapped = arr.map(|x| x * 2);
    
    // 2. Chaining operations requiring type persistence
    let filtered = mapped.filter(|x| x > 4); 
    
    // 3. reduce with implicit accumulator and element types
    let sum = filtered.reduce(|acc, val| acc + val);
    verify_inference("Vec map/filter/reduce", sum, 6);
}

fn test_enums() {
    print("--- Enum Inference ---");
    // 1. Option construction inferring from value
    let opt_val = Option::Some(100); 
    
    // 2. map on Option with implicit lambda types
    let opt_mapped = opt_val.map(|v| v + 50); 
    
    // 3. unwrap_or_else with implicit return type
    let final_val = opt_mapped.unwrap_or_else(|| 0);
    verify_inference("Option map and unwrap", final_val, 150);

    // 4. Result mapping
    let res: Result<i64, String> = Result::Ok(10);
    // map_err uses implicit error parameter and implicit string return
    let res_err = res.map_err(|e| "error: " + e); 
}

fn test_generics() {
    print("--- Generic Context Inference ---");
    // 1. Empty Vec initialization predicting type from future appends 
    let my_vec = Vec::new();
    my_vec.push(42); 
    
    // 2. Nested generics (Vec<Option<i64>>)
    let nested = Vec::new(); 
    nested.push(Option::Some(7));
    nested.push(Option::None());
}

fn test_concurrency() {
    print("--- Concurrency Inference ---");
    // 1. Thread spawn implicitly capturing & returning a tuple
    let handle = Thread::spawn(|| {
        (10, 20)
    });
    let res = handle.join(); 
    let unpacked = res.unwrap_or_else(|| (0, 0));
    verify_inference("Thread returning tuple via Result", unpacked.0 + unpacked.1, 30);
    
    // 2. Mutex implicitly typed operations
    let m = Mutex::new(10);
    m.modify(|val| val * 5); 
    let m_val = m.read(|v| v);
    verify_inference("Mutex read/modify", m_val, 50);
}

fn main() {
    test_collections();
    test_enums();
    test_generics();
    test_concurrency();
}
