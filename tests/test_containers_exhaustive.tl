extern fn exit(code: i32);

fn assert(cond: bool, msg: String) {
    if !cond {
        print("Assertion Failed: ");
        println(msg);
        exit(1);
    }
}

fn assert_eq_i64(a: i64, b: i64, msg: String) {
    if a != b {
        print("Assertion Failed: ");
        print(msg);
        print(" (Expected ");
        print(b);
        print(", Got ");
        print(a);
        println(")");
        exit(1);
    }
}

fn assert_eq_f32(a: f32, b: f32, msg: String) {
    let diff = a - b;
    if diff < -0.0001 || diff > 0.0001 {
        print("Assertion Failed: ");
        print(msg);
        print(" (Expected ");
        print(b);
        print(", Got ");
        print(a);
        println(")");
        exit(1);
    }
}

fn assert_eq_bool(a: bool, b: bool, msg: String) {
    if a != b {
        print("Assertion Failed: ");
        println(msg);
        exit(1);
    }
}

// --- Vec Tests ---

fn test_vec_i64() {
    println("Testing Vec<i64>...");
    let v: Vec<i64> = Vec::new();
    println("Created vec");
    assert(v.is_empty(), "Vec should be empty initially");
    println("Checked empty");
    
    v.push(10);
    println("Pushed 10");
    v.push(20);
    v.push(30);
    
    assert_eq_i64(v.len(), 3, "Len should be 3");
    assert(!v.is_empty(), "Vec should not be empty");
    assert(v.contains(20), "Vec should contain 20");
    assert(!v.contains(99), "Vec should not contain 99");
    
    assert_eq_i64(v.get(0), 10, "Get 0");
    assert_eq_i64(v.get(1), 20, "Get 1");
    assert_eq_i64(v.get(2), 30, "Get 2");
    
    let p = v.pop();
    assert_eq_i64(p, 30, "Pop should return 30");
    assert_eq_i64(v.len(), 2, "Len after pop");
    
    v.insert(0, 5); // [5, 10, 20]
    assert_eq_i64(v.get(0), 5, "Insert at 0");
    assert_eq_i64(v.get(1), 10, "Shifted element");
    
    let r = v.remove(1); // Remove 10 -> [5, 20]
    assert_eq_i64(r, 10, "Remove should return 10");
    assert_eq_i64(v.get(1), 20, "Element shifted back");
    
    v.clear();
    assert(v.is_empty(), "Vec should be empty after clear");
    assert_eq_i64(v.len(), 0, "Len 0 after clear");
}

fn test_vec_f32() {
    println("Testing Vec<f32>...");
    let v: Vec<f32> = Vec::new();
    v.push(1.5);
    v.push(2.5);
    
    assert_eq_f32(v.get(0), 1.5, "f32 get 0");
    assert_eq_f32(v.get(1), 2.5, "f32 get 1");
    assert(v.contains(1.5), "contains f32");
    
    let p = v.pop();
    assert_eq_f32(p, 2.5, "pop f32");
}

fn test_vec_string() {
    println("Testing Vec<String>...");
    let v: Vec<String> = Vec::new();
    v.push("Hello");
    v.push("World");
    
    assert_eq_i64(v.len(), 2, "String vec len");

    let s0 = v.get(0);
    // tl_string_eq check implicitly
    if s0 != "Hello" {
        println("Vec<String> get(0) failed");
        exit(1);
    }
    
    assert(v.contains("World"), "String vec contains");
    
    let p = v.pop();
    if p != "World" {
         println("Vec<String> pop failed");
         exit(1);
    }
}

fn test_vec_nested() {
    println("Testing Vec<Vec<i64>>...");
    let v: Vec<Vec<i64>> = Vec::new();
    let inner1: Vec<i64> = Vec::new();
    inner1.push(1);
    inner1.push(2);
    
    let inner2: Vec<i64> = Vec::new();
    inner2.push(3);
    inner2.push(4);
    
    v.push(inner1);
    v.push(inner2);
    
    let r1 = v.get(0);
    assert_eq_i64(r1.get(0), 1, "Nested vec access 0-0");
    assert_eq_i64(r1.len(), 2, "Nested vec len");
    
    let r2 = v.get(1);
    assert_eq_i64(r2.get(0), 3, "Nested vec access 1-0");
}

fn test_vec_stress() {
    println("Testing Vec Stress...");
    let v: Vec<i64> = Vec::new();
    let n = 1000;
    let mut i = 0;
    loop {
        if i >= n { break; }
        v.push(i);
        i = i + 1;
    }
    
    assert_eq_i64(v.len(), n, "Stress len");
    assert_eq_i64(v.get(500), 500, "Stress get");
    
    let mut j = 0;
    loop {
        if j >= n { break; }
        v.pop();
        j = j + 1;
    }
    assert(v.is_empty(), "Stress empty");
}

// --- HashMap Tests ---

fn test_hashmap_i64() {
    println("Testing HashMap<String, i64>...");
    let m: HashMap<String, i64> = HashMap::new();
    
    m.insert("one", 1);
    m.insert("two", 2);
    m.insert("three", 3);
    
    assert_eq_i64(m.len(), 3, "Map len");
    assert(m.contains_key("two"), "Map contains key");
    assert(!m.contains_key("four"), "Map missing key");
    
    assert_eq_i64(m.get("one"), 1, "Map get one");
    
    // Update
    m.insert("one", 100);
    assert_eq_i64(m.get("one"), 100, "Map update");
    
    let r = m.remove("two"); 
    assert_eq_i64(r, 2, "Map remove two");
    
    assert(!m.contains_key("two"), "Map removed key");
}

fn test_hashmap_string() {
    println("Testing HashMap<String, String>...");
    let m: HashMap<String, String> = HashMap::new();
    m.insert("k1", "v1");
    m.insert("k2", "v2");
    
    let v = m.get("k1");
    if v != "v1" {
        println("Map string value failed");
        exit(1);
    }
}

fn main() {
    test_vec_i64();
    test_vec_f32();
    test_vec_string();
    test_vec_nested();
    test_vec_stress();
    
    test_hashmap_i64();
    test_hashmap_string();
    
    println("ALL TESTS PASSED");
}
