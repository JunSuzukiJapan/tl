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

extern fn exit(code: i32);

fn main() {
    println("Testing HashMap<String, i64>");
    let m: HashMap<String, i64> = HashMap::new();
    
    m.insert("one", 1);
    m.insert("two", 2);
    
    assert_eq_i64(m.len(), 2, "Len check");
    assert_eq_i64(m.get("one"), 1, "Get one");
    assert_eq_i64(m.get("two"), 2, "Get two");
    
    m.insert("one", 100);
    assert_eq_i64(m.get("one"), 100, "Update one");
    
    let r = m.remove("two");
    assert_eq_i64(r, 2, "Remove two");
    assert_eq_i64(m.len(), 1, "Len after remove");
    
    println("HashMap<String, i64> Passed!");
}
