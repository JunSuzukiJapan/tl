fn main() {
    let mut map = HashMap::new();
    map.insert(1, 10);
    map.insert(2, 20);
    
    let entries = map.entries();
    assert(entries.len() == 2, "entries.len() == 2");
    
    let entry0 = entries[0].unwrap();
    let entry1 = entries[1].unwrap();
    
    // Hash maps are unordered, so we check if the sum matches
    assert((entry0.0 + entry1.0) == 3, "sum matches");
    assert((entry0.1 + entry1.1) == 30, "sum values matches");
    
    print("HashMap::entries OK\n");
}
