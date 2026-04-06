fn test_btreemap_basic() {
    let mut map: BTreeMap<i64, i64> = BTreeMap::new();
    
    if map.is_empty() {
        println("OK: map is empty");
    } else {
        println("FAIL: map should be empty");
    }
    
    map.insert(10, 100);
    map.insert(20, 200);
    map.insert(5, 50);
    
    if map.len() == 3 {
        println("OK: map len is 3");
    } else {
        println("FAIL: map len is not 3");
    }
    
    match map.get(10) {
        Option::Some(v) => if v == 100 { println("OK: get(10) == 100"); } else { println("FAIL: get(10) == {}", v); },
        Option::None => println("FAIL: get(10) not found"),
    }
    match map.get(20) {
        Option::Some(v) => if v == 200 { println("OK: get(20) == 200"); } else { println("FAIL: get(20) == {}", v); },
        Option::None => println("FAIL: get(20) not found"),
    }
    match map.get(5) {
        Option::Some(v) => if v == 50 { println("OK: get(5) == 50"); } else { println("FAIL: get(5) == {}", v); },
        Option::None => println("FAIL: get(5) not found"),
    }
    
    // Key that does not exist
    match map.get(15) {
        Option::Some(v) => println("FAIL: get(15) found {}", v),
        Option::None => println("OK: get(15) not found"),
    }
    
    // Update existing key
    map.insert(10, 1000);
    match map.get(10) {
        Option::Some(v) => if v == 1000 { println("OK: updated get(10) == 1000"); } else { println("FAIL: updated get(10) == {}", v); },
        Option::None => println("FAIL: updated get(10) not found"),
    }
}

fn test_btreemap_split() {
    let mut map: BTreeMap<i64, i64> = BTreeMap::new();
    
    // Insert more than 7 elements to trigger a split (since max keys per node is 7)
    let mut i = 1;
    while i <= 15 {
        map.insert(i, i * 10);
        i = i + 1;
    }
    
    if map.len() == 15 {
        println("OK: map len after splits is 15");
    } else {
        println("FAIL: map len is {}", map.len());
    }
    
    // Verify all keys are still retrievable
    let mut ok = true;
    let mut j = 1;
    while j <= 15 {
        match map.get(j) {
            Option::Some(v) => {
                if v != j * 10 {
                    println("FAIL: get({}) == {}", j, v);
                    ok = false;
                }
            },
            Option::None => {
                println("FAIL: get({}) not found", j);
                ok = false;
            }
        }
        j = j + 1;
    }
    if ok {
        println("OK: all 15 elements found correctly after splits");
    }
    
    // Test clear
    map.clear();
    if map.is_empty() {
        println("OK: map is empty after clear");
    } else {
        println("FAIL: map is not empty after clear");
    }
}

fn main() {
    test_btreemap_basic();
    test_btreemap_split();
}
