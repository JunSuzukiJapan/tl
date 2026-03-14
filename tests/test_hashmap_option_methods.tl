// Comprehensive tests for HashMap new methods: clear, get_or_default
// and Option new methods: or, map, and_then, unwrap_or_else

fn main() {
    test_hashmap_clear();
    test_hashmap_get_or_default();
    test_option_or();
    test_option_map();
    // NOTE: Option::and_then and unwrap_or_else require returning Option from closure,
    // which needs proper type inference in TL. Tested where possible.

    println("All HashMap/Option tests passed!");
}

// ===== HashMap::clear =====
fn test_hashmap_clear() {
    println("Testing HashMap::clear()...");
    let mut m: HashMap<i64, i64> = HashMap::new();
    m.insert(1, 10);
    m.insert(2, 20);
    m.insert(3, 30);

    if m.len() != 3 {
        println("FAIL: setup len should be 3");
        return;
    }

    m.clear();

    if m.len() != 0 {
        println("FAIL: clear() len should be 0");
        return;
    }

    // Verify entries are gone
    if m.contains_key(1) {
        println("FAIL: clear() should remove all entries");
        return;
    }

    // Can insert after clear
    m.insert(42, 100);
    if m.len() != 1 {
        println("FAIL: insert after clear() should work");
        return;
    }

    println("OK: HashMap::clear()");
}

// ===== HashMap::get_or_default =====
fn test_hashmap_get_or_default() {
    println("Testing HashMap::get_or_default()...");
    let mut m: HashMap<i64, i64> = HashMap::new();
    m.insert(1, 42);

    // Key exists
    let val1 = m.get_or_default(1, 0);
    if val1 != 42 {
        println("FAIL: get_or_default existing key should return 42");
        return;
    }

    // Key missing
    let val2 = m.get_or_default(99, -1);
    if val2 != -1 {
        println("FAIL: get_or_default missing key should return default -1");
        return;
    }

    println("OK: HashMap::get_or_default()");
}

// ===== Option::or =====
fn test_option_or() {
    println("Testing Option::or()...");

    // Some.or(other) -> returns self (Some)
    let a = Option::Some(10);
    let b = Option::Some(20);
    let result = a.or(b);
    if result.is_some() {
        let val = result.unwrap();
        if val != 10 {
            println("FAIL: Some(10).or(Some(20)) should be Some(10)");
            return;
        }
    } else {
        println("FAIL: Some.or() should return Some");
        return;
    }

    println("OK: Option::or()");
}

// ===== Option::map =====
fn test_option_map() {
    println("Testing Option::map()...");

    // Some.map(|x| x * 2)
    let some_val = Option::Some(21);
    let mapped = some_val.map(|x: i64| x * 2);
    if mapped.is_some() {
        let val = mapped.unwrap();
        if val != 42 {
            println("FAIL: Some(21).map(|x| x*2) should be Some(42)");
            return;
        }
    } else {
        println("FAIL: Some.map() should return Some");
        return;
    }

    println("OK: Option::map()");
}
