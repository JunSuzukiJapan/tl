fn test_primitive() {
    println("Testing Option<i64>...");
    let some_val: Option<i64> = Option::Some(42);
    let none_val: Option<i64> = Option::None;

    if some_val.is_some() {
        println("OK: is_some()");
    } else {
        println("FAIL: is_some()");
    }

    if none_val.is_none() {
        println("OK: is_none()");
    } else {
        println("FAIL: is_none()");
    }

    let val = some_val.unwrap();
    if val == 42 {
        println("OK: unwrap()");
    } else {
        println("FAIL: unwrap()");
    }

    let default_val = none_val.unwrap_or(100);
    if default_val == 100 {
        println("OK: unwrap_or()");
    } else {
        println("FAIL: unwrap_or()");
    }
}

// ===== Option::map =====
fn test_option_map() {
    println("Testing Option::map()...");
    let ok: Option<i64> = Option::Some(21);
    let mapped = ok.map(|x: i64| x * 2);
    if mapped.is_some() {
        if mapped.unwrap() == 42 {
            println("OK: Option::map() some");
        } else {
            println("FAIL: Option::map() some");
        }
    } else {
        println("FAIL: Option::map() returned None");
    }

    let none: Option<i64> = Option::None;
    let none_mapped = none.map(|x: i64| x * 2);
    if none_mapped.is_none() {
        println("OK: Option::map() none");
    } else {
        println("FAIL: Option::map() none");
    }
}

// ===== Option::and_then =====
fn test_option_and_then() {
    println("Testing Option::and_then()...");
    let ok: Option<i64> = Option::Some(10);
    let chained = ok.and_then(|x: i64| Option::Some(x * 3));
    if chained.is_some() {
        if chained.unwrap() == 30 {
            println("OK: Option::and_then() some");
        } else {
            println("FAIL: Option::and_then() some");
        }
    } else {
        println("FAIL: Option::and_then() returned None");
    }
    
    let none: Option<i64> = Option::None;
    let none_chained = none.and_then(|x: i64| Option::Some(x * 3));
    if none_chained.is_none() {
        println("OK: Option::and_then() none");
    } else {
        println("FAIL: Option::and_then() none");
    }
}

// ===== Option::unwrap_or_else =====
fn test_option_unwrap_or_else() {
    println("Testing Option::unwrap_or_else()...");
    let ok: Option<i64> = Option::Some(42);
    let val1 = ok.unwrap_or_else(|| 100);
    if val1 == 42 {
        println("OK: Option::unwrap_or_else() some");
    } else {
        println("FAIL: Option::unwrap_or_else() some");
    }

    let none: Option<i64> = Option::None;
    let val2 = none.unwrap_or_else(|| 100);
    if val2 == 100 {
        println("OK: Option::unwrap_or_else() none");
    } else {
        println("FAIL: Option::unwrap_or_else() none");
    }
}

fn main() {
    test_primitive();
    test_option_map();
    test_option_and_then();
    test_option_unwrap_or_else();
    println("\nAll Option tests finished.");
}
