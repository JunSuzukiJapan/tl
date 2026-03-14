// Comprehensive tests for Vec new methods: extend, slice, clone, index_of, reverse, join, reduce
// Tests all newly implemented Vec methods

fn main() {
    test_index_of();
    test_reverse();
    test_extend();
    test_slice();
    test_clone();
    test_reduce();
    test_join();

    println("All Vec new method tests passed!");
}

// ===== index_of =====
fn test_index_of() {
    println("Testing index_of()...");
    let mut v: Vec<i64> = Vec::new();
    v.push(10);
    v.push(20);
    v.push(30);
    v.push(20);

    // Found at beginning
    let idx = v.index_of(10);
    if idx != 0 {
        println("FAIL: index_of(10) should be 0");
        return;
    }

    // Found in middle (first occurrence)
    let idx2 = v.index_of(20);
    if idx2 != 1 {
        println("FAIL: index_of(20) should be 1 (first occurrence)");
        return;
    }

    // Found at end
    let idx3 = v.index_of(30);
    if idx3 != 2 {
        println("FAIL: index_of(30) should be 2");
        return;
    }

    // Not found
    let idx4 = v.index_of(99);
    if idx4 != -1 {
        println("FAIL: index_of(99) should be -1");
        return;
    }

    // Empty vec
    let empty: Vec<i64> = Vec::new();
    let idx5 = empty.index_of(1);
    if idx5 != -1 {
        println("FAIL: index_of on empty vec should be -1");
        return;
    }

    println("OK: index_of()");
}

// ===== reverse =====
fn test_reverse() {
    println("Testing reverse()...");
    let mut v: Vec<i64> = Vec::new();
    v.push(1);
    v.push(2);
    v.push(3);
    v.push(4);

    v.reverse();

    // Check reversed order
    if v.get(0).unwrap() != 4 {
        println("FAIL: reverse v[0] should be 4");
        return;
    }
    if v.get(1).unwrap() != 3 {
        println("FAIL: reverse v[1] should be 3");
        return;
    }
    if v.get(2).unwrap() != 2 {
        println("FAIL: reverse v[2] should be 2");
        return;
    }
    if v.get(3).unwrap() != 1 {
        println("FAIL: reverse v[3] should be 1");
        return;
    }

    // Odd-length vec
    let mut v2: Vec<i64> = Vec::new();
    v2.push(10);
    v2.push(20);
    v2.push(30);
    v2.reverse();

    if v2.get(0).unwrap() != 30 {
        println("FAIL: reverse odd-length v[0] should be 30");
        return;
    }
    if v2.get(2).unwrap() != 10 {
        println("FAIL: reverse odd-length v[2] should be 10");
        return;
    }

    // Single element
    let mut v3: Vec<i64> = Vec::new();
    v3.push(42);
    v3.reverse();
    if v3.get(0).unwrap() != 42 {
        println("FAIL: reverse single element should be unchanged");
        return;
    }

    println("OK: reverse()");
}

// ===== extend =====
fn test_extend() {
    println("Testing extend()...");
    let mut v1: Vec<i64> = Vec::new();
    v1.push(1);
    v1.push(2);

    let mut v2: Vec<i64> = Vec::new();
    v2.push(3);
    v2.push(4);
    v2.push(5);

    v1.extend(v2);

    // Check length
    if v1.len() != 5 {
        println("FAIL: extended vec len should be 5");
        return;
    }

    // Check values
    if v1.get(0).unwrap() != 1 {
        println("FAIL: extend v[0] should be 1");
        return;
    }
    if v1.get(2).unwrap() != 3 {
        println("FAIL: extend v[2] should be 3");
        return;
    }
    if v1.get(4).unwrap() != 5 {
        println("FAIL: extend v[4] should be 5");
        return;
    }

    // Extend with empty vec
    let empty: Vec<i64> = Vec::new();
    v1.extend(empty);
    if v1.len() != 5 {
        println("FAIL: extend with empty should not change len");
        return;
    }

    println("OK: extend()");
}

// ===== slice =====
fn test_slice() {
    println("Testing slice()...");
    let mut v: Vec<i64> = Vec::new();
    v.push(10);
    v.push(20);
    v.push(30);
    v.push(40);
    v.push(50);

    // Normal slice
    let s1 = v.slice(1, 4);
    if s1.len() != 3 {
        println("FAIL: slice(1,4) len should be 3");
        return;
    }
    if s1.get(0).unwrap() != 20 {
        println("FAIL: slice(1,4)[0] should be 20");
        return;
    }
    if s1.get(2).unwrap() != 40 {
        println("FAIL: slice(1,4)[2] should be 40");
        return;
    }

    // Slice from beginning
    let s2 = v.slice(0, 2);
    if s2.len() != 2 {
        println("FAIL: slice(0,2) len should be 2");
        return;
    }
    if s2.get(0).unwrap() != 10 {
        println("FAIL: slice(0,2)[0] should be 10");
        return;
    }

    // Slice to end
    let s3 = v.slice(3, 5);
    if s3.len() != 2 {
        println("FAIL: slice(3,5) len should be 2");
        return;
    }
    if s3.get(0).unwrap() != 40 {
        println("FAIL: slice(3,5)[0] should be 40");
        return;
    }

    // Empty slice (start >= end)
    let s4 = v.slice(3, 3);
    if s4.len() != 0 {
        println("FAIL: slice(3,3) should be empty");
        return;
    }

    // Out of bounds clamping
    let s5 = v.slice(-1, 100);
    if s5.len() != 5 {
        println("FAIL: slice(-1,100) should be clamped to full vec");
        return;
    }

    println("OK: slice()");
}

// ===== clone =====
fn test_clone() {
    println("Testing clone()...");
    let mut v: Vec<i64> = Vec::new();
    v.push(100);
    v.push(200);
    v.push(300);

    let c = v.clone();

    // Check length
    if c.len() != 3 {
        println("FAIL: clone len should be 3");
        return;
    }

    // Check values
    if c.get(0).unwrap() != 100 {
        println("FAIL: clone[0] should be 100");
        return;
    }
    if c.get(1).unwrap() != 200 {
        println("FAIL: clone[1] should be 200");
        return;
    }
    if c.get(2).unwrap() != 300 {
        println("FAIL: clone[2] should be 300");
        return;
    }

    // Modify original, clone should be independent
    v.push(400);
    if c.len() != 3 {
        println("FAIL: clone should be independent of original");
        return;
    }

    // Clone empty vec
    let empty: Vec<i64> = Vec::new();
    let ec = empty.clone();
    if ec.len() != 0 {
        println("FAIL: clone of empty vec should be empty");
        return;
    }

    println("OK: clone()");
}

// ===== reduce =====
fn test_reduce() {
    println("Testing reduce()...");
    let mut v: Vec<i64> = Vec::new();
    v.push(1);
    v.push(2);
    v.push(3);
    v.push(4);
    v.push(5);

    // Sum: 1+2+3+4+5 = 15
    let sum = v.reduce(|acc: i64, x: i64| acc + x);
    if sum != 15 {
        println("FAIL: reduce sum should be 15");
        return;
    }

    // Product: 1*2*3*4*5 = 120
    let product = v.reduce(|acc: i64, x: i64| acc * x);
    if product != 120 {
        println("FAIL: reduce product should be 120");
        return;
    }

    // Max: find maximum value
    let mut v2: Vec<i64> = Vec::new();
    v2.push(3);
    v2.push(7);
    v2.push(1);
    v2.push(9);
    v2.push(4);
    let max_val = v2.reduce(|acc: i64, x: i64| if x > acc { x } else { acc });
    if max_val != 9 {
        println("FAIL: reduce max should be 9");
        return;
    }

    // Single element
    let mut v3: Vec<i64> = Vec::new();
    v3.push(42);
    let single = v3.reduce(|acc: i64, x: i64| acc + x);
    if single != 42 {
        println("FAIL: reduce single element should be 42");
        return;
    }

    println("OK: reduce()");
}

// ===== join =====
fn test_join() {
    println("Testing join()...");

    // Join integers (auto to_string conversion)
    let mut nums: Vec<i64> = Vec::new();
    nums.push(1);
    nums.push(2);
    nums.push(3);
    let joined = nums.join("-");
    if joined == "1-2-3" {
        println("OK: join integers with dash");
    } else {
        println("FAIL: join integers with dash");
        return;
    }

    // Join with comma
    let joined2 = nums.join(", ");
    if joined2 == "1, 2, 3" {
        println("OK: join integers with comma");
    } else {
        println("FAIL: join integers with comma");
        return;
    }

    // Join with empty separator
    let joined3 = nums.join("");
    if joined3 == "123" {
        println("OK: join with empty sep");
    } else {
        println("FAIL: join with empty sep");
        return;
    }

    // Join single element
    let mut single: Vec<i64> = Vec::new();
    single.push(42);
    let joined4 = single.join(",");
    if joined4 == "42" {
        println("OK: join single element");
    } else {
        println("FAIL: join single element");
        return;
    }

    // Join empty vec
    let empty: Vec<i64> = Vec::new();
    let joined5 = empty.join(",");
    if joined5 == "" {
        println("OK: join empty vec");
    } else {
        println("FAIL: join empty vec");
    }

    // Join strings
    let mut words: Vec<String> = Vec::new();
    words.push("hello");
    words.push("world");
    words.push("tl");
    let joined6 = words.join(", ");
    if joined6 == "hello, world, tl" {
        println("OK: join strings");
    } else {
        println("FAIL: join strings");
        return;
    }

    // Join strings with empty sep
    let joined7 = words.join("");
    if joined7 == "helloworldtl" {
        println("OK: join strings with empty sep");
    } else {
        println("FAIL: join strings with empty sep");
        return;
    }

    println("OK: join()");
}
