fn main() {
    print("Testing Vec<i32>...\n");

    let mut v: Vec<i32> = Vec::with_capacity(10);
    if v.capacity() != 10 {
         print("FAIL: capacity mismatch\n");
    } else {
         print("OK: capacity\n");
    }

    v.push(1 as i32);
    v.push(2 as i32);
    v.push(3 as i32);

    if v.len() != 3 {
         print("FAIL: len mismatch\n");
    } else {
         print("OK: len\n");
    }

    if v.get(0).unwrap() != 1 as i32 {
         print("FAIL: v[0]\n");
    }
    if v.get(1).unwrap() != 2 as i32 {
         print("FAIL: v[1]\n");
    }
    if v.get(2).unwrap() != 3 as i32 {
         print("FAIL: v[2]\n");
    }
    print("OK: get values\n");
    
    // Pop
    let p = v.pop().unwrap();
    if p != 3 as i32 {
         print("FAIL: pop value\n");
    }
    if v.len() != 2 {
         print("FAIL: pop len\n");
    }
    print("OK: pop\n");

    // Lazy Init
    let mut v2: Vec<i32> = Vec::new();
    if v2.capacity() != 0 {
         print("FAIL: lazy init capacity\n");
    } else {
         print("OK: lazy init\n");
    }
    v2.push(42 as i32);
    if v2.capacity() < 1 {
         print("FAIL: lazy alloc\n");
    }
    if v2.get(0).unwrap() != 42 as i32 {
         print("FAIL: v2[0]\n");
    }
    print("OK: lazy push\n");
}
