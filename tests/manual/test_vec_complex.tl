struct Point {
    x: i64, 
    y: i64,
}

fn main() {
    print("Testing Vec<Point>...\n");
    let mut v: Vec<Point> = Vec::new();
    
    let p1 = Point { x: 10, y: 20 };
    let p2 = Point { x: 30, y: 40 };
    
    v.push(p1);
    v.push(p2);
    
    if v.len() != 2 {
        print("FAIL: len mismatch\n");
    } else {
        print("OK: len\n");
    }
    
    let first = v.get(0).unwrap();
    if first.x != 10 {
         print("FAIL: first.x mismatch\n");
    } else {
         print("OK: first.x\n");
    }

    let second = v.get(1).unwrap();
    if second.y != 40 {
        print("FAIL: second.y mismatch\n");
    } else {
        print("OK: second.y\n");
    }
    
    print("Testing Vec<String>...\n");
    // TODO: Verify strings once basic struct works
}
