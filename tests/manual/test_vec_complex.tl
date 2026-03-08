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
    let mut vs: Vec<String> = Vec::new();
    vs.push("hello");
    vs.push("world");
    vs.push("tl");

    if vs.len() != 3 {
        print("FAIL: Vec<String> len mismatch\n");
    } else {
        print("OK: Vec<String> len\n");
    }

    let s0 = vs.get(0).unwrap();
    if s0 != "hello" {
        print("FAIL: Vec<String> get(0) mismatch\n");
    } else {
        print("OK: Vec<String> get(0)\n");
    }

    let s2 = vs.get(2).unwrap();
    if s2 != "tl" {
        print("FAIL: Vec<String> get(2) mismatch\n");
    } else {
        print("OK: Vec<String> get(2)\n");
    }
}
