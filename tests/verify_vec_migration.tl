
struct Point {
    x: i64,
    y: i64,
}

fn main() {
    // Test Vec<i64>
    let v_i64: Vec<i64> = Vec::new();
    v_i64.push(10);
    v_i64.push(20);
    if v_i64.len() != 2 {
        println("Fail: Vec<i64> len mismatch");
    }
    if v_i64.get(0) != 10 {
        println("Fail: Vec<i64> get(0) mismatch");
    }
    if v_i64.pop() != 20 {
        println("Fail: Vec<i64> pop mismatch");
    }

    // Test Vec<U8> (special optimization target)
    /*
    let v_u8: Vec<U8> = Vec::new();
    v_u8.push(100);
    v_u8.push(200);
    if v_u8.len() != 2 {
        println("Fail: Vec<U8> len mismatch");
    }
    if v_u8.get(0) != 100 {
        println("Fail: Vec<U8> get(0) mismatch");
    }
    */
    
    // Test Vec<Point> (pointer type)
    let v_ptr: Vec<Point> = Vec::new();
    let p1 = Point { x: 1, y: 2 };
    v_ptr.push(p1);
    if v_ptr.len() != 1 {
        println("Fail: Vec<Point> len mismatch");
    }
    let p_out = v_ptr.get(0);
    if p_out.x != 1 {
        println("Fail: Vec<Point> content mismatch");
    }

    println("Vec verification passed");
}
