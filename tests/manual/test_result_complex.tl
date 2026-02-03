
struct Point {
    x: i32,
    y: i32,
}

impl Point {
    fn new(x: i32, y: i32) -> Point {
        Point { x: x, y: y }
    }
}

fn main() {
    println("Testing Result<String, String>...");
    
    // 1. Result<String, String> - Ok
    let ok_str: Result<String, String> = Result::Ok("success");
    if ok_str.is_ok() {
        println("OK: is_ok for String");
    } else {
        println("ERR: is_ok failed for String");
    }

    let val = ok_str.unwrap();
    // Comparison for strings might need a method call if == is not implemented for String structs,
    // but typically == works for primitives. String is a struct in TL often.
    // If == fails, I might need to unbox.
    // However, binary op compile_bin_op handles (String, String) if enabled.
    if val == "success" {
        println("OK: unwrap String");
    } else {
        println("ERR: unwrap String mismatch");
    }

    // 2. Result<String, String> - Err
    let err_str: Result<String, String> = Result::Err("failure");
    if err_str.is_err() {
        println("OK: is_err for String");
    } else {
        println("ERR: is_err failed for String");
    }

    // 3. Result<Point, String>
    /* */
    println("Testing Result<Point, String>...");
    let p = Point::new(10 as i32, 20 as i32);
    let ok_point: Result<Point, String> = Result::Ok(p);
    
    let p_res = ok_point.unwrap();
    /* */
    if p_res.x == 10 as i32 {
        if p_res.y == 20 as i32 {
            println("OK: unwrap Point");
        } else {
            println("ERR: Point.y mismatch");
        }
    } else {
        println("ERR: Point.x mismatch");
    }

    // 4. Deep Clone Check (Implicit via variable usage)
    // If memory management is wrong, double-free might occur at end of scope.
    println("Done.");
}
