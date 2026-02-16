fn safe_div(a: i64, b: i64) -> Result<i64, String> {
    if b == 0 {
        return Result::Err("Division by zero");
    }
    return Result::Ok(a / b);
} 

fn test_try(a: i64, b: i64) -> Result<i64, String> {
    let x = safe_div(a, b)?;
    return Result::Ok(x + 100);
}

fn main() {
    let r1 = test_try(10, 2);
    if r1.is_ok() {
        println("10/2 + 100 = {}", r1.unwrap());
    } else {
        println("Error");
    }

    let r2 = test_try(10, 0);
    println("10/0 returned successfully");
    // r2 drops here. If stack/memory corrupted, it might crash.
}
