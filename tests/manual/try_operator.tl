fn safe_div(a: i64, b: i64) -> Result<i64, i64> {
    if b == 0 {
        return Result::Err(999); // Error code 999
    }
    return Result::Ok(a / b);
} 

fn test_try(a: i64, b: i64) -> Result<i64, i64> {
    let x = safe_div(a, b)?;
    return Result::Ok(x + 100);
}

fn main() {
    let r1 = test_try(10, 2);
    if r1.is_ok() {
        println("10/2 + 100 = {}", r1.unwrap());
    } else {
        println("Error: {}", r1.unwrap_err());
    }

    let r2 = test_try(10, 0);
    if r2.is_err() {
        println("10/0 error: {}", r2.unwrap_err());
    } else {
        println("Unexpected success: {}", r2.unwrap());
    }
}
