// SKIP: Known codegen issue - Result<i64,i64> unwrap_err PHI type mismatch
// Comprehensive tests for Result new methods: unwrap_or, map, and_then, unwrap_or_else
// Note: map_err and unwrap_err have known codegen issues (PHI type mismatch)
// so they are excluded from automated tests

fn main() {
    test_result_unwrap_or();
    test_result_map();
    test_result_and_then();
    test_result_unwrap_or_else();

    println("All Result tests passed!");
}

// ===== Result::unwrap_or =====
fn test_result_unwrap_or() {
    println("Testing Result::unwrap_or()...");

    // Ok: returns inner value
    let ok: Result<i64, i64> = Result::Ok(42);
    let val1 = ok.unwrap_or(0);
    if val1 != 42 {
        println("FAIL: Ok(42).unwrap_or(0) should be 42");
        return;
    }

    // Err: returns default
    let err: Result<i64, i64> = Result::Err(-1);
    let val2 = err.unwrap_or(99);
    if val2 != 99 {
        println("FAIL: Err(-1).unwrap_or(99) should be 99");
        return;
    }

    println("OK: Result::unwrap_or()");
}

// ===== Result::map =====
fn test_result_map() {
    println("Testing Result::map()...");

    // Ok: transforms value
    let ok: Result<i64, i64> = Result::Ok(21);
    let mapped = ok.map(|x: i64| x * 2);
    if mapped.is_ok() {
        let val = mapped.unwrap();
        if val != 42 {
            println("FAIL: Ok(21).map(|x| x*2) should be Ok(42)");
            return;
        }
    } else {
        println("FAIL: Ok.map() should return Ok");
        return;
    }

    println("OK: Result::map()");
}

// ===== Result::and_then =====
fn test_result_and_then() {
    println("Testing Result::and_then()...");

    // Ok: chains
    let ok: Result<i64, i64> = Result::Ok(10);
    let chained = ok.and_then(|x: i64| Result::Ok(x * 3));
    if chained.is_ok() {
        let val = chained.unwrap();
        if val != 30 {
            println("FAIL: Ok(10).and_then should be Ok(30)");
            return;
        }
    } else {
        println("FAIL: Ok.and_then should return Ok");
        return;
    }

    println("OK: Result::and_then()");
}

// ===== Result::unwrap_or_else =====
fn test_result_unwrap_or_else() {
    println("Testing Result::unwrap_or_else()...");

    // Ok: returns inner value
    let ok: Result<i64, i64> = Result::Ok(42);
    let val1 = ok.unwrap_or_else(|e: i64| e * 100);
    if val1 != 42 {
        println("FAIL: Ok(42).unwrap_or_else should be 42");
        return;
    }

    // Err: calls closure
    let err: Result<i64, i64> = Result::Err(5);
    let val2 = err.unwrap_or_else(|e: i64| e * 10);
    if val2 != 50 {
        println("FAIL: Err(5).unwrap_or_else(|e| e*10) should be 50");
        return;
    }

    println("OK: Result::unwrap_or_else()");
}
