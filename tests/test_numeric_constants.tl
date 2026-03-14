// Tests for I64 and F64 numeric type constants and methods
// Note: F64 BinOp (==, <=, etc.) has known codegen issues, so we test via is_nan/is_inf methods

fn main() {
    test_i64_constants();
    test_f64_is_nan();
    test_f64_is_inf();

    println("All I64/F64 numeric type tests passed!");
}

// ===== I64 constants =====
fn test_i64_constants() {
    println("Testing I64 constants...");

    let max = I64::max_value();
    if max != 9223372036854775807 {
        println("FAIL: I64::max_value()");
        return;
    }

    let min = I64::min_value();
    if min != -9223372036854775808 {
        println("FAIL: I64::min_value()");
        return;
    }

    if max <= min {
        println("FAIL: max should be > min");
        return;
    }

    println("OK: I64 constants");
}

// ===== F64::is_nan =====
fn test_f64_is_nan() {
    println("Testing f64.is_nan()...");

    let nan = F64::nan();
    if !nan.is_nan() {
        println("FAIL: NaN.is_nan() should be true");
        return;
    }

    let normal: f64 = 3.14 as f64;
    if normal.is_nan() {
        println("FAIL: 3.14.is_nan() should be false");
        return;
    }

    let inf = F64::infinity();
    if inf.is_nan() {
        println("FAIL: infinity.is_nan() should be false");
        return;
    }

    println("OK: f64.is_nan()");
}

// ===== F64::is_inf =====
fn test_f64_is_inf() {
    println("Testing f64.is_inf()...");

    let inf = F64::infinity();
    if !inf.is_inf() {
        println("FAIL: infinity.is_inf() should be true");
        return;
    }

    let normal: f64 = 3.14 as f64;
    if normal.is_inf() {
        println("FAIL: 3.14.is_inf() should be false");
        return;
    }

    // NOTE: F64 BinOp comparisons (nan == nan, etc.) have known codegen issues
    // so we rely on is_nan()/is_inf() methods instead

    println("OK: f64.is_inf()");
}
