// Tests for I64 and F64 numeric type constants and methods

fn main() {
    test_i64_constants();
    test_f64_constants();
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

// ===== F64 constants =====
fn test_f64_constants() {
    println("Testing F64 constants...");

    let max = F64::max_value();
    let zero_f64: f64 = 0.0 as f64;
    if max <= zero_f64 {
        println("FAIL: F64::max_value() should be positive");
        return;
    }

    let min = F64::min_value();
    if min >= zero_f64 {
        println("FAIL: F64::min_value() should be negative");
        return;
    }

    let nan = F64::nan();
    // NaN != NaN (OEQ should be false for NaN)
    if nan == nan {
        println("FAIL: F64::nan() should not equal itself");
        return;
    }

    let inf = F64::infinity();
    if inf <= max {
        println("FAIL: F64::infinity() should be > max_value");
        return;
    }

    println("OK: F64 constants");
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

    let nan = F64::nan();
    if nan.is_inf() {
        println("FAIL: NaN.is_inf() should be false");
        return;
    }

    println("OK: f64.is_inf()");
}
