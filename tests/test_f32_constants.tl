// Tests for I32 and F32 numeric type constants and methods
// Separated from I64/F64 tests to avoid type mismatch in semantic analysis

fn main() {
    test_i32_constants();
    test_f32_constants();
    test_f32_is_nan();
    test_f32_is_inf();

    println("All I32/F32 tests passed!");
}

// ===== I32 constants =====
fn test_i32_constants() {
    println("Testing I32 constants...");

    let max = I32::max_value();
    if max <= 0 as i32 {
        println("FAIL: I32::max_value() should be positive");
        return;
    }

    let min = I32::min_value();
    if min >= 0 as i32 {
        println("FAIL: I32::min_value() should be negative");
        return;
    }

    println("OK: I32 constants");
}

// ===== F32 constants =====
fn test_f32_constants() {
    println("Testing F32 constants...");

    let max = F32::max_value();
    if max <= 0.0 as f32 {
        println("FAIL: F32::max_value() should be positive");
        return;
    }

    let min = F32::min_value();
    if min >= 0.0 as f32 {
        println("FAIL: F32::min_value() should be negative");
        return;
    }

    let nan = F32::nan();
    if nan == nan {
        println("FAIL: F32::nan() should not equal itself");
        return;
    }

    let inf = F32::infinity();
    if inf <= max {
        println("FAIL: F32::infinity() should be > max_value");
        return;
    }

    println("OK: F32 constants");
}

// ===== F32::is_nan =====
fn test_f32_is_nan() {
    println("Testing f32.is_nan()...");

    let nan = F32::nan();
    if !nan.is_nan() {
        println("FAIL: F32 NaN.is_nan() should be true");
        return;
    }

    let normal: f32 = 3.14 as f32;
    if normal.is_nan() {
        println("FAIL: F32 3.14.is_nan() should be false");
        return;
    }

    println("OK: f32.is_nan()");
}

// ===== F32::is_inf =====
fn test_f32_is_inf() {
    println("Testing f32.is_inf()...");

    let inf = F32::infinity();
    if !inf.is_inf() {
        println("FAIL: F32 infinity.is_inf() should be true");
        return;
    }

    let normal: f32 = 3.14 as f32;
    if normal.is_inf() {
        println("FAIL: F32 3.14.is_inf() should be false");
        return;
    }

    println("OK: f32.is_inf()");
}
