// SKIP: Known compiler issue - generic type inference
fn main() {
    print("Testing Result<i32, i32>...\n");

    // Case 1: Ok
    let ok: Result<i32, i32> = Result::Ok(42 as i32);
    if !ok.is_ok() {
        print("FAIL: Expected Ok\n");
    } else {
        print("OK: is_ok check\n");
    }

    let val = ok.unwrap();
    if val != 42 as i32 {
        print("FAIL: Value mismatch\n");
    } else {
        print("OK: Value match\n");
    }

    // Case 2: Err
    let err: Result<i32, i32> = Result::Err(0 as i32);
    if !err.is_err() {
        print("FAIL: Expected Err\n");
    } else {
        print("OK: is_err check\n");
    }
    
    // Copy check
    let ok2 = ok;
    print("OK: Copy Result\n");
}
