fn main() {
    print("Testing Result<I64, I64>...");
    
    // Test Ok
    let ok: Result<I64, I64> = Result::Ok(100);
    if ok.is_ok() {
        print("ok.is_ok(): OK");
    } else {
        print("ok.is_ok(): FAIL");
    }
    
    if ok.is_err() {
        print("ok.is_err(): FAIL");
    } else {
        print("ok.is_err(): OK");
    }
    
    if ok.unwrap() == 100 {
        print("ok.unwrap(): OK");
    } else {
        print("ok.unwrap(): FAIL");
    }

    // Test Err
    let err: Result<I64, I64> = Result::Err(404);
    if err.is_ok() {
        print("err.is_ok(): FAIL");
    } else {
        print("err.is_ok(): OK");
    }
    
    if err.is_err() {
        print("err.is_err(): OK");
    } else {
        print("err.is_err(): FAIL");
    }

    print("\nTesting Result<String, I64>...");
    let ok_str: Result<String, I64> = Result::Ok("Success");
    if ok_str.is_ok() {
         print("ok_str.is_ok(): OK");
    } else {
         print("ok_str.is_ok(): FAIL");
    }
    
    if ok_str.unwrap() == "Success" {
         print("ok_str.unwrap(): OK");
    } else {
         print("ok_str.unwrap(): FAIL");
    }
}
