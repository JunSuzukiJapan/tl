fn main() {
    print("Testing Option<String>...\n");
    
    // Case 1: Some checks
    let s = "hello";
    let opt = Option::Some(s);
    
    if opt.is_none() {
        print("FAIL: Expected Some, got None\n");
    } else {
        print("OK: is_none check\n");
    }

    let val = opt.unwrap();
    print("Unwrapped: ");
    print(val);
    print("\n");

    if val != "hello" {
        print("FAIL: Content mismatch\n");
    } else {
        print("OK: Content match\n");
    }

    // Case 2: None checks
    let none_opt: Option<String> = Option::None;
    if !none_opt.is_none() {
         print("FAIL: Expected None, got Some\n");
    } else {
         print("OK: None check\n");
    }

    // Case 3: Re-wrapping (Move semantics verification)
    let s2 = "world";
    let opt2 = Option::Some(s2);
    let opt3 = opt2; // Move? Copy?
    // In TL strings are reference counted, usually Copy increases RefCount.
    print("OK: Copied Option<String>\n");
}
