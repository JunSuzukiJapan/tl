fn test_primitive() {
    print("Testing Option<I64>...");
    let some_val = Option::Some(42);
    // FIXME: Option::None requires explicit generics or type inference improvements
    // let none_val: Option<I64> = Option::None;

    if some_val.is_some() {
        print("some_val is some: OK");
    } else {
        print("some_val is some: FAILED");
    }
    
    // if none_val.is_none() {
    //    print("none_val is none: OK");
    // } else {
    //    print("none_val is none: FAILED");
    // }

    let val = some_val.unwrap();
    if val == 42 {
        print("unwrap: OK");
    } else {
        print("unwrap: FAILED");
    }
    
    // let default_val = none_val.unwrap_or(100);
    // if default_val == 100 {
    //    print("unwrap_or: OK");
    // } else {
    //    print("unwrap_or: FAILED");
    // }
}

fn test_string() {
    /*
    print("\nTesting Option<String>...");
    let s = "Hello";
    let some_str = Option::Some(s);
    let none_str: Option<String> = Option::None;

    if some_str.is_some() {
        print("some_str is some: OK");
    } else {
        print("some_str is some: FAIL");
    }

    let unwrapped = some_str.unwrap();
    if unwrapped == "Hello" {
        print("some_str.unwrap(): OK");
    } else {
        print("some_str.unwrap(): FAIL");
    }

    let default_str = none_str.unwrap_or("World");
    if default_str == "World" {
        print("none_str.unwrap_or(): OK");
    } else {
        print("none_str.unwrap_or(): FAIL");
    }
    */
}

fn main() {
    test_primitive();
    test_string();
    print("\nAll Option tests finished.");
}
