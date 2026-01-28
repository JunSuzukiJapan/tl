fn test_primitive() {
    print("Testing Option<I64>...");
    let some_val = Option::some(42);
    let none_val: Option<I64> = Option::none();

    if some_val.is_some() {
        print("some_val is some: OK");
    } else {
        print("some_val is some: FAIL");
    }

    if none_val.is_none() {
        print("none_val is none: OK");
    } else {
        print("none_val is none: FAIL");
    }

    let val = some_val.unwrap();
    if val == 42 {
        print("some_val.unwrap(): OK");
    } else {
        print("some_val.unwrap(): FAIL");
    }

    let default = none_val.unwrap_or(100);
    if default == 100 {
        print("none_val.unwrap_or(): OK");
    } else {
        print("none_val.unwrap_or(): FAIL");
    }
}

fn test_string() {
    print("\nTesting Option<String>...");
    let s = "Hello";
    let some_str = Option::some(s);
    let none_str: Option<String> = Option::none();

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
}

fn main() {
    test_primitive();
    test_string();
    print("\nAll Option tests finished.");
}
