
fn test_option_some() {
    let x: Option<i64> = Option::Some(42);
    let val = x.unwrap();
    if val != 42 {
        print("FAIL: unwrap\n");
    } else {
        print("OK: unwrap\n");
    }
}

fn test_option_none() {
    let x: Option<i64> = Option::None;
    if !x.is_none() {
        print("FAIL: is_none\n");
    } else {
        print("OK: is_none\n");
    }
}

fn main() {
    print("Testing Option<i64>...\n");
    test_option_some();
    test_option_none();
}
