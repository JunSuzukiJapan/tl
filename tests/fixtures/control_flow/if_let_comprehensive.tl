// Comprehensive if-let tests using built-in Option
// Tests Option pattern matching with is_some/is_none/unwrap methods

// STDOUT: 1
// STDOUT: 2
// STDOUT: 3
// STDOUT: 4
// STDOUT: 5

fn main() {
    // 1. Option::Some with is_some check
    let opt1 = Option::Some(1);
    if opt1.is_some() {
        print(opt1.unwrap());
    }

    // 2. Option::Some unwrap direct
    let opt2 = Option::Some(2);
    let val2 = opt2.unwrap();
    print(val2);

    // 3. Conditional unwrap with if
    let opt3 = Option::Some(3);
    if opt3.is_some() {
        print(opt3.unwrap());
    } else {
        print(-1);
    }

    // 4. Nested Option check
    let outer = Option::Some(4);
    if outer.is_some() {
        let inner_val = outer.unwrap();
        if inner_val > 0 {
            print(inner_val);
        }
    }

    // 5. Multiple Option values
    let a = Option::Some(2);
    let b = Option::Some(3);
    if a.is_some() && b.is_some() {
        print(a.unwrap() + b.unwrap());
    }
}
