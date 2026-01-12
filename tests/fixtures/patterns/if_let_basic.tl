// STDOUT: 7
// STDOUT: 3
// STDOUT: none

enum MaybeInt {
    Some { value: i64 },
    None
}

fn main() {
    let a = MaybeInt::Some { value: 7 };
    if let MaybeInt::Some { value } = a {
        print(value);
    } else {
        print(0);
    }

    let b = MaybeInt::None;
    if let MaybeInt::Some { value } = b {
        print(value);
    } else {
        print(3);
    }

    if let MaybeInt::None = b {
        print("none");
    }
}
