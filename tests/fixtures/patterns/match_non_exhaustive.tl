// EXIT: 1
// STDERR: Non-exhaustive match

enum MaybeInt {
    Some { value: i64 },
    None
}

fn main() {
    let a = MaybeInt::Some { value: 4 };
    let v = match a {
        MaybeInt::Some { value } => value
    };
    print(v);
}
