// EXIT: 1
// STDERR: Type mismatch

enum MaybeInt {
    Some { value: i64 },
    None
}

fn main() {
    let a = MaybeInt::Some { value: 5 };
    let v = if let MaybeInt::Some { value } = a {
        value
    };
    print(v);
}
