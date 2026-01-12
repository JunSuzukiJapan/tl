// EXIT: 1
// STDERR: Field val

enum MaybeInt {
    Some { value: i64 },
    None
}

fn main() {
    let a = MaybeInt::Some { value: 1 };
    if let MaybeInt::Some { val } = a {
        print(val);
    }
}
