// EXIT: 1
// STDERR: type mismatch

enum MaybeInt {
    Some { value: i64 },
    None
}

fn main() {
    let x = 1;
    if let MaybeInt::Some { value } = x {
        print(value);
    }
}
