// EXIT: 1
// STDERR: unreachable match arm

enum MaybeInt {
    Some { value: i64 },
    None
}

fn main() {
    let a = MaybeInt::None;
    let v = match a {
        _ => 0,
        MaybeInt::None => 1
    };
    print(v);
}
