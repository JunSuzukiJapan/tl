// EXIT: 1
// STDERR: Duplicate match arm

enum MaybeInt {
    Some { value: i64 },
    None
}

fn main() {
    let a = MaybeInt::None;
    let v = match a {
        MaybeInt::None => 0,
        MaybeInt::None => 1,
        _ => 2
    };
    print(v);
}
