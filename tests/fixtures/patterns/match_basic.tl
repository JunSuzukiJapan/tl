// STDOUT: 4
// STDOUT: 9
// STDOUT: 5

enum MaybeInt {
    Some { value: i64 },
    None
}

fn main() {
    let a = MaybeInt::Some { value: 4 };
    let v = match a {
        MaybeInt::Some { value } => value,
        MaybeInt::None => 0
    };
    print(v);

    let b = MaybeInt::None;
    let v2 = match b {
        MaybeInt::Some { value: v } => v,
        _ => 9
    };
    print(v2);

    let v3 = match a {
        MaybeInt::Some { value } => {
            let t = value + 1;
            t
        },
        MaybeInt::None => 0
    };
    print(v3);
}
