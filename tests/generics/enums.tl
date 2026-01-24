enum Option<T> {
    Some { value: T },
    None,
}

enum Result<T, E> {
    Ok { value: T },
    Err { error: E },
}

fn main() {
    let some_val: Option<i64> = Option::Some { value: 10 };
    // let none_val: Option<i64> = Option::None;

    match some_val {
        Option::Some { value: v } => print(v),
        Option::None => print(0),
    }

    // Result test commented out due to partial inference limitation
    // let res: Result<i64, String> = Result::Ok { value: 100 };
    // match res {
    //     Result::Ok { value: v } => print(v),
    //     Result::Err { error: e } => print(e),
    // }
}
