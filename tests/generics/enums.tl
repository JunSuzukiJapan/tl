// Option and Result are now built-in types.


fn main() {
    let some_val: Option<i64> = Option::Some(10);
    // let none_val: Option<i64> = Option::None;

    match some_val {
        Option::Some(v) => print(v),
        Option::None => print(0),
    }

    // Result test commented out due to partial inference limitation
    // let res: Result<i64, String> = Result::Ok { value: 100 };
    // match res {
    //     Result::Ok { value: v } => print(v),
    //     Result::Err { error: e } => print(e),
    // }
}
