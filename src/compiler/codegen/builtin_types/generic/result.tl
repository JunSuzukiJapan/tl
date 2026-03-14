enum Result<T, E> {
    Ok(T),
    Err(E),
}

impl<T, E> Result<T, E> {
    fn is_ok(self) -> bool {
        match self {
            Result::Ok(x) => true,
            Result::Err(e) => false,
        }
    }

    fn is_err(self) -> bool {
        match self {
            Result::Ok(x) => false,
            Result::Err(e) => true,
        }
    }

    fn unwrap(self) -> T {
        match self {
            Result::Ok(x) => x,
            Result::Err(e) => {
                panic("called unwrap() on Result::Err");
            },
        }
    }

    fn unwrap_err(self) -> E {
        match self {
            Result::Ok(x) => {
                panic("called unwrap_err() on Result::Ok");
            },
            Result::Err(e) => e,
        }
    }

    fn unwrap_or(self, default: T) -> T {
        match self {
            Result::Ok(x) => x,
            Result::Err(e) => default,
        }
    }

    // NOTE: map, map_err, and_then, unwrap_or_else はクロージャ引数と
    // メソッドレベルの型パラメータを必要とするため、
    // expr.rs でのインライン実装が必要。
}
