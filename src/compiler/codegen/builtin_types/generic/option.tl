// src/compiler/codegen/builtin_types/option.tl

enum Option<T> {
    Some(T),
    None,
}

impl<T> Option<T> {
    fn is_some(self) -> bool {
        match self {
            Option::Some(x) => true,
            Option::None => false,
        }
    }

    fn is_none(self) -> bool {
        match self {
            Option::Some(x) => false,
            Option::None => true,
        }
    }

    fn unwrap(self) -> T {
        match self {
            Option::Some(x) => x,
            Option::None => {
                // panic! returns Never type, so type checker is satisfied
                panic("called unwrap() on Option::None");
            },
        }
    }

    fn unwrap_or(self, default: T) -> T {
        match self {
            Option::Some(x) => x,
            Option::None => default,
        }
    }

    fn or(self, other: Option<T>) -> Option<T> {
        match self {
            Option::Some(x) => Option::Some(x),
            Option::None => other,
        }
    }

    // NOTE: map はメソッドレベルの型パラメータ U を必要とするため、
    // codegen の compile_option_map_method (expr.rs) でインラインIR生成している。
    // 使用例: opt.map(|x: i64| x * 2)

    // NOTE: and_then, unwrap_or_else もクロージャ引数を取るため、
    // expr.rs でのインライン実装が必要。
}
