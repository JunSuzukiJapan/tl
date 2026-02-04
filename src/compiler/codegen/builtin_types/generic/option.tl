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
}
