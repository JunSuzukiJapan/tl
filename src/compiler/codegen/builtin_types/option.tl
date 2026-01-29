// src/compiler/codegen/builtin_types/option.tl
enum Option<T> {
    Some(T),
    None,
}

impl<T> Option<T> {
    extern fn is_some(self) -> bool;
    extern fn is_none(self) -> bool;
    extern fn unwrap(self) -> T;
    extern fn unwrap_or(self, default: T) -> T;
}
