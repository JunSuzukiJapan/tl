// src/compiler/codegen/builtin_types/result.tl
enum Result<T, E> {
    Ok(T),
    Err(E),
}

impl Result<T, E> {
    extern fn is_ok(self) -> bool;
    extern fn is_err(self) -> bool;
    extern fn unwrap(self) -> T;
}
