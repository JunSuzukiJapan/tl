enum Option<T> {
    Some(T),
    None,
}

impl Option<T> {
    extern fn is_some(self) -> bool;
    extern fn is_none(self) -> bool;
    extern fn unwrap(self) -> T;
    extern fn unwrap_or(self, default: T) -> T;
}
