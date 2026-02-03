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
            Result::Err(e) => self.unwrap(), // Panic or loop
        }
    }

    fn unwrap_err(self) -> E {
        match self {
            Result::Ok(x) => self.unwrap_err(), // Panic or loop
            Result::Err(e) => e,
        }
    }
}
