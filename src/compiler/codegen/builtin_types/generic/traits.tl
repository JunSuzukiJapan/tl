// Standard trait definitions for TL language

trait Index<Idx, Output> {
    fn index(self, idx: Idx) -> Output;
}

trait IndexMut<Idx, Value> {
    fn set(self, idx: Idx, value: Value);
}

trait Iterable<T> {
    fn len(self) -> i64;
    fn get(self, index: i64) -> T;
}

// ── Common Standard Traits ──

trait Display {
    fn to_string(self) -> String;
}

trait From<T> {
    fn from(value: T) -> Self;
}

// ── Operator Overloading Traits ──

trait Add<RHS> {
    fn add(self, rhs: RHS) -> Self;
}

trait Sub<RHS> {
    fn sub(self, rhs: RHS) -> Self;
}

trait Mul<RHS> {
    fn mul(self, rhs: RHS) -> Self;
}

trait Div<RHS> {
    fn div(self, rhs: RHS) -> Self;
}

trait Rem<RHS> {
    fn rem(self, rhs: RHS) -> Self;
}

trait Eq<RHS> {
    fn eq(self, rhs: RHS) -> bool;
}

trait PartialOrd<RHS> {
    fn partial_cmp(self, rhs: RHS) -> bool;
}
