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
