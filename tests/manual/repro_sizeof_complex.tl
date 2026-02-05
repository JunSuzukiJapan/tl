// SKIP: Known compiler issue - generic type inference
extern fn malloc(size: i64) -> ptr<u8>;

struct Vec<T> { ptr: ptr<T> }

impl<T> Vec<T> {
    fn with_capacity(cap: i64) -> Vec<T> {
        let s = T::sizeof();
        Vec { ptr: 0 as ptr<T> }
    }
}
fn main() {}
