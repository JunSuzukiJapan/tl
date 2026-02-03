struct S { x: i64 }
impl S {
    fn test(self) {
        if self.x == 0 {}
    }
}
fn main() {}
