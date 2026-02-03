struct S { x: i64 }
impl S {
    fn test(mut self) {
        self.x = 1;
    }
}
fn main() {}
