struct S { arr: ptr<i64> }
impl S {
    fn test(self) {
        let mut p = self.arr;
        p[0] = 1;
    }
}
fn main() {}
