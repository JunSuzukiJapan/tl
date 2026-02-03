struct S {}
impl S {
    fn test<T>() {
        let x = T::sizeof();
    }
}
fn main() {
    print("Run\n");
}
