
struct MyStruct { h: i64 }
impl MyStruct {
    fn new() -> MyStruct { MyStruct { h: 123 } }
    fn method(self) -> i64 { self.h }
}

fn take_struct(s: MyStruct) -> i64 {
    s.method()
}

fn main() {
    let s = MyStruct::new();
    let v = take_struct(s);
    println("Result: {}", v);
}
