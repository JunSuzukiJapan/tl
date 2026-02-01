struct MyBox<T> {
    val: T
}

impl<T> MyBox<T> {
    fn new(v: T) -> MyBox<T> {
        MyBox { val: v }
    }
    
    fn get(self) -> T {
        self.val
    }
}

fn main() {
    let b = MyBox::new(123);
    let v = b.get();
    if v != 123 {
        // fail
        let x = 1 / 0; 
    }
    println("Generic Static OK");
}
