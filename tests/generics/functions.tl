fn id<T>(x: T) -> T {
    x
}

fn pick_first<A, B>(a: A, b: B) -> A {
    a
}

fn make_box<T>(val: T) -> Box<T> {
    Box { inner: val }
}

struct Box<T> {
    inner: T,
}

fn main() {
    let x = id(42);
    let y = id(3.14);
    let z = id(true);

    let first = pick_first(10, 20.0);
    
    let b = make_box(100);
    
    print(x);
}
