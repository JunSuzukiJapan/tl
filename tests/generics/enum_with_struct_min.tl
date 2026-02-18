// Minimal test for Issue 3 - unwrap() SRET
struct Pair<T> {
    first: T,
    second: T,
}

impl<T> Pair<T> {
    fn new(a: T, b: T) -> Pair<T> {
        Pair { first: a, second: b }
    }
}

fn main() {
    let p: Pair<i64> = Pair::new(77, 88);
    let opt: Option<Pair<i64>> = Option::Some(p);
    let unwrapped = opt.unwrap();
    print("first=");
    print(unwrapped.first);
    print(" second=");
    println(unwrapped.second);
}
