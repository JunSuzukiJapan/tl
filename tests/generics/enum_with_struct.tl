// Test: generic enum with generic struct variants
// Verifies all 3 compiler fixes:
//   1. User-defined generic enum (Container<T>) definition & monomorphization
//   2. Nested generic types (Option<Pair<T>>) monomorphization
//   3. Generic struct unwrap() SRET allocation

struct Pair<T> {
    first: T,
    second: T,
}

impl<T> Pair<T> {
    fn new(a: T, b: T) -> Pair<T> {
        Pair { first: a, second: b }
    }
}

// Issue 1: User-defined generic enum
enum Container<T> {
    Single(T),
    Double(Pair<T>),
    Empty
}

fn main() {
    // === Issue 1: User-defined generic enum ===
    let c1: Container<i64> = Container::Single(42);
    match c1 {
        Container::Single(v) => {
            print("Single=");
            println(v);
        },
        Container::Double(p) => {
            println("FAIL: expected Single");
        },
        Container::Empty => {
            println("FAIL: expected Single");
        }
    }
    // STDOUT: Single=42

    let p1: Pair<i64> = Pair::new(10, 20);
    let c2: Container<i64> = Container::Double(p1);
    match c2 {
        Container::Single(v) => {
            println("FAIL: expected Double");
        },
        Container::Double(p) => {
            print("Double first=");
            print(p.first);
            print(" second=");
            println(p.second);
        },
        Container::Empty => {
            println("FAIL: expected Double");
        }
    }
    // STDOUT: Double first=10 second=20

    let c3: Container<i64> = Container::Empty;
    match c3 {
        Container::Single(v) => {
            println("FAIL: expected Empty");
        },
        Container::Double(p) => {
            println("FAIL: expected Empty");
        },
        Container::Empty => {
            println("Empty");
        }
    }
    // STDOUT: Empty

    // === Issue 2: Nested generic type (Option<Pair<i64>>) ===
    let p2: Pair<i64> = Pair::new(100, 200);
    let opt: Option<Pair<i64>> = Option::Some(p2);
    match opt {
        Option::Some(val) => {
            print("Option::Some first=");
            print(val.first);
            print(" second=");
            println(val.second);
        },
        Option::None => {
            println("FAIL: expected Some");
        }
    }
    // STDOUT: Option::Some first=100 second=200

    let none_opt: Option<Pair<i64>> = Option::None;
    match none_opt {
        Option::Some(val) => {
            println("FAIL: expected None");
        },
        Option::None => {
            println("Option::None OK");
        }
    }
    // STDOUT: Option::None OK

    // === Issue 3: Generic struct unwrap() SRET ===
    let p3: Pair<i64> = Pair::new(77, 88);
    let opt3: Option<Pair<i64>> = Option::Some(p3);
    let unwrapped = opt3.unwrap();
    print("unwrap first=");
    print(unwrapped.first);
    print(" second=");
    println(unwrapped.second);
    // STDOUT: unwrap first=77 second=88

    println("All generic enum+struct tests passed!");
    // STDOUT: All generic enum+struct tests passed!
}
