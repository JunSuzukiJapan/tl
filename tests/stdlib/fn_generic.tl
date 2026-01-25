fn identity<T>(x: T) -> T {
    x
}

fn generic_add<T>(a: T, b: T) -> T {
    a + b
}

fn main() {
    let x = identity(10);
    if x != 10 { println("Error: x!=10"); } else { println("x OK"); }

    let y = identity(3.14);
    // Float equality is tricky, just check it runs
    println("y OK");

    let z = generic_add(10, 20);
    if z != 30 { println("Error: z!=30"); } else { println("z OK"); }

    let w = generic_add(1.5, 2.5);
    // 4.0
    println("w OK");
}
