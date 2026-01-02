fn add(a: f32, b: f32) -> f32 {
    let res = a + b;
    return res;
}

fn main() {
    let x = 10.0;
    let y = 20.0;
    let z = add(x, y);
    print([z]);

    let f = factorial(5.0);
    print([f]);
}

fn factorial(n: f32) -> f32 {
    if n < 1.0 {
        return 1.0;
    }
    return n * factorial(n - 1.0);
}
