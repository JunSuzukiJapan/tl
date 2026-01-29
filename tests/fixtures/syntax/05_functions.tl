fn add(a: i64, b: i64) -> i64 {
    a + b
}

fn hello() {
    print("Function called"); // STDOUT: Function called
}

fn factorial(n: i64) -> i64 {
    if n <= 1 {
        return 1;
    }
    return n * factorial(n - 1);
}

fn main() {
    let sum = add(10, 20);
    print(sum); // STDOUT: 30

    hello();

    let f = factorial(5); // 120
    print(f); // STDOUT: 120
}
