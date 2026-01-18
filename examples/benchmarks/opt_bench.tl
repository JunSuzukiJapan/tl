fn main() -> i64 {
    println("Running opt benchmark...");
    let mut x = 0.0;
    for i in range(0, 50000) {
        let a = [1.0, 2.0];
        x = x + a[0];
    }
    print("Result: ");
    println(x);
    return 0;
}
