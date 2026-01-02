fn main() -> i64 {
    let start = tl_system_time();
    let x = 0.0;
    for i in range(0, 50000) {
        let a = [1.0, 2.0];
        x = x + a[0];
    }
    let end = tl_system_time();
    print("Time:");
    print(end - start);
    print("Result:");
    print(x);
    return 0;
}
