// Vec Closure Methods Test
fn main() {
    let mut v: Vec<i64> = Vec::new();
    v.push(1);
    v.push(2);
    v.push(3);
    v.push(4);
    v.push(5);

    // Test any: has even number?
    let has_even = v.any(|x: i64| x % 2 == 0);
    println("{}", has_even);

    // Test all: all positive?
    let all_positive = v.all(|x: i64| x > 0);
    println("{}", all_positive);

    // Test all: all even? (should be false)
    let all_even = v.all(|x: i64| x % 2 == 0);
    println("{}", all_even);
}
