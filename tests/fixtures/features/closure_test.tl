// Closure Test
fn main() {
    // Test 1: Simple closure assigned to variable
    let add_one = |x: i64| -> i64 { x + 1 };
    let result = add_one(5);
    println("{}", result);

    // Test 2: Closure with multiple args
    let add = |x: i64, y: i64| -> i64 { x + y };
    let sum = add(3, 4);
    println("{}", sum);

    // Test 3: Closure called multiple times
    let double = |x: i64| -> i64 { x * 2 };
    println("{}", double(5));
    println("{}", double(10));
}
