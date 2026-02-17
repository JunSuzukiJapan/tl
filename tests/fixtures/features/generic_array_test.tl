// Generic array test

// Non-generic function with array parameter
fn sum_array(arr: [i64; 3]) -> i64 {
    arr[0] + arr[1] + arr[2]
}

// Generic function with array parameter
fn first<T>(arr: [T; 2]) -> T {
    arr[0]
}

fn main() {
    let a: [i64; 3] = [10, 20, 30];
    print(sum_array(a));    // expect: 60

    let b: [i64; 2] = [42, 99];
    print(first(b));        // expect: 42
}
