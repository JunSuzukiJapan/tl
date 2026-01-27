fn main() {
    let mut v: Vec<i64> = Vec::new();
    v.push(1);
    v.push(2);
    v.push(3);

    let len = process_vec(v);
    print(len); // Should be 3

    // Inline mutation to test if function boundary is the issue
    v.push(100);
    print(v.len()); // Should be 4
    if v.len() > 3 {
        print(v.get(3)); // Should be 100
    }
}

fn process_vec(v: Vec<i64>) -> i64 {
    return v.len();
}

fn mutate_vec(v: Vec<i64>) -> Vec<i64> {
    v.push(100);
    return v;
}
