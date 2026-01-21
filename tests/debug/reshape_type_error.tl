// Test case for semantic analysis: reshape with wrong argument type
fn main() {
    let x = [1.0, 2.0, 3.0, 4.0];
    // This should fail: [1, 2].print_2() returns Void, not a shape array
    x.reshape([1, 2].print_2());
}
