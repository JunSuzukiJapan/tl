// Test case for semantic analysis: reshape with correct syntax
fn main() {
    let x = [1.0, 2.0, 3.0, 4.0];
    // Correct: reshape first, then print
    x.reshape([2, 2]).print_2();
}
