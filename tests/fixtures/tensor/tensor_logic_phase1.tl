// Phase 1 Test: Tensor access in logic predicates
// Note: This is a design exploration - full integration requires more work

// Define a tensor
let values: Tensor<f64, 1> = [0.5, 1.0, 1.5, 2.0];

// In the future, we want to be able to write:
// high_value(i) :- values[i] > 1.0;
// ?- high_value(2);

fn main() -> i64 {
    print(values[2]);  // Should print 1.5
    return 0;
}
