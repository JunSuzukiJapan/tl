// STDOUT: autograd_test OK
// Autograd Test

fn main() {
    // 1. Create a tensor with gradient tracking
    let mut w = Tensor::randn([1], true);

    // 2. Define a simple function: y = w^3
    let w2 = w * w;
    let y = w2 * w;

    // 3. Backward
    y.backward();

    // 4. Get gradient
    let g = w.grad();

    // 5. Update weights (Gradient Descent)
    w -= g;

    println("autograd_test OK");
}
