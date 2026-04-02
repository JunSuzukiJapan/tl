// tests/test_tril_batch.tl
fn main() {
    print("Testing tril with batched tensor...");
    let x = Tensor::randn([2, 3, 3], true);

    print("x:");
    x.display();

    let diag = 0;
    
    // forward pass like train_paper.tl
    let ones = Tensor::ones_like(x);
    let upper_tri = ones - ones.tril(0);
    print("upper_tri:");
    upper_tri.display();
    
    // Create a mock loss
    let diff = x.masked_fill(upper_tri, -f32::INFINITY);
    print("diff:");
    diff.display();
    
    let loss = x.sum();
    print("loss (sum of diff):");
    loss.display();

    // backward pass
    print("Running backward...");
    loss.backward();

    let grad = x.grad();
    print("x.grad():");
    grad.display();

    print("Success!");
}
