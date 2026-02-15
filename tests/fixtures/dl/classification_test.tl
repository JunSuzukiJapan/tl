// Test: classification with autograd (forward + backward + grad)
struct Linear {
    W: Tensor<f32, 2>,
    b: Tensor<f32, 1>,
}

fn Linear_new(in_dim: i64, out_dim: i64) -> Linear {
    let W = Tensor::randn([in_dim, out_dim], true);
    let b = Tensor::randn([out_dim], true);
    Linear(W.clone(), b.clone())
}

fn forward(model: Linear, x: Tensor<f32, 2>) -> Tensor<f32, 2> {
    let out = x.matmul(model.W);
    let y = out + model.b;
    y
}

fn main() {
    Param::set_device(Device::Auto);

    // Data: 3 classes, 2 features
    let X = Tensor::randn([2, 2], false); 
    print(X);

    // Target: one-hot labels for 3 classes
    let target = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    print(target);

    let model = Linear_new(2, 3);

    // Forward
    let logits = forward(model, X);
    print(logits);

    // Softmax
    let probs = logits.softmax(1);
    print(probs);

    // Loss
    // NOTE: current backend expects probability-like inputs + one-hot labels.
    let loss = probs.cross_entropy(target);
    print(loss);

    // Backward
    loss.backward();

    print(model.W.grad());

    println("=== Classification test passed ===");
}
