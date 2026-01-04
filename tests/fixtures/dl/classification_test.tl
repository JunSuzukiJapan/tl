struct Linear {
    W: Tensor<f32, 2>,
    b: Tensor<f32, 1>,
}

// Data: 3 classes, 2 features
// X: [2, 2]
let X = Tensor::randn([2, 2], false); 
print(X);

// Target: [2] (indices 0, 1)
// Represented as float tensor for now: [0.0, 1.0]
let target = [0.0, 1.0];
print(target);

fn Linear_new(in_dim: i64, out_dim: i64) -> Linear {
    let W = Tensor::randn([in_dim, out_dim], true);
    let b = Tensor::randn([out_dim], true);
    return Linear(W.clone(), b.clone());
}

fn forward(model: Linear, x: Tensor<f32, 2>) -> Tensor<f32, 2> {
    // x: [B, In], W: [In, Out] -> [B, Out]
    let out = matmul(x, model.W);
    // Add bias (broadcasting)
    let y = out + model.b;
    return y;
}

let model = Linear_new(2, 3);

// Forward
let logits = forward(model, X);
print(logits);

// Softmax (just to test it works)
let probs = softmax(logits, 1);
print(probs);

// Loss
let loss = cross_entropy(logits, target);
print(loss);

// Backward
backward(loss);

print(model.W.grad());
