struct Linear {
    W: Tensor<f32, 2>,
    b: Tensor<f32, 1>,
}

// Factory function instead of static method
fn Linear_new(in_dim: i64, out_dim: i64) -> Linear {
    // Hardcoded shape for test stability/simplicity
    // In real lib, we'd use dynamic shape if randn supported it fully
    let W = randn([2, 1], true); 
    let b = randn([1], true);
    Linear(W.clone(), b.clone())
}

impl Linear {
    fn forward(x: Tensor<f32, 2>) -> Tensor<f32, 2> {
        let xw = matmul(x, self.W);
        // Broadcasting add: [1, 1] + [1] -> [1, 1]
        let y = xw + self.b;
        return y;
    }
}

tensor X: f32 = [1.0, 2.0];
let X = reshape(X, [1, 2]);

// Create model
let model = Linear_new(2, 1);

// Forward
let pred = model.forward(X);
print(pred);
