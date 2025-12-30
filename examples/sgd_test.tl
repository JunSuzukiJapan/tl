
struct Linear {
    W: Tensor<f32, 2>,
    b: Tensor<f32, 1>,
}

fn Linear_new(in_dim: i64, out_dim: i64) -> Linear {
    let W = randn([in_dim, out_dim], true);
    let b = randn([out_dim], true);
    // MUST clone to keep tensors alive after W, b locals are dropped
    Linear(W.clone(), b.clone())
}

struct SGD { lr: f32 }

impl SGD {
    // Implicit self
    fn step(model: Linear) -> Linear {  
        let gW = model.W.grad();
        let gb = model.b.grad();
        
        let step_W = gW * self.lr;
        let new_W = model.W - step_W;
        
        let step_b = gb * self.lr;
        let new_b = model.b - step_b;
        
        let dW = new_W.detach(true);
        let db = new_b.detach(true);

        // Return new model with CLONED tensors to avoid use-after-free
        return Linear(dW.clone(), db.clone());
    }
}

fn forward(model: Linear, x: Tensor<f32, 2>) -> Tensor<f32, 2> {
    let y = matmul(x, model.W);
    // return y + model.b; // Broadcasting not fully supported yet in bin_op?
    // Let's assume broadcasting works now or use manual add if needed.
    // Actually prior steps said broadcasting scalar->tensor works.
    // Tensor(2,1) + Tensor(1) should work if broadcasting logic is there.
    return y + model.b;
}

// Main
let model = Linear_new(2, 1);
let sgd = SGD(0.01); 

// Data: y = 2*x1 - x2 + 0.5
let X = randn([16, 2], false); // Batch size 16
let true_W = randn([2, 1], false); // Weights not matching target exactly but arbitrary
let true_b = randn([1], false);

print(model.W);

// Unrolled iteration 1
let y_pred = forward(model, X);
let loss = sum(y_pred * y_pred);
print(loss);
backward(loss);
let model = sgd.step(model);

// Unrolled iteration 2
let y_pred = forward(model, X);
let loss = sum(y_pred * y_pred);
print(loss);
backward(loss);
let model = sgd.step(model);

// Unrolled iteration 3
let y_pred = forward(model, X);
let loss = sum(y_pred * y_pred);
print(loss);
backward(loss);
let model = sgd.step(model);

// Unrolled iteration 4
let y_pred = forward(model, X);
let loss = sum(y_pred * y_pred);
print(loss);
backward(loss);
let model = sgd.step(model);

// Unrolled iteration 5
let y_pred = forward(model, X);
let loss = sum(y_pred * y_pred);
print(loss);
backward(loss);
let model = sgd.step(model);

print(model.W);
print(model.b);
