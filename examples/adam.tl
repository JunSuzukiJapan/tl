
struct Linear {
    W: Tensor<f32, 2>,
    b: Tensor<f32, 1>,
}

fn Linear_new(in_dim: i64, out_dim: i64) -> Linear {
    let W = Tensor::randn([in_dim, out_dim], true);
    let b = Tensor::randn([out_dim], true);
    Linear(W.clone(), b.clone())
}

struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: i64,
    m_W: Tensor<f32, 2>,
    v_W: Tensor<f32, 2>,
    m_b: Tensor<f32, 1>,
    v_b: Tensor<f32, 1>,
}

impl Adam {
    fn step(model: Linear) {
        // self.t += 1; // Not supported on i64 yet? Manual:
        // self.t = self.t + 1; // Assign integers supported? 
        // If not, just ignore t for now (only needed for bias correction).
        
        // W update
        let g = model.W.grad();
        
        self.m_W *= self.beta1;
        let one_minus_beta1 = 1.0 - self.beta1;
        let term2_m = g * one_minus_beta1;
        self.m_W += term2_m;

        self.v_W *= self.beta2;
        let g2 = g * g;
        let one_minus_beta2 = 1.0 - self.beta2;
        let term2_v = g2 * one_minus_beta2;
        self.v_W += term2_v;

        let v_sqrt = sqrt(self.v_W);
        let denom = v_sqrt + self.eps;
        let step_size = self.m_W / denom;
        let update = step_size * self.lr;
        
        // Explicit Detach Mechanism to reset graph tracking for next iteration
        let w_next = model.W - update;
        let w_leaf = w_next.detach(true);
        model.W = w_leaf;

        // b update
        let gb = model.b.grad();
        
        self.m_b *= self.beta1;
        let term2_mb = gb * one_minus_beta1;
        self.m_b += term2_mb;

        self.v_b *= self.beta2;
        let gb2 = gb * gb;
        let term2_vb = gb2 * one_minus_beta2;
        self.v_b += term2_vb;
        
        let vb_sqrt = sqrt(self.v_b);
        let denomb = vb_sqrt + self.eps;
        let step_sizeb = self.m_b / denomb;
        let updateb = step_sizeb * self.lr;
        
        let b_next = model.b - updateb;
        let b_leaf = b_next.detach(true);
        model.b = b_leaf;
    }
}

// Main
let model = Linear_new(2, 1);

// Zeros
let m_W = Tensor::randn([2, 1], false) * 0.0;
let v_W = Tensor::randn([2, 1], false) * 0.0;
let m_b = Tensor::randn([1], false) * 0.0;
let v_b = Tensor::randn([1], false) * 0.0;

let optim = Adam(0.01, 0.9, 0.999, 0.00000001, 1, m_W, v_W, m_b, v_b);

let X = Tensor::randn([16, 2], false);

print(model.W);

// Iter 1
let y_pred = matmul(X, model.W) + model.b;
let loss = sum(y_pred * y_pred);
print(loss);
backward(loss);
optim.step(model);

// Iter 2
let y_pred = matmul(X, model.W) + model.b;
let loss = sum(y_pred * y_pred);
print(loss);
backward(loss);
optim.step(model);

// Iter 3
let y_pred = matmul(X, model.W) + model.b;
let loss = sum(y_pred * y_pred);
print(loss);
backward(loss);
optim.step(model);

print(model.W);
