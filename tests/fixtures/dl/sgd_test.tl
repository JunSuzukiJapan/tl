// STDOUT: sgd_test OK

struct Linear {
    W: Tensor<f32, 2>,
    b: Tensor<f32, 1>,
}

fn Linear_new(in_dim: i64, out_dim: i64) -> Linear {
    let W = Tensor::randn([in_dim, out_dim], true);
    let b = Tensor::randn([out_dim], true);
    Linear(W.clone(), b.clone())
}

struct SGD { lr: f32 }

impl SGD {
    fn step(self, model: Linear) -> Linear {  
        let gW = model.W.grad();
        let gb = model.b.grad();
        
        let step_W = gW * self.lr;
        let new_W = model.W - step_W;
        
        let step_b = gb * self.lr;
        let new_b = model.b - step_b;
        
        let dW = new_W.detach(true);
        let db = new_b.detach(true);

        return Linear(dW.clone(), db.clone());
    }
}

fn forward(model: Linear, x: Tensor<f32, 2>) -> Tensor<f32, 2> {
    let y = x.matmul(model.W);
    return y + model.b;
}

fn main() {
    let model = Linear_new(2, 1);
    let sgd = SGD(0.01); 

    let X = Tensor::randn([16, 2], false);

    // 3 iterations of SGD
    let y_pred = forward(model, X);
    let loss = (y_pred * y_pred).sumall();
    loss.backward();
    let model = sgd.step(model);

    let y_pred = forward(model, X);
    let loss = (y_pred * y_pred).sumall();
    loss.backward();
    let model = sgd.step(model);

    let y_pred = forward(model, X);
    let loss = (y_pred * y_pred).sumall();
    loss.backward();
    let model = sgd.step(model);

    println("sgd_test OK");
}
