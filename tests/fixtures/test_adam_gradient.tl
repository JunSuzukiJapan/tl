extern fn tl_adam_step(
    param: Tensor<f32, 2>, 
    grad: Tensor<f32, 2>, 
    m: Tensor<f32, 2>, 
    v: Tensor<f32, 2>, 
    step: i32, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32
);

fn main() {
    let w = Tensor::randn([10, 10], true);
    let mut w_train = w.detach(true);
    let x = Tensor::randn([10, 10], true);
    let m = Tensor::randn([10, 10], true)*0.0;
    let v = Tensor::randn([10, 10], true)*0.0;
    let loss = w_train.matmul(x).sum();
    loss.backward();
    println("Weight Before:");
    w_train.print();
    println("Grad Before:");
    w_train.grad().print();
    tl_adam_step(w_train, w_train.grad(), m, v, 1, 0.001, 0.9, 0.999, 1e-8, 0.0);
    println("Weight After:");
    w_train.print();
    println("Grad After:");
    w_train.grad().print();
}
