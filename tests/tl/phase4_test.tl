fn main() {
    // leaky_relu, elu, mish
    let mut data = Vec::new();
    data.push(-2.0);
    data.push(-1.0);
    data.push(0.0);
    data.push(1.0);
    data.push(2.0);
    let mut s = Vec::new();
    s.push(5);
    let x = Tensor::from_vec(data, s);
    let lr = x.leaky_relu();
    print(lr);
    let e = x.elu();
    print(e);
    let m = x.mish();
    print(m);

    // mse_loss
    let pred = Tensor::full([4], 3.0);
    let target = Tensor::full([4], 1.0);
    let mse = Tensor::mse_loss(pred, target);
    print(mse);

    // l1_loss
    let l1 = Tensor::l1_loss(pred, target);
    print(l1);
}
