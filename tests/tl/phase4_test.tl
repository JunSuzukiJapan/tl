fn main() {
    // leaky_relu
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

    // elu
    let e = x.elu();
    print(e);

    // mish
    let m = x.mish();
    print(m);
}
