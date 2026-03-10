fn main() {
    // leaky_relu test
    let mut data = Vec::new();
    data.push(-2.0);
    data.push(-1.0);
    data.push(0.0);
    data.push(1.0);
    data.push(2.0);
    let mut s = Vec::new();
    s.push(5);
    let x = Tensor::from_vec(data, s);

    // default slope = 0.01
    let lr1 = x.leaky_relu();
    print(lr1);

    // custom slope = 0.1
    let lr2 = x.leaky_relu(0.1);
    print(lr2);
}
