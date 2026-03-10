fn main() {
    let mut data = Vec::new();
    data.push(-6.0);
    data.push(-3.0);
    data.push(0.0);
    data.push(3.0);
    data.push(6.0);
    let mut s = Vec::new();
    s.push(5);
    let x = Tensor::from_vec(data, s);

    // hardswish: x * max(0, min(6, x+3)) / 6
    let hs = x.hardswish();
    print(hs);

    // hardsigmoid: max(0, min(1, (x+3)/6))
    let hsig = x.hardsigmoid();
    print(hsig);
}
