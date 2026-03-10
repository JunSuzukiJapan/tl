fn main() {
    // prod test
    let mut data = Vec::new();
    data.push(2.0);
    data.push(3.0);
    data.push(4.0);
    let mut s = Vec::new();
    s.push(3);
    let x = Tensor::from_vec(data, s);
    let p = x.prod();
    print(p);

    // fill_ test
    let t = Tensor::full([2, 3], 1.0);
    print(t);
    t.fill_(42.0);
    print(t);
}
