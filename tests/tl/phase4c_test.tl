fn main() {
    // pad test
    let mut data = Vec::new();
    data.push(1.0);
    data.push(2.0);
    data.push(3.0);
    let mut s = Vec::new();
    s.push(3);
    let x = Tensor::from_vec(data, s);
    let padded = x.pad(2, 1);
    print(padded);
}
