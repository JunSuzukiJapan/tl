fn main() {
    // masked_fill test
    let mut mask_data = Vec::new();
    mask_data.push(1.0);
    mask_data.push(0.0);
    mask_data.push(1.0);
    mask_data.push(0.0);
    let mut shape = Vec::new();
    shape.push(2);
    shape.push(2);
    let mask = Tensor::from_vec(mask_data, shape);
    let t = Tensor::full([2, 2], 5.0);
    let filled = t.masked_fill(mask, -999.0);
    print(filled);

    // var/std test
    let mut data = Vec::new();
    data.push(1.0);
    data.push(2.0);
    data.push(3.0);
    data.push(4.0);
    let mut s = Vec::new();
    s.push(4);
    let x = Tensor::from_vec(data, s);
    print(x);
    let v = x.var(0);
    print(v);
    let sd = x.std(0);
    print(sd);
}
