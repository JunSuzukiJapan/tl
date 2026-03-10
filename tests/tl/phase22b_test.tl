fn main() {
    // cumsum test
    let mut data = Vec::new();
    data.push(1.0);
    data.push(2.0);
    data.push(3.0);
    data.push(4.0);
    let mut s = Vec::new();
    s.push(4);
    let x = Tensor::from_vec(data, s);
    let cs = x.cumsum(0);
    print(cs);

    // norm test
    let mut d2 = Vec::new();
    d2.push(3.0);
    d2.push(4.0);
    let mut s2 = Vec::new();
    s2.push(2);
    let y = Tensor::from_vec(d2, s2);
    let n = y.norm(2.0);
    print(n);

    // topk test
    let mut d3 = Vec::new();
    d3.push(5.0);
    d3.push(1.0);
    d3.push(9.0);
    d3.push(3.0);
    d3.push(7.0);
    let mut s3 = Vec::new();
    s3.push(5);
    let z = Tensor::from_vec(d3, s3);
    let tk = z.topk(3);
    print(tk);

    // logical ops
    let mut la = Vec::new();
    la.push(1.0);
    la.push(0.0);
    la.push(1.0);
    la.push(0.0);
    let mut lb = Vec::new();
    lb.push(1.0);
    lb.push(1.0);
    lb.push(0.0);
    lb.push(0.0);
    let mut ls = Vec::new();
    ls.push(4);
    let a = Tensor::from_vec(la, ls);
    let mut ls2 = Vec::new();
    ls2.push(4);
    let b = Tensor::from_vec(lb, ls2);
    let land = Tensor::logical_and(a, b);
    print(land);
    let lor = Tensor::logical_or(a, b);
    print(lor);
    let lnot = a.logical_not();
    print(lnot);
}
