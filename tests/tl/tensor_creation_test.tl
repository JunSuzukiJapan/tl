fn main() {
    let t = Tensor::full([2, 3], 3.14);
    print(t);

    let e = Tensor::eye(3);
    print(e);

    let a = Tensor::arange(0.0, 10.0, 2.0);
    print(a);

    let b = Tensor::arange(1.0, 5.0, 0.5);
    print(b);

    let src = Tensor::randn([2, 4]);
    let zl = Tensor::zeros_like(src);
    print(zl);

    let ol = Tensor::ones_like(src);
    print(ol);

    let mut data = Vec::new();
    data.push(1.0);
    data.push(2.0);
    data.push(3.0);
    data.push(4.0);
    data.push(5.0);
    data.push(6.0);
    let mut shape = Vec::new();
    shape.push(2);
    shape.push(3);
    let fv = Tensor::from_vec(data, shape);
    print(fv);

    let ls = Tensor::linspace(0.0, 1.0, 5);
    print(ls);

    let r = Tensor::rand([3, 2]);
    print(r);

    let rl = Tensor::rand_like(src);
    print(rl);

    let rnl = Tensor::randn_like(src);
    print(rnl);
}
