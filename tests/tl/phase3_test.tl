fn main() {
    // view test (reshape alias)
    let t = Tensor::full([2, 3], 1.0);
    let mut shape = Vec::new();
    shape.push(3);
    shape.push(2);
    let v = t.view(shape);
    print(v);

    // stack test
    let a = Tensor::full([3], 1.0);
    let b = Tensor::full([3], 2.0);
    let s = Tensor::stack(a, b, 0);
    print(s);
}
