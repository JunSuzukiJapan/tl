fn main() {
    let mut cond_data = Vec::new();
    cond_data.push(1.0);
    cond_data.push(0.0);
    cond_data.push(1.0);
    cond_data.push(0.0);
    cond_data.push(1.0);
    cond_data.push(0.0);
    let mut shape = Vec::new();
    shape.push(2);
    shape.push(3);
    let cond = Tensor::from_vec(cond_data, shape);

    let x = Tensor::full([2, 3], 10.0);
    let y = Tensor::full([2, 3], -1.0);
    let result = Tensor::where_cond(cond, x, y);
    print(result);
}
