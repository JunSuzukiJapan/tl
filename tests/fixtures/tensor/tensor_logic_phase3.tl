fn main() -> i64 {
    let values: Tensor<i64, 1> = [0, 1, 2, 3, 4];
    // let a = sum(values[i] for i in values); // Old syntax
    let a = values.sum();
    print(a);
    0
}
