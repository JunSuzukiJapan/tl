fn tmp_test() {
    Tensor::zeros([1000, 1000]);
}
fn main() {
    for i in 0..1000 {
        tmp_test();
    }
}
