fn main() {
    let T = Tensor::zeros([1]);
    let A = T;
    tl_print_string("Success: No Double Free");
}
