fn make_tensor() -> Tensor {
    Tensor::ones([3, 3], true)
}

fn main() {
    let mut i = 0;
    while i < 100 {
        let t = make_tensor();
        i = i + 1;
    }
    print("Done - no crash!");
}
