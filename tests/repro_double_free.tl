fn make_tensor() -> Tensor {
    let t = Tensor::ones([100, 100]);
    t
}

fn main() {
    let mut i = 0;
    while i < 100 {
        let t = make_tensor();
        let s = t.sum();
        i = i + 1;
    }
    print("Done - no crash!");
}
