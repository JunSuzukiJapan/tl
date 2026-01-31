fn main() {
    let mut t = Tensor::zeros([2, 2]);
    t[0, 0] = 10.0;
    let v = t[0, 0];
    print(v);
    if v == 10.0 {
        print("OK");
    } else {
        print("FAIL");
    }
}
