fn main() {
    // mse_loss
    let pred = Tensor::full([4], 3.0);
    let target = Tensor::full([4], 1.0);
    let mse = Tensor::mse_loss(pred, target);
    print(mse);

    // l1_loss
    let l1 = Tensor::l1_loss(pred, target);
    print(l1);

    // bce_loss: pred=0.9, target=1.0 -> -log(0.9) ≈ 0.105
    let p2 = Tensor::full([4], 0.9);
    let t2 = Tensor::full([4], 1.0);
    let bce = Tensor::bce_loss(p2, t2);
    print(bce);
}
