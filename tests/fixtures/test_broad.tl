fn main() {
    let mut board = Tensor::randn([8, 8], true);
    let probs = board.softmax(1);
    let col_sums = probs.sum(0);
    let col_loss = (col_sums - 1.0).pow(2).sumall();
    col_loss.backward();
    println("col_loss: {}", col_loss.item());
    println("board.grad sum: {}", board.grad().sumall().item());
    
    // just to check if board.grad is all zeros
    let s = board.grad().abs().sumall().item();
    println("board.grad total abs: {}", s);
}
