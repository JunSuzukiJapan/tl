fn main() {
    print("Testing backward...");
    
    // 単純なテンソル操作とbackward
    let W = (Tensor::randn([3, 3], true)*0.1).detach(true);
    let x = Tensor::randn([1, 3], true);
    let x_3d = x.reshape([1, 1, 3]);
    
    print("Forward...");
    let y = x_3d.matmul(W);
    let y_2d = y.reshape([1, 3]);
    
    // 簡単な損失関数
    let target = [1.0, 0.0, 0.0];
    let t = target.reshape([1, 3]);
    let diff = y_2d - t;
    let prod = diff * diff;
    let loss = prod.sum();
    
    print("Loss:");
    print(loss);
    
    print("Backward...");
    loss.backward();
    
    print("Grad:");
    let g = W.grad();
    print(g);
    
    print("Done!");
}
