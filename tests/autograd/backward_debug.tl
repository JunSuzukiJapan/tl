// 元のcrashパターン: struct.W.grad() の return
struct Linear {
    W: Tensor<f32, 2>,
    b: Tensor<f32, 1>,
}

fn get_grad(model: Linear) -> Tensor<f32, 2> {
    let gW = model.W.grad();
    return gW;
}

fn main() {
    let W = Tensor::randn([2, 1], true);
    let b = Tensor::randn([1], false);
    let model = Linear { W: W, b: b };
    
    let X = Tensor::randn([4, 2], false);
    let y = X.matmul(model.W);
    let loss = y.sumall();
    loss.backward();
    
    println("backward done");
    
    // Test 1: main 内で grad 取得
    let g_main = model.W.grad();
    println("main grad:");
    g_main.print();
    
    // Test 2: 関数内で grad 取得して return
    let g_fn = get_grad(model);
    println("fn grad:");
    g_fn.print();
    
    // Test 3: grad から detach へ
    let gW2 = model.W.grad();
    let dW2 = model.W.detach();
    let result = dW2 - gW2;
    println("dW - gW:");
    result.print();
    
    println("=== All passed ===");
}
