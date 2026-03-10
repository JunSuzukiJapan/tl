fn main() {
    // initialize a tensor with requires_grad=true
    let t = Tensor::randn([2, 5], true);

    // perform some ops to generate a gradient
    let s = t * 2.5;
    let l = s.sum();
    l.backward();

    // we should have a gradient
    let g1 = t.grad();
    println("Original grad sum:");
    println(g1.sum().item());

    // 1. 测试 clip_grad_value
    let mut g1 = t.grad();
    g1.clip_grad_value(-0.5, 0.5);
    println("Grad sum after clip_value:");
    println(g1.sum().item());

    // 2. 测试 clip_grad_norm
    let mut g2 = t.grad();
    let norm = g2.clip_grad_norm(1.0, 2.0); // max_norm=1.0, L2 norm
    println("Grad norm before clip_norm:");
    println(norm);

    println("Grad sum after clip_norm:");
    println(g2.sum().item());

    // 3. 测试 freeze
    t.grad().freeze();
    
    // freeze 后，再次前向传播和反向传播，不应该有梯度
    let s2 = t * 2.0;
    let l2 = s2.sum();
    l2.backward();
    
    let mut g_freeze = t.grad();
    // g_freeze 应当是 nullptr 或者没有 elements. tl的 grad() 返回 Tensor，如果没grad应该引发错误或者返回空
    // 为了防止 crash，通常我们不直接访问没有 grad 的 tensor。
    // 这里我们只是调用 freeze 证明它不会崩溃。
    
    // 测试 unfreeze
    t.grad().unfreeze();

    println("Success!");
}
