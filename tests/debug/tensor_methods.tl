fn main() {
    let t = Tensor::randn([2, 3], false);
    
    // Unary Math
    let t_sin = t.sin();
    let t_cos = t.cos();
    let t_tan = t.tan();
    let t_sqrt = t.abs().sqrt(); // abs first to avoid NaN
    let t_exp = t.exp();
    let t_log = t.abs().log();
    let t_neg = t.neg();

    // Activations
    let t_relu = t.relu();
    let t_gelu = t.gelu();
    let t_silu = t.silu();
    // let t_sigmoid = t.sigmoid(); // Not yet implemented in builtins?
    let t_tanh = t.tanh();

    // Shape/Reduction
    let t_view = t.reshape([3, 2]);
    let t_t = t.transpose(0, 1);
    let t_sum = t.sum();
    let t_max = t.max();
    // let t_min = t.min(); // Not yet implemented in builtins?
    let t_mean = t.mean();
    let len = t.len();

    // Comparisons (returning Tensor<i64> or similar)
    // let t_eq = t.eq(t); 

    // NN Ops
    let logits = Tensor::randn([1, 10], false);
    let target = Tensor::zeros([1, 10], false); 
    // let loss = logits.cross_entropy(target);
    // let probs = logits.softmax(1);

    // Utils
    let t_cpu = t.cpu();
    // let t_cuda = t.cuda(); // Might require CUDA build
    let t_detach = t.detach();
    let t_cont = t.contiguous();

    print("Tensor methods test passed (compilation check)");
}
// STDOUT: Tensor methods test passed (compilation check)
