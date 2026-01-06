extern fn tl_tensor_print_2(t: Tensor<f32, 2>);

fn main() {
    print("Debug Variables Start");
    let i = 13;
    let o = 384;
    
    let t1 = Tensor::randn([i, o], true);
    print("i=13, o=384. randn([i, o]):");
    tl_tensor_print_2(t1);

    let t2 = Tensor::randn([o, i], true);
    print("i=13, o=384. randn([o, i]):");
    tl_tensor_print_2(t2);
}
