
fn main() {
    print("Debug Variables Start");
    let i = 13;
    let o = 384;
    
    let t1 = Tensor::randn([i, o], true);
    print("i=13, o=384. randn([i, o]):");
    t1.print_2();

    let t2 = Tensor::randn([o, i], true);
    print("i=13, o=384. randn([o, i]):");
    t2.print_2();
}
