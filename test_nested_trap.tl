struct Linear { W: Tensor<f32, 2>, b: Tensor<f32, 1> }
impl Linear { 
    fn new() -> Linear { 
        Linear(Tensor::randn([10, 10], false), Tensor::randn([10], false)) 
    } 
}

struct CausalSelfAttention { q_proj: Linear, k_proj: Linear, v_proj: Linear, p_proj: Linear }
impl CausalSelfAttention { 
    fn new() -> CausalSelfAttention { 
        CausalSelfAttention(Linear::new(), Linear::new(), Linear::new(), Linear::new()) 
    }
}

fn main() {
    print("Start Causal Init");
    let a = CausalSelfAttention::new();
    print("End Causal Init");
}
