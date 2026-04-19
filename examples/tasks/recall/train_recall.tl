// Associative Recall Task with Transformer
// Task: Input "k1 v1 k2 v2 ... k3 ?" -> Output "v3"
// This tests the model's ability to "attend" back to where "k3" appeared and retrieve "v3".

struct Linear {
    W: Tensor<f32, 2>, b: Tensor<f32, 1>,
    mW: Tensor<f32, 2>, vW: Tensor<f32, 2>,
    mb: Tensor<f32, 1>, vb: Tensor<f32, 1>
}

impl Linear {
    fn new(i: i64, o: i64) -> Linear {
        Linear(
            (Tensor::randn([i, o], true) * 0.1).detach(true), 
            (Tensor::randn([o], true) * 0.0).detach(true),
            Tensor::zeros([i, o], false), Tensor::zeros([i, o], false),
            Tensor::zeros([o], false), Tensor::zeros([o], false)
        )
    }
    
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> {
        x.matmul(self.W) + self.b
    }
    
    fn step(self, step_n: i64, lr: f32) -> Linear {
        let mut s = self;
        s.W.adam_step(s.W.grad(), s.mW, s.vW, step_n, lr, 0.9, 0.999, 1e-8, 0.0);
        s.b.adam_step(s.b.grad(), s.mb, s.vb, step_n, lr, 0.9, 0.999, 1e-8, 0.0);
        s
    }
}

struct Embedding {
    w: Tensor<f32, 2>,
    mw: Tensor<f32, 2>, vw: Tensor<f32, 2>
}

impl Embedding {
    fn new(v: i64, d: i64) -> Embedding {
        Embedding(
            (Tensor::randn([v, d], true) * 0.02).detach(true),
            Tensor::zeros([v, d], false), Tensor::zeros([v, d], false)
        )
    }
    
    fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> {
        i.embedding(self.w)
    }
    
    fn step(self, step_n: i64, lr: f32) -> Embedding {
        let mut s = self;
        s.w.adam_step(s.w.grad(), s.mw, s.vw, step_n, lr, 0.9, 0.999, 1e-8, 0.0);
        s
    }
}

struct LayerNorm {
    w: Tensor<f32, 1>, b: Tensor<f32, 1>,
    mw: Tensor<f32, 1>, vw: Tensor<f32, 1>,
    mb: Tensor<f32, 1>, vb: Tensor<f32, 1>
}

impl LayerNorm {
    fn new(d: i64) -> LayerNorm {
        LayerNorm(
            (Tensor::randn([d], true)*0.0+1.0).detach(true), 
            (Tensor::randn([d], true)*0.0).detach(true),
            Tensor::zeros([d], false), Tensor::zeros([d], false),
            Tensor::zeros([d], false), Tensor::zeros([d], false)
        )
    }
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> {
        x.layer_norm(self.w, self.b, 0.00001 as f64)
    }
    fn step(self, step_n: i64, lr: f32) -> LayerNorm {
        let mut s = self;
        s.w.adam_step(s.w.grad(), s.mw, s.vw, step_n, lr, 0.9, 0.999, 1e-8, 0.0);
        s.b.adam_step(s.b.grad(), s.mb, s.vb, step_n, lr, 0.9, 0.999, 1e-8, 0.0);
        s
    }
}

struct CausalSelfAttention {
    c_attn: Linear, 
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    c_proj: Linear, 
}

impl CausalSelfAttention {
    fn new(d: i64) -> CausalSelfAttention {
        CausalSelfAttention(
            Linear::new(1, 1), 
            Linear::new(d, d), 
            Linear::new(d, d), 
            Linear::new(d, d), 
            Linear::new(d, d)
        )
    }

    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> {
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        let k_t = k.transpose(1, 2);
        let att = q.matmul(k_t) * 0.125;
        
        // Causal Mask
        let ones = att * 0.0 + 1.0;
        let causal_mask = ones.tril(0); 
        let neg_inf_mask = (causal_mask - 1.0) * 10000.0; 
        
        let masked = att + neg_inf_mask;
        let att_p = masked.softmax(2);
        let y = att_p.matmul(v);
        
        self.c_proj.forward(y)
    }

    fn step(self, step_n: i64, lr: f32) -> CausalSelfAttention {
        let mut s = self;
        s.q_proj = s.q_proj.step(step_n, lr);
        s.k_proj = s.k_proj.step(step_n, lr);
        s.v_proj = s.v_proj.step(step_n, lr);
        s.c_proj = s.c_proj.step(step_n, lr);
        s
    }
}

struct MLP {
    c_fc: Linear,
    c_proj: Linear
}

impl MLP {
    fn new(d: i64) -> MLP {
        MLP(Linear::new(d, d * 4), Linear::new(d * 4, d))
    }
    
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> {
        return self.c_proj.forward(self.c_fc.forward(x).relu());
    }
    
    fn step(self, step_n: i64, lr: f32) -> MLP {
        let mut s = self;
        s.c_fc = s.c_fc.step(step_n, lr);
        s.c_proj = s.c_proj.step(step_n, lr);
        s
    }
}

struct Block {
    ln1: LayerNorm,
    attn: CausalSelfAttention,
    ln2: LayerNorm,
    mlp: MLP
}

impl Block {
    fn new(d: i64) -> Block {
        Block(LayerNorm::new(d), CausalSelfAttention::new(d), LayerNorm::new(d), MLP::new(d))
    }
    
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> {
        let x = x + self.attn.forward(self.ln1.forward(x));
        let x = x + self.mlp.forward(self.ln2.forward(x));
        x
    }
    
    fn step(self, step_n: i64, lr: f32) -> Block {
        let mut s = self;
        s.ln1 = s.ln1.step(step_n, lr);
        s.attn = s.attn.step(step_n, lr);
        s.ln2 = s.ln2.step(step_n, lr);
        s.mlp = s.mlp.step(step_n, lr);
        s
    }
}

struct GPT {
    wte: Embedding, 
    wpe: Embedding, 
    bg: Block,      
    ln_f: LayerNorm,
    head: Linear
}

impl GPT {
    fn new(vocab_size: i64, d_model: i64, max_len: i64) -> GPT {
        GPT(
            Embedding::new(vocab_size, d_model),
            Embedding::new(max_len, d_model),
            Block::new(d_model),
            LayerNorm::new(d_model),
            Linear::new(d_model, vocab_size)
        )
    }
    
    fn forward(self, idx: Tensor<f32, 2>) -> Tensor<f32, 3> {
        let pos_arr = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let pos = pos_arr.reshape([1, 13]);
        
        let tok_emb = self.wte.forward(idx);
        let pos_emb = self.wpe.forward(pos);
        
        let x = tok_emb + pos_emb;
        let x = self.bg.forward(x);
        let x = self.ln_f.forward(x);
        let logits = self.head.forward(x);
        logits
    }
    
    fn step(self, step_n: i64, lr: f32) -> GPT {
        let mut s = self;
        s.wte = s.wte.step(step_n, lr);
        s.wpe = s.wpe.step(step_n, lr);
        s.bg = s.bg.step(step_n, lr);
        s.ln_f = s.ln_f.step(step_n, lr);
        s.head = s.head.step(step_n, lr);
        s
    }
}

fn gen_data(step: i64) -> Tensor<f32, 2> {
    let seed = step * 12345 + 6789;
    
    let k1 = ((seed + 1) * 7) - ((((seed + 1) * 7) / 20) * 20) + 10;
    let v1 = ((seed + 2) * 3) - ((((seed + 2) * 3) / 10) * 10);
    
    let k2 = ((seed + 3) * 7) - ((((seed + 3) * 7) / 20) * 20) + 10;
    let v2 = ((seed + 4) * 3) - ((((seed + 4) * 3) / 10) * 10);
    
    let k3 = ((seed + 5) * 7) - ((((seed + 5) * 7) / 20) * 20) + 10;
    let v3 = ((seed + 6) * 3) - ((((seed + 6) * 3) / 10) * 10);
    
    let k4 = ((seed + 7) * 7) - ((((seed + 7) * 7) / 20) * 20) + 10;
    let v4 = ((seed + 8) * 3) - ((((seed + 8) * 3) / 10) * 10);
    
    let k5 = ((seed + 9) * 7) - ((((seed + 9) * 7) / 20) * 20) + 10;
    let v5 = ((seed + 10) * 3) - ((((seed + 10) * 3) / 10) * 10);
    
    let k6 = ((seed + 11) * 7) - ((((seed + 11) * 7) / 20) * 20) + 10;
    let v6 = ((seed + 12) * 3) - ((((seed + 12) * 3) / 10) * 10);
    
    let pick = seed - ((seed / 6) * 6);
    let mut q_k = 0;
    let mut q_ans = 0;
    if pick == 0 { q_k = k1; q_ans = v1; }
    if pick == 1 { q_k = k2; q_ans = v2; }
    if pick == 2 { q_k = k3; q_ans = v3; }
    if pick == 3 { q_k = k4; q_ans = v4; }
    if pick == 4 { q_k = k5; q_ans = v5; }
    if pick == 5 { q_k = k6; q_ans = v6; }
    
    let arr = [
        k1 as f32, v1 as f32, k2 as f32, v2 as f32, k3 as f32, v3 as f32, 
        k4 as f32, v4 as f32, k5 as f32, v5 as f32, k6 as f32, v6 as f32, 
        q_k as f32, q_ans as f32
    ];
    arr.reshape([1, 14])
}

struct TrainState {
    global_step: i64,
    model: GPT
}

fn train_epoch(state: TrainState, lr: f32, steps: i64) -> TrainState {
    let mut st = state;
    for s in range(0, steps) {
        let i = st.global_step;
        let data = gen_data(i); 
        
        let in_seq = [
             data[0, 0], data[0, 1],
             data[0, 2], data[0, 3],
             data[0, 4], data[0, 5],
             data[0, 6], data[0, 7],
             data[0, 8], data[0, 9],
             data[0, 10], data[0, 11],
             data[0, 12]
        ];
        let X = in_seq.reshape([1, 13]);
        let logits = st.model.forward(X);
        let logits_flat = logits.reshape([13, 30]);
        
        // Exact loss on the last target token only
        let last_logit = logits_flat.slice(0, 12, 13, 1);
        let Y_arr = [data[0, 13]];
        let Y = Y_arr.reshape([1]);
        
        let loss = last_logit.cross_entropy(Y);
        
        loss.backward();
        
        if s % 16 == 15 {
             st.model = st.model.step(st.global_step, lr);
             Tensor::clear_grads();
             st.global_step = st.global_step + 1;
        }
        
        let loss_val = loss.item(); 
        if s % 50 == 0 {
             print("Step:"); print(s);
             print("Loss:"); print(loss_val);
        }
    }
    st
}

fn main() {
    let vocab_size = 30;
    let d_model = 64;
    let max_len = 32;
    
    let mut state = TrainState(1, GPT::new(vocab_size, d_model, max_len));
    let lr = 0.002;
    
    print("Starting Associative Recall Training (Adam)...");
    
    for epoch in range(0, 100) {
        print("Epoch:"); print(epoch);
        state = train_epoch(state, lr, 250);
    }
    
    print("Training Complete.");
    print("Saving model weights to recall_weights.safetensors...");
    Param::save(state.model, "recall_weights.safetensors");
    print("Model saved.");
}
