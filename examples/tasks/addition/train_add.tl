// 1桁足し算タスク - 2-Layer GPT (Adam, grad_accum=8)
// ---------------
// Model Components
// ---------------

struct Linear { 
    W: Tensor<f32, 2>, b: Tensor<f32, 1>,
    mW: Tensor<f32, 2>, vW: Tensor<f32, 2>,
    mb: Tensor<f32, 1>, vb: Tensor<f32, 1>
}
impl Linear { 
    fn new(i: i64, o: i64) -> Linear { 
        Linear(
            (Tensor::randn([i, o], true)*0.1).detach(true), // W
            (Tensor::randn([o], true)*0.0).detach(true),    // b
            Tensor::zeros([i, o], false),                   // mW
            Tensor::zeros([i, o], false),                   // vW
            Tensor::zeros([o], false),                      // mb
            Tensor::zeros([o], false)                       // vb
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

struct LayerNorm { 
    w: Tensor<f32, 1>, b: Tensor<f32, 1>,
    mw: Tensor<f32, 1>, vw: Tensor<f32, 1>,
    mb: Tensor<f32, 1>, vb: Tensor<f32, 1>
}
impl LayerNorm {
    fn new(d: i64) -> LayerNorm { 
        LayerNorm(
            (Tensor::randn([d], true)*0.0+1.0).detach(true),  // w
            (Tensor::randn([d], true)*0.0).detach(true),      // b
            Tensor::zeros([d], false),                        // mw
            Tensor::zeros([d], false),                        // vw
            Tensor::zeros([d], false),                        // mb
            Tensor::zeros([d], false)                         // vb
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

struct CausalSelfAttention { q_proj: Linear, k_proj: Linear, v_proj: Linear, p_proj: Linear }
impl CausalSelfAttention { 
    fn new(d: i64) -> CausalSelfAttention { 
        CausalSelfAttention(
            Linear::new(d, d), // q_proj
            Linear::new(d, d), // k_proj
            Linear::new(d, d), // v_proj
            Linear::new(d, d)  // p_proj
        ) 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        let q = self.q_proj.forward(x); 
        let k = self.k_proj.forward(x); 
        let v = self.v_proj.forward(x); 
        let kt = k.transpose(1, 2);
        let scores = q.matmul(kt) * 0.0884; // 1/sqrt(128)
        
        let ones = scores * 0.0 + 1.0;
        let causal_mask = ones.tril(0); 
        let neg_inf_mask = (causal_mask - 1.0) * 10000.0; 
        let masked = scores + neg_inf_mask;
        
        let attn_weights = masked.softmax(2);
        self.p_proj.forward(attn_weights.matmul(v)) 
    } 
    fn step(self, step_n: i64, lr: f32) -> CausalSelfAttention { 
        let mut s = self; 
        s.k_proj = s.k_proj.step(step_n, lr); 
        s.p_proj = s.p_proj.step(step_n, lr); 
        s.q_proj = s.q_proj.step(step_n, lr); 
        s.v_proj = s.v_proj.step(step_n, lr); 
        s 
    } 
}

struct MLP { f: Linear, p: Linear }
impl MLP { 
    fn new(d: i64) -> MLP { 
        MLP(
            Linear::new(d, d*4), // f
            Linear::new(d*4, d)  // p
        ) 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        self.p.forward(self.f.forward(x).relu()) 
    } 
    fn step(self, step_n: i64, lr: f32) -> MLP { 
        let mut s = self; 
        s.f = s.f.step(step_n, lr); 
        s.p = s.p.step(step_n, lr); 
        s 
    } 
}

struct Block { l1: LayerNorm, a: CausalSelfAttention, l2: LayerNorm, m: MLP }
impl Block { 
    fn new(d: i64) -> Block { 
        Block(
            LayerNorm::new(d),           // l1
            CausalSelfAttention::new(d), // a
            LayerNorm::new(d),           // l2
            MLP::new(d)                  // m
        ) 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        let x = x + self.a.forward(self.l1.forward(x)); 
        x + self.m.forward(self.l2.forward(x)) 
    } 
    fn step(self, step_n: i64, lr: f32) -> Block { 
        let mut s = self; 
        s.a = s.a.step(step_n, lr); 
        s.l1 = s.l1.step(step_n, lr); 
        s.l2 = s.l2.step(step_n, lr); 
        s.m = s.m.step(step_n, lr); 
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
            (Tensor::randn([v, d], true)*0.1).detach(true), // w
            Tensor::zeros([v, d], false),                   // mw
            Tensor::zeros([v, d], false)                    // vw
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

// 2-Layer GPT
struct GPT { w: Embedding, wp: Embedding, b1: Block, b2: Block, l: LayerNorm, h: Linear }
impl GPT { 
    fn new(v: i64, d: i64) -> GPT { 
        GPT(
            Embedding::new(v, d),  // w
            Embedding::new(12, d), // wp
            Block::new(d),         // b1
            Block::new(d),         // b2
            LayerNorm::new(d),     // l
            Linear::new(d, v)      // h
        ) 
    } 
    fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> { 
        let pos_data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let pos = pos_data.reshape([1, 12]);
        let tok_emb = self.w.forward(i);
        let pos_emb = self.wp.forward(pos);
        let x = tok_emb + pos_emb;
        let x = self.b1.forward(x);
        let x = self.b2.forward(x);
        self.h.forward(self.l.forward(x)) 
    } 
    fn step(self, step_n: i64, lr: f32) -> GPT { 
        let mut s = self; 
        s.b1 = s.b1.step(step_n, lr); 
        s.b2 = s.b2.step(step_n, lr); 
        s.h = s.h.step(step_n, lr); 
        s.l = s.l.step(step_n, lr); 
        s.w = s.w.step(step_n, lr); 
        s.wp = s.wp.step(step_n, lr); 
        s 
    } 
}

// ---------------
// Training
// ---------------

fn get_random_digit() -> i64 {
    let r = Tensor::randn([1], true)[0];
    if r < -1.28 { return 0; }
    if r >= -1.28 && r < -0.84 { return 1; }
    if r >= -0.84 && r < -0.52 { return 2; }
    if r >= -0.52 && r < -0.25 { return 3; }
    if r >= -0.25 && r < 0.0   { return 4; }
    if r >= 0.0   && r < 0.25  { return 5; }
    if r >= 0.25  && r < 0.52  { return 6; }
    if r >= 0.52  && r < 0.84  { return 7; }
    if r >= 0.84  && r < 1.28  { return 8; }
    return 9;
}

struct TrainState {
    global_step: i64,
    model: GPT
}

fn train_epoch(state: TrainState, lr: f32, steps: i64) -> TrainState {
    let mut st = state;
    for s in range(0, steps) {
        let i = get_random_digit();
        let j = get_random_digit();
        
        let sum = i + j;
        let i_d10 = i / 10;
        let i_d1 = i - (i_d10 * 10);
        let j_d10 = j / 10;
        let j_d1 = j - (j_d10 * 10);
        let s_d100 = sum / 100;
        let rem = sum - (s_d100 * 100);
        let s_d10 = rem / 10;
        let s_d1 = rem - (s_d10 * 10);
        
        let v_i_d1 = i_d1 as f32;
        let v_i_d10 = i_d10 as f32;
        let v_j_d1 = j_d1 as f32;
        let v_j_d10 = j_d10 as f32;
        let v_s_d1 = s_d1 as f32;
        let v_s_d10 = s_d10 as f32;
        let v_s_d100 = s_d100 as f32;
        
        let data = [
            v_i_d1, v_i_d10, 10.0, 
            v_j_d1, v_j_d10, 11.0, 
            v_s_d1, v_s_d10, v_s_d100, 
            12.0, 12.0, 12.0
        ];
        
        let X = data.reshape([1, 12]);
        let logits = st.model.forward(X);
        let logits_flat = logits.reshape([12, 13]);

        // 位置5-7(答えのdigit)のみで損失計算
        let logits_ans = logits_flat.slice(0, 5, 8, 1);
        let Y_ans = [v_s_d1, v_s_d10, v_s_d100];
        let Y_flat = Y_ans.reshape([3]);
        let loss = logits_ans.cross_entropy(Y_flat);
        
        loss.backward();
        
        // grad_accum=8: 8サンプルごとに更新
        if s % 8 == 7 {
            st.model = st.model.step(st.global_step, lr);
            Tensor::clear_grads();
            st.global_step = st.global_step + 1;
        }
        
        if s % 100 == 0 {
             print("    Step:"); print(s);
             print("    Loss:"); loss.print();
        }
    }
    return st;
}

fn main() {
    let vocab_size = 13;
    let d_model = 128; 
    let mut state = TrainState(1, GPT::new(vocab_size, d_model));
    
    // 800 steps/epoch * 8 grad_accum = 100 updates/epoch
    // 100 epochs * 800 samples = 80000 total samples
    let epochs = 100;

    print("Training 1-digit addition (2-layer GPT, Adam, d=128, grad_accum=8)");

    for epoch in range(0, epochs) {
        print("Epoch:"); print(epoch); 
        // lr scheduling: warmup 5 epochs, then constant, then decay
        let mut lr = 0.0005;
        if epoch < 5 {
            // warmup: linearly increase from 0.0001 to 0.0005
            lr = 0.0001 + (epoch as f32) * 0.0001;
        }
        if epoch >= 70 {
            lr = 0.0001;
        }
        state = train_epoch(state, lr, 800);
    }
    
    print("Training Complete!");
    Param::save(state.model, "model_add.safetensors");
}
