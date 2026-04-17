// Sequence Reversal Task
// Input:  [d1, d2, d3, SEP, d3, d2, d1]
// Model outputs next token.

struct Linear { W: Tensor<f32, 2>, b: Tensor<f32, 1> }
impl Linear { 
    fn new(i: i64, o: i64) -> Linear { 
        // Xavier initialization-ish
        return Linear((Tensor::randn([i, o], true)*0.1).detach(true), (Tensor::randn([o], true)*0.0).detach(true)); 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        x.matmul(self.W) + self.b 
    } 
    fn step(self, lr: f32) -> Linear { 
        let mut s = self;
        let gW = s.W.grad(); 
        let gb = s.b.grad(); 
        s.W = (s.W - gW * lr).detach(true); 
        s.b = (s.b - gb * lr).detach(true); 
        s
    } 
}

struct Embedding { w: Tensor<f32, 2> }
impl Embedding { 
    fn new(v: i64, d: i64) -> Embedding { 
        Embedding((Tensor::randn([v, d], true)*0.1).detach(true)) 
    } 
    fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> { 
        i.embedding(self.w) 
    } 
    fn step(self, lr: f32) -> Embedding { 
        let mut s = self;
        let g = s.w.grad(); 
        s.w = (s.w - g * lr).detach(true); 
        s
    } 
}

struct LayerNorm { w: Tensor<f32, 1>, b: Tensor<f32, 1> }
impl LayerNorm { 
    fn new(d: i64) -> LayerNorm { 
        LayerNorm((Tensor::randn([d], true)*0.0+1.0).detach(true), (Tensor::randn([d], true)*0.0).detach(true)) 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        return x * self.w + self.b; 
    } 
    fn step(self, lr: f32) -> LayerNorm { 
        let mut s = self;
        let gw = s.w.grad();
        let gb = s.b.grad(); 
        s.w = (s.w - gw * lr).detach(true);
        s.b = (s.b - gb * lr).detach(true); 
        s
    } 
}

struct CausalSelfAttention { q_proj: Linear, k_proj: Linear, v_proj: Linear, o_proj: Linear }
impl CausalSelfAttention { 
    fn new(d: i64) -> CausalSelfAttention { 
        // d_model=128. 4 heads -> 32 dim per head.
        return CausalSelfAttention(Linear::new(d, d), Linear::new(d, d), Linear::new(d, d), Linear::new(d, d)); 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        // x: [1, T, 128]
        let Q = self.q_proj.forward(x); // [1, T, 128]
        let K = self.k_proj.forward(x);
        let V = self.v_proj.forward(x);

        let Q_split = Q.reshape([9, 4, 32]); 
        let Q_heads = Q_split.transpose(0, 1); // [4, 9, 32]
        
        let K_split = K.reshape([9, 4, 32]);
        let K_heads = K_split.transpose(0, 1); // [4, 9, 32]
        
        let V_split = V.reshape([9, 4, 32]);
        let V_heads = V_split.transpose(0, 1); // [4, 9, 32]
        
        let K_heads_T = K_heads.transpose(1, 2);
        
        // Scale: 1/sqrt(32) = 0.176
        let K_scaled = (K_heads_T * 0.176).contiguous();
        let Q_cont = Q_heads.contiguous();
        
        // Softmax normalization requires -Infinity for masking, but tril(0) gives 0.0
        let logits = Q_cont.matmul(K_scaled);
        // Create an algebraic mask: 0.0 for lower triangle, -10000.0 for upper triangle
        let ones = logits * 0.0 + 1.0;
        let causal_mask = ones.tril(0); 
        let neg_inf_mask = (causal_mask - 1.0) * 10000.0; 
        let masked = logits + neg_inf_mask;
        
        let probs = masked.softmax(2); 
        
        let probs_cont = probs.contiguous();
        let V_cont = V_heads.contiguous();
        let y_heads = probs_cont.matmul(V_cont);
        
        let y_trans = y_heads.transpose(0, 1); // [9, 4, 32]
        let y_out = y_trans.reshape([1, 9, 128]);
        
        self.o_proj.forward(y_out) 
    } 
    fn step(self, lr: f32) -> CausalSelfAttention { 
        let mut s = self;
        s.q_proj = s.q_proj.step(lr); 
        s.k_proj = s.k_proj.step(lr); 
        s.v_proj = s.v_proj.step(lr);
        s.o_proj = s.o_proj.step(lr);
        s
    } 
}

struct MLP { f: Linear, p: Linear }
impl MLP { 
    fn new(d: i64) -> MLP { 
        MLP(Linear::new(d, d*4), Linear::new(d*4, d)) 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        self.p.forward(self.f.forward(x).relu()) 
    } 
    fn step(self, lr: f32) -> MLP { 
        let mut s = self;
        s.f = s.f.step(lr); 
        s.p = s.p.step(lr); 
        s
    } 
}

struct Block { l1: LayerNorm, a: CausalSelfAttention, l2: LayerNorm, m: MLP }
impl Block { 
    fn new(d: i64) -> Block { 
        Block(LayerNorm::new(d), CausalSelfAttention::new(d), LayerNorm::new(d), MLP::new(d)) 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        let x = x + self.a.forward(self.l1.forward(x)); 
        x + self.m.forward(self.l2.forward(x)) 
    } 
    fn step(self, lr: f32) -> Block { 
        let mut s = self;
        s.l1 = s.l1.step(lr); 
        s.a = s.a.step(lr); 
        s.l2 = s.l2.step(lr); 
        s.m = s.m.step(lr); 
        s
    } 
}

struct PositionalEmbedding { w: Tensor<f32, 2> }
impl PositionalEmbedding {
    fn new(max_len: i64, d: i64) -> PositionalEmbedding {
        PositionalEmbedding((Tensor::randn([max_len, d], true)*0.02).detach(true))
    }
    fn forward(self, p: Tensor<f32, 2>) -> Tensor<f32, 3> {
        p.embedding(self.w)
    }
    fn step(self, lr: f32) -> PositionalEmbedding {
        let mut s = self;
        let g = s.w.grad();
        s.w = (s.w - g * lr).detach(true);
        s
    }
}

struct Transformer { w: Embedding, p: PositionalEmbedding, b1: Block, b2: Block, l: LayerNorm, h: Linear }
impl Transformer { 
    fn new(v: i64, max_len: i64, d: i64) -> Transformer { 
        Transformer(
            Embedding::new(v, d), 
            PositionalEmbedding::new(max_len, d), 
            Block::new(d), 
            Block::new(d), 
            LayerNorm::new(d), 
            Linear::new(d, v)
        ) 
    } 
    fn forward(self, i: Tensor<f32, 2>, p: Tensor<f32, 2>) -> Tensor<f32, 3> { 
        let tok_emb = self.w.forward(i);
        let pos_emb = self.p.forward(p);
        let x = tok_emb + pos_emb;
        let x = self.b1.forward(x);
        let x = self.b2.forward(x);
        self.h.forward(self.l.forward(x)) 
    } 
    fn step(self, lr: f32) -> Transformer { 
        let mut s = self;
        s.w = s.w.step(lr); 
        s.p = s.p.step(lr);
        s.b1 = s.b1.step(lr); 
        s.b2 = s.b2.step(lr);
        s.l = s.l.step(lr); 
        s.h = s.h.step(lr); 
        s
    } 
}

fn get_random_digit() -> f32 {
    let r = Tensor::randn([1], true)[0];
    if r < -1.28 { return 0.0; }
    if r >= -1.28 && r < -0.84 { return 1.0; }
    if r >= -0.84 && r < -0.52 { return 2.0; }
    if r >= -0.52 && r < -0.25 { return 3.0; }
    if r >= -0.25 && r < 0.0   { return 4.0; }
    if r >= 0.0   && r < 0.25  { return 5.0; }
    if r >= 0.25  && r < 0.52  { return 6.0; }
    if r >= 0.52  && r < 0.84  { return 7.0; }
    if r >= 0.84  && r < 1.28  { return 8.0; }
    if r >= 1.28 { return 9.0; }
    return 9.0;
}

fn train_epoch(model: Transformer, lr: f32, steps: i64) -> Transformer {
    let mut m = model;
    let SEP = 10.0;
    let PAD = 11.0;
    let p0 = 0.0; let p1 = 1.0; let p2 = 2.0; let p3 = 3.0; let p4 = 4.0;
    let p5 = 5.0; let p6 = 6.0; let p7 = 7.0; let p8 = 8.0;
    let P_arr = [p0, p1, p2, p3, p4, p5, p6, p7, p8];
    let P_t = P_arr.reshape([1, 9]);

    for s in range(0, steps) {
        let d1 = get_random_digit();
        let d2 = get_random_digit();
        let d3 = get_random_digit();
        let d4 = get_random_digit();
        
        // Input: [d1, d2, d3, d4, SEP, d4, d3, d2, d1]
        // Target: [d2, d3, d4, SEP, d4, d3, d2, d1, PAD]
        let X_arr = [d1, d2, d3, d4, SEP, d4, d3, d2, d1];
        let Y_arr = [d2, d3, d4, SEP, d4, d3, d2, d1, PAD];
        let X = X_arr.reshape([1, 9]);
        let Y = Y_arr.reshape([1, 9]);
        
        let logits = m.forward(X, P_t);
        let logits_flat = logits.reshape([9, 12]); 
        let Y_flat = Y.reshape([9]);
        let loss = logits_flat.cross_entropy(Y_flat);
        
        loss.backward();
        
        if s % 8 == 7 {
            m = m.step(lr * 8.0); // Scale LR by batch size
            Tensor::clear_grads();
        }
        
        if s % 50 == 0 {
             print("    Step:"); print(s);
             print("    Loss:"); loss.print();
        }
    }
    return m;
}

fn main() {
    let vocab_size = 12;
    let max_len = 16;
    let d_model = 128; 
    let mut model = Transformer::new(vocab_size, max_len, d_model);
    
    let lr = 0.001;
    let epochs = 400;

    print("Training Sequence Reversal (Generalized)...");

    for epoch in range(0, epochs) {
        print("Epoch:"); print(epoch); 
        model = train_epoch(model, lr, 200);
    }
    print("Training Complete.");
    
    print("Saving struct weights...");
    Param::save(model, "reverse_model.safetensors");
    print("Saved.");
}
