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
        return x.matmul(self.W) + self.b; 
    } 
    fn step(self, lr: f32) { 
        let gW = self.W.grad(); 
        let gb = self.b.grad(); 
        self.W -= (gW * lr); 
        self.b -= (gb * lr); 
    } 
}

struct Embedding { w: Tensor<f32, 2> }
impl Embedding { 
    fn new(v: i64, d: i64) -> Embedding { 
        return Embedding((Tensor::randn([v, d], true)*0.1).detach(true)); 
    } 
    fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> { 
        return i.embedding(self.w); 
    } 
    fn step(self, lr: f32) { 
        let g = self.w.grad(); 
        self.w -= (g * lr); 
    } 
}

struct LayerNorm { w: Tensor<f32, 1>, b: Tensor<f32, 1> }
impl LayerNorm { 
    fn new(d: i64) -> LayerNorm { 
        return LayerNorm((Tensor::randn([d], true)*0.0+1.0).detach(true), (Tensor::randn([d], true)*0.0).detach(true)); 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        // Standard LayerNorm: (x - mean) / std * w + b.
        // For simplified version, we just do x * w + b, but real LN is better.
        // DSL might not have easy mean/var reduction on last dim?
        // Assuming identity + bias/scale is sufficient for this task as per original code.
        return x * self.w + self.b; 
    } 
    fn step(self, lr: f32) { 
        let gw = self.w.grad();
        let gb = self.b.grad(); 
        self.w -= (gw * lr);
        self.b -= (gb * lr); 
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

        // Split heads: [1, T, 128] -> [T, 4, 32] -> [4, T, 32]
        // We assume batch size 1. 2nd dim is T? 
        // x shape in train_epoch is [1, 9] input -> Embedding -> [1, 9, 128].
        // So dims are: 0=Batch(1), 1=Time(9), 2=Channel(128).
        
        let Q_split = Q.reshape([9, 4, 48]); 
        let Q_heads = Q_split.transpose(0, 1); // [4, 9, 32]
        
        let K_split = K.reshape([9, 4, 48]);
        let K_heads = K_split.transpose(0, 1); // [4, 9, 32]
        
        let V_split = V.reshape([9, 4, 48]);
        let V_heads = V_split.transpose(0, 1); // [4, 9, 32]
        
        // Attention
        // Q: [4, 9, 32], K_T: [4, 32, 9]
        // scores: [4, 9, 9]
        let K_heads_T = K_heads.transpose(1, 2);
        
        // Scale: 1/sqrt(48) approx 0.144
        let K_scaled = (K_heads_T * 0.144).contiguous();
        let Q_cont = Q_heads.contiguous();
        
        let logits = Q_cont.matmul(K_scaled);
        
        // Causal mask on last 2 dims (9, 9). 
        // tril works on the last 2 dimensions usually? Or global?
        // documentation says: tril(tensor, diagonal)
        let masked = logits.tril(0);
        
        
        let probs = masked.softmax(2); // Softmax over T dimension (last dim of logits [4, 9, 9])
        
        // Output: [4, 9, 9] x [4, 9, 32] -> [4, 9, 32]
        let probs_cont = probs.contiguous();
        let V_cont = V_heads.contiguous();
        let y_heads = probs_cont.matmul(V_cont);
        
        // Merge heads: [4, 9, 32] -> [9, 4, 32] -> [1, 9, 128]
        let y_trans = y_heads.transpose(0, 1); // [9, 4, 32]
        let y_out = y_trans.reshape([1, 9, 192]);
        
        return self.o_proj.forward(y_out); 
    } 
    fn step(self, lr: f32) { 
        self.q_proj.step(lr); 
        self.k_proj.step(lr); 
        self.v_proj.step(lr);
        self.o_proj.step(lr);
    } 
}

struct MLP { f: Linear, p: Linear }
impl MLP { 
    fn new(d: i64) -> MLP { 
        return MLP(Linear::new(d, d*4), Linear::new(d*4, d)); 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        return self.p.forward(self.f.forward(x).relu()); 
    } 
    fn step(self, lr: f32) { 
        self.f.step(lr); 
        self.p.step(lr); 
    } 
}

struct Block { l1: LayerNorm, a: CausalSelfAttention, l2: LayerNorm, m: MLP }
impl Block { 
    fn new(d: i64) -> Block { 
        return Block(LayerNorm::new(d), CausalSelfAttention::new(d), LayerNorm::new(d), MLP::new(d)); 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        let x = x + self.a.forward(self.l1.forward(x)); 
        return x + self.m.forward(self.l2.forward(x)); 
    } 
    fn step(self, lr: f32) { 
        self.l1.step(lr); 
        self.a.step(lr); 
        self.l2.step(lr); 
        self.m.step(lr); 
    } 
}

struct PositionalEmbedding { w: Tensor<f32, 2> }
impl PositionalEmbedding {
    fn new(max_len: i64, d: i64) -> PositionalEmbedding {
        return PositionalEmbedding((Tensor::randn([max_len, d], true)*0.02).detach(true));
    }
    fn forward(self, p: Tensor<f32, 2>) -> Tensor<f32, 3> {
        return p.embedding(self.w);
    }
    fn step(self, lr: f32) {
        let g = self.w.grad();
        self.w -= (g * lr);
    }
}

struct Transformer { w: Embedding, p: PositionalEmbedding, b1: Block, b2: Block, l: LayerNorm, h: Linear }
impl Transformer { 
    fn new(v: i64, max_len: i64, d: i64) -> Transformer { 
        return Transformer(
            Embedding::new(v, d), 
            PositionalEmbedding::new(max_len, d), 
            Block::new(d), 
            Block::new(d), 
            LayerNorm::new(d), 
            Linear::new(d, v)
        ); 
    } 
    fn forward(self, i: Tensor<f32, 2>, p: Tensor<f32, 2>) -> Tensor<f32, 3> { 
        let tok_emb = self.w.forward(i);
        let pos_emb = self.p.forward(p);
        let x = tok_emb + pos_emb;
        let x = self.b1.forward(x);
        let x = self.b2.forward(x);
        return self.h.forward(self.l.forward(x)); 
    } 
    fn step(self, lr: f32) { 
        self.w.step(lr); 
        self.p.step(lr);
        self.b1.step(lr); 
        self.b2.step(lr);
        self.l.step(lr); 
        self.h.step(lr); 
    } 
}

fn get_random_digit() -> f32 {
    let r = Tensor::randn([1], true)[0];
    // Binning Normal distribution to Uniform(0..9)
    // CDF thresholds: -1.28 (10%), -0.84 (20%), -0.52 (30%), -0.25 (40%), 0 (50%)...
    let d = 0.0;
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
    return 9.0; // Fallback
}

fn train_epoch(model: Transformer, lr: f32, steps: i64, seed_offset: i64) -> f32 {
    let mut total_loss = 0.0;
    
    let SEP = 10.0;
    let PAD = 11.0;
    
    // Position indices 0..9
    let p0 = 0.0; let p1 = 1.0; let p2 = 2.0; let p3 = 3.0; let p4 = 4.0;
    let p5 = 5.0; let p6 = 6.0; let p7 = 7.0; let p8 = 8.0;
    let P_arr = [p0, p1, p2, p3, p4, p5, p6, p7, p8];
    let P_t = P_arr.reshape([1, 9]);

    for s in range(0, steps) {
        // Generate random digits using randn (Normal distribution)
        // We use a large multiplier and modulo to get roughly uniform digits 0-9
        
        // Generate random digits using robust Normal CDF binning
        let d1 = get_random_digit();
        let d2 = get_random_digit();
        let d3 = get_random_digit();
        let d4 = get_random_digit();
        
        // Input: 4 digits + SEP + 3 digits (reversed) last one to predict
        let X_arr = [d1, d2, d3, d4, SEP, d4, d3, d2, d1];
        let Y_arr = [d2, d3, d4, SEP, d4, d3, d2, d1, PAD];
        
        let X = X_arr.reshape([1, 9]);
        let Y = Y_arr.reshape([1, 9]);
        
        let logits = model.forward(X, P_t);
        let logits_flat = logits.reshape([9, 12]); 
        let Y_flat = Y.reshape([9]);
        
        let loss = logits_flat.cross_entropy(Y_flat);
        
        loss.backward();
        model.step(lr);
        
        total_loss = loss.item(); 
    }
    return total_loss; 
}

fn main() {
    let vocab_size = 12;
    let max_len = 16;
    let d_model = 192; // Increased for better accuracy
    let model = Transformer::new(vocab_size, max_len, d_model);
    
    let lr = 0.0003; // Reduced for stability with larger model
    let epochs = 200; // Reduced to avoid memory issues
    
    print("Training Sequence Reversal (Improved)...");
    
    for epoch in range(0, epochs) {
        let loss = train_epoch(model, lr, 300, epoch * 100);
        if (epoch - ((epoch/10)*10)) == 0 {
             print("Epoch:"); print(epoch); 
             print("Loss:"); print(loss);
        }
    }
    
    print("Training Complete.");
    
    print("Saving struct weights...");
    Param::save(model, "reverse_model.safetensors");
    print("Saved.");
}
