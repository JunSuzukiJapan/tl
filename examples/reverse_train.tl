// Sequence Reversal Task
// Input:  [d1, d2, d3, SEP, d3, d2, d1]
// Model outputs next token.

struct Linear { W: Tensor<f32, 2>, b: Tensor<f32, 1> }
impl Linear { 
    fn new(i: i64, o: i64) -> Linear { 
        return Linear((Tensor::randn([i, o], true)*0.1).detach(true), (Tensor::randn([o], true)*0.0).detach(true)); 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        return matmul(x, self.W) + self.b; 
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
        return embedding(i, self.w); 
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
        return x + self.b; // Simplified LN (no div by var for stability in toy example or full LN)
        // Note: Real LN requires mean/var calculation. For now using simplified bias/scale.
        // Actually, let's just stick to identity+bias for simplicity unless we need real LN.
    } 
    fn step(self, lr: f32) { 
        let gb = self.b.grad(); 
        self.b -= (gb * lr); 
    } 
}

struct CausalSelfAttention { a: Linear, p: Linear }
impl CausalSelfAttention { 
    fn new(d: i64) -> CausalSelfAttention { 
        return CausalSelfAttention(Linear::new(d, d*3), Linear::new(d*3, d)); 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        let qkv = self.a.forward(x); 
        // Simply split logic isn't here, assuming simplified single head attention mechanism or just dense
        // Reusing train_add logic which seemed to be a dense approximation or specific hack.
        // Let's implement a standard single head attention if possible, or copied from train_add.
        // train_add: let q=qkv; let k=qkv; let v=qkv; ...
        let q = qkv; 
        let k = qkv; 
        let v = qkv; 
        let logits = matmul(q, transpose(k, 1, 2)) * 0.125;
        let masked = tril(logits, 0); // Causal mask
        let probs = softmax(masked, 2);
        let y = matmul(probs, v); 
        return self.p.forward(y); 
    } 
    fn step(self, lr: f32) { 
        self.a.step(lr); 
        self.p.step(lr); 
    } 
}

struct MLP { f: Linear, p: Linear }
impl MLP { 
    fn new(d: i64) -> MLP { 
        return MLP(Linear::new(d, d*4), Linear::new(d*4, d)); 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        return self.p.forward(relu(self.f.forward(x))); 
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
        return embedding(p, self.w);
    }
    fn step(self, lr: f32) {
        let g = self.w.grad();
        self.w -= (g * lr);
    }
}

struct Transformer { w: Embedding, p: PositionalEmbedding, b: Block, l: LayerNorm, h: Linear }
impl Transformer { 
    fn new(v: i64, max_len: i64, d: i64) -> Transformer { 
        return Transformer(Embedding::new(v, d), PositionalEmbedding::new(max_len, d), Block::new(d), LayerNorm::new(d), Linear::new(d, v)); 
    } 
    fn forward(self, i: Tensor<f32, 2>, p: Tensor<f32, 2>) -> Tensor<f32, 3> { 
        let tok_emb = self.w.forward(i);
        let pos_emb = self.p.forward(p);
        return self.h.forward(self.l.forward(self.b.forward(tok_emb + pos_emb))); 
    } 
    fn step(self, lr: f32) { 
        self.w.step(lr); 
        self.p.step(lr);
        self.b.step(lr); 
        self.l.step(lr); 
        self.h.step(lr); 
    } 
}

fn train_epoch(model: Transformer, lr: f32, steps: i64, seed_offset: i64) -> f32 {
    let total_loss = 0.0;
    
    let SEP = 10.0;
    let PAD = 11.0;
    let MAX_DIGITS = 4; 

    // Fixed position indices for length 9: 0..8
    // But we padded to 10? Data array in code below has 10 elements.
    // "let data = ... (size 10)"
    // Indices 0..9.
    
    // Construct pos array manually as I64? Or F32? Embedding expects F32 usually in this DSL context?
    // train_add used f32 floats for embedding input.
    let p0 = 0.0; let p1 = 1.0; let p2 = 2.0; let p3 = 3.0; let p4 = 4.0;
    let p5 = 5.0; let p6 = 6.0; let p7 = 7.0; let p8 = 8.0;
    
    // We used length 9 for X logic below (X_arr).
    // let X_arr = [d1, d2, d3, d4, SEP, d4, d3, d2, d1]; (Size 9)
    let P_arr = [p0, p1, p2, p3, p4, p5, p6, p7, p8];
    let P_t = reshape(P_arr, 1, 9);

    for s in range(0, steps) {
        let r = s + seed_offset; // Initial seed
        
        // Generate d1
        let d1_val = (r * 7 + 3) - (((r * 7 + 3) / 10) * 10);
        let d1 = pow(d1_val, 1.0).get(0);
        
        // Advance r mixed
        let r = r * 17 + 11;

        let d2_val = (r * 13 + 5) - (((r * 13 + 5) / 10) * 10);
        let d2 = pow(d2_val, 1.0).get(0);
        
        let r = r * 19 + 7;

        let d3_val = (r * 23 + 9) - (((r * 23 + 9) / 10) * 10);
        let d3 = pow(d3_val, 1.0).get(0);
        
        let r = r * 29 + 3;

        let d4_val = (r * 31 + 1) - (((r * 31 + 1) / 10) * 10);
        let d4 = pow(d4_val, 1.0).get(0);
        
        let X_arr = [d1, d2, d3, d4, SEP, d4, d3, d2, d1];
        let Y_arr = [d2, d3, d4, SEP, d4, d3, d2, d1, PAD];
        
        let X = reshape(X_arr, 1, 9);
        let Y = reshape(Y_arr, 1, 9);
        
        let logits = model.forward(X, P_t);
        let logits_flat = reshape(logits, 9, 12); 
        let Y_flat = reshape(Y, 9);
        
        let loss = cross_entropy(logits_flat, Y_flat);
        
        backward(loss);
        model.step(lr);
        
        total_loss = loss.get(0); 
    }
    return total_loss; 
}

fn main() {
    let vocab_size = 12;
    let max_len = 16;
    let d_model = 64;    
    let model = Transformer::new(vocab_size, max_len, d_model);
    
    let lr = 0.01;
    let epochs = 35;
    
    print("Training Sequence Reversal with PosEmb...");
    
    for epoch in range(0, epochs) {
        let loss = train_epoch(model, lr, 200, epoch * 100);
        if (epoch - ((epoch/5)*5)) == 0 {
             print("Epoch:"); print(epoch); 
             print("Loss:"); print(loss);
        }
    }
    
    print("Training Complete.");
    
    print("Saving struct weights...");
    save_weights(model, "reverse_model.safetensors");
    print("Saved.");
}
