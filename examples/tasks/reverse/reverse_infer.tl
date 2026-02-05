// SKIP: Requires external resources or model files
// Sequence Reversal Inference
//
// Matches improved architecture instructions.

struct Linear { W: Tensor<f32, 2>, b: Tensor<f32, 1> }
impl Linear { 
    fn new(i: i64, o: i64) -> Linear { 
        Linear((Tensor::randn([i, o], true)*0.1).detach(), (Tensor::randn([o], true)*0.0).detach()) 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        x.matmul(self.W) + self.b 
    } 
}

struct Embedding { w: Tensor<f32, 2> }
impl Embedding { 
    fn new(v: i64, d: i64) -> Embedding { 
        Embedding((Tensor::randn([v, d], true)*0.1).detach()) 
    } 
    fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> { 
        i.embedding(self.w) 
    } 
}

struct LayerNorm { w: Tensor<f32, 1>, b: Tensor<f32, 1> }
impl LayerNorm { 
    fn new(d: i64) -> LayerNorm { 
        LayerNorm((Tensor::randn([d], true)*0.0+1.0).detach(), (Tensor::randn([d], true)*0.0).detach()) 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        x * self.w + self.b 
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
        
        let Q_split = Q.reshape([10, 4, 48]);  // NOTE: infer uses len 10
        let Q_heads = Q_split.transpose(0, 1); // [4, 10, 32]
        
        let K_split = K.reshape([10, 4, 48]);
        let K_heads = K_split.transpose(0, 1); // [4, 10, 32]
        
        let V_split = V.reshape([10, 4, 48]);
        let V_heads = V_split.transpose(0, 1); // [4, 10, 32]
        
        // Attention
        let K_heads_T = K_heads.transpose(1, 2);
        
        // Scale: 1/sqrt(48) approx 0.144
        let K_scaled = (K_heads_T * 0.144).contiguous();
        let Q_cont = Q_heads.contiguous();
        let logits = Q_cont.matmul(K_scaled);
        
        let masked = logits.tril(0);
        let probs = masked.softmax(2); 
        let probs_cont = probs.contiguous();
        let V_cont = V_heads.contiguous();
        let y_heads = probs_cont.matmul(V_cont);
        
        // Merge heads
        let y_trans = y_heads.transpose(0, 1); // [10, 4, 32]
        let y_out = y_trans.reshape([1, 10, 192]);
        
        self.o_proj.forward(y_out) 
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
}

struct PositionalEmbedding { w: Tensor<f32, 2> }
impl PositionalEmbedding {
    fn new(max_len: i64, d: i64) -> PositionalEmbedding {
        PositionalEmbedding((Tensor::randn([max_len, d], true)*0.02).detach())
    }
    fn forward(self, p: Tensor<f32, 2>) -> Tensor<f32, 3> {
        p.embedding(self.w)
    }
}

struct SeqModel { w: Embedding, p: PositionalEmbedding, b1: Block, b2: Block, l: LayerNorm, h: Linear }
impl SeqModel { 
    fn new(v: i64, max_len: i64, d: i64) -> SeqModel { 
        SeqModel(
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
}

fn inference(model: SeqModel) {
    let SEP = 10.0;
    let PAD = 11.0;
    
    // Test case: 1 2 3 4 -> Expect 4 3 2 1
    
    let d1 = 1.0;
    let d2 = 2.0;
    let d3 = 3.0;
    let d4 = 4.0;
    
    let input = [d1, d2, d3, d4, SEP, d4, d3, d2, d1, PAD];
    print("Full Context Input:");
    print(input);

    // Positions 0..9
    let p0 = 0.0; let p1 = 1.0; let p2 = 2.0; let p3 = 3.0; let p4 = 4.0;
    let p5 = 5.0; let p6 = 6.0; let p7 = 7.0; let p8 = 8.0; let p9 = 9.0;
    
    let P_arr = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9];
    let P_t = P_arr.reshape([1, 10]);

    let X = input.reshape([1, 10]);
    let logits = model.forward(X, P_t); // [1, 10, Vocab]
    let logits_flat = logits.reshape([10, 12]);
    
    print("Checking predictions...");
    
    // Step 1: Pos 4 -> Expect 4
    let l_step1 = logits_flat.slice(4, 1);
    print("Step 1 (Pos 4) -> Expect '4':");
    print(l_step1);

    // Step 2: Pos 5 -> Expect 3
    let l_step2 = logits_flat.slice(5, 1);
    print("Step 2 (Pos 5) -> Expect '3':");
    print(l_step2);

    // Step 3: Pos 6 -> Expect 2
    let l_step3 = logits_flat.slice(6, 1);
    print("Step 3 (Pos 6) -> Expect '2':");
    print(l_step3);

    // Step 4: Pos 7 -> Expect 1
    let l_step4 = logits_flat.slice(7, 1);
    print("Step 4 (Pos 7) -> Expect '1':");
    print(l_step4);
}

fn main() {
    let vocab_size = 12;
    let d_model = 192;
    let max_len = 16;
    let model = SeqModel::new(vocab_size, max_len, d_model);
    
    print("Loading weights...");
    Param::load("reverse_model.safetensors");
    print("Weights loaded.");
    
    inference(model);
}
