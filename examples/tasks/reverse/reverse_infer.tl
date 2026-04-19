// Sequence Reversal 推論 (reverse_train.tl で学習したモデルを使用)
// Sequence Reversal Inference

struct Linear { 
    W: Tensor<f32, 2>, b: Tensor<f32, 1>,
    mW: Tensor<f32, 2>, vW: Tensor<f32, 2>,
    mb: Tensor<f32, 1>, vb: Tensor<f32, 1>
}
impl Linear { 
    fn new(i: i64, o: i64) -> Linear { 
        Linear(
            (Tensor::randn([i, o], true)*0.1).detach(true), 
            (Tensor::randn([o], true)*0.0).detach(true),
            Tensor::zeros([i, o], false), Tensor::zeros([i, o], false),
            Tensor::zeros([o], false), Tensor::zeros([o], false)
        ) 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        x.matmul(self.W) + self.b 
    } 
}

struct Embedding { 
    w: Tensor<f32, 2>,
    mw: Tensor<f32, 2>, vw: Tensor<f32, 2>
}
impl Embedding { 
    fn new(v: i64, d: i64) -> Embedding { 
        Embedding(
            (Tensor::randn([v, d], true)*0.02).detach(true), 
            Tensor::zeros([v, d], false), Tensor::zeros([v, d], false)
        ) 
    } 
    fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> { 
        i.embedding(self.w) 
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
}

struct CausalSelfAttention { q_proj: Linear, k_proj: Linear, v_proj: Linear, o_proj: Linear }
impl CausalSelfAttention { 
    fn new(d: i64) -> CausalSelfAttention { 
        CausalSelfAttention(Linear::new(d, d), Linear::new(d, d), Linear::new(d, d), Linear::new(d, d)) 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        let Q = self.q_proj.forward(x); 
        let K = self.k_proj.forward(x);
        let V = self.v_proj.forward(x);

        let Q_split = Q.reshape([9, 4, 32]);
        let Q_heads = Q_split.transpose(0, 1);

        let K_split = K.reshape([9, 4, 32]);
        let K_heads = K_split.transpose(0, 1);

        let V_split = V.reshape([9, 4, 32]);
        let V_heads = V_split.transpose(0, 1); 
        
        let K_heads_T = K_heads.transpose(1, 2);
        let K_scaled = (K_heads_T * 0.176).contiguous();
        let Q_cont = Q_heads.contiguous();
        
        let logits = Q_cont.matmul(K_scaled);
        let ones = logits * 0.0 + 1.0;
        let causal_mask = ones.tril(0); 
        let neg_inf_mask = (causal_mask - 1.0) * 10000.0; 
        let masked = logits + neg_inf_mask;
        
        let probs = masked.softmax(2); 
        let probs_cont = probs.contiguous();
        let V_cont = V_heads.contiguous();
        let y_heads = probs_cont.matmul(V_cont);
        
        let y_trans = y_heads.transpose(0, 1); 
        let y_out = y_trans.reshape([1, 9, 128]);
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

struct PositionalEmbedding { 
    w: Tensor<f32, 2>,
    mw: Tensor<f32, 2>, vw: Tensor<f32, 2>
}
impl PositionalEmbedding {
    fn new(max_len: i64, d: i64) -> PositionalEmbedding {
        PositionalEmbedding(
            (Tensor::randn([max_len, d], true)*0.02).detach(true),
            Tensor::zeros([max_len, d], false), Tensor::zeros([max_len, d], false)
        )
    }
    fn forward(self, p: Tensor<f32, 2>) -> Tensor<f32, 3> {
        p.embedding(self.w)
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

fn inference(model: Transformer, tests: i64) {
    let SEP = 10.0;
    let PAD = 11.0;
    
    // Positions 0..8 (seq_len=9, matching training)
    let p0 = 0.0; let p1 = 1.0; let p2 = 2.0; let p3 = 3.0; let p4 = 4.0;
    let p5 = 5.0; let p6 = 6.0; let p7 = 7.0; let p8 = 8.0;
    let P_arr = [p0, p1, p2, p3, p4, p5, p6, p7, p8];
    let P_t = P_arr.reshape([1, 9]);

    let mut correct_count = 0;

    for t in range(0, tests) {
        let d1 = get_random_digit();
        let d2 = get_random_digit();
        let d3 = get_random_digit();
        let d4 = get_random_digit();
        
        let input = [d1, d2, d3, d4, SEP, d4, d3, d2, d1];
        print("======== Test Case ========");
        print("Input (d1..d4):");
        print([d1, d2, d3, d4]);

        let X = input.reshape([1, 9]);
        let logits = model.forward(X, P_t); // [1, 9, Vocab]
        let logits_flat = logits.reshape([9, 12]);
        
        let p4_logits = logits_flat.slice(0, 4, 5, 1).reshape([12]);
        let p5_logits = logits_flat.slice(0, 5, 6, 1).reshape([12]);
        let p6_logits = logits_flat.slice(0, 6, 7, 1).reshape([12]);
        let p7_logits = logits_flat.slice(0, 7, 8, 1).reshape([12]);
        
        let pred_4 = p4_logits.argmax(0)[0];
        let pred_5 = p5_logits.argmax(0)[0];
        let pred_6 = p6_logits.argmax(0)[0];
        let pred_7 = p7_logits.argmax(0)[0];
        
        print("Expected Output (reversed):");
        print([d4, d3, d2, d1]);
        
        print("Model Predictions (positions 4,5,6,7):");
        print([pred_4, pred_5, pred_6, pred_7]);
        
        let d4_i = d4 as i64;
        let d3_i = d3 as i64;
        let d2_i = d2 as i64;
        let d1_i = d1 as i64;
        
        if pred_4 == d4_i && pred_5 == d3_i && pred_6 == d2_i && pred_7 == d1_i {
            print("Result: HIT");
            correct_count = correct_count + 1;
        } else {
            print("Result: MISS");
        }
    }

    print("Inference Complete.");
    print("Accuracy:");
    print(correct_count);
    print("Correct out of");
    print(tests);
}

fn main() {
    let vocab_size = 12;
    let max_len = 16;
    let d_model = 128;
    let model = Transformer::new(vocab_size, max_len, d_model);
    
    print("Loading weights...");
    Param::load(model, "reverse_model.safetensors");
    print("Weights loaded.");
    
    inference(model, 5);
}
