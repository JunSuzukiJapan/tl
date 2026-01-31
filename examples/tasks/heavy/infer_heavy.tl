fn argmax(t: Tensor<f32, 1>) -> i64 {
    let mut max_val = -1000000.0;
    let mut idx = 0;
    for k in range(0, 13) {
        let v = t[k];
        if v > max_val {
            max_val = v;
            idx = k;
        }
    }
    return idx;
}

// 4層GPT (train_heavy.tlと同じ定義)
struct Linear { W: Tensor<f32, 2>, b: Tensor<f32, 1> }
impl Linear { fn new(i: i64, o: i64) -> Linear { Linear((Tensor::randn([i, o], true)*0.1).detach(), (Tensor::randn([o], true)*0.0).detach()) } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { x.matmul(self.W) + self.b } }
struct Embedding { w: Tensor<f32, 2> }
impl Embedding { fn new(v: i64, d: i64) -> Embedding { Embedding((Tensor::randn([v, d], true)*0.1).detach()) } fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> { i.embedding(self.w) } }
struct LayerNorm { w: Tensor<f32, 1>, b: Tensor<f32, 1> }
impl LayerNorm { fn new(d: i64) -> LayerNorm { LayerNorm((Tensor::randn([d], true)*0.0+1.0).detach(), (Tensor::randn([d], true)*0.0).detach()) } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { x + self.b } }
struct CausalSelfAttention { q_proj: Linear, k_proj: Linear, v_proj: Linear, p_proj: Linear }
impl CausalSelfAttention { fn new(d: i64) -> CausalSelfAttention { CausalSelfAttention(Linear::new(d, d), Linear::new(d, d), Linear::new(d, d), Linear::new(d, d)) } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { let q = self.q_proj.forward(x); let k = self.k_proj.forward(x); let v = self.v_proj.forward(x); let k_t = k.transpose(1, 2); let qk = q.matmul(k_t) * 0.05103; let masked = qk.tril(0); let attn = masked.softmax(2); let y = attn.matmul(v); self.p_proj.forward(y) } }
struct MLP { f: Linear, p: Linear }
impl MLP { fn new(d: i64) -> MLP { MLP(Linear::new(d, d*4), Linear::new(d*4, d)) } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { self.p.forward(self.f.forward(x).relu()) } }
struct Block { l1: LayerNorm, a: CausalSelfAttention, l2: LayerNorm, m: MLP }
impl Block { fn new(d: i64) -> Block { Block(LayerNorm::new(d), CausalSelfAttention::new(d), LayerNorm::new(d), MLP::new(d)) } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { let x = x + self.a.forward(self.l1.forward(x)); x + self.m.forward(self.l2.forward(x)) } }

// 4-Layer GPT
struct GPTHeavy { w: Embedding, wp: Embedding, b1: Block, b2: Block, b3: Block, b4: Block, l: LayerNorm, h: Linear }
impl GPTHeavy { 
    fn new(v: i64, d: i64) -> GPTHeavy { 
        GPTHeavy(
            Embedding::new(v, d), Embedding::new(12, d), 
            Block::new(d), Block::new(d), Block::new(d), Block::new(d),
            LayerNorm::new(d), Linear::new(d, v)
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
        let x = self.b3.forward(x);
        let x = self.b4.forward(x);
        self.h.forward(self.l.forward(x)) 
    } 
}

fn main() {
    let vocab_size = 13;
    let d_model = 384; 
    
    print("Inference with Heavy Model (4-Layers, d_model=384)");
    let model = GPTHeavy::new(vocab_size, d_model);
    
    Param::load_all("model_heavy.safetensors");
    print("Parameters Loaded from model_heavy.safetensors");
    
    // Autoregressive Inference Logic using PAD=12
    
    // Test 12 + 34 = 46 (Reverse: 6, 4, 0)
    print("Test: 12 + 34 = 46");
    print("Expected (Reverse): 6, 4, 0");
    
    let data1 = [2.0, 1.0, 10.0, 4.0, 3.0, 11.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0];
    let input1 = data1.reshape([1, 12]);
    let logits1 = model.forward(input1);
    let logits1_flat = logits1.reshape([12, 13]);
    let p1 = argmax(logits1_flat.slice(5, 1).reshape([13]));
    
    let v1 = p1 as f32;
    let data2 = [2.0, 1.0, 10.0, 4.0, 3.0, 11.0, v1, 12.0, 12.0, 12.0, 12.0, 12.0];
    let input2 = data2.reshape([1, 12]);
    let logits2 = model.forward(input2);
    let logits2_flat = logits2.reshape([12, 13]);
    let p2 = argmax(logits2_flat.slice(6, 1).reshape([13]));
    
    let v2 = p2 as f32;
    let data3 = [2.0, 1.0, 10.0, 4.0, 3.0, 11.0, v1, v2, 12.0, 12.0, 12.0, 12.0];
    let input3 = data3.reshape([1, 12]);
    let logits3 = model.forward(input3);
    let logits3_flat = logits3.reshape([12, 13]);
    let p3 = argmax(logits3_flat.slice(7, 1).reshape([13]));
    
    print("Predicted:"); print(p1); print(p2); print(p3);
    
    // Test 99 + 1 = 100 (Reverse: 0, 0, 1)
    print("Test: 99 + 1 = 100");
    print("Expected (Reverse): 0, 0, 1");
    // 9, 9, 10, 1, 0, 11 (Reverse)
    let q_data1 = [9.0, 9.0, 10.0, 1.0, 0.0, 11.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0];
    let q_input1 = q_data1.reshape([1, 12]);
    let q_logits1 = model.forward(q_input1);
    let q_logits1_flat = q_logits1.reshape([12, 13]);
    let q1 = argmax(q_logits1_flat.slice(5, 1).reshape([13]));
    
    let qv1 = q1 as f32;
    let q_data2 = [9.0, 9.0, 10.0, 1.0, 0.0, 11.0, qv1, 12.0, 12.0, 12.0, 12.0, 12.0];
    let q_input2 = q_data2.reshape([1, 12]);
    let q_logits2 = model.forward(q_input2);
    let q_logits2_flat = q_logits2.reshape([12, 13]);
    let q2 = argmax(q_logits2_flat.slice(6, 1).reshape([13]));
    
    let qv2 = q2 as f32;
    let q_data3 = [9.0, 9.0, 10.0, 1.0, 0.0, 11.0, qv1, qv2, 12.0, 12.0, 12.0, 12.0];
    let q_input3 = q_data3.reshape([1, 12]);
    let q_logits3 = model.forward(q_input3);
    let q_logits3_flat = q_logits3.reshape([12, 13]);
    let q3 = argmax(q_logits3_flat.slice(7, 1).reshape([13]));
    
    print("Predicted:"); print(q1); print(q2); print(q3);
    
    print("Done.");
}
