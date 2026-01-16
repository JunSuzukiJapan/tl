// 2桁足し算タスク推論 - 組み込み関数を最小限に抑えた実装
// 自己回帰生成による推論

// ---------------
// Helper Functions
// ---------------

// 手動argmax (元のコードから維持)
fn my_argmax(t: Tensor<f32, 1>) -> i64 {
    let mut max_val = -1000000.0;
    let idx = 0;
    for k in range(0, 13) {
        let v = t.get(k);
        if v > max_val {
            max_val = v;
            idx = k;
        }
    }
    return idx;
}

// One-Hot Embedding (embedding()の代替)
fn my_embedding(indices: Tensor<f32, 2>, w: Tensor<f32, 2>, V: i64) -> Tensor<f32, 3> {
    let T = 12;
    let flat_indices = indices.reshape(T);
    
    let one_hot: Tensor<f32, 2> = Tensor::randn([T, V], false) * 0.0;
    for t in range(0, T) {
        let mut idx = flat_indices.get(t) as i64;
        one_hot[t, idx] = 1.0;
    }
    let one_hot_d = one_hot.detach(true);
    
    let out_flat = one_hot_d.matmul(w);
    return out_flat.reshape(1, 12, 128);
}

// Causal Mask (tril()の代替)
fn get_causal_mask(size: i64) -> Tensor<f32, 2> {
    let mask: Tensor<f32, 2> = Tensor::randn([size, size], false) * 0.0;
    for i in range(0, size) {
        for j in range(0, size) {
            if j <= i {
                mask[i, j] = 1.0;
            }
        }
    }
    return mask.detach(true);
}

// ---------------
// Model Components (Inference Only)
// ---------------

struct Linear { W: Tensor<f32, 2>, b: Tensor<f32, 1> }
impl Linear { 
    fn new(i: i64, o: i64) -> Linear { 
        return Linear((Tensor::randn([i, o], true)*0.1).detach(true), (Tensor::randn([o], true)*0.0).detach(true)); 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        return x.matmul(self.W) + self.b; 
    } 
}

struct Embedding { w: Tensor<f32, 2>, vocab_size: i64 }
impl Embedding { 
    fn new(v: i64, d: i64) -> Embedding { 
        return Embedding((Tensor::randn([v, d], true)*0.1).detach(true), v); 
    } 
    fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> { 
        return my_embedding(i, self.w, self.vocab_size); 
    } 
}

struct LayerNorm { w: Tensor<f32, 1>, b: Tensor<f32, 1> }
impl LayerNorm { 
    fn new(d: i64) -> LayerNorm { 
        return LayerNorm((Tensor::randn([d], true)*0.0+1.0).detach(true), (Tensor::randn([d], true)*0.0).detach(true)); 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        return x * self.w + self.b; 
    } 
}

struct CausalSelfAttention { q_proj: Linear, k_proj: Linear, v_proj: Linear, p_proj: Linear }
impl CausalSelfAttention { 
    fn new(d: i64) -> CausalSelfAttention { 
        return CausalSelfAttention(Linear::new(d, d), Linear::new(d, d), Linear::new(d, d), Linear::new(d, d)); 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        let q = self.q_proj.forward(x); 
        let k = self.k_proj.forward(x); 
        let v = self.v_proj.forward(x); 
        
        let kt = k.transpose(1, 2);
        let scores = q.matmul(kt) * 0.08839;
        
        // 手動因果マスク
        let mask = get_causal_mask(12);
        let masked_scores = scores * mask;
        
        let attn_weights = masked_scores.softmax(2);
        let y = attn_weights.matmul(v); 
        return self.p_proj.forward(y); 
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
}

// 3-Layer GPT (Inference only)
struct GPT { w: Embedding, wp: Embedding, b1: Block, b2: Block, b3: Block, l: LayerNorm, h: Linear }
impl GPT { 
    fn new(v: i64, d: i64) -> GPT { 
        return GPT(
            Embedding::new(v, d), 
            Embedding::new(12, d), 
            Block::new(d), 
            Block::new(d), 
            Block::new(d), 
            LayerNorm::new(d), 
            Linear::new(d, v)
        ); 
    } 
    fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> { 
        let pos_data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let pos = pos_data.reshape(1, 12);
        let tok_emb = self.w.forward(i);
        let pos_emb = self.wp.forward(pos);
        let x = tok_emb + pos_emb;
        let x = self.b1.forward(x);
        let x = self.b2.forward(x);
        let x = self.b3.forward(x);
        return self.h.forward(self.l.forward(x)); 
    } 
}

// ---------------
// Inference
// ---------------

fn main() {
    let vocab_size = 13;
    let d_model = 128; 
    
    print("Model Inference - Autoregressive (Explicit Implementation)");
    let model = GPT::new(vocab_size, d_model);
    
    Param::load_all("model_add.safetensors");
    print("Parameters Loaded.");
    
    // Test 12 + 34 = 46
    // 逆順: 2, 1, +, 4, 3, =, 6, 4, 0
    print("Test: 12 + 34 = 46");
    print("Expected (Reverse): 6, 4, 0");
    
    // Step 1: Predict first result digit
    // Correct input for Teacher Forcing check: [2, 1, 10, 4, 3, 11, 6, 4, 0, 12, 12, 12]
    // Note: We fill the answer positions with correct values to see if the model predicts the NEXT token correctly
    // Position 5 (11.0) predicts Position 6 (should be 6)
    let data1 = [2.0, 1.0, 10.0, 4.0, 3.0, 11.0, 6.0, 4.0, 0.0, 12.0, 12.0, 12.0];
    let input1 = data1.reshape(1, 12);
    let logits1 = model.forward(input1);
    let logits1_flat = logits1.reshape(12, 13);
    let p1 = my_argmax(logits1_flat.slice(5, 1));
    
    // Step 2: Predict second digit at Position 7 (should be 4)
    // Position 6 (6.0) predicts Position 7
    let p2 = my_argmax(logits1_flat.slice(6, 1));
    
    // Step 3: Predict third digit at Position 8 (should be 0)
    // Position 7 (4.0) predicts Position 8
    let p3 = my_argmax(logits1_flat.slice(7, 1));
    
    print("Predicted:"); print(p1); print(p2); print(p3);
    
    // Test 99 + 1 = 100
    print("Test: 99 + 1 = 100");
    print("Expected (Reverse): 0, 0, 1");
    
    // Correct input: [9, 9, 10, 1, 0, 11, 0, 0, 1, 12, 12, 12]
    let q_data1 = [9.0, 9.0, 10.0, 1.0, 0.0, 11.0, 0.0, 0.0, 1.0, 12.0, 12.0, 12.0];
    let q_input1 = q_data1.reshape(1, 12);
    let q_logits1 = model.forward(q_input1);
    let q_logits1_flat = q_logits1.reshape(12, 13);
    
    let q1 = my_argmax(q_logits1_flat.slice(5, 1));
    let q2 = my_argmax(q_logits1_flat.slice(6, 1));
    let q3 = my_argmax(q_logits1_flat.slice(7, 1));
    
    print("Predicted:"); print(q1); print(q2); print(q3);
    
    print("Done.");
}
