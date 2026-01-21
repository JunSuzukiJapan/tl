// 2桁足し算タスク - 組み込み関数を最小限に抑えた実装
// 置換した組み込み関数: embedding(), tril(), cross_entropy(), pow()
// 維持した組み込み関数: relu(), softmax() (数値安定性のため)

// ---------------
// Helper Functions
// ---------------

// One-Hot Embedding (embedding()の代替)
fn my_embedding(indices: Tensor<f32, 2>, w: Tensor<f32, 2>, V: i64) -> Tensor<f32, 3> {
    let T = 12;
    let flat_indices = indices.reshape([T]);
    
    let mut one_hot: Tensor<f32, 2> = Tensor::randn([T, V], false) * 0.0;
    for t in range(0, T) {
        let idx = flat_indices.get(t) as i64;
        one_hot[t, idx] = 1.0;
    }
    let one_hot_d = one_hot.detach(true);
    
    let out_flat = one_hot_d.matmul(w);
    return out_flat.reshape([1, 12, 128]);
}

// Causal Mask (tril()の代替)
fn get_causal_mask(size: i64) -> Tensor<f32, 2> {
    let mut mask: Tensor<f32, 2> = Tensor::randn([size, size], false) * 0.0;
    for i in range(0, size) {
        for j in range(0, size) {
            if j <= i {
                mask[i, j] = 1.0;
            }
        }
    }
    return mask.detach(true);
}

// Cross Entropy (手動実装)
fn my_cross_entropy(logits: Tensor<f32, 3>, targets: Tensor<f32, 2>) -> Tensor<f32, 1> {
    let T = 12;
    let V = 13;
    
    let probs = logits.softmax(2);
    let log_probs = probs.log();
    
    let log_probs_flat = log_probs.reshape([T, V]);
    let targets_flat = targets.reshape([T]);
    
    let mut mask: Tensor<f32, 2> = Tensor::randn([T, V], false) * 0.0;
    for i in range(0, T) {
        let t = targets_flat.get(i) as i64;
        mask[i, t] = 1.0;
    }
    let mask_d = mask.detach(true);
    
    let selected = mask_d * log_probs_flat;
    let sum_log_probs = selected.sum();
    
    // -1 * mean を計算
    let mut neg_one: Tensor<f32, 1> = Tensor::randn([1], false) * 0.0;
    neg_one[0] = -1.0;
    let neg_one_d = neg_one.detach(true);
    
    let mut n_float: Tensor<f32, 1> = Tensor::randn([1], false) * 0.0;
    n_float[0] = 12.0;
    let n_float_d = n_float.detach(true);
    
    let mean = sum_log_probs / n_float_d;
    let neg_mean = mean * neg_one_d;
    
    return neg_mean;
}

// ---------------
// Model Components
// ---------------

struct Linear { W: Tensor<f32, 2>, b: Tensor<f32, 1> }
impl Linear { 
    fn new(i: i64, o: i64) -> Linear { 
        return Linear((Tensor::randn([i, o], true)*0.1).detach(true), (Tensor::randn([o], true)*0.0).detach(true)); 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        return x.matmul(self.W) + self.b; 
    } 
    fn step(self, lr: f32) -> Linear { 
        let mut s = self; 
        s.W = (s.W - s.W.grad() * lr).detach(true); 
        s.b = (s.b - s.b.grad() * lr).detach(true); 
        return s; 
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
    fn step(self, lr: f32) -> Embedding { 
        let mut s = self; 
        s.w = (s.w - s.w.grad() * lr).detach(true); 
        return s; 
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
    fn step(self, lr: f32) -> LayerNorm { 
        let mut s = self; 
        s.w = (s.w - s.w.grad() * lr).detach(true);
        s.b = (s.b - s.b.grad() * lr).detach(true); 
        return s; 
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
        
        // 手動因果マスク (tril()の代替)
        let mask = get_causal_mask(12);
        let masked_scores = scores * mask;
        
        let attn_weights = masked_scores.softmax(2);
        let y = attn_weights.matmul(v); 
        return self.p_proj.forward(y); 
    } 
    fn step(self, lr: f32) -> CausalSelfAttention { 
        let mut s = self; 
        s.q_proj = s.q_proj.step(lr); 
        s.k_proj = s.k_proj.step(lr); 
        s.v_proj = s.v_proj.step(lr); 
        s.p_proj = s.p_proj.step(lr); 
        return s; 
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
    fn step(self, lr: f32) -> MLP { 
        let mut s = self; 
        s.f = s.f.step(lr); 
        s.p = s.p.step(lr); 
        return s; 
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
    fn step(self, lr: f32) -> Block { 
        let mut s = self; 
        s.l1 = s.l1.step(lr); 
        s.a = s.a.step(lr); 
        s.l2 = s.l2.step(lr); 
        s.m = s.m.step(lr); 
        return s; 
    } 
}

// 3-Layer GPT
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
        let pos = pos_data.reshape([1, 12]);
        let tok_emb = self.w.forward(i);
        let pos_emb = self.wp.forward(pos);
        let x = tok_emb + pos_emb;
        let x = self.b1.forward(x);
        let x = self.b2.forward(x);
        let x = self.b3.forward(x);
        return self.h.forward(self.l.forward(x)); 
    } 
    fn step(self, lr: f32) -> GPT { 
        let s = self; 
        s.w = s.w.step(lr); 
        s.wp = s.wp.step(lr); 
        s.b1 = s.b1.step(lr); 
        s.b2 = s.b2.step(lr); 
        s.b3 = s.b3.step(lr); 
        s.l = s.l.step(lr); 
        s.h = s.h.step(lr); 
        return s; 
    } 
}

// ---------------
// Training
// ---------------

fn train_step(model: GPT, lr: f32, i: i64, j: i64) -> GPT {
    let sum = i + j;
    
    let i_d10 = i / 10;
    let i_d1 = i - (i_d10 * 10);
    let j_d10 = j / 10;
    let j_d1 = j - (j_d10 * 10);
    let s_d100 = sum / 100;
    let rem = sum - (s_d100 * 100);
    let s_d10 = rem / 10;
    let s_d1 = rem - (s_d10 * 10);

    // int→f32変換 (pow()の代替)
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
    
    let target = [
        v_i_d10, 10.0, 
        v_j_d1, v_j_d10, 11.0, 
        v_s_d1, v_s_d10, v_s_d100, 
        12.0, 12.0, 12.0, 12.0
    ];

    let X = data.reshape([1, 12]);
    let Y = target.reshape([1, 12]);
    
    let logits = model.forward(X);
    let loss = my_cross_entropy(logits, Y);
    
    loss.backward();
    model = model.step(lr);
    return model;
}

fn train_epoch(model: GPT, lr: f32, epoch: i64) -> GPT {
    let total_steps = 500;
    let stride = 137;
    let offset = epoch * 79; 

    for s in range(0, total_steps) {
        let raw = s * stride + offset;
        let idx = raw - ((raw / 10000) * 10000);
        let i = idx / 100;
        let j = idx - ((idx / 100) * 100);
        model = train_step(model, lr, i, j);
    }
    
    print("Epoch Done.");
    return model;
}

fn get_memory() -> i64 {
    return System::memory_mb();
}

fn main() {
    let vocab_size = 13;
    let d_model = 128; 
    let mut model = GPT::new(vocab_size, d_model);
    
    let lr = 0.0005;
    let epochs = 100;

    print("Training 2-digit addition (Explicit Implementation) - 100 Epochs");

    for epoch in range(0, epochs) {
        model = train_epoch(model, lr, epoch);
        if epoch - ((epoch / 5) * 5) == 0 {
             let mem = get_memory();
             print("Epoch:"); print(epoch);
             print("Mem(MB):"); print(mem);
             Param::save_all("model_add.safetensors");
        }
    }
    
    print("Training Complete!");
    Param::save_all("model_add.safetensors");
}
