// 2桁足し算タスク推論 - train_add.tl と同じモデル構造
// 自己回帰生成による推論

// ---------------
// Helper Functions
// ---------------

fn my_argmax(t: Tensor<f32, 1>) -> i64 {
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
    return mask.detach(false);
}

// ---------------
// Model Components (train_add.tl と同一)
// ---------------

struct Linear { W: Tensor<f32, 2>, b: Tensor<f32, 1> }
impl Linear { 
    fn new(i: i64, o: i64) -> Linear { 
        Linear((Tensor::randn([i, o], true)*0.1).detach(true), (Tensor::randn([o], true)*0.0).detach(true))
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        x.matmul(self.W) + self.b 
    } 
}

struct CausalSelfAttention { q_proj: Linear, k_proj: Linear, v_proj: Linear, p_proj: Linear }
impl CausalSelfAttention { 
    fn new(d: i64) -> CausalSelfAttention { 
        CausalSelfAttention(Linear::new(d, d), Linear::new(d, d), Linear::new(d, d), Linear::new(d, d)) 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        let q = self.q_proj.forward(x); 
        let k = self.k_proj.forward(x); 
        let v = self.v_proj.forward(x); 
        
        let kt = k.transpose(1, 2);
        let scores = q.matmul(kt) * 0.08839;
        
        let mask = get_causal_mask(12);
        let masked_scores = scores * mask;
        
        let attn_weights = masked_scores.softmax(2);
        let y = attn_weights.matmul(v); 
        self.p_proj.forward(y) 
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

struct Block { a: CausalSelfAttention, m: MLP }
impl Block { 
    fn new(d: i64) -> Block { 
        Block(CausalSelfAttention::new(d), MLP::new(d)) 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        let x = x + self.a.forward(x); 
        x + self.m.forward(x) 
    } 
}

// 3-Layer GPT (train_add.tl と同一構造)
struct GPT { b1: Block, b2: Block, b3: Block, h: Linear }
impl GPT { 
    fn new(v: i64, d: i64) -> GPT { 
        GPT(
            Block::new(d), 
            Block::new(d), 
            Block::new(d), 
            Linear::new(d, v)
        ) 
    } 
    fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> { 
        let x = Tensor::randn([1, 12, 128], false);
        let x = self.b1.forward(x);
        let x = self.b2.forward(x);
        let x = self.b3.forward(x);
        self.h.forward(x)
    } 
}

// ---------------
// Inference
// ---------------

fn main() {
    let model_path = "model_add.safetensors";
    
    let vocab_size = 13;
    let d_model = 128; 
    
    print("Model Inference - Addition (train_add.tl structure)");
    let model = GPT::new(vocab_size, d_model);
    
    Param::load(model, model_path);
    print("Parameters Loaded.");
    
    // Test 12 + 34 = 46
    print("Test: 12 + 34 = 46");
    print("Expected (Reverse): 6, 4, 0");
    
    let data1 = [2.0, 1.0, 10.0, 4.0, 3.0, 11.0, 6.0, 4.0, 0.0, 12.0, 12.0, 12.0];
    let input1 = data1.reshape([1, 12]);
    let logits1 = model.forward(input1);
    let logits1_flat = logits1.reshape([12, 13]);
    let p1 = my_argmax(logits1_flat.slice(0, 5, 6, 1).reshape([13]));
    let p2 = my_argmax(logits1_flat.slice(0, 6, 7, 1).reshape([13]));
    let p3 = my_argmax(logits1_flat.slice(0, 7, 8, 1).reshape([13]));
    
    print("Predicted:"); print(p1); print(p2); print(p3);
    
    // Test 99 + 1 = 100
    print("Test: 99 + 1 = 100");
    print("Expected (Reverse): 0, 0, 1");
    
    let q_data1 = [9.0, 9.0, 10.0, 1.0, 0.0, 11.0, 0.0, 0.0, 1.0, 12.0, 12.0, 12.0];
    let q_input1 = q_data1.reshape([1, 12]);
    let q_logits1 = model.forward(q_input1);
    let q_logits1_flat = q_logits1.reshape([12, 13]);
    
    let q1 = my_argmax(q_logits1_flat.slice(0, 5, 6, 1).reshape([13]));
    let q2 = my_argmax(q_logits1_flat.slice(0, 6, 7, 1).reshape([13]));
    let q3 = my_argmax(q_logits1_flat.slice(0, 7, 8, 1).reshape([13]));
    
    print("Predicted:"); print(q1); print(q2); print(q3);
    
    print("Done.");
}
