// Heavy Model 推論 (teacher forced)
// Vocab: 0-9(数字), 10(+), 11(=), 12(PAD) → vocab_size=13

struct Linear { 
    W: Tensor<f32, 2>, b: Tensor<f32, 1>,
    mW: Tensor<f32, 2>, vW: Tensor<f32, 2>,
    mb: Tensor<f32, 1>, vb: Tensor<f32, 1>
}
impl Linear { 
    fn new(i: i64, o: i64) -> Linear { 
        Linear(
            Tensor::zeros([i, o], false), Tensor::zeros([o], false),
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
        Embedding(Tensor::zeros([v, d], false), Tensor::zeros([v, d], false), Tensor::zeros([v, d], false))
    } 
    fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> { i.embedding(self.w) } 
}

struct LayerNorm { 
    w: Tensor<f32, 1>, b: Tensor<f32, 1>,
    mw: Tensor<f32, 1>, vw: Tensor<f32, 1>,
    mb: Tensor<f32, 1>, vb: Tensor<f32, 1>
}
impl LayerNorm { 
    fn new(d: i64) -> LayerNorm { 
        LayerNorm(
            Tensor::zeros([d], false), Tensor::zeros([d], false),
            Tensor::zeros([d], false), Tensor::zeros([d], false),
            Tensor::zeros([d], false), Tensor::zeros([d], false)
        ) 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        x.layer_norm(self.w, self.b, 0.00001 as f64) 
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
        let scores = q.matmul(kt) * 0.088388; // 1/sqrt(128)
        let ones = scores * 0.0 + 1.0;
        let causal_mask = ones.tril(0); 
        let neg_inf_mask = (causal_mask - 1.0) * 10000.0; 
        let masked = scores + neg_inf_mask;
        let attn_weights = masked.softmax(2);
        self.p_proj.forward(attn_weights.matmul(v)) 
    } 
}

struct MLP { f: Linear, p: Linear }
impl MLP { 
    fn new(d: i64) -> MLP { MLP(Linear::new(d, d*4), Linear::new(d*4, d)) } 
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

struct GPTHeavy { w: Embedding, wp: Embedding, b1: Block, b2: Block, l: LayerNorm, h: Linear }
impl GPTHeavy { 
    fn new(v: i64, d: i64) -> GPTHeavy { 
        GPTHeavy(
            Embedding::new(v, d), Embedding::new(12, d),
            Block::new(d), Block::new(d),
            LayerNorm::new(d), Linear::new(d, v)
        ) 
    } 
    fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> { 
        let pos_data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let pos = pos_data.reshape([1, 12]);
        let x = self.w.forward(i) + self.wp.forward(pos);
        let x = self.b1.forward(x);
        let x = self.b2.forward(x);
        self.h.forward(self.l.forward(x)) 
    } 
}

fn inference_tf(model: GPTHeavy, i: i64, j: i64) {
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
    
    // Teacher forced: 正解の答えを含む入力
    let data = [
        v_i_d1, v_i_d10, 10.0, 
        v_j_d1, v_j_d10, 11.0, 
        v_s_d1, v_s_d10, v_s_d100, 
        12.0, 12.0, 12.0
    ];
    
    let X = data.reshape([1, 12]);
    let logits = model.forward(X);
    
    let l_d1 = logits.slice(1, 5, 6, 1).reshape([13]);
    let l_d10 = logits.slice(1, 6, 7, 1).reshape([13]);
    let l_d100 = logits.slice(1, 7, 8, 1).reshape([13]);
    
    let p_d1 = l_d1.argmax(0)[0] as i64;
    let p_d10 = l_d10.argmax(0)[0] as i64;
    let p_d100 = l_d100.argmax(0)[0] as i64;
    
    let pred_sum = p_d100 * 100 + p_d10 * 10 + p_d1;
    
    print("-------");
    print("Input : "); print(i); print("+"); print(j);
    print("Target: "); print(sum);
    print("Pred  : "); print(pred_sum);
    
    if sum == pred_sum {
        print("OK");
    } else {
        print("FAIL");
    }
}

fn main() {
    let vocab_size = 13;
    let d_model = 128; 
    
    print("Heavy Model Inference (Teacher Forced)");
    let model = GPTHeavy::new(vocab_size, d_model);
    
    if File::exists("model_heavy.safetensors") {
        Param::load(model, "model_heavy.safetensors");
        print("Loaded model");
    } else {
        print("ERROR: model not found");
        return;
    }
    
    inference_tf(model, 12, 34);  // 46
    inference_tf(model, 99, 1);   // 100
    inference_tf(model, 45, 55);  // 100
    inference_tf(model, 8, 9);    // 17
    inference_tf(model, 73, 27);  // 100
    inference_tf(model, 66, 88);  // 154
    inference_tf(model, 0, 0);    // 0
    inference_tf(model, 50, 50);  // 100
    inference_tf(model, 37, 48);  // 85
    inference_tf(model, 91, 9);   // 100
    
    print("Done.");
}
