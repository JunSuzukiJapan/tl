// 2桁足し算タスク - 2-Layer GPT (Adam)
// ---------------
// Model Components
// ---------------

struct Linear { 
    W: Tensor<f32, 2>, b: Tensor<f32, 1>,
    mW: Tensor<f32, 2>, vW: Tensor<f32, 2>,
    mb: Tensor<f32, 1>, vb: Tensor<f32, 1>
}
impl Linear { 
    fn new(i: i64, o: i64) -> Linear { 
        Linear(
            Tensor::zeros([i, o], false), // W
            Tensor::zeros([o], false),    // b
            Tensor::zeros([i, o], false), // mW
            Tensor::zeros([i, o], false), // vW
            Tensor::zeros([o], false),    // mb
            Tensor::zeros([o], false)     // vb
        )
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        x.matmul(self.W) + self.b 
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
            Tensor::zeros([d], false), // w
            Tensor::zeros([d], false), // b
            Tensor::zeros([d], false), // mw
            Tensor::zeros([d], false), // vw
            Tensor::zeros([d], false), // mb
            Tensor::zeros([d], false)  // vb
        ) 
    }
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        x.layer_norm(self.w, self.b, 0.00001 as f64) 
    }
}

struct CausalSelfAttention { q_proj: Linear, k_proj: Linear, v_proj: Linear, p_proj: Linear }
impl CausalSelfAttention { 
    fn new(d: i64) -> CausalSelfAttention { 
        CausalSelfAttention(
            Linear::new(d, d), // q_proj
            Linear::new(d, d), // k_proj
            Linear::new(d, d), // v_proj
            Linear::new(d, d)  // p_proj
        ) 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        let q = self.q_proj.forward(x); 
        let k = self.k_proj.forward(x); 
        let v = self.v_proj.forward(x); 
        let kt = k.transpose(1, 2);
        let scores = q.matmul(kt) * 0.08838;
        
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
    fn new(d: i64) -> MLP { 
        MLP(
            Linear::new(d, d*4), // f
            Linear::new(d*4, d)  // p
        ) 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        self.p.forward(self.f.forward(x).relu()) 
    } 
}

struct Block { l1: LayerNorm, a: CausalSelfAttention, l2: LayerNorm, m: MLP }
impl Block { 
    fn new(d: i64) -> Block { 
        Block(
            LayerNorm::new(d),           // l1
            CausalSelfAttention::new(d), // a
            LayerNorm::new(d),           // l2
            MLP::new(d)                  // m
        ) 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        let x = x + self.a.forward(self.l1.forward(x)); 
        x + self.m.forward(self.l2.forward(x)) 
    } 
}

struct Embedding { 
    w: Tensor<f32, 2>,
    mw: Tensor<f32, 2>, vw: Tensor<f32, 2>
}
impl Embedding {
    fn new(v: i64, d: i64) -> Embedding {
        Embedding(
            Tensor::zeros([v, d], false), // w
            Tensor::zeros([v, d], false), // mw
            Tensor::zeros([v, d], false)  // vw
        )
    }
    fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> {
        i.embedding(self.w)
    }
}

// 2-Layer GPT
struct GPT { w: Embedding, wp: Embedding, b1: Block, b2: Block, l: LayerNorm, h: Linear }
impl GPT { 
    fn new(v: i64, d: i64) -> GPT { 
        GPT(
            Embedding::new(v, d),  // w
            Embedding::new(12, d), // wp
            Block::new(d),         // b1
            Block::new(d),         // b2
            LayerNorm::new(d),     // l
            Linear::new(d, v)      // h
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
        self.h.forward(self.l.forward(x)) 
    } 
}

fn inference(model: GPT, i: i64, j: i64) {
    let sum = i + j;
    
    let i_d10 = i / 10;
    let i_d1 = i - (i_d10 * 10);
    let j_d10 = j / 10;
    let j_d1 = j - (j_d10 * 10);
    
    let v_i_d1 = i_d1 as f32;
    let v_i_d10 = i_d10 as f32;
    let v_j_d1 = j_d1 as f32;
    let v_j_d10 = j_d10 as f32;
    
    let data = [
        v_i_d1, v_i_d10, 10.0, 
        v_j_d1, v_j_d10, 11.0, 
        12.0, 12.0, 12.0, 
        12.0, 12.0, 12.0
    ];
    
    let X = data.reshape([1, 12]);
    let logits = model.forward(X);
    
    // We expect sum_d1 at pos=6, sum_d10 at pos=7, sum_d100 at pos=8
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
        print("Result: OK");
    } else {
        print("Result: FAIL");
    }
}

fn main() {
    print("Loading model...");
    let vocab_size = 13;
    let d_model = 128; 
    let model = GPT::new(vocab_size, d_model);
    Param::load(model, "model_add.safetensors");
    print("Model loaded successfully.");

    inference(model, 12, 34);
    inference(model, 99, 1);
    inference(model, 45, 55);
    inference(model, 8, 9);
    inference(model, 73, 27);
    inference(model, 66, 88);
}
