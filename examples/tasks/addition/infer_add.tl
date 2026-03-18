// 2桁足し算タスク - 推論 (train_add.tl と同一構造)

struct Linear { W: Tensor<f32, 2>, b: Tensor<f32, 1> }
impl Linear { 
    fn new(i: i64, o: i64) -> Linear { 
        Linear((Tensor::randn([i, o], true)*0.1).detach(true), (Tensor::randn([o], true)*0.0).detach(true))
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { x.matmul(self.W) + self.b }
}

struct LayerNorm { w: Tensor<f32, 1>, b: Tensor<f32, 1> }
impl LayerNorm {
    fn new(d: i64) -> LayerNorm { 
        LayerNorm((Tensor::randn([d], true)*0.0+1.0).detach(true), (Tensor::randn([d], true)*0.0).detach(true)) 
    }
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { x.layer_norm(self.w, self.b, 0.00001 as f64) }
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
        let scores = q.matmul(k.transpose(1, 2)) * 0.125;
        let attn_weights = scores.tril(0).softmax(2);
        self.p_proj.forward(attn_weights.matmul(v)) 
    } 
}

struct MLP { f: Linear, p: Linear }
impl MLP { 
    fn new(d: i64) -> MLP { MLP(Linear::new(d, d*4), Linear::new(d*4, d)) } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { self.p.forward(self.f.forward(x).relu()) } 
}

struct Block { l1: LayerNorm, a: CausalSelfAttention, l2: LayerNorm, m: MLP }
impl Block { 
    fn new(d: i64) -> Block { Block(LayerNorm::new(d), CausalSelfAttention::new(d), LayerNorm::new(d), MLP::new(d)) } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        let x = x + self.a.forward(self.l1.forward(x)); 
        x + self.m.forward(self.l2.forward(x)) 
    } 
}

struct Embedding { w: Tensor<f32, 2> }
impl Embedding {
    fn new(v: i64, d: i64) -> Embedding { Embedding((Tensor::randn([v, d], true)*0.1).detach(true)) }
    fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> { i.embedding(self.w) }
}

// 1-Layer GPT (train_add.tl と同一構造)
struct GPT { w: Embedding, wp: Embedding, b1: Block, l: LayerNorm, h: Linear }
impl GPT { 
    fn new(v: i64, d: i64) -> GPT { 
        GPT(Embedding::new(v, d), Embedding::new(12, d), Block::new(d), LayerNorm::new(d), Linear::new(d, v)) 
    } 
    fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> { 
        let pos_data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let pos = pos_data.reshape([1, 12]);
        let tok_emb = self.w.forward(i);
        let pos_emb = self.wp.forward(pos);
        let x = tok_emb + pos_emb;
        let x = self.b1.forward(x);
        self.h.forward(self.l.forward(x)) 
    } 
}

fn main() {
    let model_path = "model_add.safetensors";
    
    let vocab_size = 13;
    let d_model = 64; 
    
    print("Model Inference - Addition (train_add.tl structure)");
    let model = GPT::new(vocab_size, d_model);
    
    if File::exists(model_path) {
        Param::load(model, model_path);
        print("Parameters Loaded.");
    } else {
        print("ERROR: model_add.safetensors not found. Run train_add.tl first.");
        return;
    }
    
    // Test: 12 + 34 = 46
    print("Test: 12 + 34 = 46");
    print("Expected (Reverse): 6, 4, 0");
    
    let data1 = [2.0, 1.0, 10.0, 4.0, 3.0, 11.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0];
    let input1 = data1.reshape([1, 12]);
    let logits1 = model.forward(input1);
    let flat1 = logits1.reshape([12, 13]);
    
    print("Predicted:");
    let p6 = flat1.argmax(1);
    print(p6[6]);
    print(p6[7]);
    print(p6[8]);
    
    // Test: 99 + 1 = 100
    print("Test: 99 + 1 = 100");
    print("Expected (Reverse): 0, 0, 1");
    
    let data2 = [9.0, 9.0, 10.0, 1.0, 0.0, 11.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0];
    let input2 = data2.reshape([1, 12]);
    let logits2 = model.forward(input2);
    let flat2 = logits2.reshape([12, 13]);
    
    print("Predicted:");
    let p7 = flat2.argmax(1);
    print(p7[6]);
    print(p7[7]);
    print(p7[8]);
    
    print("Done.");
}
