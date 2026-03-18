// 2桁足し算タスク - paper/heavy と統一されたアーキテクチャ

// ---------------
// Model Components (paper/heavy と同一)
// ---------------

struct Linear { W: Tensor<f32, 2>, b: Tensor<f32, 1> }
impl Linear { 
    fn new(i: i64, o: i64) -> Linear { 
        Linear((Tensor::randn([i, o], true)*0.1).detach(true), (Tensor::randn([o], true)*0.0).detach(true))
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        x.matmul(self.W) + self.b 
    } 
    fn step(self, lr: f32) -> Linear { 
        let mut s = self; 
        s.W = (s.W - s.W.grad() * lr).detach(true); 
        s.b = (s.b - s.b.grad() * lr).detach(true); 
        s
    }
}

struct LayerNorm { w: Tensor<f32, 1>, b: Tensor<f32, 1> }
impl LayerNorm {
    fn new(d: i64) -> LayerNorm { 
        LayerNorm((Tensor::randn([d], true)*0.0+1.0).detach(true), (Tensor::randn([d], true)*0.0).detach(true)) 
    }
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        x.layer_norm(self.w, self.b, 0.00001 as f64) 
    }
    fn step(self, lr: f32) -> LayerNorm {
        let mut s = self; 
        s.w = (s.w - s.w.grad() * lr).detach(true); 
        s.b = (s.b - s.b.grad() * lr).detach(true); 
        s
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
        let scores = q.matmul(kt) * 0.125;
        let attn_weights = scores.tril(0).softmax(2);
        self.p_proj.forward(attn_weights.matmul(v)) 
    } 
    fn step(self, lr: f32) -> CausalSelfAttention { 
        let mut s = self; 
        s.q_proj = s.q_proj.step(lr); 
        s.k_proj = s.k_proj.step(lr); 
        s.v_proj = s.v_proj.step(lr); 
        s.p_proj = s.p_proj.step(lr); 
        s 
    } 
}

struct MLP { f: Linear, p: Linear }
impl MLP { 
    fn new(d: i64) -> MLP { MLP(Linear::new(d, d*4), Linear::new(d*4, d)) } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        self.p.forward(self.f.forward(x).relu()) 
    } 
    fn step(self, lr: f32) -> MLP { 
        let mut s = self; s.f = s.f.step(lr); s.p = s.p.step(lr); s 
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
    fn step(self, lr: f32) -> Block { 
        let mut s = self; 
        s.l1 = s.l1.step(lr); 
        s.a = s.a.step(lr); 
        s.l2 = s.l2.step(lr); 
        s.m = s.m.step(lr); 
        s 
    } 
}

struct Embedding { w: Tensor<f32, 2> }
impl Embedding {
    fn new(v: i64, d: i64) -> Embedding {
        Embedding((Tensor::randn([v, d], true)*0.1).detach(true))
    }
    fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> {
        i.embedding(self.w)
    }
    fn step(self, lr: f32) -> Embedding {
        let mut s = self;
        let g = s.w.grad();
        s.w = (s.w - g * lr).detach(true);
        s
    }
}

// 1-Layer GPT
struct GPT { w: Embedding, wp: Embedding, b1: Block, l: LayerNorm, h: Linear }
impl GPT { 
    fn new(v: i64, d: i64) -> GPT { 
        GPT(
            Embedding::new(v, d), 
            Embedding::new(12, d), 
            Block::new(d), 
            LayerNorm::new(d), 
            Linear::new(d, v)
        ) 
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
    fn step(self, lr: f32) -> GPT { 
        let mut s = self; 
        s.w = s.w.step(lr); 
        s.wp = s.wp.step(lr); 
        s.b1 = s.b1.step(lr); 
        s.l = s.l.step(lr); 
        s.h = s.h.step(lr); 
        s 
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
        12.0, 12.0, 12.0, 12.0, 
        v_s_d100, v_s_d10, v_s_d1, 
        12.0, 12.0, 12.0
    ];

    let X = data.reshape([1, 12]);
    let Y = target.reshape([1, 12]);
    
    let logits = model.forward(X);
    let logits_flat = logits.reshape([12, 13]);
    let Y_flat = Y.reshape([12]);
    let loss = logits_flat.cross_entropy(Y_flat);

    loss.backward();
    model.step(lr)
}

fn train_epoch(model: GPT, lr: f32, epoch: i64) -> GPT {
    let total_steps = 10;
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
    model
}

fn main() {
    let vocab_size = 13;
    let d_model = 64; 
    let mut model = GPT::new(vocab_size, d_model);
    
    let lr = 0.0005;
    let epochs = 20;

    print("Training 2-digit addition (1-layer GPT, d_model=64) - 20 Epochs");

    for epoch in range(0, epochs) {
        model = train_epoch(model, lr, epoch);
        if epoch - ((epoch / 5) * 5) == 0 {
             print("Epoch:"); print(epoch);
             print("MB:"); print(System::memory_mb());
        }
    }
    
    print("Training Complete!");
    Param::save(model, "model_add.safetensors");
}
