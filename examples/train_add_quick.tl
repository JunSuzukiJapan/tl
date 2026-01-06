struct Linear { W: Tensor<f32, 2>, b: Tensor<f32, 1> }
impl Linear { fn new(i: i64, o: i64) -> Linear { return Linear((Tensor::randn([i, o], true)*0.1).detach(true), (Tensor::randn([o], true)*0.0).detach(true)); } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { return matmul(x, self.W) + self.b; } fn step(self, lr: f32) -> Linear { let s = self; let gW = s.W.grad(); let gb = s.b.grad(); s.W = (s.W - gW * lr).detach(true); s.b = (s.b - gb * lr).detach(true); return s; } }
struct Embedding { w: Tensor<f32, 2> }
impl Embedding { fn new(v: i64, d: i64) -> Embedding { return Embedding((Tensor::randn([v, d], true)*0.1).detach(true)); } fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> { return embedding(i, self.w); } fn step(self, lr: f32) -> Embedding { let s = self; let g = s.w.grad(); s.w = (s.w - g * lr).detach(true); return s; } }
struct LayerNorm { w: Tensor<f32, 1>, b: Tensor<f32, 1> }
impl LayerNorm { fn new(d: i64) -> LayerNorm { return LayerNorm((Tensor::randn([d], true)*0.0+1.0).detach(true), (Tensor::randn([d], true)*0.0).detach(true)); } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { return x + self.b; } fn step(self, lr: f32) -> LayerNorm { let s = self; let gb = s.b.grad(); s.b = (s.b - gb * lr).detach(true); return s; } }
struct CausalSelfAttention { q_proj: Linear, k_proj: Linear, v_proj: Linear, p_proj: Linear }
impl CausalSelfAttention { fn new(d: i64) -> CausalSelfAttention { return CausalSelfAttention(Linear::new(d, d), Linear::new(d, d), Linear::new(d, d), Linear::new(d, d)); } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { let q = self.q_proj.forward(x); let k = self.k_proj.forward(x); let v = self.v_proj.forward(x); let y = matmul(softmax(tril(matmul(q, transpose(k, 1, 2))*0.08839, 0), 2), v); return self.p_proj.forward(y); } fn step(self, lr: f32) -> CausalSelfAttention { let s = self; s.q_proj = s.q_proj.step(lr); s.k_proj = s.k_proj.step(lr); s.v_proj = s.v_proj.step(lr); s.p_proj = s.p_proj.step(lr); return s; } }
struct MLP { f: Linear, p: Linear }
impl MLP { fn new(d: i64) -> MLP { return MLP(Linear::new(d, d*4), Linear::new(d*4, d)); } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { return self.p.forward(relu(self.f.forward(x))); } fn step(self, lr: f32) -> MLP { let s = self; s.f = s.f.step(lr); s.p = s.p.step(lr); return s; } }
struct Block { l1: LayerNorm, a: CausalSelfAttention, l2: LayerNorm, m: MLP }
impl Block { fn new(d: i64) -> Block { return Block(LayerNorm::new(d), CausalSelfAttention::new(d), LayerNorm::new(d), MLP::new(d)); } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { let x = x + self.a.forward(self.l1.forward(x)); return x + self.m.forward(self.l2.forward(x)); } fn step(self, lr: f32) -> Block { let s = self; s.l1 = s.l1.step(lr); s.a = s.a.step(lr); s.l2 = s.l2.step(lr); s.m = s.m.step(lr); return s; } }

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
        let pos = reshape(pos_data, 1, 12);
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

fn train_epoch(model: GPT, lr: f32, epoch: i64) -> GPT {
    let total_steps = 500;
    let stride = 137;
    let offset = epoch * 79; 
    let total_loss = pow(0.0, 1.0);

    for s in range(0, total_steps) {
        let raw = s * stride + offset;
        let idx = raw - ((raw / 10000) * 10000);
        let i = idx / 100;
        let j = idx - ((idx / 100) * 100);

        let sum = i + j;
        
        let i_d10 = i / 10;
        let i_d1 = i - (i_d10 * 10);
        let j_d10 = j / 10;
        let j_d1 = j - (j_d10 * 10);
        let s_d100 = sum / 100;
        let rem = sum - (s_d100 * 100);
        let s_d10 = rem / 10;
        let s_d1 = rem - (s_d10 * 10);

        let v_i_d1 = pow(i_d1, 1.0).get(0);
        let v_i_d10 = pow(i_d10, 1.0).get(0);
        let v_j_d1 = pow(j_d1, 1.0).get(0);
        let v_j_d10 = pow(j_d10, 1.0).get(0);
        let v_s_d1 = pow(s_d1, 1.0).get(0);
        let v_s_d10 = pow(s_d10, 1.0).get(0);
        let v_s_d100 = pow(s_d100, 1.0).get(0);
        
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

        let X = reshape(data, 1, 12);
        let Y = reshape(target, 1, 12);
        
        let logits = model.forward(X);
        let logits_flat = reshape(logits, 12, 13);
        let Y_flat = reshape(Y, 12);
        let loss = cross_entropy(logits_flat, Y_flat);
        
        backward(loss);
        model = model.step(lr);
        total_loss = loss.detach(true);
    }
    
    print("Epoch:"); print(epoch); print("Loss:"); print(total_loss);
    return model;
}

fn main() {
    let vocab_size = 13;
    let d_model = 128; 
    let model = GPT::new(vocab_size, d_model);
    
    // Resume from saved model
    load_all_params(model, "model_2digit_rev.safetensors");
    
    let lr = 0.0001;  // Even lower LR for fine-tuning
    let epochs = 50;

    print("Training 2-digit addition - 50 more epochs (resume, very low LR)");

    for epoch in range(0, epochs) {
        model = train_epoch(model, lr, epoch);
        // Save every 10 epochs
        if epoch - ((epoch / 10) * 10) == 9 {
            save_all_params(model, "model_2digit_rev.safetensors");
        }
    }
    print("Training Complete!");
    save_all_params(model, "model_2digit_rev.safetensors");
}
