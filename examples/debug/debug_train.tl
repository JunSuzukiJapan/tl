struct Linear { W: Tensor<f32, 2>, b: Tensor<f32, 1> }
impl Linear { fn new(i: i64, o: i64) -> Linear { return Linear((Tensor::randn([i, o], true)*0.1).detach(true), (Tensor::randn([o], true)*0.0).detach(true)); } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { return x.matmul(self.W) + self.b; } fn step(self, lr: f32) -> Linear { let gW = self.W.grad(); let gb = self.b.grad(); let s = self; s.W = (s.W - gW * lr).detach(true); s.b = (s.b - gb * lr).detach(true); return s; } }
struct Embedding { w: Tensor<f32, 2> }
impl Embedding { fn new(v: i64, d: i64) -> Embedding { return Embedding((Tensor::randn([v, d], true)*0.1).detach(true)); } fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> { return i.embedding(self.w); } fn step(self, lr: f32) -> Embedding { let g = self.w.grad(); let s = self; s.w = (s.w - g * lr).detach(true); return s; } }
struct LayerNorm { w: Tensor<f32, 1>, b: Tensor<f32, 1> }
impl LayerNorm { fn new(d: i64) -> LayerNorm { return LayerNorm((Tensor::randn([d], true)*0.0+1.0).detach(true), (Tensor::randn([d], true)*0.0).detach(true)); } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { return x + self.b; } fn step(self, lr: f32) -> LayerNorm { let gb = self.b.grad(); let s = self; s.b = (s.b - gb * lr).detach(true); return s; } }
struct CausalSelfAttention { q_proj: Linear, k_proj: Linear, v_proj: Linear, p_proj: Linear }
impl CausalSelfAttention { fn new(d: i64) -> CausalSelfAttention { return CausalSelfAttention(Linear::new(d, d), Linear::new(d, d), Linear::new(d, d), Linear::new(d, d)); } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { let q = self.q_proj.forward(x); let k = self.k_proj.forward(x); let v = self.v_proj.forward(x); let y = q.matmul(k.transpose(1, 2)).tril(0).softmax(2).matmul(v); return self.p_proj.forward(y); } fn step(self, lr: f32) -> CausalSelfAttention { let s = self; s.q_proj = self.q_proj.step(lr); s.k_proj = self.k_proj.step(lr); s.v_proj = self.v_proj.step(lr); s.p_proj = self.p_proj.step(lr); return s; } }
struct MLP { f: Linear, p: Linear }
impl MLP { fn new(d: i64) -> MLP { return MLP(Linear::new(d, d*4), Linear::new(d*4, d)); } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { return self.p.forward(self.f.forward(x).relu()); } fn step(self, lr: f32) -> MLP { let s = self; s.f = self.f.step(lr); s.p = self.p.step(lr); return s; } }
struct Block { l1: LayerNorm, a: CausalSelfAttention, l2: LayerNorm, m: MLP }
impl Block { fn new(d: i64) -> Block { return Block(LayerNorm::new(d), CausalSelfAttention::new(d), LayerNorm::new(d), MLP::new(d)); } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { let x = x + self.a.forward(self.l1.forward(x)); return x + self.m.forward(self.l2.forward(x)); } fn step(self, lr: f32) -> Block { let s = self; s.l1 = self.l1.step(lr); s.a = self.a.step(lr); s.l2 = self.l2.step(lr); s.m = self.m.step(lr); return s; } }

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
        print("GPT: Alloc Pos");
        let pos = pos_data.reshape([1, 12]);
        print("GPT: Calc Tok Emb");
        let tok_emb = self.w.forward(i);
        print("GPT: Calc Pos Emb");
        let pos_emb = self.wp.forward(pos);
        print("GPT: Add Emb");
        let x = tok_emb + pos_emb;
        let x = self.b1.forward(x);
        print("GPT: B1 Done");
        let x = self.b2.forward(x);
        print("GPT: B2 Done");
        let x = self.b3.forward(x);
        print("GPT: B3 Done");
        return self.h.forward(self.l.forward(x)); 
    } 
    fn step(self, lr: f32) -> GPT {
        let s = self;
        s.w = self.w.step(lr);
        s.wp = self.wp.step(lr);
        s.b1 = self.b1.step(lr);
        s.b2 = self.b2.step(lr);
        s.b3 = self.b3.step(lr);
        s.l = self.l.step(lr);
        s.h = self.h.step(lr);
        return s;
    } 
}

fn get_memory() -> i64 {
    return tl_get_memory_mb();
}

fn train_epoch(model: GPT, lr: f32, epoch: i64) {
    let mut total_loss = 0.0.pow(1.0);
    // Training on random subset
    let total_steps = 20;
    let stride = 137;
    let offset = epoch * 79; 

    for s in range(0, total_steps) {
        let raw = s * stride + offset;
        let idx = raw - ((raw / 10000) * 10000);
        let i = idx / 100;
        let j = idx - ((idx / 100) * 100);

        let sum = i + j;
        
        // Reverse Digits Logic (Least Significant Digit First)
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
            v_j_d1, v_j_d10, 11.0, 
            v_s_d1, v_s_d10, v_s_d100, 
            12.0, 12.0, 12.0, 12.0
        ];
        
        print("Iter:");
        print(s);

        print("P-ReshapeX");
        let X = data.reshape([1, 12]);
        print("P-ReshapeY");
        let Y = target.reshape([1, 12]);
        
        print("P-Forward");
        let logits = model.forward(X);
        print("Forward done");
        let logits_flat = logits.reshape([12, 13]);
        let Y_flat = Y.reshape([12]);
        let loss = logits_flat.cross_entropy(Y_flat);
        print("Loss calculated");
        let loss_detached = loss.detach(true);
        
        loss.backward();
        print("Backward done");
        let dump_grad = model.w.w.grad();
        print("Model W grad:");
        print(dump_grad);
        model = model.step(lr);
        print("Step done");
        total_loss = loss_detached;
        print("Loop End");
    }
    
    let mem_mb = get_memory();
    print("Loss:"); print(total_loss); 
    print("Memory(MB):"); print(mem_mb);
}

fn main() {
    let vocab_size = 13;
    let d_model = 128; 
    // 3 layers
    let model = GPT::new(vocab_size, d_model);
    
    // Attempt to load existing parameters to resume training
    if (1==1) { // Simple hack to execute load. Ideally check file existence but tl doesn't have `exists`.
        // We know the file exists from previous run (if it didn't hang before first save).
        // If it fails, runtime might panic. But we verified file exists (Epoch 8 saved).
        print("Resuming training from model_2digit_rev.safetensors...");
        Param::load_all(model, "model_2digit_rev.safetensors");
    }

    // Low LR for stability with deeper model
    let lr = 0.0005; 
    let epochs = 2; // Additional 20 epochs

    print("Training 2-digit addition (Reverse Digits, 3-Layer) - RESUME");

    for epoch in range(0, epochs) {
        print("Epoch:");
        print(epoch);
        train_epoch(model, lr, epoch);
        // Save every epoch to prevent loss on hang
        Param::save_all(model, "model_2digit_rev.safetensors");
    }
    print("Training Complete!");
    Param::save_all(model, "model_2digit_rev.safetensors");
}
