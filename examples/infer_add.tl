
// Helper for argmax
fn argmax(t: Tensor<f32, 1>) -> i64 {
    let max_val = -1000000.0;
    let idx = 0;
    for k in range(0, 13) {
        let v = t.get(k);
        // print(v); // Uncomment for full dump
        if v > max_val {
            max_val = v;
            idx = k;
        }
    }
    print("Argmax result:"); print(idx); print("Val:"); print(max_val);
    return idx;
}

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

fn get_memory() -> i64 {
    return tl_get_memory_mb();
}

fn main() {
    let vocab_size = 13;
    let d_model = 128; 
    
    print("Initializing Model for Inference...");
    let model = GPT::new(vocab_size, d_model);
    
    print("Initializing Runtime: Metal backend selected.");
    
    print("Loading Parameters...");
    load_all_params(model, "model_2digit_rev.safetensors");
    print("Parameters Loaded.");
    
    // Test Case: 12 + 34 = 46
    // Reverse: 2 1 + 4 3 = 6 4 0
    print("Test: 12 + 34");
    
    let x0 = 2.0; let x1 = 1.0; // 12 -> 2,1
    let x2 = 10.0; // (+)
    let x3 = 4.0; let x4 = 3.0; // 34 -> 4,3
    let x5 = 11.0; // (=)
    
    let val_pad = 12.0;

    // Step 1: Predict 1st result digit (LSD)
    let data1 = [x0, x1, x2, x3, x4, x5, val_pad, val_pad, val_pad, 12.0, 12.0, 12.0];
    let input1 = reshape(data1, 1, 12);
    let logits1 = model.forward(input1);
    let logits1_flat = reshape(logits1, 12, 13);
    let pred1 = argmax(logits1_flat.slice(5, 1));
    
    // Step 2: Predict 2nd result digit (10s)
    let val_pred1 = pow(pred1, 1.0).get(0);
    
    let data2 = [x0, x1, x2, x3, x4, x5, val_pred1, val_pad, val_pad, 12.0, 12.0, 12.0];
    let input2 = reshape(data2, 1, 12);
    let logits2 = model.forward(input2);
    let logits2_flat = reshape(logits2, 12, 13);
    let pred2 = argmax(logits2_flat.slice(6, 1));

    // Step 3: Predict 3rd result digit (100s)
    let val_pred2 = pow(pred2, 1.0).get(0);
    
    let data3 = [x0, x1, x2, x3, x4, x5, val_pred1, val_pred2, val_pad, 12.0, 12.0, 12.0];
    let input3 = reshape(data3, 1, 12);
    let logits3 = model.forward(input3);
    let logits3_flat = reshape(logits3, 12, 13);
    let pred3 = argmax(logits3_flat.slice(7, 1));

    print("Predicted (Reverse):");
    print(pred1); print(pred2); print(pred3);
    
    // 99 + 1 = 100
    // Rev: 9 9 + 1 0 = 0 0 1
    print("Test: 99 + 1");
    let y0=9.0; let y1=9.0; let y2=10.0; let y3=1.0; let y4=0.0; let y5=11.0;
    
    let d1 = [y0, y1, y2, y3, y4, y5, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0];
    let inp1 = reshape(d1, 1, 12);
    let log1 = model.forward(inp1);
    let log1_flat = reshape(log1, 12, 13);
    let p1 = argmax(log1_flat.slice(5, 1));
    
    let vp1 = pow(p1, 1.0).get(0);
    let d2 = [y0, y1, y2, y3, y4, y5, vp1, 12.0, 12.0, 12.0, 12.0, 12.0];
    let inp2 = reshape(d2, 1, 12);
    let log2 = model.forward(inp2);
    let log2_flat = reshape(log2, 12, 13);
    let p2 = argmax(log2_flat.slice(6, 1));
    
    let vp2 = pow(p2, 1.0).get(0);
    let d3 = [y0, y1, y2, y3, y4, y5, vp1, vp2, 12.0, 12.0, 12.0, 12.0];
    let inp3 = reshape(d3, 1, 12);
    let log3 = model.forward(inp3);
    let log3_flat = reshape(log3, 12, 13);
    let p3 = argmax(log3_flat.slice(7, 1));
    
    print("Predicted (Reverse):");
    print(p1); print(p2); print(p3);
    
    print("Inference Verification Complete.");
}
