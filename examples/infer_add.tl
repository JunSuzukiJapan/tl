
// Helper for argmax
fn argmax(t: Tensor<f32, 1>) -> i64 {
    let max_val = -1000000.0;
    let idx = 0;
    // Vocab size is 13
    for k in range(0, 13) {
        let v = t.get(k);
        if v > max_val {
            max_val = v;
            idx = k;
        }
    }
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

struct GPT { w: Embedding, wp: Embedding, b: Block, l: LayerNorm, h: Linear }
impl GPT { fn new(v: i64, d: i64) -> GPT { return GPT(Embedding::new(v, d), Embedding::new(12, d), Block::new(d), LayerNorm::new(d), Linear::new(d, v)); } fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> { 
    let pos_data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
    let pos = reshape(pos_data, 1, 12);
    let tok_emb = self.w.forward(i);
    let pos_emb = self.wp.forward(pos);
    return self.h.forward(self.l.forward(self.b.forward(tok_emb + pos_emb))); 
} fn step(self, lr: f32) -> GPT { let s = self; s.w = s.w.step(lr); s.wp = s.wp.step(lr); s.b = s.b.step(lr); s.l = s.l.step(lr); s.h = s.h.step(lr); return s; } }

// Wrapper for memory monitoring
fn get_memory() -> i64 {
    return tl_get_memory_mb();
}

fn main() {
    let vocab_size = 13;
    let d_model = 128; // Updated to match trained model
    let block_size = 12; 
    print("Initializing Model for Inference...");
    let model = GPT::new(vocab_size, d_model);
    
    print("Initializing Runtime: Metal backend selected.");
    
    print("Loading Parameters...");
    load_all_params(model, "model_2digit.safetensors");
    print("Parameters Loaded.");

    print("Parameters Loaded.");
    
    print("Running Inference verification 2-digit...");
    
    // Test Case: 12 + 34 = 046
    print("Input:");
    print("12");
    print("+");
    print("34");
    print("Predicted Digits:");
    
    // Encode '12' -> 1, 2.  '34' -> 3, 4.
    // Fixed format sequence: d1 d2 + d1 d2 = 
    // Indices: 0,1,2,3,4,5. 
    // 12 + 34
    // 0: 1
    // 1: 2
    // 2: 10 (+)
    // 3: 3
    // 4: 4
    // 5: 11 (=)
    // We want to predict pos 6 (d1 of result) using logits at pos 5
    // Then pos 7 using pos 6
    // Then pos 8 using pos 7
    
    let x0 = 1.0; let x1 = 2.0; let x2 = 10.0; let x3 = 3.0; let x4 = 4.0; let x5 = 11.0;
    
    let val_pad = 12.0;

    // Step 1: Predict 1st result digit (at pos 6, using logits from pos 5)
    let data1 = [x0, x1, x2, x3, x4, x5, val_pad, val_pad, val_pad, 12.0, 12.0, 12.0];
    let input1 = reshape(data1, 1, 12);
    let logits1 = model.forward(input1);
    let logits1_flat = reshape(logits1, 12, 13);
    // Slice at pos 5 to get prediction for pos 6
    let next_logits1 = logits1_flat.slice(5, 1);
    let pred1 = argmax(next_logits1);
    
    print(pred1);
    
    // Step 2: Predict 2nd result digit (at pos 7, using logits from pos 6 'pred1')
    let val_pred1 = pow(pred1, 1.0).get(0);
    
    let data2 = [x0, x1, x2, x3, x4, x5, val_pred1, val_pad, val_pad, 12.0, 12.0, 12.0];
    let input2 = reshape(data2, 1, 12);
    let logits2 = model.forward(input2);
    let logits2_flat = reshape(logits2, 12, 13);
    // Slice at pos 6 to get prediction for pos 7
    let next_logits2 = logits2_flat.slice(6, 1);
    let pred2 = argmax(next_logits2);

    print(pred2);
    
    // Step 3: Predict 3rd result digit (at pos 8, using logits from pos 7 'pred2')
    let val_pred2 = pow(pred2, 1.0).get(0);
    
    let data3 = [x0, x1, x2, x3, x4, x5, val_pred1, val_pred2, val_pad, 12.0, 12.0, 12.0];
    let input3 = reshape(data3, 1, 12);
    let logits3 = model.forward(input3);
    let logits3_flat = reshape(logits3, 12, 13);
    // Slice at pos 7 to get prediction for pos 8
    let next_logits3 = logits3_flat.slice(7, 1);
    let pred3 = argmax(next_logits3);

    print(pred3);

    // Test Case: 99 + 1 = 100
    print("Input:");
    print("99");
    print("+");
    print("1");
    print("Predicted Digits:");
    
    // 99 + 01 (Need leading zero for 1 -> 01 ?)
    // Training adds leading zeros in data gen?
    // let i = idx / 100; let i_d1 = i/10; let i_d2 = i%10;
    // So 1 -> 0, 1. Correct.
    // Input must be 99 + 01.
    // x0=9, x1=9, x2=10, x3=0, x4=1, x5=11.
    
    let y0=9.0; let y1=9.0; let y2=10.0; let y3=0.0; let y4=1.0; let y5=11.0;
    
    let d1 = [y0, y1, y2, y3, y4, y5, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0];
    let inp1 = reshape(d1, 1, 12);
    let log1 = model.forward(inp1);
    let log1_flat = reshape(log1, 12, 13);
    let p1 = argmax(log1_flat.slice(5, 1));
    print(p1);
    
    let vp1 = pow(p1, 1.0).get(0);
    let d2 = [y0, y1, y2, y3, y4, y5, vp1, 12.0, 12.0, 12.0, 12.0, 12.0];
    let inp2 = reshape(d2, 1, 12);
    let log2 = model.forward(inp2);
    let log2_flat = reshape(log2, 12, 13);
    let p2 = argmax(log2_flat.slice(6, 1));
    print(p2);
    
    let vp2 = pow(p2, 1.0).get(0);
    let d3 = [y0, y1, y2, y3, y4, y5, vp1, vp2, 12.0, 12.0, 12.0, 12.0];
    let inp3 = reshape(d3, 1, 12);
    let log3 = model.forward(inp3);
    let log3_flat = reshape(log3, 12, 13);
    let p3 = argmax(log3_flat.slice(7, 1));
    print(p3);
    
    // 5+5
    print("Input:");
    print("5");
    print("+");
    print("5");
    print("Predicted Digits:");
    // 05 + 05
    let z0=0.0; let z1=5.0; let z2=10.0; let z3=0.0; let z4=5.0; let z5=11.0;
    let dd1 = [z0, z1, z2, z3, z4, z5, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0];
    let ii1 = reshape(dd1, 1, 12);
    let llo1 = model.forward(ii1);
    let llo1_flat = reshape(llo1, 12, 13);
    let pp1 = argmax(llo1_flat.slice(5, 1));
    print(pp1);

    let vpp1 = pow(pp1, 1.0).get(0);
    let dd2 = [z0, z1, z2, z3, z4, z5, vpp1, 12.0, 12.0, 12.0, 12.0, 12.0];
    let ii2 = reshape(dd2, 1, 12);
    let llo2 = model.forward(ii2);
    let llo2_flat = reshape(llo2, 12, 13);
    let pp2 = argmax(llo2_flat.slice(6, 1));
    print(pp2);
    
    let vpp2 = pow(pp2, 1.0).get(0);
    let dd3 = [z0, z1, z2, z3, z4, z5, vpp1, vpp2, 12.0, 12.0, 12.0, 12.0];
    let ii3 = reshape(dd3, 1, 12);
    let llo3 = model.forward(ii3);
    let llo3_flat = reshape(llo3, 12, 13);
    let pp3 = argmax(llo3_flat.slice(7, 1));
    print(pp3);
    
    // 88+99
    print("Input:");
    print("88");
    print("+");
    print("99");
    print("Predicted Digits:");
    let w0=8.0; let w1=8.0; let w2=10.0; let w3=9.0; let w4=9.0; let w5=11.0;
    let ee1 = [w0, w1, w2, w3, w4, w5, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0];
    let jj1 = reshape(ee1, 1, 12);
    let ll1 = model.forward(jj1);
    let ll1_flat = reshape(ll1, 12, 13);
    let qq1 = argmax(ll1_flat.slice(5, 1));
    print(qq1);

    let vqq1 = pow(qq1, 1.0).get(0);
    let ee2 = [w0, w1, w2, w3, w4, w5, vqq1, 12.0, 12.0, 12.0, 12.0, 12.0];
    let jj2 = reshape(ee2, 1, 12);
    let ll2 = model.forward(jj2);
    let ll2_flat = reshape(ll2, 12, 13);
    let qq2 = argmax(ll2_flat.slice(6, 1));
    print(qq2);
    
    let vqq2 = pow(qq2, 1.0).get(0);
    let ee3 = [w0, w1, w2, w3, w4, w5, vqq1, vqq2, 12.0, 12.0, 12.0, 12.0];
    let jj3 = reshape(ee3, 1, 12);
    let ll3 = model.forward(jj3);
    let ll3_flat = reshape(ll3, 12, 13);
    let qq3 = argmax(ll3_flat.slice(7, 1));
    print(qq3);
    
    print("Inference Verification Complete.");
}
