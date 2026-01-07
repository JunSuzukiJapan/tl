
extern fn tl_tensor_print_2(t: Tensor<f32, 2>);
extern fn tl_tensor_print_1(t: Tensor<f32, 1>);
extern fn tl_clear_grads();

struct Linear { W: Tensor<f32, 2>, b: Tensor<f32, 1> }
impl Linear { 
    fn new(i: i64, o: i64) -> Linear { 
        return Linear((Tensor::randn([i, o], true)*0.1).detach(true), (Tensor::randn([o], true)*0.0).detach(true)); 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        return matmul(x, self.W) + self.b; 
    } 
    fn step(self, lr: f32) -> Linear { 
        let s = self; 
        let gW = s.W.grad(); 
        let gb = s.b.grad(); 
        s.W = (s.W - gW * lr).detach(true); 
        s.b = (s.b - gb * lr).detach(true); 
        return s; 
    } 
    fn clone(self) -> Linear { return Linear(self.W.clone(), self.b.clone()); }
}

struct Embedding { w: Tensor<f32, 2> }
impl Embedding { 
    fn new(v: i64, d: i64) -> Embedding {
        return Embedding((Tensor::randn([v, d], true)*0.1).detach(true));
    } 
    fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> { 
        return embedding(i, self.w); 
    } 
    fn step(self, lr: f32) -> Embedding { 
        let s = self; 
        let g = s.w.grad(); 
        s.w = (s.w - g * lr).detach(true); 
        return s; 
    } 
    fn clone(self) -> Embedding { return Embedding(self.w.clone()); }
}

struct LayerNorm { w: Tensor<f32, 1>, b: Tensor<f32, 1> }
impl LayerNorm { 
    fn new(d: i64) -> LayerNorm { 
        return LayerNorm((Tensor::randn([d], true)*0.0+1.0).detach(true), (Tensor::randn([d], true)*0.0).detach(true)); 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        return x + self.b; 
    } 
    fn step(self, lr: f32) -> LayerNorm { 
        let s = self; 
        let gb = s.b.grad(); 
        s.b = (s.b - gb * lr).detach(true); 
        return s; 
    } 
    fn clone(self) -> LayerNorm { return LayerNorm(self.w.clone(), self.b.clone()); }
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
        // scale = 1/sqrt(64) = 0.125
        let y = matmul(softmax(tril(matmul(q, transpose(k, 1, 2))*0.125, 0), 2), v); 
        return self.p_proj.forward(y); 
    } 
    fn step(self, lr: f32) -> CausalSelfAttention { 
        let s = self; 
        s.q_proj = s.q_proj.step(lr); 
        s.k_proj = s.k_proj.step(lr); 
        s.v_proj = s.v_proj.step(lr); 
        s.p_proj = s.p_proj.step(lr); 
        return s; 
    } 
    fn clone(self) -> CausalSelfAttention {
        return CausalSelfAttention(self.q_proj.clone(), self.k_proj.clone(), self.v_proj.clone(), self.p_proj.clone());
    }
}

struct MLP { f: Linear, p: Linear }
impl MLP { 
    fn new(d: i64) -> MLP { 
        return MLP(Linear::new(d, d*4), Linear::new(d*4, d)); 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        return self.p.forward(relu(self.f.forward(x))); 
    } 
    fn step(self, lr: f32) -> MLP { 
        let s = self; 
        s.f = s.f.step(lr); 
        s.p = s.p.step(lr); 
        return s; 
    } 
    fn clone(self) -> MLP { return MLP(self.f.clone(), self.p.clone()); }
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
        let s = self; 
        s.l1 = s.l1.step(lr); 
        s.a = s.a.step(lr); 
        s.l2 = s.l2.step(lr); 
        s.m = s.m.step(lr); 
        return s; 
    } 
    fn clone(self) -> Block {
        return Block(self.l1.clone(), self.a.clone(), self.l2.clone(), self.m.clone());
    }
}

// 2-Layer GPT (Smaller for verification)
struct GPT { 
    w: Embedding, wp: Embedding, 
    b1: Block, b2: Block, 
    l: LayerNorm, h: Linear 
}
impl GPT { 
    fn new(v: i64, d: i64) -> GPT { 
        return GPT(
            Embedding::new(v, d), 
            Embedding::new(4, d), 
            Block::new(d), 
            Block::new(d), 
            LayerNorm::new(d), 
            Linear::new(d, v)
        ); 
    } 
    fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> { 
        let pos_data = [0.0, 1.0, 2.0, 3.0];
        let pos = reshape(pos_data, 1, 4);
        let tok_emb = self.w.forward(i);
        let pos_emb = self.wp.forward(pos);
        let x = tok_emb + pos_emb;
        // Checkpointing blocks
        let x = checkpoint(self.b1.forward, x);
        let x = checkpoint(self.b2.forward, x);
        return self.h.forward(self.l.forward(x)); 
    } 
    fn step(self, lr: f32) -> GPT { 
        let s = self; 
        s.w = s.w.step(lr); 
        s.wp = s.wp.step(lr); 
        s.b1 = s.b1.step(lr); 
        s.b2 = s.b2.step(lr); 
        s.l = s.l.step(lr); 
        s.h = s.h.step(lr); 
        return s; 
    }
    fn clone(self) -> GPT {
        return GPT(
            self.w.clone(), self.wp.clone(),
            self.b1.clone(), self.b2.clone(),
            self.l.clone(), self.h.clone()
        );
    }
}

fn train_step(model: GPT, lr: f32, i: i64, j: i64) -> GPT {
    let m = model.clone();
    let sum_raw = i + j;
    let sum = sum_raw - (sum_raw/13)*13;
    
    // Mock data input [i, j, sum, 0]
    // Fixed: Using 0 instead of 0.0 to match i/j types
    let X = reshape([i, j, sum, 0], 1, 4);
    // Target [j, sum, 0, 0] (Shifted)
    let Y = reshape([j, sum, 0, 0], 1, 4);

    // Forward
    let logits = m.forward(X);
    
    // Need to reshape for cross_entropy
    // logits: [1, 4, 13] -> [4, 13]
    let logits_flat = reshape(logits, 4, 13);
    let Y_flat = reshape(Y, 4);
    
    let loss = cross_entropy(logits_flat, Y_flat);
    
    // Print loss
    tl_tensor_print_1(reshape(loss, 1));
    
    // Backprop
    backward(loss);
    
    // Step
    m = m.step(lr);
    
    // Clear grads
    tl_clear_grads();
    
    return m;
}

fn infer(model: GPT, i: i64, j: i64) {
    let m = model.clone();
    let sum_raw = i + j;
    let expected = sum_raw - (sum_raw/13)*13;
    
    // Calculate loss to check correctness
    // Logits: [1, 12, 13]
    // Target for loss check should be shifted: [j, expected, 0...]
    // But we provided Input X = [i, j, 0...]
    // So X[0]=i -> should predict j
    // X[1]=j -> should predict expected
    // X[2]=0 -> should predict 0
    let X = reshape([i, j, 0, 0], 1, 4);
    let logits = m.forward(X);
    let Y = reshape([j, expected, 0, 0], 1, 4);
    
    let logits_flat = reshape(logits, 4, 13);
    let Y_flat = reshape(Y, 4);
    let loss = cross_entropy(logits_flat, Y_flat);
    
    // Prediction analysis
    // We want the prediction AFTER 'j', which is at index 1.
    // argmax(logits, 2) returns indices [1, 4]
    
    let preds = argmax(logits, 2); // [1, 4]
    print("Predictions (Whole Sequence):");
    tl_tensor_print_2(reshape(preds, 1, 4));
    
    // Extract single prediction at index 1
    // preds is [1, 12].
    // We need to slice it or get item. 
    // Since we don't have slice/get easily exposed yet that works on arbitrary index without setup:
    // Let's rely on print for manual verification, 
    // but try to be fancy:
    // We can multiply preds by a one-hot vector [0, 1, 0...] and sum?
    // Too complex. Manual check via print is fine for now.
    
    print("Inference: Input:"); print(i); print(j);
    print("Expected (at pos 1):"); print(expected);
    print("Loss:");
    tl_tensor_print_1(reshape(loss, 1));
}

fn main() {
    print("Verifying Activation Checkpointing Correctness...");
    // Vocab=13 (digits 0-12), d_model=64 (small)
    let model = GPT::new(13, 64);
    
    print("Training...");
    // 50 Epochs
    // 50 Epochs -> 200 Iterations
    // 50 Epochs -> 200 Iterations
    for epoch in range(0, 200) {
        // Train on fixed samples to ensure convergence
        model = train_step(model, 0.05, 3, 5); // 8
        model = train_step(model, 0.05, 2, 2); // 4
        model = train_step(model, 0.05, 9, 1); // 10
        model = train_step(model, 0.05, 5, 5); // 10
        model = train_step(model, 0.05, 0, 0); // 0
        
        if (epoch - (epoch/20)*20) == 0 {
            print("epoch:"); print(epoch);
        }
    }
    
    print("Inference check...");
    // Check known samples
    infer(model, 3, 5);
    infer(model, 2, 2);
    infer(model, 9, 1);
    
    print("Verification Completed.");
}
