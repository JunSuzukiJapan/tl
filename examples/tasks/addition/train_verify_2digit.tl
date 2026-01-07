
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
    let sum = i + j;
    // Vocabulary size needs to cover sum
    
    // Mock data input [i, j, sum, 0]
    let X = reshape([i, j, sum, 0], 1, 4);
    // Target [j, sum, 0, 0] (Shifted)
    let Y = reshape([j, sum, 0, 0], 1, 4);

    // Forward
    let logits = m.forward(X);
    
    // Need to reshape for cross_entropy
    // logits: [1, 4, 200] -> [4, 200]
    let logits_flat = reshape(logits, 4, 200);
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
    let expected = i + j;
    
    // Calculate loss to check correctness
    let X = reshape([i, j, 0, 0], 1, 4);
    let logits = m.forward(X);
    let Y = reshape([j, expected, 0, 0], 1, 4);
    
    let logits_flat = reshape(logits, 4, 200);
    let Y_flat = reshape(Y, 4);
    let loss = cross_entropy(logits_flat, Y_flat);
    
    // Prediction analysis
    // We want the prediction AFTER 'j', which is at index 1.
    // argmax(logits, 2) returns indices [1, 4]
    
    let preds = argmax(logits, 2); // [1, 4]
    print("Predictions (Whole Sequence):");
    tl_tensor_print_2(reshape(preds, 1, 4));
    
    print("Inference: Input:"); print(i); print(j);
    print("Expected (at pos 1):"); print(expected);
    print("Loss:");
    tl_tensor_print_1(reshape(loss, 1));
}

fn main() {
    print("Verifying 2-Digit Addition (Vocab=200)...");
    // Vocab=200 (covers 99+99=198), d_model=64
    let model = GPT::new(200, 64);
    
    print("Training...");
    // 200 Iterations of fixed dataset
    for epoch in range(0, 200) {
        // Train on 2-digit examples
        model = train_step(model, 0.05, 12, 34); // 46
        model = train_step(model, 0.05, 50, 50); // 100
        model = train_step(model, 0.05, 99, 99); // 198
        model = train_step(model, 0.05, 10, 5);  // 15
        model = train_step(model, 0.05, 0, 0);   // 0
        
        if (epoch - (epoch/20)*20) == 0 {
            print("epoch:"); print(epoch);
        }
    }
    
    print("Inference check...");
    infer(model, 12, 34); // Expect 46
    infer(model, 50, 50); // Expect 100
    infer(model, 99, 99); // Expect 198
    
    print("Verification Completed.");
}
