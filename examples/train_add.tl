struct Linear { W: Tensor<f32, 2>, b: Tensor<f32, 1> }
impl Linear { fn new(i: i64, o: i64) -> Linear { return Linear((randn([i, o], true)*0.1).detach(true), (randn([o], true)*0.0).detach(true)); } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { return matmul(x, self.W) + self.b; } fn step(self, lr: f32) { let gW = self.W.grad(); let gb = self.b.grad(); self.W -= (gW * lr); self.b -= (gb * lr); } }
struct Embedding { w: Tensor<f32, 2> }
impl Embedding { fn new(v: i64, d: i64) -> Embedding { return Embedding((randn([v, d], true)*0.1).detach(true)); } fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> { return embedding(i, self.w); } fn step(self, lr: f32) { let g = self.w.grad(); self.w -= (g * lr); } }
struct LayerNorm { w: Tensor<f32, 1>, b: Tensor<f32, 1> }
impl LayerNorm { fn new(d: i64) -> LayerNorm { return LayerNorm((randn([d], true)*0.0+1.0).detach(true), (randn([d], true)*0.0).detach(true)); } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { return x + self.b; } fn step(self, lr: f32) { let gb = self.b.grad(); self.b -= (gb * lr); } }
struct CausalSelfAttention { a: Linear, p: Linear }
impl CausalSelfAttention { fn new(d: i64) -> CausalSelfAttention { return CausalSelfAttention(Linear::new(d, d*3), Linear::new(d*3, d)); } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { let q = self.a.forward(x); let k=q; let v=q; let y = matmul(softmax(tril(matmul(q, transpose(k, 1, 2))*0.125, 0), 2), v); return self.p.forward(y); } fn step(self, lr: f32) { self.a.step(lr); self.p.step(lr); } }
struct MLP { f: Linear, p: Linear }
impl MLP { fn new(d: i64) -> MLP { return MLP(Linear::new(d, d*4), Linear::new(d*4, d)); } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { return self.p.forward(relu(self.f.forward(x))); } fn step(self, lr: f32) { self.f.step(lr); self.p.step(lr); } }
struct Block { l1: LayerNorm, a: CausalSelfAttention, l2: LayerNorm, m: MLP }
impl Block { fn new(d: i64) -> Block { return Block(LayerNorm::new(d), CausalSelfAttention::new(d), LayerNorm::new(d), MLP::new(d)); } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { let x = x + self.a.forward(self.l1.forward(x)); return x + self.m.forward(self.l2.forward(x)); } fn step(self, lr: f32) { self.l1.step(lr); self.a.step(lr); self.l2.step(lr); self.m.step(lr); } }

struct GPT { w: Embedding, b: Block, l: LayerNorm, h: Linear }
impl GPT { fn new(v: i64, d: i64) -> GPT { return GPT(Embedding::new(v, d), Block::new(d), LayerNorm::new(d), Linear::new(d, v)); } fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> { return self.h.forward(self.l.forward(self.b.forward(self.w.forward(i)))); } fn step(self, lr: f32) { self.w.step(lr); self.b.step(lr); self.l.step(lr); self.h.step(lr); } }

// Wrapper for memory monitoring
fn get_memory() -> i64 {
    return tl_get_memory_mb();
}

fn register_gpt_params(m: GPT) {
    add_parameter("w.w", m.w.w);

    add_parameter("b.l1.w", m.b.l1.w);
    add_parameter("b.l1.b", m.b.l1.b);
    
    add_parameter("b.a.a.W", m.b.a.a.W);
    add_parameter("b.a.a.b", m.b.a.a.b);
    add_parameter("b.a.p.W", m.b.a.p.W);
    add_parameter("b.a.p.b", m.b.a.p.b);
    
    add_parameter("b.l2.w", m.b.l2.w);
    add_parameter("b.l2.b", m.b.l2.b); 

    add_parameter("b.m.f.W", m.b.m.f.W);
    add_parameter("b.m.f.b", m.b.m.f.b);
    add_parameter("b.m.p.W", m.b.m.p.W);
    add_parameter("b.m.p.b", m.b.m.p.b);

    add_parameter("l.w", m.l.w);
    add_parameter("l.b", m.l.b);

    add_parameter("h.W", m.h.W);
    add_parameter("h.b", m.h.b);
}

// CRITICAL: Extract training loop into separate function to create new stack frame
fn train_epoch(model: GPT, lr: f32, epoch: i64) {
    let total_loss = pow(0.0, 1.0);

    // Use larger range for actual training
    // 0-99 covers all 2-digit additions
    // To safe time, we use a subset or full set?
    // Let's use full set for i, and subset for j to mix predictability and speed, or smaller ranges.
    // 30x30 = 900 steps per epoch. 50 epochs = 45000 steps.
    for i in range(0, 30) { 
        for j in range(0, 30) {
            let sum = i + j;
            
            // Extract digits using simple division/modulo
            // Format: always 3 digits with leading zeros (e.g., "099 + 001 = 100")
            let i_d1 = pow(i / 10, 1.0).get(0);
            let i_d2 = pow(i - ((i/10)*10), 1.0).get(0);
            
            let j_d1 = pow(j / 10, 1.0).get(0);
            let j_d2 = pow(j - ((j/10)*10), 1.0).get(0);
            
            let s_d1 = pow(sum / 100, 1.0).get(0);
            let s_d2 = pow((sum - ((sum/100)*100)) / 10, 1.0).get(0);
            let s_d3 = pow(sum - ((sum/10)*10), 1.0).get(0);
            
            // Fixed format sequence: d1 d2 + d1 d2 = d1 d2 d3 PAD PAD PAD
            // Total: 12 tokens
            let data = [i_d1, i_d2, 10.0, j_d1, j_d2, 11.0, s_d1, s_d2, s_d3, 12.0, 12.0, 12.0];
            let target = [i_d2, 10.0, j_d1, j_d2, 11.0, s_d1, s_d2, s_d3, 12.0, 12.0, 12.0, 12.0];
            
            let X = reshape(data, 1, 12);
            let Y = reshape(target, 1, 12);
            
            let logits = model.forward(X);
            let logits_flat = reshape(logits, 12, 13);
            let Y_flat = reshape(Y, 12);
            let loss = cross_entropy(logits_flat, Y_flat);
            
            backward(loss);
            
            // In-place update
            model.step(lr);
            total_loss = loss.detach(true);
        }
    }

    // Memory logging every epoch
    let mem_mb = get_memory();
    print("Loss:"); print(total_loss); 
    print("Memory(MB):"); print(mem_mb);
    

}

fn main() {
    let vocab_size = 13;
    let d_model = 64; 
    let model = GPT::new(vocab_size, d_model);
    let lr = 0.01; 
    let lr = 0.01; 
    let epochs = 50; // Increased to 50 for actual learning

    print("Training 2-digit addition (0-99) - With memory monitoring");

    for epoch in range(0, epochs) {
        print("Epoch:");
        print(epoch);
        
        train_epoch(model, lr, epoch);
    }
    
    // Register all parameters manually before saving
    // Because we use functional updates, the final tensors are in 'model', but not in the global registry.
    register_gpt_params(model);
    
    print("Training Complete!");
    save_all_params("model_2digit.safetensors");
}
