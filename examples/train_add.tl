struct Linear { W: Tensor<f32, 2>, b: Tensor<f32, 1> }
impl Linear { fn new(i: i64, o: i64) -> Linear { return Linear((randn([i, o], true)*0.1).detach(true), (randn([o], true)*0.0).detach(true)); } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { return matmul(x, self.W) + self.b; } fn step(self, lr: f32) { let gW = self.W.grad(); let gb = self.b.grad(); self.W = (self.W - gW * lr).detach(true); self.b = (self.b - gb * lr).detach(true); } }
struct Embedding { w: Tensor<f32, 2> }
impl Embedding { fn new(v: i64, d: i64) -> Embedding { return Embedding((randn([v, d], true)*0.1).detach(true)); } fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> { return embedding(i, self.w); } fn step(self, lr: f32) { let g = self.w.grad(); self.w = (self.w - g * lr).detach(true); } }
struct LayerNorm { w: Tensor<f32, 1>, b: Tensor<f32, 1> }
impl LayerNorm { fn new(d: i64) -> LayerNorm { return LayerNorm((randn([d], true)*0.0+1.0).detach(true), (randn([d], true)*0.0).detach(true)); } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { return x + self.b; } fn step(self, lr: f32) { let gb = self.b.grad(); self.b = (self.b - gb * lr).detach(true); } }
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



// CRITICAL: Extract training loop into separate function to create new stack frame
fn train_epoch(model: GPT, lr: f32, epoch: i64) {
    let total_loss = pow(0.0, 1.0);

    // Use larger range for actual training
    // 0-99 covers all 2-digit additions
    // To safe time, we use a subset or full set?
    // Let's use full set for i, and subset for j to mix predictability and speed, or smaller ranges.
    // 30x30 = 900 steps per epoch. 50 epochs = 45000 steps.
    // Use single loop with pseudo-random shuffling to break data correlation
    // Total 100*100 = 10000 combinations
    let total_steps = 1000; // Reduced for faster feedback
    let stride = 137; // Prime stride to cover all indices
    let offset = epoch * 79; // Vary start point per epoch

    for s in range(0, total_steps) {
        // Calculate randomized index: (s * stride + offset) % total_steps
        let raw = s * stride + offset;
        let idx = raw - ((raw / 10000) * 10000);
        
        // Decode i and j from idx
        let i = idx / 100;
        let j = idx - ((idx / 100) * 100);

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
    let epochs = 100; // Increased epochs, smaller steps

    print("Training 2-digit addition (0-99) - With memory monitoring");

    for epoch in range(0, epochs) {
        print("Epoch:");
        print(epoch);
        
        train_epoch(model, lr, epoch);
    }
    print("Training Complete!");
    // Auto-register and save
    save_all_params(model, "model_2digit.safetensors");
}
