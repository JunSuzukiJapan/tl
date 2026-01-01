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

fn main() {
    let vocab_size = 13;
    let d_model = 64; 
    let block_size = 6;
    let model = GPT::new(vocab_size, d_model, block_size); // args mismatch but just checking parse
    print("Main Start");
    
    // Test Array Literal
    let data = [1.0, 2.0];
    print("Array Literal OK");
    
    // Test Loop
    for i in range(0, 5) {
        print(i);
    }
    print("Loop OK");
}
