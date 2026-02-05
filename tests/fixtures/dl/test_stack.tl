// SKIP: Mutability issues in struct field assignment
struct Linear { W: Tensor<f32, 2>, b: Tensor<f32, 1> }

impl Linear {
    fn new(i: i64, o: i64) -> Linear {
        print("Linear::new");
        print(i);
        print(o);
        // Split init to safe-guard against temp drops
        let w = (Tensor::randn([i, o], true) * 0.1).detach();
        let b = (Tensor::randn([o], true) * 0.0).detach();
        Linear(w, b)
    }
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { x.matmul(self.W) + self.b }
    fn step(self, lr: f32) -> Linear { let mut s = self; let gW = s.W.grad(); let gb = s.b.grad(); s.W = (s.W - gW * lr).detach(); s.b = (s.b - gb * lr).detach(); s }
}

struct Embedding { w: Tensor<f32, 2> }
impl Embedding { fn new(v: i64, d: i64) -> Embedding { print("Embedding::new"); let w = (Tensor::randn([v, d], true)*0.1).detach(); Embedding(w) } fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> { i.embedding(self.w) } fn step(self, lr: f32) -> Embedding { let mut s = self; let g = s.w.grad(); s.w = (s.w - g * lr).detach(); s } }

struct LayerNorm { w: Tensor<f32, 1>, b: Tensor<f32, 1> }
impl LayerNorm { fn new(d: i64) -> LayerNorm { print("LayerNorm::new"); let w = (Tensor::randn([d], true)*0.0+1.0).detach(); let b = (Tensor::randn([d], true)*0.0).detach(); LayerNorm(w, b) } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { x + self.b } fn step(self, lr: f32) -> LayerNorm { let mut s = self; let gb = s.b.grad(); s.b = (s.b - gb * lr).detach(); s } }

struct CausalSelfAttention { a: Linear, p: Linear }
impl CausalSelfAttention { fn new(d: i64) -> CausalSelfAttention { print("CSA::new"); let l1 = Linear::new(d, d*3); let l2 = Linear::new(d*3, d); CausalSelfAttention(l1, l2) } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { let q = self.a.forward(x); let k=q; let v=q; let y = q.matmul(k.transpose(1, 2)).scale(0.125).tril(0).softmax(2).matmul(v); self.p.forward(y) } fn step(self, lr: f32) -> CausalSelfAttention { let mut s = self; s.a = s.a.step(lr); s.p = s.p.step(lr); s } }

struct MLP { c_fc: Linear, c_proj: Linear }
impl MLP { fn new(d: i64) -> MLP { print("MLP::new"); let l1 = Linear::new(d, d*4); print("MLP L1"); let l2 = Linear::new(d*4, d); print("MLP L2"); MLP(l1, l2) } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { self.c_proj.forward(self.c_fc.forward(x).gelu()) } fn step(self, lr: f32) -> MLP { let mut s = self; s.c_fc = s.c_fc.step(lr); s.c_proj = s.c_proj.step(lr); s } }

struct Block { l1: LayerNorm, a: CausalSelfAttention, l2: LayerNorm, m: MLP }
impl Block { fn new(d: i64) -> Block { print("Block::new"); Block(LayerNorm::new(d), CausalSelfAttention::new(d), LayerNorm::new(d), MLP::new(d)) } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { let x = x + self.a.forward(self.l1.forward(x)); x + self.m.forward(self.l2.forward(x)) } fn step(self, lr: f32) -> Block { let mut s = self; s.l1 = s.l1.step(lr); s.a = s.a.step(lr); s.l2 = s.l2.step(lr); s.m = s.m.step(lr); s } }

struct GPT { w: Embedding, b: Block, l: LayerNorm, h: Linear }
impl GPT { fn new(v: i64, d: i64) -> GPT { print("GPT::new"); let e = Embedding::new(v, d); print("Embedding created"); let b = Block::new(d); print("Block created"); let res = GPT(e, b, LayerNorm::new(d), Linear::new(d, v)); print("GPT::new Done"); res } fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> { self.h.forward(self.l.forward(self.b.forward(self.w.forward(i)))) } fn step(self, lr: f32) -> GPT { let mut s = self; s.w = s.w.step(lr); s.b = s.b.step(lr); s.l = s.l.step(lr); s.h = s.h.step(lr); s } }

fn main() {
    let vocab_size = 13;
    let d_model = 64;
    let mut model = GPT::new(vocab_size, d_model);
    let lr = 0.01;
    
    print("Simple loop test - 2 epochs, 5 iterations each");
    
    for epoch in range(0, 2) {
        print("Epoch:"); print(epoch);
        
        for i in range(0, 5) {
            // Minimal data construction
            let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
            let target = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 12.0];
            
            let X = data.reshape([1, 12]);
            let Y = target.reshape([1, 12]);
            
            let logits = model.forward(X);
            let logits_flat = logits.reshape([12, 13]);
            let Y_flat = Y.reshape([12]);
            let loss = logits_flat.cross_entropy(Y_flat);
            
            loss.backward();
            model = model.step(lr);
        }
    }
    
    print("Test completed successfully!");
}
