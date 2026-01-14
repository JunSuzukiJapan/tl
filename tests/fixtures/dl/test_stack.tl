// Minimal test to see if stack overflow is from large epoch count or complex loop body
struct Linear { W: Tensor<f32, 2>, b: Tensor<f32, 1> }
impl Linear { fn new(i: i64, o: i64) -> Linear { return Linear((Tensor::randn([i, o], true)*0.1).detach(true), (Tensor::randn([o], true)*0.0).detach(true)); } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { return x.matmul(self.W) + self.b; } fn step(self, lr: f32) -> Linear { let s = self; let gW = s.W.grad(); let gb = s.b.grad(); s.W = (s.W - gW * lr).detach(true); s.b = (s.b - gb * lr).detach(true); return s; } }
struct Embedding { w: Tensor<f32, 2> }
impl Embedding { fn new(v: i64, d: i64) -> Embedding { return Embedding((Tensor::randn([v, d], true)*0.1).detach(true)); } fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> { return i.embedding(self.w); } fn step(self, lr: f32) -> Embedding { let s = self; let g = s.w.grad(); s.w = (s.w - g * lr).detach(true); return s; } }
struct LayerNorm { w: Tensor<f32, 1>, b: Tensor<f32, 1> }
impl LayerNorm { fn new(d: i64) -> LayerNorm { return LayerNorm((Tensor::randn([d], true)*0.0+1.0).detach(true), (Tensor::randn([d], true)*0.0).detach(true)); } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { return x + self.b; } fn step(self, lr: f32) -> LayerNorm { let s = self; let gb = s.b.grad(); s.b = (s.b - gb * lr).detach(true); return s; } }
struct CausalSelfAttention { a: Linear, p: Linear }
impl CausalSelfAttention { fn new(d: i64) -> CausalSelfAttention { return CausalSelfAttention(Linear::new(d, d*3), Linear::new(d*3, d)); } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { let q = self.a.forward(x); let k=q; let v=q; let y = q.matmul(k.transpose(1, 2)).mul(0.125).tril(0).softmax(2).matmul(v); return self.p.forward(y); } fn step(self, lr: f32) -> CausalSelfAttention { let s = self; s.a = s.a.step(lr); s.p = s.p.step(lr); return s; } }
struct MLP { f: Linear, p: Linear }
impl MLP { fn new(d: i64) -> MLP { return MLP(Linear::new(d, d*4), Linear::new(d*4, d)); } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { return self.p.forward(self.f.forward(x).relu()); } fn step(self, lr: f32) -> MLP { let s = self; s.f = s.f.step(lr); s.p = s.p.step(lr); return s; } }
struct Block { l1: LayerNorm, a: CausalSelfAttention, l2: LayerNorm, m: MLP }
impl Block { fn new(d: i64) -> Block { return Block(LayerNorm::new(d), CausalSelfAttention::new(d), LayerNorm::new(d), MLP::new(d)); } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { let x = x + self.a.forward(self.l1.forward(x)); return x + self.m.forward(self.l2.forward(x)); } fn step(self, lr: f32) -> Block { let s = self; s.l1 = s.l1.step(lr); s.a = s.a.step(lr); s.l2 = s.l2.step(lr); s.m = s.m.step(lr); return s; } }

struct GPT { w: Embedding, b: Block, l: LayerNorm, h: Linear }
impl GPT { fn new(v: i64, d: i64) -> GPT { return GPT(Embedding::new(v, d), Block::new(d), LayerNorm::new(d), Linear::new(d, v)); } fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> { return self.h.forward(self.l.forward(self.b.forward(self.w.forward(i)))); } fn step(self, lr: f32) -> GPT { let s = self; s.w = s.w.step(lr); s.b = s.b.step(lr); s.l = s.l.step(lr); s.h = s.h.step(lr); return s; } }

fn main() {
    let vocab_size = 13;
    let d_model = 64;
    let model = GPT::new(vocab_size, d_model);
    let lr = 0.01;
    
    print("Simple loop test - 100 epochs, 100 iterations each");
    
    for epoch in range(0, 100) {
        print("Epoch:"); print(epoch);
        
        for i in range(0, 100) {
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
