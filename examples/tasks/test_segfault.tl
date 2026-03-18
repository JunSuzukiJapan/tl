// デバッグ出力付き4ブロック SIGBUS テスト

struct Linear { W: Tensor<f32, 2>, b: Tensor<f32, 1> }
impl Linear { 
    fn new(i: i64, o: i64) -> Linear { 
        Linear((Tensor::randn([i, o], true)*0.1).detach(), (Tensor::randn([o], true)*0.0).detach()) 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { x.matmul(self.W) + self.b } 
    fn step(self, lr: f32) -> Linear { 
        let mut s = self; 
        let gW = s.W.grad(); let gb = s.b.grad(); 
        s.W = (s.W - gW * lr).detach(); s.b = (s.b - gb * lr).detach(); s 
    } 
}

struct LayerNorm { b: Tensor<f32, 1> }
impl LayerNorm { 
    fn new(d: i64) -> LayerNorm { LayerNorm((Tensor::randn([d], true)*0.0).detach()) } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { x + self.b } 
    fn step(self, lr: f32) -> LayerNorm { 
        let mut s = self; let gb = s.b.grad(); s.b = (s.b - gb * lr).detach(); s 
    } 
}

struct CausalSelfAttention { q_proj: Linear, k_proj: Linear, v_proj: Linear, p_proj: Linear }
impl CausalSelfAttention { 
    fn new(d: i64) -> CausalSelfAttention { 
        CausalSelfAttention(Linear::new(d, d), Linear::new(d, d), Linear::new(d, d), Linear::new(d, d)) 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        let q = self.q_proj.forward(x); let k = self.k_proj.forward(x); let v = self.v_proj.forward(x); 
        let k_t = k.transpose(1, 2); let qk = q.matmul(k_t) * 0.125; 
        let attn = qk.tril(0).softmax(2); self.p_proj.forward(attn.matmul(v)) 
    } 
    fn step(self, lr: f32) -> CausalSelfAttention { 
        let mut s = self; s.q_proj = s.q_proj.step(lr); s.k_proj = s.k_proj.step(lr); 
        s.v_proj = s.v_proj.step(lr); s.p_proj = s.p_proj.step(lr); s 
    } 
}

struct MLP { f: Linear, p: Linear }
impl MLP { 
    fn new(d: i64) -> MLP { MLP(Linear::new(d, d*4), Linear::new(d*4, d)) } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { self.p.forward(self.f.forward(x).relu()) } 
    fn step(self, lr: f32) -> MLP { 
        let mut s = self; s.f = s.f.step(lr); s.p = s.p.step(lr); s 
    } 
}

struct Block { l1: LayerNorm, a: CausalSelfAttention, l2: LayerNorm, m: MLP }
impl Block { 
    fn new(d: i64) -> Block { Block(LayerNorm::new(d), CausalSelfAttention::new(d), LayerNorm::new(d), MLP::new(d)) } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        let x = x + self.a.forward(self.l1.forward(x)); x + self.m.forward(self.l2.forward(x)) 
    } 
    fn step(self, lr: f32) -> Block { 
        let mut s = self; s.l1 = s.l1.step(lr); s.a = s.a.step(lr); s.l2 = s.l2.step(lr); s.m = s.m.step(lr); s 
    } 
}

struct GPT4B { w: Linear, b1: Block, b2: Block, b3: Block, b4: Block, h: Linear }
impl GPT4B { 
    fn new(d: i64) -> GPT4B { 
        GPT4B(Linear::new(13, d), Block::new(d), Block::new(d), Block::new(d), Block::new(d), Linear::new(d, 13)) 
    } 
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { 
        let x = self.b1.forward(x); let x = self.b2.forward(x); let x = self.b3.forward(x); let x = self.b4.forward(x);
        self.h.forward(x)
    } 
    fn step(self, lr: f32) -> GPT4B { 
        let mut s = self; 
        print("  step:w");
        s.w = s.w.step(lr); 
        print("  step:b1");
        s.b1 = s.b1.step(lr); 
        print("  step:b2");
        s.b2 = s.b2.step(lr); 
        print("  step:b3");
        s.b3 = s.b3.step(lr); 
        print("  step:b4");
        s.b4 = s.b4.step(lr); 
        print("  step:h");
        s.h = s.h.step(lr); 
        print("  step:done");
        s 
    }
}

fn train_step(model: GPT4B, lr: f32, idx: i64) -> GPT4B {
    print("train_step:"); print(idx);
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let target = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 12.0];
    let X = data.reshape([1, 12]); let Y = target.reshape([1, 12]);
    let input = X.embedding(Tensor::randn([13, 64], false));
    print("  fwd start");
    let logits = model.forward(input);
    print("  fwd done");
    let logits_flat = logits.reshape([12, 13]); let Y_flat = Y.reshape([12]);
    let loss = logits_flat.cross_entropy(Y_flat);
    print("  backward");
    loss.backward();
    print("  step start");
    let result = model.step(lr);
    print("  step end");
    result
}

fn main() {
    let mut model = GPT4B::new(64);
    for epoch in range(0, 2) {
        for s in range(0, 2) { 
            model = train_step(model, 0.001, s); 
        }
        print("Epoch:"); print(epoch);
    }
    print("Done!");
}
