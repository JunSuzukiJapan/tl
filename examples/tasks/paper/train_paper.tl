// Paper Architecture: 2-Layer GPT + Adam + 2桁足し算
// d_model=256, masked_fill attention, answer-only loss

struct Linear {
    W: Tensor<f32, 2>, b: Tensor<f32, 1>,
    mW: Tensor<f32, 2>, vW: Tensor<f32, 2>,
    mb: Tensor<f32, 1>, vb: Tensor<f32, 1>
}
impl Linear {
    fn new(i: i64, o: i64) -> Linear {
        Linear(
            (Tensor::randn([i, o], true)*0.1).detach(true),
            (Tensor::randn([o], true)*0.0).detach(true),
            Tensor::zeros([i, o], false),
            Tensor::zeros([i, o], false),
            Tensor::zeros([o], false),
            Tensor::zeros([o], false)
        )
    }
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> {
        x.matmul(self.W) + self.b
    }
    fn step(self, step_n: i64, lr: f32) -> Linear {
        let mut s = self;
        s.W.adam_step(s.W.grad(), s.mW, s.vW, step_n, lr, 0.9, 0.999, 1e-8, 0.0);
        s.b.adam_step(s.b.grad(), s.mb, s.vb, step_n, lr, 0.9, 0.999, 1e-8, 0.0);
        s
    }
}

struct Embedding {
    w: Tensor<f32, 2>,
    mw: Tensor<f32, 2>, vw: Tensor<f32, 2>
}
impl Embedding {
    fn new(v: i64, d: i64) -> Embedding {
        Embedding(
            (Tensor::randn([v, d], true)*0.1).detach(true),
            Tensor::zeros([v, d], false),
            Tensor::zeros([v, d], false)
        )
    }
    fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> {
        i.embedding(self.w)
    }
    fn step(self, step_n: i64, lr: f32) -> Embedding {
        let mut s = self;
        s.w.adam_step(s.w.grad(), s.mw, s.vw, step_n, lr, 0.9, 0.999, 1e-8, 0.0);
        s
    }
}

struct LayerNorm {
    w: Tensor<f32, 1>, b: Tensor<f32, 1>,
    mw: Tensor<f32, 1>, vw: Tensor<f32, 1>,
    mb: Tensor<f32, 1>, vb: Tensor<f32, 1>
}
impl LayerNorm {
    fn new(d: i64) -> LayerNorm {
        LayerNorm(
            (Tensor::randn([d], true)*0.0+1.0).detach(true),
            (Tensor::randn([d], true)*0.0).detach(true),
            Tensor::zeros([d], false),
            Tensor::zeros([d], false),
            Tensor::zeros([d], false),
            Tensor::zeros([d], false)
        )
    }
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> {
        x.layer_norm(self.w, self.b, 0.00001 as f64)
    }
    fn step(self, step_n: i64, lr: f32) -> LayerNorm {
        let mut s = self;
        s.w.adam_step(s.w.grad(), s.mw, s.vw, step_n, lr, 0.9, 0.999, 1e-8, 0.0);
        s.b.adam_step(s.b.grad(), s.mb, s.vb, step_n, lr, 0.9, 0.999, 1e-8, 0.0);
        s
    }
}

struct CausalSelfAttention { q_proj: Linear, k_proj: Linear, v_proj: Linear, p_proj: Linear }
impl CausalSelfAttention {
    fn new(d: i64) -> CausalSelfAttention {
        CausalSelfAttention(Linear::new(d, d), Linear::new(d, d), Linear::new(d, d), Linear::new(d, d))
    }
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> {
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);
        // scale = 1/sqrt(d_model) = 1/sqrt(256) = 0.0625
        let k_t = k.transpose(1, 2);
        let qk = q.matmul(k_t) * 0.0625;
        let ones = Tensor::ones_like(qk);
        let upper_tri = ones - ones.tril(0);
        let masked = qk.masked_fill(upper_tri, -f32::INFINITY);
        let attn = masked.softmax(2);
        let y = attn.matmul(v);
        self.p_proj.forward(y)
    }
    fn step(self, step_n: i64, lr: f32) -> CausalSelfAttention {
        let mut s = self;
        s.q_proj = s.q_proj.step(step_n, lr);
        s.k_proj = s.k_proj.step(step_n, lr);
        s.v_proj = s.v_proj.step(step_n, lr);
        s.p_proj = s.p_proj.step(step_n, lr);
        s
    }
}

struct MLP { f: Linear, p: Linear }
impl MLP {
    fn new(d: i64) -> MLP { MLP(Linear::new(d, d*4), Linear::new(d*4, d)) }
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> {
        self.p.forward(self.f.forward(x).relu())
    }
    fn step(self, step_n: i64, lr: f32) -> MLP {
        let mut s = self;
        s.f = s.f.step(step_n, lr); s.p = s.p.step(step_n, lr); s
    }
}

struct Block { l1: LayerNorm, a: CausalSelfAttention, l2: LayerNorm, m: MLP }
impl Block {
    fn new(d: i64) -> Block {
        Block(LayerNorm::new(d), CausalSelfAttention::new(d), LayerNorm::new(d), MLP::new(d))
    }
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> {
        let x = x + self.a.forward(self.l1.forward(x));
        x + self.m.forward(self.l2.forward(x))
    }
    fn step(self, step_n: i64, lr: f32) -> Block {
        let mut s = self;
        s.l1 = s.l1.step(step_n, lr); s.a = s.a.step(step_n, lr);
        s.l2 = s.l2.step(step_n, lr); s.m = s.m.step(step_n, lr); s
    }
}

struct GPT2 { w: Embedding, wp: Embedding, b1: Block, b2: Block, l: LayerNorm, h: Linear }
impl GPT2 {
    fn new(v: i64, d: i64) -> GPT2 {
        GPT2(Embedding::new(v, d), Embedding::new(12, d),
            Block::new(d), Block::new(d), LayerNorm::new(d), Linear::new(d, v))
    }
    fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> {
        let pos_data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let pos = pos_data.reshape([1, 12]);
        let x = self.w.forward(i) + self.wp.forward(pos);
        let x = self.b1.forward(x);
        let x = self.b2.forward(x);
        self.h.forward(self.l.forward(x))
    }
    fn step(self, step_n: i64, lr: f32) -> GPT2 {
        let mut s = self;
        s.w = s.w.step(step_n, lr); s.wp = s.wp.step(step_n, lr);
        s.b1 = s.b1.step(step_n, lr); s.b2 = s.b2.step(step_n, lr);
        s.l = s.l.step(step_n, lr); s.h = s.h.step(step_n, lr); s
    }
}

fn get_random_digit() -> i64 {
    let r = Tensor::randn([1], true)[0];
    if r < -1.28 { return 0; }
    if r >= -1.28 && r < -0.84 { return 1; }
    if r >= -0.84 && r < -0.52 { return 2; }
    if r >= -0.52 && r < -0.25 { return 3; }
    if r >= -0.25 && r < 0.0   { return 4; }
    if r >= 0.0   && r < 0.25  { return 5; }
    if r >= 0.25  && r < 0.52  { return 6; }
    if r >= 0.52  && r < 0.84  { return 7; }
    if r >= 0.84  && r < 1.28  { return 8; }
    return 9;
}

struct TrainState { global_step: i64, model: GPT2 }

fn train_epoch(state: TrainState, lr: f32, steps: i64) -> TrainState {
    let mut st = state;
    for s in range(0, steps) {
        let i = get_random_digit() * 10 + get_random_digit();
        let j = get_random_digit() * 10 + get_random_digit();

        let sum = i + j;
        let i_d10 = i / 10;
        let i_d1 = i - (i_d10 * 10);
        let j_d10 = j / 10;
        let j_d1 = j - (j_d10 * 10);
        let s_d100 = sum / 100;
        let rem = sum - (s_d100 * 100);
        let s_d10 = rem / 10;
        let s_d1 = rem - (s_d10 * 10);

        let v_i_d1 = i_d1 as f32;
        let v_i_d10 = i_d10 as f32;
        let v_j_d1 = j_d1 as f32;
        let v_j_d10 = j_d10 as f32;
        let v_s_d1 = s_d1 as f32;
        let v_s_d10 = s_d10 as f32;
        let v_s_d100 = s_d100 as f32;

        let data = [
            v_i_d1, v_i_d10, 10.0,
            v_j_d1, v_j_d10, 11.0,
            v_s_d1, v_s_d10, v_s_d100,
            12.0, 12.0, 12.0
        ];

        let X = data.reshape([1, 12]);
        let logits = st.model.forward(X);
        let logits_flat = logits.reshape([12, 13]);

        // 位置5-7(答えのdigit)のみで損失計算
        let logits_ans = logits_flat.slice(0, 5, 8, 1);
        let Y_ans = [v_s_d1, v_s_d10, v_s_d100];
        let Y_flat = Y_ans.reshape([3]);
        let loss = logits_ans.cross_entropy(Y_flat);

        loss.backward();

        if s % 4 == 3 {
            st.model = st.model.step(st.global_step, lr);
            Tensor::clear_grads();
            st.global_step = st.global_step + 1;
        }

        if s % 160 == 0 {
            print("    Loss:"); loss.print();
        }
    }
    return st;
}

fn main() {
    let vocab_size = 13;
    let d_model = 256;
    let mut state = TrainState(1, GPT2::new(vocab_size, d_model));

    if File::exists("model_paper.safetensors") {
        Param::load(state.model, "model_paper.safetensors");
        print("Loaded parameters from model_paper.safetensors");
        if File::exists("model_paper_step.txt") {
            state.global_step = File::read("model_paper_step.txt").to_i64();
            print("Restored global_step");
        }
    }

    let lr = 0.0005;
    let epochs = 1500;

    print("Paper Training: 2-layer GPT, d_model=256, Adam, lr=0.0005, epochs=1500");

    for epoch in range(0, epochs) {
        print("Epoch:"); print(epoch);
        state = train_epoch(state, lr, 160);

        if epoch % 100 == 99 {
            Param::save(state.model, "model_paper.safetensors");
            File::write("model_paper_step.txt", state.global_step.to_string());
            print("Checkpoint saved.");
        }
    }

    print("Training Complete!");
    Param::save(state.model, "model_paper.safetensors");
    File::write("model_paper_step.txt", state.global_step.to_string());
}
