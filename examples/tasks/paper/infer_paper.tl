// 2層GPT 推論 (train_paper.tl で学習したモデルを使用)

// Adam moment buffers included to match train_paper.tl struct layout
struct Linear {
    W: Tensor<f32, 2>, b: Tensor<f32, 1>,
    mW: Tensor<f32, 2>, vW: Tensor<f32, 2>,
    mb: Tensor<f32, 1>, vb: Tensor<f32, 1>
}
impl Linear {
    fn new(i: i64, o: i64) -> Linear {
        Linear(
            Tensor::zeros([i, o], false),
            Tensor::zeros([o], false),
            Tensor::zeros([i, o], false),
            Tensor::zeros([i, o], false),
            Tensor::zeros([o], false),
            Tensor::zeros([o], false)
        )
    }
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { x.matmul(self.W) + self.b }
}

struct Embedding {
    w: Tensor<f32, 2>,
    mw: Tensor<f32, 2>, vw: Tensor<f32, 2>
}
impl Embedding {
    fn new(v: i64, d: i64) -> Embedding {
        Embedding(
            Tensor::zeros([v, d], false),
            Tensor::zeros([v, d], false),
            Tensor::zeros([v, d], false)
        )
    }
    fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> { i.embedding(self.w) }
}

struct LayerNorm {
    w: Tensor<f32, 1>, b: Tensor<f32, 1>,
    mw: Tensor<f32, 1>, vw: Tensor<f32, 1>,
    mb: Tensor<f32, 1>, vb: Tensor<f32, 1>
}
impl LayerNorm {
    fn new(d: i64) -> LayerNorm {
        LayerNorm(
            Tensor::zeros([d], false),
            Tensor::zeros([d], false),
            Tensor::zeros([d], false),
            Tensor::zeros([d], false),
            Tensor::zeros([d], false),
            Tensor::zeros([d], false)
        )
    }
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> {
        x.layer_norm(self.w, self.b, 0.00001 as f64)
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
        let k_t = k.transpose(1, 2);
        let qk = q.matmul(k_t) * 0.0625;
        let ones = Tensor::ones_like(qk);
        let upper_tri = ones - ones.tril(0);
        let masked = qk.masked_fill(upper_tri, -f32::INFINITY);
        let attn = masked.softmax(2);
        let y = attn.matmul(v);
        self.p_proj.forward(y)
    }
}

struct MLP { f: Linear, p: Linear }
impl MLP {
    fn new(d: i64) -> MLP { MLP(Linear::new(d, d*4), Linear::new(d*4, d)) }
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> {
        self.p.forward(self.f.forward(x).relu())
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
}

struct GPT2 { w: Embedding, wp: Embedding, b1: Block, b2: Block, l: LayerNorm, h: Linear }
impl GPT2 {
    fn new(v: i64, d: i64) -> GPT2 {
        GPT2(
            Embedding::new(v, d),
            Embedding::new(12, d),
            Block::new(d),
            Block::new(d),
            LayerNorm::new(d),
            Linear::new(d, v)
        )
    }
    fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> {
        let pos_data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let pos = pos_data.reshape([1, 12]);
        let x = self.w.forward(i) + self.wp.forward(pos);
        let x = self.b1.forward(x);
        let x = self.b2.forward(x);
        self.h.forward(self.l.forward(x))
    }
}

fn inference(model: GPT2, i: i64, j: i64) -> i64 {
    let sum = i + j;

    let i_d10 = i / 10;
    let i_d1 = i - (i_d10 * 10);
    let j_d10 = j / 10;
    let j_d1 = j - (j_d10 * 10);

    let v_i_d1 = i_d1 as f32;
    let v_i_d10 = i_d10 as f32;
    let v_j_d1 = j_d1 as f32;
    let v_j_d10 = j_d10 as f32;

    let mut data = [
        v_i_d1, v_i_d10, 10.0,
        v_j_d1, v_j_d10, 11.0,
        12.0, 12.0, 12.0,
        12.0, 12.0, 12.0
    ];

    // Autoregressive: predict d1, d10, d100
    let X1 = data.reshape([1, 12]);
    let logits1 = model.forward(X1);
    let logits1_flat = logits1.reshape([12, 13]);
    let p_d1 = logits1_flat.slice(0, 5, 6, 1).reshape([13]).argmax(0)[0] as i64;

    data[6] = p_d1 as f32;
    let X2 = data.reshape([1, 12]);
    let logits2 = model.forward(X2);
    let logits2_flat = logits2.reshape([12, 13]);
    let p_d10 = logits2_flat.slice(0, 6, 7, 1).reshape([13]).argmax(0)[0] as i64;

    data[7] = p_d10 as f32;
    let X3 = data.reshape([1, 12]);
    let logits3 = model.forward(X3);
    let logits3_flat = logits3.reshape([12, 13]);
    let p_d100 = logits3_flat.slice(0, 7, 8, 1).reshape([13]).argmax(0)[0] as i64;

    let pred_sum = p_d100 * 100 + p_d10 * 10 + p_d1;

    print("-------");
    print("Input : "); print(i); print("+"); print(j);
    print("Target: "); print(sum);
    print("Pred  : "); print(pred_sum);

    if sum == pred_sum {
        print("Result: HIT");
        return 1;
    } else {
        print("Result: MISS");
        return 0;
    }
}

fn main() {
    let vocab_size = 13;
    let d_model = 256;

    print("Inference with Paper Settings (2-Layer, d_model=256)");
    let model = GPT2::new(vocab_size, d_model);

    if File::exists("model_paper.safetensors") {
        Param::load(model, "model_paper.safetensors");
        print("Parameters Loaded from model_paper.safetensors");
    } else {
        print("ERROR: model_paper.safetensors not found. Run train_paper.tl first.");
        return;
    }

    let mut correct_count = 0;
    correct_count = correct_count + inference(model, 12, 34);
    correct_count = correct_count + inference(model, 99, 1);

    print("Inference Complete.");
    print("Accuracy:");
    print(correct_count);
    print("Correct out of 2");
}
