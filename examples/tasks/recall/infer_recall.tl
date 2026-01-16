// Associative Recall Task with Transformer
// Task: Input "k1 v1 k2 v2 ... k3 ?" -> Output "v3"
// This tests the model's ability to "attend" back to where "k3" appeared and retrieve "v3".

struct Linear {
    W: Tensor<f32, 2>,
    b: Tensor<f32, 1>
}

impl Linear {
    fn new(i: i64, o: i64) -> Linear {
        // Xavier initialization-ish
        return Linear((Tensor::randn([i, o], true) * 0.1).detach(true), (Tensor::randn([o], true) * 0.0).detach(true));
    }
    
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> {
        return x.matmul(self.W) + self.b;
    }
    
    fn step(self, lr: f32) -> Linear {
        let s = self;
        let gWg = s.W.grad();
        let gbg = s.b.grad();
        s.W = (s.W - gWg * lr).detach(true);
        s.b = (s.b - gbg * lr).detach(true);
        return s;
    }
}

struct Embedding {
    w: Tensor<f32, 2>
}

impl Embedding {
    fn new(v: i64, d: i64) -> Embedding {
        return Embedding((Tensor::randn([v, d], true) * 0.1).detach(true));
    }
    
    fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> {
        return i.embedding(self.w);
    }
    
    fn step(self, lr: f32) -> Embedding {
        let s = self;
        let g = s.w.grad();
        s.w = (s.w - g * lr).detach(true);
        return s;
    }
}

struct LayerNorm {
    w: Tensor<f32, 1>,
    b: Tensor<f32, 1>
}

impl LayerNorm {
    fn new(d: i64) -> LayerNorm {
        return LayerNorm((Tensor::randn([d], true)*0.0+1.0).detach(true), (Tensor::randn([d], true)*0.0).detach(true));
    }
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> {
        return x * self.w + self.b;
    }
    fn step(self, lr: f32) -> LayerNorm {
        let s = self;
        let gw = s.w.grad();
        let gb = s.b.grad();
        s.w = (s.w - gw * lr).detach(true);
        s.b = (s.b - gb * lr).detach(true);
        return s;
    }
}

struct CausalSelfAttention {
    c_attn: Linear,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    c_proj: Linear,
}

impl CausalSelfAttention {
    fn new(d: i64) -> CausalSelfAttention {
        return CausalSelfAttention(
            Linear::new(1, 1),
            Linear::new(d, d), 
            Linear::new(d, d), 
            Linear::new(d, d), 
            Linear::new(d, d)
        );
    }

    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> {
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        let k_t = k.transpose(1, 2);
        let att = q.matmul(k_t) * 0.125;
        let att = att.tril(0); 
        let att = att.softmax(2);
        let y = att.matmul(v);
        
        return self.c_proj.forward(y);
    }

    fn step(self, lr: f32) -> CausalSelfAttention {
        let s = self;
        s.q_proj = s.q_proj.step(lr);
        s.k_proj = s.k_proj.step(lr);
        s.v_proj = s.v_proj.step(lr);
        s.c_proj = s.c_proj.step(lr);
        return s;
    }
}

struct MLP {
    c_fc: Linear,
    c_proj: Linear
}

impl MLP {
    fn new(d: i64) -> MLP {
        return MLP(Linear::new(d, d * 4), Linear::new(d * 4, d));
    }
    
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> {
        return self.c_proj.forward(self.c_fc.forward(x).relu());
    }
    
    fn step(self, lr: f32) -> MLP {
        let s = self;
        s.c_fc = s.c_fc.step(lr);
        s.c_proj = s.c_proj.step(lr);
        return s;
    }
}

struct Block {
    ln1: LayerNorm,
    attn: CausalSelfAttention,
    ln2: LayerNorm,
    mlp: MLP
}

impl Block {
    fn new(d: i64) -> Block {
        return Block(LayerNorm::new(d), CausalSelfAttention::new(d), LayerNorm::new(d), MLP::new(d));
    }
    
    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> {
        let x = x + self.attn.forward(self.ln1.forward(x));
        let x = x + self.mlp.forward(self.ln2.forward(x));
        return x;
    }
    
    fn step(self, lr: f32) -> Block {
        let s = self;
        s.ln1 = s.ln1.step(lr);
        s.attn = s.attn.step(lr);
        s.ln2 = s.ln2.step(lr);
        s.mlp = s.mlp.step(lr);
        return s;
    }
}

struct GPT {
    wte: Embedding,
    wpe: Embedding,
    bg: Block,
    ln_f: LayerNorm,
    head: Linear
}

impl GPT {
    fn new(vocab_size: i64, d_model: i64, max_len: i64) -> GPT {
        return GPT(
            Embedding::new(vocab_size, d_model),
            Embedding::new(max_len, d_model),
            Block::new(d_model),
            LayerNorm::new(d_model),
            Linear::new(d_model, vocab_size)
        );
    }
    
    fn forward(self, idx: Tensor<f32, 2>) -> Tensor<f32, 3> {
        let T = 13;
        let pos_arr = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let pos = pos_arr.reshape(1, 13);
        
        let tok_emb = self.wte.forward(idx);
        let pos_emb = self.wpe.forward(pos);
        
        let x = tok_emb + pos_emb;
        let x = self.bg.forward(x);
        let x = self.ln_f.forward(x);
        let logits = self.head.forward(x);
        return logits;
    }
    
    fn step(self, lr: f32) -> GPT {
        let s = self;
        s.wte = s.wte.step(lr);
        s.wpe = s.wpe.step(lr);
        s.bg = s.bg.step(lr);
        s.ln_f = s.ln_f.step(lr);
        s.head = s.head.step(lr);
        return s;
    }
}


fn gen_data(step: i64) -> Tensor<f32, 2> {
    let seed = step * 12345 + 6789;
    
    let k1 = ((seed + 1) * 7) - ((((seed + 1) * 7) / 20) * 20) + 10;
    let v1 = ((seed + 2) * 3) - ((((seed + 2) * 3) / 10) * 10);
    
    let k2 = ((seed + 3) * 7) - ((((seed + 3) * 7) / 20) * 20) + 10;
    let v2 = ((seed + 4) * 3) - ((((seed + 4) * 3) / 10) * 10);
    
    let k3 = ((seed + 5) * 7) - ((((seed + 5) * 7) / 20) * 20) + 10;
    let v3 = ((seed + 6) * 3) - ((((seed + 6) * 3) / 10) * 10);
    
    let k4 = ((seed + 7) * 7) - ((((seed + 7) * 7) / 20) * 20) + 10;
    let v4 = ((seed + 8) * 3) - ((((seed + 8) * 3) / 10) * 10);
    
    let k5 = ((seed + 9) * 7) - ((((seed + 9) * 7) / 20) * 20) + 10;
    let v5 = ((seed + 10) * 3) - ((((seed + 10) * 3) / 10) * 10);
    
    let k6 = ((seed + 11) * 7) - ((((seed + 11) * 7) / 20) * 20) + 10;
    let v6 = ((seed + 12) * 3) - ((((seed + 12) * 3) / 10) * 10);
    
    let pick = seed - ((seed / 6) * 6);
    let q_k = 0;
    let q_ans = 0;
    if pick == 0 { q_k = k1; q_ans = v1; }
    if pick == 1 { q_k = k2; q_ans = v2; }
    if pick == 2 { q_k = k3; q_ans = v3; }
    if pick == 3 { q_k = k4; q_ans = v4; }
    if pick == 4 { q_k = k5; q_ans = v5; }
    if pick == 5 { q_k = k6; q_ans = v6; }
    
    let fk1 = k1 as f32;
    let fv1 = v1 as f32;
    let fk2 = k2 as f32;
    let fv2 = v2 as f32;
    let fk3 = k3 as f32;
    let fv3 = v3 as f32;
    let fk4 = k4 as f32;
    let fv4 = v4 as f32;
    let fk5 = k5 as f32;
    let fv5 = v5 as f32;
    let fk6 = k6 as f32;
    let fv6 = v6 as f32;
    let fqk = q_k as f32;
    let fqa = q_ans as f32;
    
    let arr = [fk1, fv1, fk2, fv2, fk3, fv3, fk4, fv4, fk5, fv5, fk6, fv6, fqk, fqa];
    return arr.reshape(1, 14);
}

fn main() {
    let vocab_size = 30;
    let d_model = 64;
    let max_len = 32;
    
    let model = GPT::new(vocab_size, d_model, max_len);
    
    print("Loading trained model weights from recall_weights.safetensors...");
    Param::load("recall_weights.safetensors");
    print("Model loaded successfully.");
    
    print("Running Inference on Validation Set...");
    print("Format: Pred / Target -> Result");
    
    let mut correct_count = 0;
    let total_count = 20;
    let start_idx = 10000;
    
    for i in range(start_idx, start_idx + total_count) {
        let val_data = gen_data(i);
        let val_in = [
            val_data.get(0), val_data.get(1),
            val_data.get(2), val_data.get(3),
            val_data.get(4), val_data.get(5),
            val_data.get(6), val_data.get(7),
            val_data.get(8), val_data.get(9),
            val_data.get(10), val_data.get(11),
            val_data.get(12)
        ];
        let val_X = val_in.reshape(1, 13);
        let val_logits = model.forward(val_X);
        
        let val_logits_flat = val_logits.reshape(13, 30);
        let val_logits_1d = val_logits_flat.reshape(390);
        let last_row = val_logits_1d.slice(360, 30); // 12 * 30 = 360
        
        // Double cast workaround for argmax U32 -> F32 issue
        let pred_t = last_row.argmax(0);
        let pred_t_i64 = pred_t as Tensor<i64, 1>;
        let pred_t_f32 = pred_t_i64 as Tensor<f32, 1>;
        let pred = pred_t_f32.get(0) as i64;
        
        let target = val_data.get(13) as i64;
        
        print("Sample:"); print(i);
        print("Pred:"); print(pred);
        print("Target:"); print(target);
        
        if pred == target {
            print("Result: Correct");
            correct_count = correct_count + 1;
        } else {
            print("Result: Wrong");
        }
    }
    
    print("Inference Complete.");
    print("Accuracy:");
    print(correct_count);
    print("Correct out of");
    print(total_count);
}
