// Transformer (NanoGPT style) Training Script for Addition

struct Linear {
    W: Tensor<f32, 2>,
    b: Tensor<f32, 1>,
}

impl Linear {
    fn new(in_dim: i64, out_dim: i64) -> Linear {
        return Linear(randn([in_dim, out_dim], true) * 0.1, randn([out_dim], true) * 0.0);
    }

    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> {
        return matmul(x, self.W) + self.b;
    }

    fn step(self, lr: f32) {
        let gW = self.W.grad();
        let gb = self.b.grad();
        self.W = (self.W - gW * lr).detach();
        self.b = (self.b - gb * lr).detach();
    }
}

struct Embedding {
    weight: Tensor<f32, 2>,
}

impl Embedding {
    fn new(vocab: i64, d_model: i64) -> Embedding {
        return Embedding(randn([vocab, d_model], true) * 0.1);
    }

    fn forward(self, idx: Tensor<f32, 2>) -> Tensor<f32, 3> {
        return embedding(idx, self.weight);
    }
    
    fn step(self, lr: f32) {
        let g = self.weight.grad();
        self.weight = (self.weight - g * lr).detach();
    }
}

struct LayerNorm {
    weight: Tensor<f32, 1>,
    bias: Tensor<f32, 1>,
}

impl LayerNorm {
    fn new(d_model: i64) -> LayerNorm {
        return LayerNorm(randn([d_model], true) * 0.0 + 1.0, randn([d_model], true) * 0.0);
    }

    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> {
        return x + self.bias;
    }
    
    fn step(self, lr: f32) {
        let gb = self.bias.grad();
        self.bias = (self.bias - gb * lr).detach();
    }
}

struct CausalSelfAttention {
    c_attn: Linear,
    c_proj: Linear,
}

impl CausalSelfAttention {
    fn new(d_model: i64) -> CausalSelfAttention {
        return CausalSelfAttention(
            Linear::new(d_model, d_model * 3),
            Linear::new(d_model * 3, d_model)
        );
    }

    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> {
        let qkv = self.c_attn.forward(x);
        let Q = qkv; 
        let K = qkv;
        let V = qkv;
        
        let K_T = transpose(K, 1, 2);
        let att = matmul(Q, K_T) * 0.125;
        let masked_att = tril(att, 0);
        let prob = softmax(masked_att, 2);
        let y = matmul(prob, V);
        return self.c_proj.forward(y);
    }
    
    fn step(self, lr: f32) {
        self.c_attn.step(lr);
        self.c_proj.step(lr);
    }
}

struct MLP {
    c_fc: Linear,
    c_proj: Linear,
}

impl MLP {
    fn new(d_model: i64) -> MLP {
        return MLP(
            Linear::new(d_model, d_model * 4),
            Linear::new(d_model * 4, d_model)
        );
    }

    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> {
        let h = self.c_fc.forward(x);
        let h = relu(h);
        return self.c_proj.forward(h);
    }
    
    fn step(self, lr: f32) {
        self.c_fc.step(lr);
        self.c_proj.step(lr);
    }
}

struct Block {
    ln1: LayerNorm,
    attn: CausalSelfAttention,
    ln2: LayerNorm,
    mlp: MLP,
}

impl Block {
    fn new(d_model: i64) -> Block {
        return Block(
            LayerNorm::new(d_model),
            CausalSelfAttention::new(d_model),
            LayerNorm::new(d_model),
            MLP::new(d_model)
        );
    }

    fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> {
        let x = x + self.attn.forward(self.ln1.forward(x));
        let x = x + self.mlp.forward(self.ln2.forward(x));
        return x;
    }
    
    fn step(self, lr: f32) {
        self.ln1.step(lr);
        self.attn.step(lr);
        self.ln2.step(lr);
        self.mlp.step(lr);
    }
}

struct GPT {
    wte: Embedding,
    wpe: Embedding,
    b1: Block,
    b2: Block,
    ln_f: LayerNorm,
    lm_head: Linear,
}

impl GPT {
    fn new(vocab_size: i64, d_model: i64, block_size: i64) -> GPT {
        return GPT(
            Embedding::new(vocab_size, d_model),
            Embedding::new(block_size, d_model),
            Block::new(d_model),
            Block::new(d_model),
            LayerNorm::new(d_model),
            Linear::new(d_model, vocab_size)
        );
    }

    fn forward(self, idx: Tensor<f32, 2>) -> Tensor<f32, 3> {
        let x = self.wte.forward(idx);
        let x = self.b1.forward(x);
        let x = self.b2.forward(x);
        let x = self.ln_f.forward(x);
        let logits = self.lm_head.forward(x);
        return logits;
    }
    
    fn step(self, lr: f32) {
        self.wte.step(lr);
        self.b1.step(lr);
        self.b2.step(lr);
        self.ln_f.step(lr);
        self.lm_head.step(lr);
    }
}

fn main() {
    let vocab_size = 14; 
    let d_model = 16;
    let block_size = 8;
    
    let model = GPT::new(vocab_size, d_model, block_size);
    let lr = 0.01;
    let epochs = 5;
    
    print("Starting Training on Random Noise...");
    
    let X = randn([2, 8], false) * 0.0 + 1.0; 
    let Y = randn([2, 8], false) * 0.0 + 2.0; 
    
    for i in range(0, epochs) {
        let logits = model.forward(X); 
        
        let logits_flat = reshape(logits, 16, 14);
        let Y_flat = reshape(Y, 16);
        
        let loss = cross_entropy(logits_flat, Y_flat);
        
        print("Epoch:");
        print(i);
        print("Loss:");
        print(loss);
        
        backward(loss);
        
        model.step(lr);
    }
}
