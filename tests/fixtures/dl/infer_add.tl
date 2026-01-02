struct Linear { W: Tensor<f32, 2>, b: Tensor<f32, 1> }

impl Linear { fn new(i: i64, o: i64) -> Linear { return Linear((randn([i, o], true)*0.1).detach(true), (randn([o], true)*0.0).detach(true)); } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { return matmul(x, self.W) + self.b; } fn step(self, lr: f32) -> Linear { let s = self; let gW = s.W.grad(); let gb = s.b.grad(); s.W = (s.W - gW * lr).detach(true); s.b = (s.b - gb * lr).detach(true); return s; } }
struct Embedding { w: Tensor<f32, 2> }
impl Embedding { fn new(v: i64, d: i64) -> Embedding { return Embedding((randn([v, d], true)*0.1).detach(true)); } fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> { return embedding(i, self.w); } fn step(self, lr: f32) -> Embedding { let s = self; let g = s.w.grad(); s.w = (s.w - g * lr).detach(true); return s; } }
struct LayerNorm { w: Tensor<f32, 1>, b: Tensor<f32, 1> }
impl LayerNorm { fn new(d: i64) -> LayerNorm { return LayerNorm((randn([d], true)*0.0+1.0).detach(true), (randn([d], true)*0.0).detach(true)); } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { return x + self.b; } fn step(self, lr: f32) -> LayerNorm { let s = self; let gb = s.b.grad(); s.b = (s.b - gb * lr).detach(true); return s; } }
struct CausalSelfAttention { a: Linear, p: Linear }
impl CausalSelfAttention { fn new(d: i64) -> CausalSelfAttention { return CausalSelfAttention(Linear::new(d, d*3), Linear::new(d*3, d)); } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { let q = self.a.forward(x); let k=q; let v=q; let y = matmul(softmax(tril(matmul(q, transpose(k, 1, 2))*0.125, 0), 2), v); return self.p.forward(y); } fn step(self, lr: f32) -> CausalSelfAttention { let s = self; s.a = s.a.step(lr); s.p = s.p.step(lr); return s; } }
struct MLP { f: Linear, p: Linear }
impl MLP { fn new(d: i64) -> MLP { return MLP(Linear::new(d, d*4), Linear::new(d*4, d)); } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { return self.p.forward(relu(self.f.forward(x))); } fn step(self, lr: f32) -> MLP { let s = self; s.f = s.f.step(lr); s.p = s.p.step(lr); return s; } }
struct Block { l1: LayerNorm, a: CausalSelfAttention, l2: LayerNorm, m: MLP }
impl Block { fn new(d: i64) -> Block { return Block(LayerNorm::new(d), CausalSelfAttention::new(d), LayerNorm::new(d), MLP::new(d)); } fn forward(self, x: Tensor<f32, 3>) -> Tensor<f32, 3> { let x = x + self.a.forward(self.l1.forward(x)); return x + self.m.forward(self.l2.forward(x)); } fn step(self, lr: f32) -> Block { let s = self; s.l1 = s.l1.step(lr); s.a = s.a.step(lr); s.l2 = s.l2.step(lr); s.m = s.m.step(lr); return s; } }

struct GPT { w: Embedding, b: Block, l: LayerNorm, h: Linear }
impl GPT { fn new(v: i64, d: i64) -> GPT { return GPT(Embedding::new(v, d), Block::new(d), LayerNorm::new(d), Linear::new(d, v)); } fn forward(self, i: Tensor<f32, 2>) -> Tensor<f32, 3> { return self.h.forward(self.l.forward(self.b.forward(self.w.forward(i)))); } fn step(self, lr: f32) -> GPT { let s = self; s.w = s.w.step(lr); s.b = s.b.step(lr); s.l = s.l.step(lr); s.h = s.h.step(lr); return s; } }

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

fn main() {
    let vocab_size = 13;
    let d_model = 64; 
    let block_size = 12; 
    print("Initializing Model for Inference...");
    let model = GPT::new(vocab_size, d_model);
    
    let model = GPT::new(vocab_size, d_model);
    
    // Register parameters to enable loading
    register_gpt_params(model);

    print("Loading Parameters...");
    load_all_params("model_2digit.safetensors");
    print("Parameters Loaded.");

    print("Running Inference verification 2-digit...");
    
    // Test cases: 12+34, 99+1, 5+5, 88+99
    // Indices: 0-3
    for t in range(0, 4) {
        let i = 0;
        let j = 0;

        if t == 0 { 
            // 12 + 34
            i = 12; j = 34;
        } 
        if t == 1 { 
            // 99 + 1
            i = 99; j = 1; 
        }
        if t == 2 { 
            // 5 + 5
            i = 5; j = 5; 
        }
        if t == 3 { 
            // 88 + 99
            i = 88; j = 99; 
        }
        
        // Extract digits using integer arithmetic (same as training)
        let i_d1 = pow(i / 10, 1.0).get(0);
        let i_d2 = pow(i - ((i/10)*10), 1.0).get(0);
        
        let j_d1 = pow(j / 10, 1.0).get(0);
        let j_d2 = pow(j - ((j/10)*10), 1.0).get(0);
        
        // Construct basic input part: "i + j ="
        let val_plus = 10.0;
        let val_eq = 11.0;
        let val_pad = 12.0;

        let x0=val_pad; let x1=val_pad; let x2=val_pad; let x3=val_pad; 
        let x4=val_pad; let x5=val_pad; let x6=val_pad; let x7=val_pad;
        let x8=val_pad; let x9=val_pad; let x10=val_pad; let x11=val_pad;

        let pos = 0;
        
        // i (Always 2 digits, leading zero if needed)
        x0 = i_d1; 
        x1 = i_d2;
        
        // +
        x2 = val_plus;

        // j (Always 2 digits, leading zero if needed)
        x3 = j_d1;
        x4 = j_d2;

        // =
        x5 = val_eq;
        
        pos = 6; // Next position to predict is x6 (first digit of result)
        
        // Printing for verification
        print("Input:"); print(i); print("+"); print(j);
        print("Predicted Digits:");

        // Step 1: Predict 1st result digit (at pos 6, using logits from pos 5 '=')
        // Input: x0..x5, pad...
        let data1 = [x0, x1, x2, x3, x4, x5, val_pad, val_pad, val_pad, 12.0, 12.0, 12.0];
        let input1 = reshape(data1, 1, 12);
        let logits1 = model.forward(input1);
        let logits1 = reshape(logits1, 12, 13);
        // Slice at pos 5 to get prediction for pos 6
        let next_logits1 = logits1.slice(5, 1);
        
        let pred1 = 0;
        let max_val = -1000000.0;
        for k in range(0, 13) {
            let val = next_logits1.get(k);
            if val > max_val { max_val = val; pred1 = k; }
        }
        print(pred1);

        // Step 2: Predict 2nd result digit (at pos 7, using logits from pos 6 'pred1')
        // Input: x0..x5, pred1, pad...
        // Convert pred1 (index) to tensor value (f32) if needed.
        // indices in training are float values of digits (0.0 - 9.0). 
        // pred1 is index k (0-12). 
        // Since vocab is 0-9, 10, 11, 12, index k maps directly to value k.0
        let val_pred1 = pow(pred1, 1.0).get(0);

        let data2 = [x0, x1, x2, x3, x4, x5, val_pred1, val_pad, val_pad, 12.0, 12.0, 12.0];
        let input2 = reshape(data2, 1, 12);
        let logits2 = model.forward(input2);
        let logits2 = reshape(logits2, 12, 13);
        // Slice at pos 6 to get prediction for pos 7
        let next_logits2 = logits2.slice(6, 1);

        let pred2 = 0;
        let max_val = -1000000.0;
        for k in range(0, 13) {
            let val = next_logits2.get(k);
            if val > max_val { max_val = val; pred2 = k; }
        }
        print(pred2);

        // Step 3: Predict 3rd result digit (at pos 8, using logits from pos 7 'pred2')
        // Input: x0..x5, pred1, pred2, pad...
        let val_pred2 = pow(pred2, 1.0).get(0);

        let data3 = [x0, x1, x2, x3, x4, x5, val_pred1, val_pred2, val_pad, 12.0, 12.0, 12.0];
        let input3 = reshape(data3, 1, 12);
        let logits3 = model.forward(input3);
        let logits3 = reshape(logits3, 12, 13);
        // Slice at pos 7 to get prediction for pos 8
        let next_logits3 = logits3.slice(7, 1);

        let pred3 = 0;
        let max_val = -1000000.0;
        for k in range(0, 13) {
            let val = next_logits3.get(k);
            if val > max_val { max_val = val; pred3 = k; }
        }
        print(pred3);
    }
    
    print("Inference Verification Complete.");
}
