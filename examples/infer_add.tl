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

fn main() {
    let vocab_size = 13;
    let d_model = 64; 
    let block_size = 12; 
    print("Initializing Model for Inference...");
    let model = GPT::new(vocab_size, d_model);
    
    print("Loading Parameters...");
    load_all_params("model_2digit.safetensors");
    print("Parameters Loaded.");

    print("Running Inference verification 2-digit...");
    
    // Test cases: 12+34, 99+1, 5+5, 88+99
    // Indices: 0-3
    for t in range(0, 4) {
        let i = 0.0;
        let j = 0.0;
        
        if t == 0 { i = 12.0; j = 34.0; } // 46
        if t == 1 { i = 99.0; j = 1.0; } // 100
        if t == 2 { i = 5.0; j = 5.0; } // 10
        if t == 3 { i = 88.0; j = 99.0; } // 187
        
        // Construct basic input part: "i + j ="
        let val_plus = 10.0;
        let val_eq = 11.0;
        let val_pad = 12.0;

        // i digits
        let i_d1 = 12.0; let i_d2 = 12.0;
        if i < 10 {
            i_d1 = pow(i, 1.0).get(0);
        } else {
             let tens = i / 10;
             let units = i - (tens * 10);
             i_d1 = pow(tens, 1.0).get(0);
             i_d2 = pow(units, 1.0).get(0);
        }

        // j digits
        let j_d1 = 12.0; let j_d2 = 12.0;
        if j < 10 {
            j_d1 = pow(j, 1.0).get(0);
        } else {
             let tens = j / 10;
             let units = j - (tens * 10);
             j_d1 = pow(tens, 1.0).get(0);
             j_d2 = pow(units, 1.0).get(0);
        }

        let x0=val_pad; let x1=val_pad; let x2=val_pad; let x3=val_pad; 
        let x4=val_pad; let x5=val_pad; let x6=val_pad; let x7=val_pad;
        let x8=val_pad; let x9=val_pad; let x10=val_pad; let x11=val_pad;

        let pos = 0;
        // i
        if i < 10 { x0 = i_d1; pos = 1; } 
        else { x0 = i_d1; x1 = i_d2; pos = 2; }
        
        // +
        if pos == 1 { x1 = val_plus; pos = 2; } 
        else { x2 = val_plus; pos = 3; }

        // j
        if j < 10 {
            if pos == 2 { x2 = j_d1; pos = 3; }
            else { x3 = j_d1; pos = 4; }
        } else {
            if pos == 2 { x2 = j_d1; x3 = j_d2; pos = 4; }
            else { x3 = j_d1; x4 = j_d2; pos = 5; }
        }

        // =
        if pos == 3 { x3 = val_eq; pos = 4; }
        else {
            if pos == 4 { x4 = val_eq; pos = 5; }
            else { x5 = val_eq; pos = 6; }
        }
        
        // Start Autoregressive decoding for 3 steps (max result length 3)
        // At pos, we predict r1.
        // Then we feed r1 at pos, predict r2 at pos+1...
        
        print("---");
        print("Input:"); print(i); print("+"); print(j);
        print("Predicted Digits:");

        let pred_pos_start = pos;
        
        // Loop max 3 digits
        for step in range(0, 3) {
            let data = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11];
            let X = reshape(data, 1, 12);
            let logits = model.forward(X);
            let out_flat = reshape(logits, 156); // 12*13
            
            // Check prediction at current pos
            let current_pos_idx = 0;
            if step == 0 { current_pos_idx = pred_pos_start; }
            if step == 1 { current_pos_idx = pred_pos_start + 1; }
            if step == 2 { current_pos_idx = pred_pos_start + 2; }
            
            if current_pos_idx >= 12 {
                // Out of bounds (e.g. if input shifted too far)
            } else {
                let offset = current_pos_idx * 13;
                let max_val = -1000.0;
                let argmax = 0;
                
                for k in range(0, 13) {
                    let idx = offset + k;
                    let v = out_flat[idx]; 
                    if v > max_val {
                        max_val = v;
                        argmax = k;
                    }
                }
                
                print(argmax);
                
                // If predicted PAD, stop? Or just continue filling.
                if argmax == 12 {
                   // break; // Can't break loop easily in tl? 
                }

                // Append argmax to input for next step
                let val_argmax = pow(argmax, 1.0).get(0);
                
                if current_pos_idx == 4 { x4 = val_argmax; }
                if current_pos_idx == 5 { x5 = val_argmax; }
                if current_pos_idx == 6 { x6 = val_argmax; }
                if current_pos_idx == 7 { x7 = val_argmax; }
                if current_pos_idx == 8 { x8 = val_argmax; }
            }
        }
    }
    
    print("Inference Verification Complete.");
}
