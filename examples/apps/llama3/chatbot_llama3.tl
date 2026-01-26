
// --- Struct Definitions ---
struct Tokenizer { _h: i64 }
struct KVCache { _h: i64 }
struct File {} 
// struct Map { _d: i64 }
// Dummy struct to act as namespace for Tensor operations
// struct Tensor {}

struct Linear { k: String }
struct RMSNorm { k: String, e: f32 }
struct MLP { p: String }
struct Attention { p: String }
struct Block { i: i64 }

// --- Standard Library Wrapper Implementations ---

fn string_concat(a: String, b: String) -> String { return a.concat(b); }
fn string_from_int(i: i64) -> String { return String::from_int(i); }

// --- App Impls ---
impl Linear {
    fn new(k: String) -> Linear { return Linear { k: k }; }
    // tensor.matmul_quantized
    fn forward(self, x: Tensor<f32, 2>, m: Map) -> Tensor<f32, 2> { 
        let w = m.get_quantized(self.k);
        return x.matmul_quantized(w); 
    }
}

impl RMSNorm {
    fn new(k: String) -> RMSNorm { 
        return RMSNorm { k: k, e: 0.00001 }; 
    }
    // Tensor::rms_norm
    fn forward(self, x: Tensor<f32, 2>, m: Map) -> Tensor<f32, 2> { 
        let w = m.get_1d(self.k);
        return x.rms_norm(w, self.e);
    }
}

impl MLP {
    fn new(p: String) -> MLP { return MLP { p: p }; }
    fn forward(self, x: Tensor<f32, 2>, m: Map) -> Tensor<f32, 2> {
        let g = Linear::new(string_concat(self.p, "_gate.weight"));
        let u = Linear::new(string_concat(self.p, "_up.weight"));
        let d = Linear::new(string_concat(self.p, "_down.weight"));
        
        let gx = g.forward(x.clone(), m);
        let sx = gx.silu();
        let ux = u.forward(x.clone(), m);
        let mu = sx.mul(ux);
        return d.forward(mu, m);
    }
}

impl Attention {
    fn new(p: String) -> Attention { return Attention { p: p }; }
    fn forward(self, x: Tensor<f32, 2>, cos: Tensor<f32, 2>, sin: Tensor<f32, 2>, sl: i64, ch: KVCache, li: i64, sp: i64, m: Map) -> Tensor<f32, 2> {
        let q = Linear::new(string_concat(self.p, "_q.weight"));
        let k = Linear::new(string_concat(self.p, "_k.weight"));
        let v = Linear::new(string_concat(self.p, "_v.weight"));
        let o = Linear::new(string_concat(self.p, "_output.weight"));

        // Llama 3: 32 heads, 128 dim/head = 4096 hidden.
        // KV Heads: 8.
        let q_4d = q.forward(x.clone(), m).reshape([1, sl, 32, 128]);
        let k_4d = k.forward(x.clone(), m).reshape([1, sl, 8, 128]);
        let v_4d = v.forward(x.clone(), m).reshape([1, sl, 8, 128]);
        
        let cs = cos.clone().narrow(0, sp, sl);
        let ss = sin.clone().narrow(0, sp, sl);
        
        // RoPE
        let qr = q_4d.clone().apply_rope(cs.clone(), ss.clone()); 
        let kr = k_4d.clone().apply_rope(cs.clone(), ss.clone());
        
        let kt = kr.transpose(1, 2);
        let vt = v_4d.transpose(1, 2); // v_4d
        
        let k_tot = if sp == 0 { kt } else { ch.get_k(li).cat_4d(kt.clone(), 2) };
        let v_tot = if sp == 0 { vt } else { ch.get_v(li).cat_4d(vt.clone(), 2) };
        
        ch.update(li, k_tot.clone(), v_tot.clone());
        
        let qt = qr.transpose(1, 2);
        
        let kw = k_tot.clone().repeat_interleave(4, 1);
        let vw = v_tot.clone().repeat_interleave(4, 1);
        
        let sm = if sl > 1 { 
            let mm = qt.clone().matmul_4d(kw.clone().transpose(2, 3));
            let sc = mm.scale(0.08839);
            let mask = Tensor::new_causal_mask(sl).reshape([1, 1, sl, sl]); // Reshape on Tensor::new_causal_mask
            sc.add_4d(mask)
        } else { 
            let mm = qt.clone().matmul_4d(kw.clone().transpose(2, 3));
            mm.scale(0.08839)
        };
        
        let sm_soft = sm.softmax(3);
        let ao = sm_soft.matmul_4d(vw).transpose(1, 2);
        
        return o.forward(ao.reshape([sl, 4096]), m);
    }
}

impl Block {
    // Stateless Block just needs index
    fn new(i: i64) -> Block { return Block { i: i }; }
    
    fn forward(self, x: Tensor<f32, 2>, cos: Tensor<f32, 2>, sin: Tensor<f32, 2>, batch_len: i64, ch: KVCache, sp: i64, weights: Map) -> Tensor<f32, 2> {
        let p = string_concat("blk.", string_from_int(self.i));
        
        let an = RMSNorm::new(string_concat(p, ".attn_norm.weight"));
        let ffn = RMSNorm::new(string_concat(p, ".ffn_norm.weight"));
        let at = Attention::new(string_concat(p, ".attn"));
        let mlp = MLP::new(string_concat(p, ".ffn"));
        
        let attn_out = at.forward(an.forward(x.clone(), weights), cos, sin, batch_len, ch, self.i, sp, weights);
        let h = x.clone().add(attn_out);
        let ffn_out = mlp.forward(ffn.forward(h.clone(), weights), weights);
        return h.add(ffn_out);
    }
}

fn main() {
    println("Llama 3.1 8B Chatbot");
    Param::set_device(Device::Auto);
    let mut model_path = "~/.llm/models/llama3/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf";
    let tokenizer_path = "~/.llm/models/llama3/tokenizer.json";
    
    let config_path = "chatbot_config.txt";
    
    // Try config first
    if File::exists(config_path) {
        let saved_path = File::read(config_path);
        if File::exists(saved_path) {
            model_path = saved_path;
        }
    }

    if !File::exists(model_path) {
        println("Model file not found at:");
        println(model_path);
        println("Please select:");
        println("1. Download Model");
        println("2. Load from another folder");
        
        while true {
            let choice = read_line("Select [1/2]> ");
            if choice == "1" {
                println("Downloading model...");
                println("Enter download directory path (or press Enter for default ~/.llm/models/):");
                let dl_dir_in = read_line("Path> ");
                let dl_dir = if dl_dir_in == "" { "~/.llm/models/" } else { dl_dir_in };
                
                let filename = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf";
                let save_path = string_concat(dl_dir, filename);
                
                let url = "https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf";
                if File::download(url, save_path) {
                    println("Download complete.");
                    model_path = save_path;
                    File::write(config_path, model_path);
                    break;
                } else {
                    println("Download failed.");
                    println("Please try manual download from:");
                    println(url);
                    return;
                }
            } else {
                if choice == "2" {
                    let new_path = read_line("Enter model absolute path> ");
                    if File::exists(new_path) {
                        model_path = new_path;
                        File::write(config_path, model_path);
                        break;
                    } else {
                        println("File not found. Please try again.");
                    }
                } else {
                    println("Invalid selection.");
                }
            }
        }
    }
    
    let weights = Map::load(model_path);
    let tok = Tokenizer::new(tokenizer_path);
    
    let cos = Tensor::rope_new_cos(128, 8192, 500000.0);
    let sin = Tensor::rope_new_sin(128, 8192, 500000.0);
    
    let on = RMSNorm::new("output_norm.weight");
    println("Loaded output norm");
    let oh = Linear::new("output.weight");
    println("Loaded output head");
    let emb = weights.get("token_embd.weight");
    println("Loaded embedding");
    
    while true {
        println("Mem: {} MB (pool={}, in_use={})", System::memory_mb(), System::pool_count(), System::refcount_count());
        let input_str = read_line("User> ");
        if (input_str == "exit" || input_str == "quit") {
            return;
        }
        
        let p1 = "<|start_header_id|>user<|end_header_id|>\n\n";
        let p2 = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
        let prompt = string_concat(p1, string_concat(input_str, p2));
        
        let mut tokens = tok.encode(prompt);
        let cache = KVCache::new(32); 
        let mut cur_tokens = tokens.clone();
        let mut gen_idx = 0;
        let mut gen_str = "";
        print("Assistant> ");
        while gen_idx < 100 {
            let total_len = tokens.len();
            let batch_len = cur_tokens.len();
            
            let start_pos = total_len - batch_len;
            
            let xi = cur_tokens.embedding(emb);
            let mut x = xi.clone();
            let mut l = 0;
            while l < 32 {
                let blk = Block::new(l);
                x = blk.forward(x, cos.clone(), sin.clone(), batch_len, cache, start_pos, weights);
                l = l + 1;
            }
            let lo = oh.forward(on.forward(x, weights), weights);
            
            let nxt = lo.narrow(0, batch_len - 1, 1).sample(0.7, 0.9);
            
            let s_nxt = tok.decode(nxt);
            print(s_nxt);
            
            gen_str = string_concat(gen_str, s_nxt);
            
            let nxt_val = nxt.item_i64();
            if (nxt_val == 128009 || nxt_val == 128001) {
                gen_idx = 200;
            }
            
            tokens = tokens.cat_i64(nxt, 0);
            
            cur_tokens = nxt;
            gen_idx = gen_idx + 1;
        }
        println("");
        // cache.free(); // Automatic cleanup handles this
    }
}
