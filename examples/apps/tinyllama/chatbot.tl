
// --- Struct Definitions ---
struct Tokenizer { _h: i64 }
struct KVCache { _h: i64 }
struct File {} 
// struct Map { _d: i64 } // Builtin
struct Tensor {} // Dummy namespace

struct Linear { w: Tensor<f32, 2> }
struct RMSNorm { w: Tensor<f32, 1>, e: f32 }
struct MLP { g: Linear, u: Linear, d: Linear }
struct Attention { q: Linear, k: Linear, v: Linear, o: Linear }
struct Block { an: RMSNorm, ffn: RMSNorm, at: Attention, mlp: MLP }

// --- Standard Library Wrapper Implementations ---

fn string_concat(a: String, b: String) -> String { return a.concat(b); }
fn string_from_int(i: i64) -> String { return String::from_int(i); }
fn string_contains(h: String, n: String) -> bool { return h.contains(n); }

// --- App Impls ---
impl Linear {
    fn from_map(m: Map, k: String) -> Linear { return Linear { w: m.get(k).transpose_2d(0, 1) }; }
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> { 
        return x.matmul(self.w); 
    }
}

impl RMSNorm {
    fn new(m: Map, k: String) -> RMSNorm { return RMSNorm { w: m.get_1d(k), e: 0.00001 }; }
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> { return x.rms_norm(self.w, self.e); }
}

impl MLP {
    fn from_map(m: Map, p: String) -> MLP {
        let g = Linear::from_map(m, string_concat(p, "_gate.weight"));
        let u = Linear::from_map(m, string_concat(p, "_up.weight"));
        let d = Linear::from_map(m, string_concat(p, "_down.weight"));
        return MLP {
            g: g,
            u: u,
            d: d
        };
    }
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> {
        let gx = self.g.forward(x.clone());
        let sx = gx.silu();
        let ux = self.u.forward(x.clone());
        let mu = sx.mul(ux);
        return self.d.forward(mu);
    }
}

impl Attention {
    fn from_map(m: Map, p: String) -> Attention {
        let q = Linear::from_map(m, string_concat(p, "_q.weight"));
        let k = Linear::from_map(m, string_concat(p, "_k.weight"));
        let v = Linear::from_map(m, string_concat(p, "_v.weight"));
        let o = Linear::from_map(m, string_concat(p, "_output.weight"));
        return Attention { q: q, k: k, v: v, o: o };
    }
    fn forward(self, x: Tensor<f32, 2>, cos: Tensor<f32, 2>, sin: Tensor<f32, 2>, sl: i64, ch: KVCache, li: i64, sp: i64) -> Tensor<f32, 2> {
        let q_4d = self.q.forward(x.clone()).reshape([1, sl, 32, 64]);
        let k_4d = self.k.forward(x.clone()).reshape([1, sl, 4, 64]);
        let v_4d = self.v.forward(x.clone()).reshape([1, sl, 4, 64]);
        
        let cs = cos.narrow(0, sp, sl);
        let ss = sin.narrow(0, sp, sl);
        
        let qr = q_4d.apply_rope(cs, ss);
        let kr = k_4d.apply_rope(cs, ss);
        
        let kt = kr.transpose(1, 2);
        let vt = v_4d.transpose(1, 2);
        
        let k_tot = if sp == 0 { kt } else { ch.get_k(li).cat_4d(kt, 2) };
        let v_tot = if sp == 0 { vt } else { ch.get_v(li).cat_4d(vt, 2) };
        
        ch.update(li, k_tot.clone(), v_tot.clone());
        
        let qt = qr.transpose(1, 2);
        
        let kw = k_tot.repeat_interleave(8, 1);
        let vw = v_tot.repeat_interleave(8, 1);
        
        let sm = if sl > 1 { 
            let mm = qt.matmul_4d(kw.transpose(2, 3));
            let sc = mm.scale(0.125);
            let mask = Tensor::new_causal_mask(sl).reshape([1, 1, sl, sl]);
            sc.add_4d(mask)
        } else { 
            let mm = qt.matmul_4d(kw.transpose(2, 3));
            mm.scale(0.125)
        };
        
        let sm_soft = sm.softmax(3);
        let ao = sm_soft.matmul_4d(vw).transpose(1, 2);
        
        return self.o.forward(ao.reshape([sl, 2048]));
    }
}

impl Block {
    fn from_map(m: Map, li: i64) -> Block {
        let p = string_concat("blk.", string_from_int(li));
        let an = RMSNorm::new(m, string_concat(p, ".attn_norm.weight"));
        let ffn = RMSNorm::new(m, string_concat(p, ".ffn_norm.weight"));
        let at = Attention::from_map(m, string_concat(p, ".attn"));
        let mlp = MLP::from_map(m, string_concat(p, ".ffn"));
        return Block {
            an: an,
            ffn: ffn,
            at: at,
            mlp: mlp
        };
    }
    fn forward(self, x: Tensor<f32, 2>, cos: Tensor<f32, 2>, sin: Tensor<f32, 2>, sl: i64, ch: KVCache, li: i64, sp: i64) -> Tensor<f32, 2> {
        let attn_out = self.at.forward(self.an.forward(x.clone()), cos, sin, sl, ch, li, sp);
        let x2 = x.add(attn_out);
        let ffn_out = self.mlp.forward(self.ffn.forward(x2.clone()));
        return x2.add(ffn_out);
    }
}

fn main() {
    println("TinyLlama Chatbot");
    Param::set_device(Device::Auto);

    let mut model_path = "~/.llm/models/tinyllama/tinyllama-1.1b-chat-q4_0.gguf";
    let tokenizer_path = "~/.llm/models/tinyllama/tokenizer.json";
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
                
                let filename = "tinyllama-1.1b-chat-q4_0.gguf";
                let save_path = string_concat(dl_dir, filename);
                
                let url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
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
    let cos = Tensor::rope_new_cos(64, 2048, 10000.0);
    let sin = Tensor::rope_new_sin(64, 2048, 10000.0);
    let on = RMSNorm::new(weights, "output_norm.weight");
    let oh = Linear::from_map(weights, "output.weight");
    let emb = weights.get("token_embd.weight");

    // Pre-load blocks (Unrolled for perf/safety)
    let b0 = Block::from_map(weights, 0);
    let b1 = Block::from_map(weights, 1);
    let b2 = Block::from_map(weights, 2);
    let b3 = Block::from_map(weights, 3);
    let b4 = Block::from_map(weights, 4);
    let b5 = Block::from_map(weights, 5);
    let b6 = Block::from_map(weights, 6);
    let b7 = Block::from_map(weights, 7);
    let b8 = Block::from_map(weights, 8);
    let b9 = Block::from_map(weights, 9);
    let b10 = Block::from_map(weights, 10);
    let b11 = Block::from_map(weights, 11);
    let b12 = Block::from_map(weights, 12);
    let b13 = Block::from_map(weights, 13);
    let b14 = Block::from_map(weights, 14);
    let b15 = Block::from_map(weights, 15);
    let b16 = Block::from_map(weights, 16);
    let b17 = Block::from_map(weights, 17);
    let b18 = Block::from_map(weights, 18);
    let b19 = Block::from_map(weights, 19);
    let b20 = Block::from_map(weights, 20);
    let b21 = Block::from_map(weights, 21);
    
    while true {
        let input_str = read_line("User> ");
        if (input_str == "exit" || input_str == "quit") {
            return;
        }
        
        // Vicuna/Alpaca format (avoids special token splitting issues)
        let p1 = "A chat between a curious user and an assistant.\n\nUSER: ";
        let p2 = "\nASSISTANT:";
        let prompt = string_concat(p1, string_concat(input_str, p2));
        let mut tokens = tok.encode(prompt);
        let cache = KVCache::new(22);
        let mut cur_tokens = tokens.clone();
        let mut gen_idx = 0;
        let mut gen_str = "";
        print("Assistant> ");
        while gen_idx < 100 {
            let total_len = tokens.len();
            let batch_len = cur_tokens.len();
            let start_pos = total_len - batch_len;
            let xi = cur_tokens.embedding(emb);
            let mut x = xi;
            x = b0.forward(x, cos, sin, batch_len, cache, 0, start_pos);
            x = b1.forward(x, cos, sin, batch_len, cache, 1, start_pos);
            x = b2.forward(x, cos, sin, batch_len, cache, 2, start_pos);
            x = b3.forward(x, cos, sin, batch_len, cache, 3, start_pos);
            x = b4.forward(x, cos, sin, batch_len, cache, 4, start_pos);
            x = b5.forward(x, cos, sin, batch_len, cache, 5, start_pos);
            x = b6.forward(x, cos, sin, batch_len, cache, 6, start_pos);
            x = b7.forward(x, cos, sin, batch_len, cache, 7, start_pos);
            x = b8.forward(x, cos, sin, batch_len, cache, 8, start_pos);
            x = b9.forward(x, cos, sin, batch_len, cache, 9, start_pos);
            x = b10.forward(x, cos, sin, batch_len, cache, 10, start_pos);
            x = b11.forward(x, cos, sin, batch_len, cache, 11, start_pos);
            x = b12.forward(x, cos, sin, batch_len, cache, 12, start_pos);
            x = b13.forward(x, cos, sin, batch_len, cache, 13, start_pos);
            x = b14.forward(x, cos, sin, batch_len, cache, 14, start_pos);
            x = b15.forward(x, cos, sin, batch_len, cache, 15, start_pos);
            x = b16.forward(x, cos, sin, batch_len, cache, 16, start_pos);
            x = b17.forward(x, cos, sin, batch_len, cache, 17, start_pos);
            x = b18.forward(x, cos, sin, batch_len, cache, 18, start_pos);
            x = b19.forward(x, cos, sin, batch_len, cache, 19, start_pos);
            x = b20.forward(x, cos, sin, batch_len, cache, 20, start_pos);
            x = b21.forward(x, cos, sin, batch_len, cache, 21, start_pos);
            let lo = oh.forward(on.forward(x));
            let nxt = lo.narrow(0, batch_len - 1, 1).sample(0.7, 0.9);
            let tid = nxt.item_i64();
            let s_nxt = tok.decode(nxt);
            print(s_nxt);
            
            // Stop String Check (Vicuna format)
            gen_str = string_concat(gen_str, s_nxt);
            if string_contains(gen_str, "USER:") {
                gen_idx = 200; // Break loop
            }
            if string_contains(gen_str, "ASSISTANT:") {
                gen_idx = 200; // Break loop
            }

            if tid == 2 { gen_idx = 200; }
            tokens = tokens.cat_i64(nxt, 0);
            cur_tokens = nxt;
            gen_idx = gen_idx + 1;
        }
        cache.free();
        println("");
    }
}
