
// --- Struct Definitions ---
struct Tokenizer { _h: i64 }
struct KVCache { _h: i64 }
struct File {} 
struct Map { _d: i64 }
// Dummy struct to act as namespace for Tensor operations
struct Tensor {}

struct Linear { w: i64 }
struct RMSNorm { w: Tensor<f32, 1>, e: f32 }
struct MLP { g: Linear, u: Linear, d: Linear }
struct Attention { q: Linear, k: Linear, v: Linear, o: Linear }
struct Block { an: RMSNorm, ffn: RMSNorm, at: Attention, mlp: MLP }

// --- Standard Library Wrapper Implementations ---

impl Tokenizer {
    extern fn tl_tokenizer_new(path: String) -> i64;
    extern fn tl_tokenizer_encode(tok: i64, prompt: String) -> Tensor<i64, 1>;
    extern fn tl_tokenizer_decode(tok: i64, ids: Tensor<i64, 1>) -> String;

    fn new(path: String) -> Tokenizer {
        return Tokenizer { _h: Tokenizer::tl_tokenizer_new(path) };
    }
    fn encode(self, prompt: String) -> Tensor<i64, 1> {
        return Tokenizer::tl_tokenizer_encode(self._h, prompt);
    }
    fn decode(self, ids: Tensor<i64, 1>) -> String {
        return Tokenizer::tl_tokenizer_decode(self._h, ids);
    }
}

impl Map {
    extern fn tl_tensor_map_get(map: Map, key: String) -> Tensor<f32, 2>;
    extern fn tl_tensor_map_get_1d(map: Map, key: String) -> Tensor<f32, 1>;
    extern fn tl_tensor_map_get_quantized(map: Map, key: String) -> i64;
    extern fn tl_gguf_load(path: String) -> Map;

    fn load(path: String) -> Map {
        return Map::tl_gguf_load(path);
    }
    fn get(self, key: String) -> Tensor<f32, 2> {
        return Map::tl_tensor_map_get(self, key);
    }
    fn get_1d(self, key: String) -> Tensor<f32, 1> {
        return Map::tl_tensor_map_get_1d(self, key);
    }
    fn get_quantized(self, key: String) -> i64 {
        return Map::tl_tensor_map_get_quantized(self, key);
    }
}

impl KVCache {
    extern fn tl_kv_cache_new(layers: i64) -> i64;
    extern fn tl_kv_cache_free(cache: i64);
    extern fn tl_kv_cache_get_k(cache: i64, layer: i64) -> Tensor<f32, 4>;
    extern fn tl_kv_cache_get_v(cache: i64, layer: i64) -> Tensor<f32, 4>;
    extern fn tl_kv_cache_update(cache: i64, layer: i64, k: Tensor<f32, 4>, v: Tensor<f32, 4>);

    fn new(layers: i64) -> KVCache {
        return KVCache { _h: KVCache::tl_kv_cache_new(layers) };
    }
    fn free(self) {
        KVCache::tl_kv_cache_free(self._h);
    }
    fn get_k(self, layer: i64) -> Tensor<f32, 4> {
        return KVCache::tl_kv_cache_get_k(self._h, layer);
    }
    fn get_v(self, layer: i64) -> Tensor<f32, 4> {
        return KVCache::tl_kv_cache_get_v(self._h, layer);
    }
    fn update(self, layer: i64, k: Tensor<f32, 4>, v: Tensor<f32, 4>) {
        KVCache::tl_kv_cache_update(self._h, layer, k, v);
    }
}

impl File {
    extern fn tl_file_exists_i64(path: String) -> i64;
    extern fn tl_read_file(path: String) -> String;
    extern fn tl_write_file(path: String, content: String) -> i64;
    extern fn tl_download_file(url: String, path: String) -> i64;

    fn exists(path: String) -> bool {
        return File::tl_file_exists_i64(path) == 1;
    }
    fn read(path: String) -> String {
        return File::tl_read_file(path);
    }
    fn write(path: String, content: String) -> bool {
        return File::tl_write_file(path, content) == 1;
    }
    fn download(url: String, path: String) -> bool {
        return File::tl_download_file(url, path) == 1;
    }
}

impl Tensor {
    extern fn tl_tensor_embedding(idx: Tensor<i64, 1>, table: Tensor<f32, 2>) -> Tensor<f32, 2>;
    extern fn tl_tensor_rms_norm(x: Tensor<f32, 2>, w: Tensor<f32, 1>, e: f32) -> Tensor<f32, 2>;
    extern fn tl_tensor_matmul(a: Tensor<f32, 2>, b: Tensor<f32, 2>) -> Tensor<f32, 2>;
    extern fn tl_tensor_add(a: Tensor<f32, 2>, b: Tensor<f32, 2>) -> Tensor<f32, 2>;
    extern fn tl_tensor_silu(t: Tensor<f32, 2>) -> Tensor<f32, 2>;
    extern fn tl_tensor_mul(a: Tensor<f32, 2>, b: Tensor<f32, 2>) -> Tensor<f32, 2>;
    extern fn tl_tensor_scale(a: Tensor<f32, 4>, s: f32) -> Tensor<f32, 4>;
    extern fn tl_tensor_transpose_2d(t: Tensor<f32, 2>, d1: i64, d2: i64) -> Tensor<f32, 2>;
    extern fn tl_tensor_transpose(t: Tensor<f32, 4>, d1: i64, d2: i64) -> Tensor<f32, 4>;
    extern fn tl_tensor_apply_rope(x: Tensor<f32, 4>, cos: Tensor<f32, 2>, sin: Tensor<f32, 2>) -> Tensor<f32, 4>;
    extern fn tl_tensor_repeat_interleave(t: Tensor<f32, 4>, repeats: i64, dim: i64) -> Tensor<f32, 4>;
    extern fn tl_tensor_new_causal_mask(dim: i64) -> Tensor<f32, 2>;
    extern fn tl_tensor_narrow(t: Tensor<f32, 2>, dim: i64, start: i64, len: i64) -> Tensor<f32, 2>;
    extern fn tl_tensor_cat_4d(a: Tensor<f32, 4>, b: Tensor<f32, 4>, d: i64) -> Tensor<f32, 4>;
    extern fn tl_tensor_rope_new_cos(d: i64, l: i64, t: f32) -> Tensor<f32, 2>;
    extern fn tl_tensor_rope_new_sin(d: i64, l: i64, t: f32) -> Tensor<f32, 2>;
    extern fn tl_tensor_argmax(t: Tensor<f32, 2>, d: i64, k: bool) -> Tensor<i64, 1>;
    extern fn tl_tensor_len(t: Tensor<i64, 1>) -> i64;
    extern fn tl_tensor_item_i64(t: Tensor<i64, 1>) -> i64;
    extern fn tl_tensor_cat_i64(a: Tensor<i64, 1>, b: Tensor<i64, 1>, d: i64) -> Tensor<i64, 1>;
    extern fn tl_tensor_sample(t: Tensor<f32, 2>, temp: f32, top_p: f32) -> Tensor<i64, 1>;
    extern fn tl_tensor_matmul_4d(a: Tensor<f32, 4>, b: Tensor<f32, 4>) -> Tensor<f32, 4>;
    extern fn tl_tensor_add_4d(a: Tensor<f32, 4>, b: Tensor<f32, 4>) -> Tensor<f32, 4>;
    extern fn tl_tensor_softmax(t: Tensor<f32, 4>, d: i64) -> Tensor<f32, 4>;
    extern fn tl_qtensor_matmul(input: Tensor<f32, 2>, weight: i64) -> Tensor<f32, 2>;


    fn new_rope_cos(d: i64, l: i64, t: f32) -> Tensor<f32, 2> { return Tensor::tl_tensor_rope_new_cos(d, l, t); }
    fn new_rope_sin(d: i64, l: i64, t: f32) -> Tensor<f32, 2> { return Tensor::tl_tensor_rope_new_sin(d, l, t); }
    fn new_causal_mask(dim: i64) -> Tensor<f32, 2> { return Tensor::tl_tensor_new_causal_mask(dim); }

    fn embedding(idx: Tensor<i64, 1>, table: Tensor<f32, 2>) -> Tensor<f32, 2> { return Tensor::tl_tensor_embedding(idx, table); }
    fn rms_norm(x: Tensor<f32, 2>, w: Tensor<f32, 1>, e: f32) -> Tensor<f32, 2> { return Tensor::tl_tensor_rms_norm(x, w, e); }
    fn matmul(a: Tensor<f32, 2>, b: Tensor<f32, 2>) -> Tensor<f32, 2> { return Tensor::tl_tensor_matmul(a, b); }
    fn add(a: Tensor<f32, 2>, b: Tensor<f32, 2>) -> Tensor<f32, 2> { return Tensor::tl_tensor_add(a, b); }
    fn silu(t: Tensor<f32, 2>) -> Tensor<f32, 2> { return Tensor::tl_tensor_silu(t); }
    fn mul(a: Tensor<f32, 2>, b: Tensor<f32, 2>) -> Tensor<f32, 2> { return Tensor::tl_tensor_mul(a, b); }
    fn scale(a: Tensor<f32, 4>, s: f32) -> Tensor<f32, 4> { return Tensor::tl_tensor_scale(a, s); }
    fn transpose(t: Tensor<f32, 4>, d1: i64, d2: i64) -> Tensor<f32, 4> { return Tensor::tl_tensor_transpose(t, d1, d2); }
    fn transpose_2d(t: Tensor<f32, 2>, d1: i64, d2: i64) -> Tensor<f32, 2> { return Tensor::tl_tensor_transpose_2d(t, d1, d2); }

    fn apply_rope(x: Tensor<f32, 4>, cos: Tensor<f32, 2>, sin: Tensor<f32, 2>) -> Tensor<f32, 4> { return Tensor::tl_tensor_apply_rope(x, cos, sin); }
    fn repeat_interleave(t: Tensor<f32, 4>, repeats: i64, dim: i64) -> Tensor<f32, 4> { return Tensor::tl_tensor_repeat_interleave(t, repeats, dim); }
    fn narrow(t: Tensor<f32, 2>, dim: i64, start: i64, len: i64) -> Tensor<f32, 2> { return Tensor::tl_tensor_narrow(t, dim, start, len); }
    fn cat_4d(a: Tensor<f32, 4>, b: Tensor<f32, 4>, d: i64) -> Tensor<f32, 4> { return Tensor::tl_tensor_cat_4d(a, b, d); }
    
    fn argmax(t: Tensor<f32, 2>, d: i64, k: bool) -> Tensor<i64, 1> { return Tensor::tl_tensor_argmax(t, d, k); }
    fn len(t: Tensor<i64, 1>) -> i64 { return Tensor::tl_tensor_len(t); }
    fn item_i64(t: Tensor<i64, 1>) -> i64 { return Tensor::tl_tensor_item_i64(t); }
    fn cat_i64(a: Tensor<i64, 1>, b: Tensor<i64, 1>, d: i64) -> Tensor<i64, 1> { return Tensor::tl_tensor_cat_i64(a, b, d); }
    fn sample(t: Tensor<f32, 2>, temp: f32, top_p: f32) -> Tensor<i64, 1> { return Tensor::tl_tensor_sample(t, temp, top_p); }
    
    fn matmul_4d(a: Tensor<f32, 4>, b: Tensor<f32, 4>) -> Tensor<f32, 4> { return Tensor::tl_tensor_matmul_4d(a, b); }
    fn add_4d(a: Tensor<f32, 4>, b: Tensor<f32, 4>) -> Tensor<f32, 4> { return Tensor::tl_tensor_add_4d(a, b); }
    fn softmax(t: Tensor<f32, 4>, d: i64) -> Tensor<f32, 4> { return Tensor::tl_tensor_softmax(t, d); }
    
    fn matmul_quantized(input: Tensor<f32, 2>, weight: i64) -> Tensor<f32, 2> { return Tensor::tl_qtensor_matmul(input, weight); }
}

struct System {}
impl System {
    extern fn println(s: String);
    extern fn print(s: String);
    extern fn tl_read_line(prompt: String) -> String;
    extern fn tl_string_concat(a: String, b: String) -> String;
    extern fn tl_string_from_int(i: i64) -> String;
    extern fn tl_string_contains(h: String, n: String) -> bool;
}

fn println(s: String) { System::println(s); }
fn print(s: String) { System::print(s); }
fn read_line(prompt: String) -> String { return System::tl_read_line(prompt); }
fn string_concat(a: String, b: String) -> String { return System::tl_string_concat(a, b); }
fn string_from_int(i: i64) -> String { return System::tl_string_from_int(i); }

// --- App Impls ---
impl Linear {
    fn from_map(m: Map, k: String) -> Linear { return Linear { w: m.get_quantized(k) }; }
    // Tensor::matmul_quantized
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> { 
        return Tensor::matmul_quantized(x, self.w); 
    }
}

impl RMSNorm {
    fn new(m: Map, k: String) -> RMSNorm { return RMSNorm { w: m.get_1d(k), e: 0.00001 }; }
    // Tensor::rms_norm
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> { return Tensor::rms_norm(x, self.w, self.e); }
}

impl MLP {
    fn from_map(m: Map, p: String) -> MLP {
        return MLP {
            g: Linear::from_map(m, string_concat(p, "_gate.weight")),
            u: Linear::from_map(m, string_concat(p, "_up.weight")),
            d: Linear::from_map(m, string_concat(p, "_down.weight"))
        };
    }
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> {
        let gx = self.g.forward(x.clone());
        let sx = Tensor::silu(gx);
        let ux = self.u.forward(x.clone());
        let mu = Tensor::mul(sx, ux);
        return self.d.forward(mu);
    }
}

impl Attention {
    fn from_map(m: Map, p: String) -> Attention {
        return Attention {
            q: Linear::from_map(m, string_concat(p, "_q.weight")),
            k: Linear::from_map(m, string_concat(p, "_k.weight")),
            v: Linear::from_map(m, string_concat(p, "_v.weight")),
            o: Linear::from_map(m, string_concat(p, "_output.weight"))
        };
    }
    fn forward(self, x: Tensor<f32, 2>, cos: Tensor<f32, 2>, sin: Tensor<f32, 2>, sl: i64, ch: KVCache, li: i64, sp: i64) -> Tensor<f32, 2> {
        // Llama 3: 32 heads, 128 dim/head = 4096 hidden.
        // KV Heads: 8.
        let q_4d = self.q.forward(x.clone()).reshape([1, sl, 32, 128]);
        let k_4d = self.k.forward(x.clone()).reshape([1, sl, 8, 128]);
        let v_4d = self.v.forward(x.clone()).reshape([1, sl, 8, 128]);
        
        let cs = Tensor::narrow(cos, 0, sp, sl);
        let ss = Tensor::narrow(sin, 0, sp, sl);
        
        // RoPE
        let qr = Tensor::apply_rope(q_4d, cs, ss); // Fixed: q_4d, not q_res (variable renamed or consistent?)
        // In previous attempt I used q_res. Here I use q_4d etc.
        let kr = Tensor::apply_rope(k_4d, cs, ss);
        
        let kt = Tensor::transpose(kr, 1, 2);
        let vt = Tensor::transpose(v_4d, 1, 2); // v_4d
        
        let k_tot = if sp == 0 { kt } else { Tensor::cat_4d(ch.get_k(li), kt, 2) };
        let v_tot = if sp == 0 { vt } else { Tensor::cat_4d(ch.get_v(li), vt, 2) };
        
        ch.update(li, k_tot.clone(), v_tot.clone());
        
        let qt = Tensor::transpose(qr, 1, 2);
        
        let kw = Tensor::repeat_interleave(k_tot, 4, 1);
        let vw = Tensor::repeat_interleave(v_tot, 4, 1);
        
        let sm = if sl > 1 { 
            let mm = Tensor::matmul_4d(qt, Tensor::transpose(kw, 2, 3));
            let sc = Tensor::scale(mm, 0.08839);
            let mask = Tensor::new_causal_mask(sl).reshape([1, 1, sl, sl]); // Reshape on Tensor::new_causal_mask
            Tensor::add_4d(sc, mask)
        } else { 
            let mm = Tensor::matmul_4d(qt, Tensor::transpose(kw, 2, 3));
            Tensor::scale(mm, 0.08839)
        };
        
        let sm_soft = Tensor::softmax(sm, 3);
        let ao = Tensor::transpose(Tensor::matmul_4d(sm_soft, vw), 1, 2);
        
        return self.o.forward(ao.reshape([sl, 4096]));
    }
}

impl Block {
    fn from_map(m: Map, li: i64) -> Block {
        let p = string_concat("blk.", string_from_int(li));
        return Block {
            an: RMSNorm::new(m, string_concat(p, ".attn_norm.weight")),
            ffn: RMSNorm::new(m, string_concat(p, ".ffn_norm.weight")),
            at: Attention::from_map(m, string_concat(p, ".attn")),
            mlp: MLP::from_map(m, string_concat(p, ".ffn"))
        };
    }
    fn forward(self, x: Tensor<f32, 2>, cos: Tensor<f32, 2>, sin: Tensor<f32, 2>, batch_len: i64, ch: KVCache, li: i64, sp: i64) -> Tensor<f32, 2> {
        let attn_out = self.at.forward(self.an.forward(x.clone()), cos, sin, batch_len, ch, li, sp);
        let h = Tensor::add(x.clone(), attn_out);
        let ffn_out = self.mlp.forward(self.ffn.forward(h.clone()));
        return Tensor::add(h, ffn_out);
    }
}

fn main() {
    println("Llama 3.1 8B Chatbot");
    Param::set_device(Device::Auto);
    let mut model_path = "~/.llm/models/models--lmstudio-community--Meta-Llama-3.1-8B-Instruct-GGUF/snapshots/8601e6db71269a2b12255ebdf09ab75becf22cc8/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf";
    let tokenizer_path = "~/.llm/models/llama3_tokenizer.json";
    
    let config_path = "chatbot_config.txt";
    
    // Try config first
    if File::exists(config_path) {
        let saved_path = File::read(config_path);
        if File::exists(saved_path) {
            model_path = saved_path;
        }
    }

    if File::exists(model_path) == false {
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
    
    let cos = Tensor::new_rope_cos(128, 8192, 500000.0);
    let sin = Tensor::new_rope_sin(128, 8192, 500000.0);
    
    let on = RMSNorm::new(weights, "output_norm.weight");
    println("Loaded output norm");
    let oh = Linear::from_map(weights, "output.weight");
    println("Loaded output head");
    let emb = weights.get("token_embd.weight");
    println("Loaded embedding");
    
    while true {
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
            let total_len = Tensor::len(tokens);
            let batch_len = Tensor::len(cur_tokens);
            
            let start_pos = total_len - batch_len;
            
            let xi = Tensor::embedding(cur_tokens, emb);
            let mut x = xi;
            let mut l = 0;
            while l < 32 {
                let blk = Block::from_map(weights, l);
                x = blk.forward(x, cos, sin, batch_len, cache, l, start_pos);
                l = l + 1;
            }
            let lo = oh.forward(on.forward(x));
            
            let nxt = Tensor::sample(Tensor::narrow(lo, 0, batch_len - 1, 1), 0.7, 0.9);
            
            let s_nxt = tok.decode(nxt);
            print(s_nxt);
            
            gen_str = string_concat(gen_str, s_nxt);
            
            let nxt_val = Tensor::item_i64(nxt);
            if (nxt_val == 128009 || nxt_val == 128001) {
                gen_idx = 200;
            }
            
            tokens = Tensor::cat_i64(tokens, nxt, 0);
            
            cur_tokens = nxt;
            gen_idx = gen_idx + 1;
        }
        println("");
        cache.free();
    }
}
