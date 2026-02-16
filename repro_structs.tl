
struct MyTokenizer { _h: i64 }
struct MyKVCache { _h: i64 }
struct MyFile {} 

struct Linear { w: Tensor<i8, 2> }
struct RMSNorm { w: Tensor<f32, 1>, e: f32 }
struct MLP { g: Linear, u: Linear, d: Linear }
struct Attention { q: Linear, k: Linear, v: Linear, o: Linear }
struct Block { an: RMSNorm, ffn: RMSNorm, at: Attention, mlp: MLP }

fn main() {
    println("Structs defined. Start.");
}
