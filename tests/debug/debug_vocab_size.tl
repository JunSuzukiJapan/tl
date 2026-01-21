struct Map { _d: i64 }

fn main() {
    let weights = Map::load("~/.llm/models/tinyllama-1.1b-chat-q4_0.gguf");
    let emb = weights.get("output.weight");
    println(emb);
}
