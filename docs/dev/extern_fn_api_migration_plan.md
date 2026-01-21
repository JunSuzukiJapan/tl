# tl_* API メソッド化 設計メモ (案)

目的: tl_* のグローバル関数呼び出しを廃止し、適切なクラスメソッド/インスタンスメソッドに整理する。
(例: Tokenizer::new / tok.encode / map.get)

## 1. 所属先の決定ルール
- **リソース/ハンドルを保持するもの**: `struct` を持ち、**インスタンスメソッド**化。
- **生成・読み込み**: **クラスメソッド**化。
- **純粋なユーティリティ**: 既存の `System`/`Args`/`Arena` などの静的 API に寄せる。
- **Tensor 操作**: 可能な限り `Tensor` の**インスタンスメソッド**。生成系は `Tensor::` の**クラスメソッド**。
- **String 操作**: `String` の**インスタンス**/`String::` の**クラス**。

## 2. API 移行マッピング (所属先/種別/シグネチャ)

### Tokenizer
- **クラス**: `Tokenizer::new(path: String) -> Tokenizer`
- **インスタンス**: `tokenizer.encode(prompt: String) -> Tensor<i64, 1>`
- **インスタンス**: `tokenizer.decode(ids: Tensor<i64, 1>) -> String`

### Map (GGUF)
- **クラス**: `Map::load(path: String) -> Map`
- **インスタンス**: `map.get(key: String) -> Tensor<f32, 2>`
- **インスタンス**: `map.get_1d(key: String) -> Tensor<f32, 1>`
- **インスタンス**: `map.get_quantized(key: String) -> i64`

### KVCache
- **クラス**: `KVCache::new(layers: i64) -> KVCache`
- **インスタンス**: `cache.free()`
- **インスタンス**: `cache.get_k(layer: i64) -> Tensor<f32, 4>`
- **インスタンス**: `cache.get_v(layer: i64) -> Tensor<f32, 4>`
- **インスタンス**: `cache.update(layer: i64, k: Tensor<f32, 4>, v: Tensor<f32, 4>)`

### File / Path / Http
- **クラス**: `File::exists(path: String) -> bool`
- **クラス**: `File::read(path: String) -> String`
- **クラス**: `File::write(path: String, content: String) -> bool`
- **クラス**: `File::download(url: String, path: String) -> bool`
- **クラス**: `File::read_binary(path: String) -> Vec<u8>`
- **クラス**: `Path::exists(path: String) -> bool`
- **クラス**: `Http::download(url: String, path: String) -> bool`

### String
- **クラス**: `String::from_int(i: i64) -> String`
- **インスタンス**: `s.concat(other: String) -> String`
- **インスタンス**: `s.contains(needle: String) -> bool`
- **インスタンス**: `s.to_i64() -> i64`

### System / Args / Arena
- **クラス**: `System::memory_mb() -> i64`
- **グローバル**: `print(s: String)` / `println(s: String)` / `read_line(prompt: String) -> String`
- **クラス**: `Args::count() -> i64`
- **クラス**: `Args::get(i: i64) -> String`
- **クラス**: `Arena::get_offset() -> i64`
- **クラス**: `Arena::alloc(bytes: i64) -> i64`
- **クラス**: `Arena::init(bytes: i64)`
- **クラス**: `Arena::is_active() -> bool`

### Tensor (class methods)
- `Tensor::new_causal_mask(dim: i64) -> Tensor<f32, 2>`
- `Tensor::rope_new_cos(d: i64, l: i64, t: f32) -> Tensor<f32, 2>`
- `Tensor::rope_new_sin(d: i64, l: i64, t: f32) -> Tensor<f32, 2>`
- `Tensor::clear_grads()`
- `Tensor::from_vec_u8(vec: Vec<u8>, offset: i64, shape: Tensor<i64, 1>, rank: i64) -> Tensor<f32, 2>`
- `Tensor::from_u8_labels(vec: Vec<u8>, offset: i64, count: i64) -> Tensor<i64, 1>`

### Tensor (instance methods)
- `t.embedding(table: Tensor<f32, 2>) -> Tensor<f32, 2>`
- `t.rms_norm(w: Tensor<f32, 1>, e: f32) -> Tensor<f32, 2>`
- `t.matmul(b: Tensor<f32, 2>) -> Tensor<f32, 2>`
- `t.add(b: Tensor<f32, 2>) -> Tensor<f32, 2>`
- `t.silu() -> Tensor<f32, 2>`
- `t.mul(b: Tensor<f32, 2>) -> Tensor<f32, 2>`
- `t.matmul_quantized(weight: i64) -> Tensor<f32, 2>`
- `t.cat_i64(b: Tensor<i64, 1>, d: i64) -> Tensor<i64, 1>`
- `t.cat_4d(b: Tensor<f32, 4>, d: i64) -> Tensor<f32, 4>`
- `t.scale(s: f32) -> Tensor<f32, 4>`
- `t.transpose(d1: i64, d2: i64) -> Tensor<f32, 4>`
- `t.transpose_2d(d1: i64, d2: i64) -> Tensor<f32, 2>`
- `t.apply_rope(cos: Tensor<f32, 2>, sin: Tensor<f32, 2>) -> Tensor<f32, 4>`
- `t.repeat_interleave(repeats: i64, dim: i64) -> Tensor<f32, 4>`
- `t.narrow(dim: i64, start: i64, len: i64) -> Tensor<f32, 2>`
- `t.argmax(d: i64, k: bool) -> Tensor<i64, 1>`
- `t.sample(temp: f32, top_p: f32) -> Tensor<i64, 1>`
- `t.matmul_4d(b: Tensor<f32, 4>) -> Tensor<f32, 4>`
- `t.add_4d(b: Tensor<f32, 4>) -> Tensor<f32, 4>`
- `t.softmax(d: i64) -> Tensor<f32, 4>`
- `t.len() -> i64`
- `t.item_i64() -> i64`
- `t.get_shape() -> Tensor<i64, 1>`
- `t.print_1()` / `t.print_2()` / `t.print_3()`

### Vec<u8>
- `vec.len() -> i64`
- `vec.read_i32_be(offset: i64) -> i64`
- `vec.free()`

## 3. 実装ステップ (案)
1) コンパイラの semantic で型ごとのメソッドを許可
2) codegen 側で `Tokenizer/Map/KVCache/File/String/System/Args/Arena/Vec<u8>` をメソッド呼び出しとして出力
3) `.tl` の呼び出しを `tl_*` から `Class::method / instance.method` に置換
4) テスト/例の動作確認
