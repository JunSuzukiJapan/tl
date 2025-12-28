# Tensor Logic 言語仕様書 (Rust-like Syntax)

## 1. はじめに
Tensor Logic（テンソル論理）は、論理、ニューラルネットワーク、および確率的グラフィカルモデルを統合するAI記述言語です。本仕様は、Rust言語の影響を受けた静的型付け構文を採用しています（ただしライフタイムは除外）。

## 2. コア構文 (Core Syntax)

### 2.1. テンソル方程式
テンソル演算は `let` バインディング、または既存テンソルへの射影代入として記述します。

**基本形式:**
```rust
// 新しいテンソルの定義
let lhs[i, j] = expression;

// 既存テンソルへの射影（累積）
lhs[i, j] += expression;
```

**例:**
```rust
// 論理: Aunt(x, z) <- Sister(x, y) ^ Parent(y, z)
let aunt[x, z]: bool = step(sister[x, y] * parent[y, z]);
```

### 2.2. 型システム
厳格な静的型付けを採用します。

*   **基本型**:
    *   `f32`, `f64`: 実数値 (標準)。
    *   `f16`, `bf16`: 半精度浮動小数点数 (LLM学習の高速化・メモリ節約用)。
    *   `bool`: ブール値 (0 または 1)。
    *   `i8`: 8ビット整数 (モデルの量子化推論用)。
    *   `i32`, `i64`: 整数型 (インデックスやカウント用。大規模モデルにはi64が必要)。
    *   `u8`, `u16`, `u32`, `usize`: 符号なし整数 (バイト処理、トークンID、メモリサイズ用)。

*   **テンソル型**:
    *   `Tensor<Type, Rank>`: 指定された型とランクを持つテンソル。
    *   例: `Tensor<f32, 2>` (2階の実数テンソル), `Tensor<bool, 3>` (3階のブールテンソル)
    *   **型変換**: 算術演算において `bool` は自動的に `f32` (0.0/1.0) または `i32` (0/1) に昇格して計算されます。
    *   `Vec<Type>`: 構造体などのオブジェクトを管理するための動的配列（数値計算にはTensorを使用し、モジュール管理などにVecを使用）。

### 2.3. インデックス
インデックス変数は方程式内で自動的に推論されますが、明示的な範囲指定も可能です。

*   **スコープ**: 左辺 (LHS) に現れるインデックスは自由変数 (free)、右辺 (RHS) のみは束縛変数 (bound) です。


### 2.4. テンソルリテラル
テンソルは配列リテラル記法を用いて定義できます。ネストしたブラケットで多次元構造を表現します。

```rust
// 1階テンソル (ベクトル)
let v: Tensor<i32, 1> = [1, 2, 3];

// 2階テンソル (行列)
let m: Tensor<f32, 2> = [
    [1.0, 0.0],
    [0.0, 1.0]
];
```

## 3. 意味論 (Semantics)

### 3.1. 暗黙の結合と和
*   **積 (Join)**: `*` 演算子で明示的に記述します（Python風の空白による暗黙積は廃止）。
*   **和 (Summation)**: 縮約記法に従い、束縛変数について自動的に和がとられます。

```rust
// C_{xz} = sum_y (A_{xy} * B_{yz})
let c[x, z] = a[x, y] * b[y, z];
```

### 3.2. 射影演算子
*   `+=`: 加算 (Sum projection)
*   `max=`: 最大値 (Max projection)
*   `avg=`: 平均 (Average projection)

## 4. 汎用プログラミング構文

### 4.1. 関数定義
`fn` キーワードを使用し、引数と戻り値の型を明記します。

```rust
fn limit_activation(input: Tensor<f32, 2>, threshold: f32) -> Tensor<f32, 2> {
    let result[i, j] = if input[i, j] > threshold {
        threshold
    } else {
        input[i, j]
    };
    result
}
```

### 4.2. 制御フロー (`if`)
式として機能します。

```rust
let val = if x > 0.0 { 1.0 } else { 0.0 };
```

### 4.3. ループ (`for`)
反復的な更新（RNNなど）に使用します。

```rust
// RNNの例
// input: Tensor<f32, 3> [Times, Batch, Dim]
// state: Tensor<f32, 2> [Batch, Hidden]
fn rnn_forward(input: Tensor<f32, 3>, w_in: Tensor<f32, 2>, w_rec: Tensor<f32, 2>) -> Tensor<f32, 3> {
    let t_steps: usize = input.shape(0);
    // 可変なテンソルとして初期化
    let mut h_state: Tensor<f32, 2> = zeros([input.shape(1), w_rec.shape(0)]);
    let mut outputs: Tensor<f32, 3> = zeros([t_steps, input.shape(1), w_rec.shape(0)]);

    for t in 0..t_steps {
        // 現在のステップの入力
        let x_t[b, d] = input[t, b, d];
        
        // 状態更新: h_new = tanh(W_in * x + W_rec * h_prev)
        let pre_act[b, h] = w_in[d, h] * x_t[b, d] + w_rec[k, h] * h_state[b, k];
        
        // 更新 (再代入ではなく新しい状態を計算して代入)
        h_state[b, h] = tanh(pre_act[b, h]);
        
        // 出力を記録
        outputs[t, b, h] = h_state[b, h];
    }
    outputs
}
```

## 5. 具体例 (Examples)

### 5.1. 論理ルール (Datalog)
```rust
fn transitive_closure(edge: Tensor<bool, 2>) -> Tensor<bool, 2> {
    let mut path: Tensor<bool, 2> = edge.clone();
    
    // 固定点に達するまで繰り返す (簡略化のためループ回数固定の例)
    for _ in 0..10 {
        // path(x, z) <- edge(x, y) ^ path(y, z)
        // max= を使用して論理和更新を行う (+= だと重複カウントになるため)
        path[x, z] max= step(edge[x, y] * path[y, z]);
    }
    path
}
```

### 5.2. Transformers (Self-Attention)
```rust
fn self_attention(query: Tensor<f32, 2>, key: Tensor<f32, 2>, value: Tensor<f32, 2>) -> Tensor<f32, 2> {
    let d_k: f32 = query.shape(1) as f32;
    
    // Attention Score: A[p, q] = softmax(Q[p, d] * K[q, d] / sqrt(d))
    // q は束縛変数、d も束縛変数
    let scores[p, q] = softmax(query[p, d] * key[q, d] / d_k.sqrt());
    
    // Output: O[p, d] = A[p, q] * V[q, d]
    let output[p, d] = scores[p, q] * value[q, d];
    
    output
}
```

### 5.3. Graph Neural Networks (GNN)
```rust
fn gnn_layer(h: Tensor<f32, 2>, adj: Tensor<f32, 2>, w: Tensor<f32, 2>) -> Tensor<f32, 2> {
    // Aggregation: Message passing
    // agg[n, l] = sum_{m} (A[n, m] * h[m, l])
    let agg[n, l] = adj[n, m] * h[m, l];
    
    // Update: h_new[n, k] = relu(sum_{l} (W[l, k] * agg[n, l]))
    let h_new[n, k] = relu(w[l, k] * agg[n, l]);
    
    h_new
}
```

## 6. 構造体とメソッド

オブジェクト指向プログラミングと同様に、データ（テンソル）と動作（メソッド）をカプセル化するために `struct` と `impl` を使用します。ニューラルネットワークのモデル定義に必須です。

### 6.1. 構造体定義
```rust
// ジェネリクスが使用可能。型パラメータ T は通常 f32, f16 等
struct Linear<T> {
    weight: Tensor<T, 2>,
    bias: Tensor<T, 1>,
}
```

### 6.2. メソッド実装
`impl` ブロック内でメソッドを定義します。`self` を通じてフィールドにアクセスします。

```rust
impl<T> Linear<T> {
    // コンストラクタ (慣習的に new を使用)
    fn new(in_dim: usize, out_dim: usize) -> Linear<T> {
        Linear {
            // パラメータ初期化 (xavier_uniformなどの組み込み関数を使用可能)
            weight: random([out_dim, in_dim]),
            bias: zeros([out_dim]),
        }
    }

    // フォワードパス
    fn forward(self, input: Tensor<f32, 2>) -> Tensor<f32, 2> {
        // y = x W^T + b
        // implicit summation: y[b, o] = input[b, i] * self.weight[o, i] + self.bias[o]
        let output[b, o] = input[b, i] * self.weight[o, i] + self.bias[o];
        output
    }
}
```

## 7. LLMネイティブサポート

Tensor Logicは外部ライブラリ（PyTorchなど）に依存せず、言語機能として大規模言語モデルの学習と推論を完結させるための機能を備えています。

### 7.1. データセットとトークナイズ
標準ライブラリ `std::data`, `std::text` が提供されます。

```rust
// テキスト読み込みとトークナイズ
let tokenizer = Tokenizer::new("gpt2"); // 組み込みトークナイザ
let raw_text = read_text("corpus.txt");
let tokens = tokenizer.encode(raw_text); // Tensor<i32, 1>

// データセット作成
// Dataset構造体はバッチ化とシャッフルを自動化します
let dataset = Dataset::new(tokens, block_size=1024);
```

### 7.3. カスタムトークナイザー
Tensor Logicの基本機能を用いて、独自のトークナイザーを実装することも可能です。文字列をバイト列 (`Tensor<u8, 1>`) として扱うことができます。

```rust
struct SimpleTokenizer {
    vocab: Tensor<u8, 2>, // [VocabSize, MaxTokenLen] (簡易的な辞書)
}

impl SimpleTokenizer {
    fn new() -> SimpleTokenizer {
        // ... 辞書初期化 ...
        SimpleTokenizer { vocab: ... }
    }

    fn encode(self, text: String) -> Tensor<i32, 1> {
        // 文字列をバイトテンソルに変換
        let bytes: Tensor<u8, 1> = text.bytes(); 
        let len = bytes.shape(0);
        let mut tokens: Tensor<i32, 1> = zeros([len]); // 最大長で確保
        
        // ... バイト列を走査してトークン化するロジック ...
        // (forループとテンソルマッチングを使用)
        
        tokens
    }
}
```

### 7.4. 学習ループと自動微分
モデル内の可変テンソルは自動的にパラメータとして追跡されます。

以下は、ユーザー定義の `GPT` 構造体を使用した学習ループの例です。

```rust
// GPTモデル定義 (ユーザー定義)
struct GPT {
    token_emb: Tensor<f32, 2>,
    pos_emb: Tensor<f32, 2>,
    layers: Vec<TransformerBlock>, // レイヤーのリスト
    head: Linear,
}

impl GPT {
    fn new(vocab_size: usize, n_embd: usize, n_layer: usize) -> GPT {
        let mut layers = Vec::new();
        for _ in 0..n_layer {
            layers.push(TransformerBlock::new(n_embd));
        }
        
        GPT {
            token_emb: random([vocab_size, n_embd]),
            pos_emb: random([1024, n_embd]),
            layers: layers,
            head: Linear::new(n_embd, vocab_size),
        }
    }

    fn forward(self, input: Tensor<i32, 2>) -> Tensor<f32, 3> {
        let b = input.shape(0);
        let t = input.shape(1);
        
        let mut x[b, t, d] = self.token_emb[input[b, t], d] + self.pos_emb[t, d];
        
        // Vec内のレイヤーを順次適用
        for layer in self.layers {
            x = layer.forward(x);
        }
        
        let logits[b, t, v] = self.head.forward(x)[b, t, v];
        
        logits
    }
}

fn train_gpt() {
    // データセット準備
    let dataset = Dataset::from_file("corpus.txt", block_size=256);
    
    // モデル初期化
    let mut model = GPT::new(vocab_size=50257, n_embd=768);
    
    // オプティマイザ設定 (AdamW)
    let mut optim = AdamW::new(model, lr=3e-4);

    // 学習ループ
    for batch in dataset.loader(batch_size=32, shuffle=true) {
        // Forward
        let logits = model.forward(batch.input);
        
        // Loss計算 (Cross Entropy)
        let loss = cross_entropy(logits, batch.target);
        
        // Backward (自動微分)
        // model内の全パラメータの .grad が計算・蓄積される
        loss.backward();
        
        // パラメータ更新
        optim.step();
        
        // 勾配リセット
        optim.zero_grad();
        
        print("Loss: ", loss);
    }
    
    // モデル保存
    model.save("gpt_model.tl");
}
```

## 8. ハードウェアと分散学習
大規模モデルの学習にはGPUや分散環境が不可欠です。

### 8.1. デバイス管理
テンソルの配置デバイスを明示的に指定できます。

```rust
// GPU上で初期化
let t = zeros([1024, 1024], device="cuda:0");

// デバイス間移動
let cpu_t = t.to("cpu");
```

### 8.2. 分散集合通信
データ並列やモデル並列のために、MPIライクな集合通信プリミティブを提供します。

```rust
// 全プロセスのテンソルを合計して同期
let sum = all_reduce(grad_tensor, op="sum");
```

## 9. 安全性とデバッグ
形状不一致によるバグを防ぐための機能です。

### 9.1. シェイプアサーション
期待する形状を明記し、実行時に検証します。次元には変数も指定可能です。

```rust
let b = 32;
let h = 768;
// ... 計算 ...
assert_shape(output, [b, h]); // [32, 768] でなければエラー
```

## 10. 標準ライブラリ
*   `step(x)`: $x > 0 \to 1, \text{else } 0$
*   `relu(x)`: $\max(0, x)$
*   `sig(x)`: $1 / (1 + e^{-x})$
*   `zeros(shape)`: ゼロテンソル生成
*   `ones(shape)`: 1テンソル生成
*   `random(shape)`: ランダムテンソル生成
*   `cross_entropy(logits, target)`: 交差エントロピー誤差
*   `AdamW`, `SGD`: オプティマイザ
*   `Tokenizer`: テキスト処理
*   `Dataset`: データローダ機能
*   `argmax(tensor, dim)`: 指定次元の最大インデックス
*   `multinomial(probs, num_samples)`: 多項分布からのサンプリング
*   `concat(tensors, dim)`: テンソルの結合
*   `sqrt(x)`, `exp(x)`, `log(x)`: 数学関数
*   `String.bytes()`: 文字列を `Tensor<u8, 1>` に変換

