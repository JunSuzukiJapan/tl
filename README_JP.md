# TL Programming Language

LLVMにJITコンパイルされる、ファーストクラスのテンソルサポートを備えたテンソル論理プログラミング言語。

## 機能
- **テンソル演算**: Candleを介した `tensor<f32>[128, 128]`, `matmul`, `topk` など。
- **論理プログラミング**: テンソル統合を備えたDatalogスタイルのルール。
- **ハイブリッド実行**: 論理項はテンソルデータにアクセス可能 (`data[i]`)。
- **JITコンパイル**: LLVM (Inkwell) を使用した高性能実行。
- **GPUサポート**: Metal (macOS) バックエンドをサポート。CUDAは将来的に対応予定。
- **最適化**: 積極的なJIT最適化と高速な論理推論。

## インストール

crates.io からインストール:

```bash
cargo install tl-lang
```

これにより `tl` コマンドがインストールされます。

## 前提条件 (macOS)
ビルドする前に、Homebrew経由で必要な依存関係がインストールされていることを確認してください。

1. **LLVM 18 と OpenSSL のインストール**:
   ```bash
   brew install llvm@18 openssl
   ```

2. **環境変数の設定**:
   ビルドシステムがLLVM 18を見つけられるように、以下をシェル設定（例：`~/.zshrc`）に追加してください：
   ```bash
   export LLVM_SYS_181_PREFIX=$(brew --prefix llvm@18)
   ```
   cargoコマンドを実行する前にシェルをリロード（`source ~/.zshrc`）してください。

## ビルドと実行
```bash
# 特定の例を実行
cargo run -- examples/hybrid_test.tl

# GPUを使用 (macOSのMetal)
cargo run --features metal -- examples/gpu_test.tl
```

## 文法

TLの文法は、Rustに非常によく似ています。ただし、ライフタイムはありません。

### 基本構文

```rust
fn main() {
    let x = 5;
    println("{}", x);
}
```

### テンソル演算

```rust
fn main() {
    let x = [1.0, 2.0, 3.0];
    let y = [4.0, 5.0, 6.0];
    let z = x + y;
    println("{}", z);
}
```

### if文

```rust
fn main() {
    let x = 5;
    if x > 0 {
        println("{}", x);
    }
}
```

### while文

```rust
fn main() {
    let mut x = 5;
    while x > 0 {
        println("{}", x);
        x = x - 1;
    }
}
```

### for文

```rust
fn main() {
    let mut x = 5;
    for i in 0..x {
        println("{}", i);
    }
}
```

### 関数定義

```rust
fn main() {
    let x = 5;
    let y = add(x, 1);
    println("{}", y);
}

fn add(x: i64, y: i64) -> i64 {
    x + y
}
```

### テンソル内包表記

```rust
fn main() {
    let t = [1.0, 2.0, 3.0, 4.0];
    // 2.0より大きい要素は2倍にし、それ以外は0.0にする（マスク処理）
    let res = [i | i <- 0..4, t[i] > 2.0 { t[i] * 2.0 }];
    println("{}", res);
}
```

テンソル内包表記に関しては詳しくは[ドキュメント](docs/jp/tensor_comprehension_jp.md)をご覧ください。

## VSCode拡張機能
`vscode-tl` でシンタックスハイライト拡張機能が提供されています。

### インストール
1. VSCodeで `vscode-tl` ディレクトリを開きます。
2. **F5** を押して、新しいウィンドウでシンタックスハイライトを確認します。
3. または手動でインストール:
   - `cd vscode-tl`
   - `npm install -g vsce` (必要であれば)
   - `vsce package`
   - `code --install-extension tensor-logic-0.1.0.vsix`

## サンプルコード: Nクイーン問題 (テンソル最適化による解法)

TensorLogicは、論理的な制約をテンソル演算による連続最適化問題として解くことができます。
以下は、勾配降下法を用いてNクイーン問題を解くプログラムの例です。

```rust
fn main() {
    let N = 8; // ボードサイズ (8x8)
    let solutions_to_find = 5; // 見つける解の数
    let mut found_count = 0;

    println("Finding {} solutions for N-Queens...", solutions_to_find);

    while found_count < solutions_to_find {
        let lr = 0.5;
        let epochs = 3000;

        // 1. ボードの確率分布を初期化 (ランダムノイズ)
        let mut board = Tensor::randn([N, N], true);

        // 最適化ループ
        for i in 0..epochs {
             let probs = board.softmax(1);

             // 制約1: 列制約
             let col_sums = probs.sum(0);
             let col_loss = (col_sums - 1.0).pow(2).sum();

             // 制約2: 対角線制約 (Tensor Comprehension)
             let anti_diag_sums = [k | k <- 0..(2 * N - 1), r <- 0..N, c <- 0..N, r + c == k { probs[r, c] }];
             let main_diag_sums = [k | k <- 0..(2 * N - 1), r <- 0..N, c <- 0..N, r - c + N - 1 == k { probs[r, c] }];

             let anti_diag_loss = (anti_diag_sums - 1.0).relu().pow(2).sum();
             let main_diag_loss = (main_diag_sums - 1.0).relu().pow(2).sum();
             
             let total_loss = col_loss + anti_diag_loss + main_diag_loss;

             // breakで早期終了
             if total_loss.item() < 1e-4 {
                 break;
             }

             total_loss.backward();
             let g = board.grad();
             board = board - g * lr;
             board = board.detach();
             board.enable_grad();
        }

        // --- 結果判定と表示 ---
        let probs = board.softmax(1);
        
        // 損失を再計算してチェック
        let col_sums = probs.sum(0);
        let col_loss = (col_sums - 1.0).pow(2).sum();
        let anti_diag_sums = [k | k <- 0..(2 * N - 1), r <- 0..N, c <- 0..N, r + c == k { probs[r, c] }];
        let main_diag_sums = [k | k <- 0..(2 * N - 1), r <- 0..N, c <- 0..N, r - c + N - 1 == k { probs[r, c] }];
        let anti_diag_loss = (anti_diag_sums - 1.0).relu().pow(2).sum();
        let main_diag_loss = (main_diag_sums - 1.0).relu().pow(2).sum();
        let total_loss = col_loss + anti_diag_loss + main_diag_loss;

        if total_loss.item() < 1e-3 {
            found_count = found_count + 1;
            println("Solution #{}", found_count);
            
            let mut rows = 0;
            while rows < N {
                let mut cols = 0;
                while cols < N {
                   if probs[rows, cols] > 0.5 {
                       print(" Q ");
                   } else {
                       print(" . ");
                   }
                   cols = cols + 1;
                }
                println("");
                rows = rows + 1;
            }
            println("----------------");
        }
    }
}
```

## ドキュメント

- [TensorLogicの利点](docs/jp/tensor_logic_advantages_jp.md)
- [言語仕様](docs/jp/reference/language_spec.md)
- [標準ライブラリ](docs/jp/reference/standard_library.md)
- [論理プログラミング](docs/jp/logic_programming.md)

# LICENSE

[MIT](LICENSE)

# References

- [Tensor Logic: The Language of AI](https://arxiv.org/abs/2510.12269)
- [AIの未来を変えるテンソル論理とは？ニューラルと記号推論を統一する新言語（2510.12269）【論文解説シリーズ】](https://www.youtube.com/watch?v=rkBLPYqPkP4)

