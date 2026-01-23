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

> [!WARNING]
> **Metal使用時の注意**: 長時間のループ処理では、Metalドライバの挙動によりRSS（メモリ使用量）が増加することがあります。これはTLのメモリリークではありません。安定したメモリ使用量が必要な場合は `TL_DEVICE=cpu` を使用してください。詳細は [Metal RSS Growth Notes](docs/dev/METAL_RSS_GROWTH_NOTES.md) を参照。

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
        let epochs = 1500; // 高速なリトライ戦略

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

             // パフォーマンス向上のため定期的にチェック
             if i % 100 == 0 {
                  // 早期終了: クイーンがN個見つかったら停止 (prob > 0.5)
                 let current_queens = [r, c | r <- 0..N, c <- 0..N { 
                     if probs[r, c] > 0.5 { 1.0 } else { 0.0 } 
                 }].sum().item();

                 if current_queens == 8.0 {
                     break;
                 }
             }

             total_loss.backward();
             let g = board.grad();
             board = board - g * lr;
             board = board.detach();
             board.enable_grad();
        }

        // --- 結果判定と表示 ---
        let probs = board.softmax(1);
        
        // クイーンの数をカウントして検証
        let mut valid = true;
        let mut total_queens = 0;
        let mut rows = 0;
        while rows < N {
            let mut queen_count = 0;
            let mut cols = 0;
            while cols < N {
               if probs[rows, cols] > 0.5 {
                   queen_count = queen_count + 1;
                   total_queens = total_queens + 1;
               }
               cols = cols + 1;
            }
            if queen_count != 1 {
                valid = false;
            }
            rows = rows + 1;
        }

        if valid && total_queens == N {
            found_count = found_count + 1;
            println("Solution #{}", found_count);
            
            let mut rows2 = 0;
            while rows2 < N {
                let mut cols2 = 0;
                while cols2 < N {
                   if probs[rows2, cols2] > 0.5 {
                       print(" Q ");
                   } else {
                       print(" . ");
                   }
                   cols2 = cols2 + 1;
                }
                println("");
                rows2 = rows2 + 1;
            }
            println("----------------");
        }
    }
}
```

## サンプルコード: ニューロシンボリックAI (空間推論)

以下の例は、**視覚**（テンソル）と**推論**（論理）を融合させる方法を示しています。
生の座標データから空間的な関係を検出し、論理ルール（推移律）を使用して隠れた事実を推論します。

```rust
// 1. 記号的知識ベース（「思考」）
// 概念（オブジェクト）の定義
object(1, cup).
object(2, box).
object(3, table).

// 再帰的な論理ルール：推移律
// XがYの上にあるなら、XはYに「積まれている」。
// XがZの上にあり、ZがYに積まれているなら、XはYに「積まれている」。
stacked_on(top, bot) :- on_top_of(top, bot).
stacked_on(top, bot) :- on_top_of(top, mid), stacked_on(mid, bot).


fn main() {
    println("--- Neuro-Symbolic AI Demo: Spatial Reasoning ---");

    // 2. 知覚（テンソルデータ）
    // ビジョンモデルによって検出されたバウンディングボックス（シミュレーション） [x, y, w, h]
    let cup_bbox   = [10.0, 20.0, 4.0, 4.0];  // Cup is high (y=20)
    let box_bbox   = [10.0, 10.0, 10.0, 10.0]; // Box is middle (y=10)
    let table_bbox = [10.0, 0.0, 50.0, 10.0];  // Table is low (y=0)

    println("\n[Visual Scene]");
    println("   [Cup]   (y=20)");
    println("     |     ");
    println("   [Box]   (y=10)");
    println("     |     ");
    println("[=========] (Table y=0)");
    println("");

    // 3. ニューロシンボリック融合
    // 座標から空間的な関係を検出し、事実（Fact）として注入する

    // 検出: Cup on Box?
    if cup_bbox[1] > box_bbox[1] {
        on_top_of(1, 2). 
        println("Detected: Cup is on_top_of Box");
    }

    // 検出: Box on Table?
    if box_bbox[1] > table_bbox[1] {
        on_top_of(2, 3).
        println("Detected: Box is on_top_of Table");
    }

    // 4. 論理推論
    println("\n[Logical Inference]");
    println("Querying: Is Cup stacked on Table? (?stacked_on(1, 3))");

    // 論理クエリの実行
    // 「カップがテーブルの上にある」とは明示されていませんが、論理が「箱」を介してそれを推論します。
    let res = ?stacked_on(1, 3);

    if res.item() > 0.5 {
        println("Result: YES (Confidence: {:.4})", res.item());
        println("Reasoning: Cup -> Box -> Table (Transitivity)");
    } else {
        println("Result: NO");
    }
}
```

## ドキュメント

- [TensorLogicの利点](docs/jp/tensor_logic_advantages_jp.md)
- [テンソル内包表記の設計思想](docs/jp/tensor_comprehension_design.md)
- [言語仕様](docs/jp/reference/language_spec.md)
- [APIリファレンス](docs/jp/reference/functions_and_types.md)
- [論理プログラミング](docs/jp/logic_programming.md)

# LICENSE

[MIT](LICENSE)

# References

- [Tensor Logic: The Language of AI](https://arxiv.org/abs/2510.12269)
- [AIの未来を変えるテンソル論理とは？ニューラルと記号推論を統一する新言語（2510.12269）【論文解説シリーズ】](https://www.youtube.com/watch?v=rkBLPYqPkP4)

