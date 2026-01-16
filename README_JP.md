# TensorLogic (TL)

LLVMにJITコンパイルされる、ファーストクラスのテンソルサポートを備えたテンソル論理プログラミング言語。

## 機能
- **テンソル演算**: Candleを介した `tensor<f32>[128, 128]`, `matmul`, `topk` など。
- **論理プログラミング**: テンソル統合を備えたDatalogスタイルのルール。
- **ハイブリッド実行**: 論理項はテンソルデータにアクセス可能 (`data[i]`)。
- **JITコンパイル**: LLVM (Inkwell) を使用した高性能実行。
- **GPUサポート**: Metal (macOS) バックエンドをサポート。CUDAは将来的に対応予定。
- **最適化**: 積極的なJIT最適化と高速な論理推論。

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
cargo run --features metal -- examples/gpu_test.tl --device metal
```

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
    let found_count = 0;

    print("Finding "); print(solutions_to_find); println(" solutions for N-Queens...");

    while found_count < solutions_to_find {
        let lr = 0.5;
        let epochs = 2000;

        // 1. ボードの確率分布を初期化 (ランダムノイズ)
        let board = Tensor::randn([N, N], true);

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

             // 早期終了判定
             if total_loss.item() < 1e-4 {
                 // 収束した場合のみループを抜けて表示へ
                 // (breakはコンパイラバージョンによって動作が異なる場合があるため、epochsまで回すか、フラグ管理)
                 // ここでは簡略化のためループを継続し、後で判定
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
            print("Solution #"); println(found_count);
            
            let rows = 0;
            while rows < N {
                let cols = 0;
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


# LICENSE

[MIT](LICENSE)

#Reference

- [AIの未来を変えるテンソル論理とは？ニューラルと記号推論を統一する新言語（2510.12269）【論文解説シリーズ】](https://www.youtube.com/watch?v=rkBLPYqPkP4)

- [Tensor Logic: The Language of AI](https://arxiv.org/abs/2510.12269)
