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
cargo run -- run examples/hybrid_test.tl

# GPUを使用 (macOSのMetal)
cargo run --features metal -- run examples/gpu_test.tl --device metal
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
