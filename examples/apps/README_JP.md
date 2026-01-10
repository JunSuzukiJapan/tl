# TensorLogic (TL) Examples: Apps

このディレクトリには、TensorLogic (TL) 言語で実装された実用に近いアプリケーションが含まれています。

## 1. TinyLlama Chatbot (`tinyllama/chatbot.tl`)

TinyLlama-1.1B モデルを使用した、対話型のチャットボットアプリケーションです。KV Cache や RoPE (Rotary Positional Embedding) など、現代的な LLM の推論に必要なコンポーネントが TL 言語で直接実装されています。

### 主な特徴

- **リアルタイム・ストリーミング**: 生成されたトークンを逐次表示します (`print` 関数を使用)。
- **KV Cache 実装**: 推論速度を向上させるため、過去の Key/Value 状態を保存し再利用します。
- **ChatML 形式**: TinyLlama に適したプロンプトフォーマット (`<|user|>`, `<|assistant|>`) を使用。
- **GGUF サポート**: 量子化された GGUF モデルファイルを直接読み込みます。

### 実行方法

1.  **モデルファイルの準備**:
    デフォルトで `~/.llm/models/` ディレクトリに以下のファイルが必要です。
    - `tinyllama-1.1b-chat-q4_0.gguf`
    - `tokenizer.json`
    (パスを変更する場合は、`chatbot.tl` の `main` 関数内のパスを編集してください)

2.  **コンパイルと実行**:
    ```bash
    # 標準実行
    cargo run --release --bin tl -- run examples/apps/tinyllama/chatbot.tl

    # Metal (Apple Silicon GPU) アクセラレーションを使用する場合
    cargo run --release --features metal --bin tl -- run examples/apps/tinyllama/chatbot.tl
    ```

3.  **対話**:
    - `User>` プロンプトが表示されたらテキストを入力します。
    - `exit` または `quit` と入力すると終了します。

## 2. Llama 3 Chatbot (`llama3/chatbot_llama3.tl`)

Llama 3.1 8B Instruct モデルを使用したチャットボットです。

### 主な特徴
- **Llama 3 プロンプト**: 標準的な Llama 3 instruct フォーマットを使用。
- **サンプリング**: Top-P サンプリングをサポート。
- **メモリ安全性**: KVキャッシュを明示的に管理し、リークを防止。

### 実行方法
```bash
cargo run --release --bin tl -- run examples/apps/llama3/chatbot_llama3.tl
```

### 注意事項

- 初回の推論にはモデルとトークナイザーのロード時間が必要です。
- 現在の実装は `argmax` サンプリング（最も確率の高いトークンを常に選択）を使用しています。

---

## 技術的な詳細

このチャットボットは、TL 言語の以下の強力な機能を活用しています。

- **カスタム構造体とメソッド**: `Linear`, `RMSNorm`, `MLP`, `Attention`, `Block` などのモジュールを構造体として定義し、再利用可能な推論パイプラインを構築しています。
- **FFI (Foreign Function Interface)**: 高速な行列演算やトークナイザー処理、KV Cache 管理のために、Rust で書かれたランタイム関数を `extern` 宣言で呼び出しています。
- **Tensor 操作**: 多次元テンソルの変形 (`reshape`)、転置 (`transpose`)、連結 (`cat`) などを組み合わせて複雑なトランスフォーマー回路を記述しています。
