# Llama 3 Chatbot (`chatbot_llama3.tl`)

Llama 3.1 8B Instruct モデルを使用したチャットボットアプリケーションです。

## 主な特徴

- **Llama 3 プロンプト**: 標準的な Llama 3 instruct フォーマット (`<|begin_of_text|><|start_header_id|>...`) を使用。
- **サンプリング**: Top-P サンプリングをサポート。
- **メモリ安全性**: KV キャッシュを明示的に管理し、メモリリークを防止。
- **停止条件**: Llama 3 固有の停止トークンを検出。

## 実行方法

1.  **モデルファイルの準備**:
    デフォルトで `~/.llm/models/` ディレクトリに以下のファイルが必要です。
    - `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`
    - `llama3_tokenizer.json`
    (パスを変更する場合は、`chatbot_llama3.tl` の `main` 関数内のパスを編集してください)

2.  **コンパイルと実行**:
    ```bash
    cargo run --release --bin tl -- examples/apps/llama3/chatbot_llama3.tl
    ```

3.  **対話**:
    - `User>` プロンプトが表示されたらテキストを入力します。
    - `exit` または `quit` と入力すると終了します。
