# テストガイド

TensorLogic (TL) コンパイラのテスト方法を説明します。

## Rust ユニットテスト

```bash
cargo test --workspace --exclude tl_cuda
```

> [!NOTE]
> `tl_cuda` はローカル環境では除外して実行します。

## TL ファイル検証スクリプト

プロジェクト内の全 `.tl` ファイル (examples + tests) を逐次実行し、ランタイムエラーを検出します。

### 基本使用方法

```bash
python3 scripts/verify_tl_files.py
```

### 主要オプション

| オプション | 説明 | デフォルト |
|---|---|---|
| `--timeout N` | 各ファイルのタイムアウト秒数 | 30 |
| `--cooldown N` | テスト間のクールダウン秒数 | 1.0 |
| `--static` | 静的コンパイルモードで実行 | JIT |
| `--safe-mode` | 10テストごとにセーフティ休憩を挟む | 有効 |

### 使用例

```bash
# タイムアウトとクールダウンを指定
python3 scripts/verify_tl_files.py --timeout 30 --cooldown 1.0
```

> [!CAUTION]
> **並列実行は禁止**。JIT コンパイルの特性上、並列実行するとメモリが枯渇し不安定になります。
> スクリプトは逐次実行を前提に設計されています。

### 期待される失敗

`tests/fixtures/patterns/*` の一部ファイルはエラー報告のテスト用に設計されており、非ゼロ終了コードを返します。スクリプトの `should_skip` ロジックで適切にスキップされます。

## セグフォのデバッグ

セグメンテーションフォールトが発生した場合は、`.agent/workflows/debug-segfault.md` を参照してください。
