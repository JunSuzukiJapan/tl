# Task: 2-Digit Addition

This task trains a Transformer model to perform 2-digit addition (e.g., `12 + 34 = 46`).

## Overview

A 1-layer GPT-style Transformer を使用して、2桁の足し算を学習・推論するデモ。

## Implementation Details

- **Scripts**:
    - `train_add.tl`: 学習ループ（20エポック、各10ステップ）
    - `infer_add.tl`: 重みをロードして autoregressive 推論
    - `train_verify_2digit.tl`: 安定性テスト用スクリプト
- **Model**: 1-Layer GPT (CausalSelfAttention + MLP + LayerNorm)
- **Vocabulary Size**: 13 (数字 0-9, `+`=10, `=`=11, `PAD`=12)
- **Embedding Size**: 64
- **Token Format**: `[i_d1, i_d10, +, j_d1, j_d10, =, s_d1, s_d10, s_d100, PAD, PAD, PAD]`
    - 各数字を逆順の桁（1の位, 10の位）で表現

## Usage

学習:
```bash
cargo run --release --bin tl -- examples/tasks/addition/train_add.tl
```

推論:
```bash
cargo run --release --bin tl -- examples/tasks/addition/infer_add.tl
```

モデルの重みは `model_add.safetensors` として保存される。
