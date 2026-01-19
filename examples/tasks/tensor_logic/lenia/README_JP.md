# Lenia / 連続セル・オートマトン

この例は、TensorLogic を使用した **Lenia** (Continuous Cellular Automata: 連続セル・オートマトン) のシンプルな実装デモです。

以下の機能を使用しています:
*   **Tensor Operations**: 近傍のセンシングに `conv2d` (畳み込み) を使用。
*   **Element-wise Math**: 成長関数の計算に `exp`, `pow` を使用。
*   **Time Integration**: 状態更新にオイラー法を使用。
*   **Clamping**: 状態値を 0.0 から 1.0 の間に保つために使用。

## 実行方法

```bash
cargo run --release -- examples/tasks/tensor_logic/lenia/lenia.tl
```

スクリプトは 100 ステップ実行され、活動状況を示すために一定間隔でグリッドの総質量 (total mass) を出力します。
