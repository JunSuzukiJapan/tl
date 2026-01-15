# Tensor Logic における Markov Logic Network (MLN)

このサンプルは、**Tensor Logic** を用いた Markov Logic Network (MLN) の簡易実装デモです。論理ルールをテンソル演算として表現し、連続緩和と勾配降下法を用いて確率論的推論を行う方法を示しています。

## 概要

Markov Logic Network は、一階述語論理と確率的グラフィカルモデルを組み合わせたものです。この実装では、以下のルールを持つ小さな知識ベースを定義しています。

1.  **Smokes(x) => Cancer(x)**: 喫煙者は癌になりやすい。
2.  **Friends(x, y) => (Smokes(x) <=> Smokes(y))**: 友人は似た喫煙傾向を持つ。

述語 (`Smokes`, `Cancer`) の真理値を連続的な確率 $[0, 1]$（Sigmoidで緩和されたロジット）として扱い、勾配降下法を用いてルールの重み付き充足度（エネルギー）を最大化します。

## 主な概念

-   **Tensorとしての述語**: `Smokes` と `Cancer` は形状 `[N]` のTensorとして表現されます。
-   **代数としての論理**:
    -   含意 ($A \Rightarrow B$) は $1 - A + AB$ (Reichenbachの含意) としてモデル化されます。
    -   同値 ($A \Leftrightarrow B$) は類似度としてモデル化されます。
-   **微分可能な推論**: MCMCのような離散サンプリングの代わりに、述語のロジットに対して `requires_grad=true` を設定し、論理制約を満たすように直接最適化（推論）を行います。

## 実行方法

```bash
cargo run --release -- run examples/tasks/tensor_logic/mln/mln.tl
```

## 期待される出力

プログラムは `N=20` 人の個体に対して推論を行います。システムが論理ルールと証拠（Evidence）を満たす状態に収束するにつれて、Loss（負のエネルギー）が減少していく様子が確認できます。

```text
Initializing MLN for N=20
Generating network... Done.
Starting inference...
Epoch 0 Loss: -91.32773
Epoch 50 Loss: -125.97
...
```
