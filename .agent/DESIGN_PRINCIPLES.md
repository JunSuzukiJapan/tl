# TL言語 設計原則

## 勾配追跡の型レベル分離（最重要）

TL言語では、**勾配追跡の有無を型システムで静的に区別**する:

- **`Tensor`** — 勾配を追跡しない。推論・データ処理用。
- **`GradTensor`** — 勾配を追跡する。学習（訓練）用。

### この設計の結果

1. **`no_grad` ブロックは不要**。PyTorchの `torch.no_grad()` のようなランタイムフラグは存在しない。
2. **勾配の有無はコンパイル時に決定**される。実行時に切り替えることはない。
3. **推論コードは `Tensor` だけを使えば自動的に勾配計算から解放**される（メモリ節約・高速化）。
4. `Param::parameters() -> Vec<Tensor>` のような一括取得APIは不要。各パラメータは `GradTensor` 型で個別に管理する。

### コード例

```tl
// 推論: Tensor型 → 勾配なし
let t = Tensor::randn([3, 3], false);
let out = model.forward(t);

// 学習: GradTensor型 → 勾配あり
let gt = Tensor::randn([3, 3], true);  // GradTensor
let loss = model.forward(gt).sum();
loss.backward();
```

> [!CAUTION]
> この設計原則に反する機能（ランタイムでの勾配有効/無効切り替え等）を提案・実装しないこと。
