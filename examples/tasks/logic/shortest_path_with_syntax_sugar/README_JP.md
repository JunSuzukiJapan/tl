# 構文シュガーを使った最短経路

この例は、TensorLogicのProlog風構文とトロピカル行列ロジックを組み合わせています。

## ファイル

- `facts.tl`: 論理事実を使ったグラフエッジの宣言。
- `main.tl`: 論理クエリのデモと、トロピカル行列演算による最短経路計算。

## アプローチ

グラフ構造は論理事実で宣言：
```
edge(a, b).
edge(b, c).
```

最短経路はトロピカル行列演算（min-plus半環）で計算：
```rust
let new_dist = [i, j, k | ... { dist[i, k] + dist[k, j] }].min(2);
```

## 実行方法

```bash
cd examples/tasks/logic/shortest_path_with_syntax_sugar
tl main.tl
```
