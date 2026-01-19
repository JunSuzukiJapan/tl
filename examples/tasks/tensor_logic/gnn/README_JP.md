# Graph Neural Network (GNN) for Shortest Path

この例は、Tensor Logic (`tl`) を用いてGraph Neural Network (GNN) を実装し、教師なし学習（Neural Bellman-Ford）によって最短経路問題を解くデモです。

## 概要

プログラムの動作ステップは以下の通りです：
1.  **グラフ生成**: `N` ノードのランダムな隣接行列 `A` を生成します。
2.  **モデル定義**: 各ノードの距離を表す学習可能なTensor `val` を初期化します。
3.  **教師なし学習**: Bellman-Ford方程式 `d[v] = min(d[u] + w(u, v))` を満たすように学習します。
    微分可能な "SoftMin" 近似を使用しています。
4.  **制約条件**: `val[source] = 0` (境界条件) および `val[v] >= 0` を制約としてロスに加えます。

## 実装の詳細

-   **メッセージパッシング**: `repeat_interleave` や `transpose` (ブロードキャスト) を駆使して、すべてのエッジ `(u, v)` 間のコスト `val[u] + weight` を効率的に計算します。
-   **スパース処理**: 隣接していないノード間をマスクし、SoftMin計算から除外しています。
-   **ワークアラウンド**: コンパイラの制約により、一部の操作（`repeat_interleave`の不具合回避のためのブロードキャスト利用など）を行っています。

## 実行方法

以下のコマンドで実行できます：

```bash
cargo run --release -- examples/tasks/tensor_logic/gnn/gnn.tl
```

## 学び

-   **Tensor Logicの実践**: 複雑なアルゴリズム（グラフ探索）を、Tensor演算の組み合わせだけで記述できることを示しました。
-   **Unsupervised Learning**: 正解データ（Ground Truth）を与えなくても、ルール（方程式）の整合性を最小化することで解を導けることを実証しました。
