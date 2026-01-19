# TensorLogicにおける論理プログラミング

この例では、TensorLogicのテンソル演算を使用して、Prologのような論理プログラミングを実装する方法を示します。

## コンセプト: 行列論理 (Matrix Logic)

Prologのような記号的なユニフィケーションやバックトラック探索の代わりに、**隣接行列 (Adjacency Matrices)** と **行列積 (Matrix Multiplication)** を使用して事実を表現し、関係を推論します。このアプローチにより、GPUアクセラレーションを活用して大規模な論理推論を高速に行うことが可能になります。

## 例: 家系図 (Family Tree)

以下のような単純な家系図を定義します：
- Alice -> Bob
- Bob -> Charlie
- Charlie -> Diana

そして以下を推論します：
- **祖父母 (GrandParent) の関係**: `Parent x Parent` の行列積によって計算されます。
- **先祖 (Ancestor) の関係**: 推移閉包（反復的な行列積）によって計算されます。

## 使い方

```bash
cargo run --release --bin tl -- examples/tasks/logic/family_tree/family_tree.tl
```
