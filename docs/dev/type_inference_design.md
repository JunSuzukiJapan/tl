# TL言語 型推論システム設計 (Type Inference System Design)

## 1. 概要
本ドキュメントでは、TLコンパイラにおける「遅延型解決（Lazy Type Resolution）」および「単一化（Unification）」を用いた型推論システムの設計について記述する。
従来のTLコンパイラは前方参照のみのシングルパス解析を行っており、`let v = Vec::new();` のような「宣言時には型が決定せず、後の使用方法によって型が定まる」パターンの推論が不可能であった。
本設計では、型変数（`Type::Undefined`）と制約解消メカニズムを導入することで、この問題を解決し、より柔軟な記述を可能にする。

## 2. アーキテクチャ

推論プロセスは大きく分けて「収集フェーズ（Constraint Collection）」と「具象化フェーズ（Reification）」の2段階で構成されるが、実装上はセマンティック解析の1パスの中に組み込まれる。

### 2.1 コア概念

1.  **Type::Undefined(u64)**:
    *   未決定の型を表すプレースホルダー。一意なIDを持つ。
    *   例: `Vec::new()` は `Struct("Vec", [Undefined(0)])` という型を返す。
2.  **Inference Map**:
    *   `Map<u64, Type>`。型変数のIDと、特定された具体的な型（または別の型変数）との対応表。
    *   `SemanticAnalyzer` が保持する。
3.  **Unification (単一化)**:
    *   2つの型 `T1` と `T2` を比較し、矛盾なく適合させるプロセス。
    *   片方が `Undefined(id)` であれば、もう片方の型に束縛（Bind）し、`Inference Map` を更新する。

## 3. 推論フロー

### ステップ1: 未定義型の生成 (Generation)
型が明示されていないジェネリック関数の呼び出し時に、`Undefined` 型を生成する。

```rust
// AST: let mut v = Vec::new();
// Semantic Analyzer:
// 1. Vec::new() を解析。
// 2. 引数がないため、T を推論できない。
// 3. 新しい型変数 $0 (id=0) を生成。
// 4. 式の型として Vec<$0> を返す。
// 5. 変数 `v` は Vec<$0> 型としてシンボルテーブルに登録される。
```

### ステップ2: 制約の適用と単一化 (Constraint Application & Unification)
変数が使用される箇所で、型制約を適用する。

```rust
// AST: v.push(10);
// Semantic Analyzer:
// 1. `v` の型を検索 -> Vec<$0>
// 2. `push` メソッドのシグネチャを確認 -> fn push(self, val: T)
// 3. 実引数 `10` の型は `i64`。
// 4. メソッドの引数 `val: T` (つまり $0) と 実引数 `i64` を Unify する。
//    Unify($0, i64)
// 5. $0 が未定義なので、$0 = i64 と決定。Inference Map に登録。
```

### ステップ3: 具象化 (Reification)
スコープ（関数定義）の終了時に、AST内に残っている `Undefined` 型を確定した型に置き換える。

```rust
// 関数終了時
// 1. AST (Stmt/Expr) をトラバース。
// 2. `Vec<$0>` を発見。
// 3. Inference Map を参照し、$0 -> i64 に置換。
// 4. 最終的な型は `Vec<i64>` となる。
// 5. もし Inference Map にエントリがない（一度も使われなかった）場合、エラーとする（"Type verify error: failed to infer type"）。
```

## 4. 実装詳細

### 4.1 SemanticAnalyzer の拡張
```rust
struct SemanticAnalyzer {
    // ...
    undefined_counter: u64,
    inference_map: HashMap<u64, Type>,
}
```

### 4.2 Unify メソッド
`are_types_compatible` 内、または引数チェック時に呼び出される。
```rust
fn unify(&mut self, t1: &Type, t2: &Type) -> bool {
    let t1 = self.resolve_inferred_type(t1);
    let t2 = self.resolve_inferred_type(t2);
    match (t1, t2) {
        (Undefined(id), ty) => { self.inference_map.insert(id, ty); true }
        (ty, Undefined(id)) => { self.inference_map.insert(id, ty); true }
        (Struct(n1, args1), Struct(n2, args2)) => {
            // 再帰的に Unify
            if n1 == n2 && args1.len() == args2.len() {
                zip(args1, args2).all(|(a, b)| self.unify(a, b))
            } else { false }
        }
        // ... 他の型
    }
}
```

### 4.3 AST Reification
解析の最後に呼ばれるパス。
```rust
fn resolve_stmt_types(&mut self, stmt: &mut Stmt) {
    match inner {
        Let { type_annotation, .. } => {
            *type_annotation = self.resolve_inferred_type(type_annotation);
        }
        // ... Expr 内の TypeNode も更新
    }
    // Expr も再帰的にトラバース
}
```

## 5. 考慮事項と制約

*   **スコープ**: 現在の設計では関数ローカルな推論を想定。関数の境界を超えた（戻り値の推論など）グローバルな推論は行わない。
*   **多相性**: Hindley-Milner のような完全な多相型システムではなく、あくまで「省略されたジェネリック引数の穴埋め」に特化する。
*   **エラーメッセージ**: 推論失敗時のエラーメッセージを分かりやすくするために、`Undefined` 生成元の情報を保持するなどが将来的に必要になる可能性がある。

## 6. 今後の拡張 (Future Work)
*   `HashMap<K, V>` などの複数型変数への対応（本設計で自然に対応可能）。
*   クロージャの引数型推論。
