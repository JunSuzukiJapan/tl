# TensorLogic AST設計ドキュメント

## 概要

TensorLogicインタープリター用の抽象構文木（AST）の設計と実装について説明します。

## モジュール構成

```
src/ast/
├── mod.rs      - コアAST型定義
├── span.rs     - ソース位置情報
└── visitor.rs  - ビジターパターン実装
```

## AST階層構造

### 1. プログラム構造

```rust
Program
├── declarations: Vec<Declaration>
│   ├── TensorDecl      // tensor x: float32[10] = ...
│   ├── RelationDecl    // relation Parent(x, y) embed ...
│   ├── RuleDecl        // Ancestor(x,y) <- Parent(x,y)
│   ├── EmbeddingDecl   // embedding person { ... }
│   └── FunctionDecl    // function f(...) -> ... { ... }
└── main_block: Option<MainBlock>
    └── statements: Vec<Statement>
```

### 2. 型システム

#### 2.1 TensorType

```rust
TensorType {
    base_type: BaseType,      // float32, float64, int32, ...
    dimensions: Vec<Dimension>, // [10, 20, ?]
    learnable: LearnableStatus  // learnable | frozen
}
```

**BaseType:**
- `Float32`, `Float64`
- `Int32`, `Int64`
- `Bool`
- `Complex64`

**Dimension:**
- `Fixed(usize)` - 固定サイズ: `10`
- `Variable(Identifier)` - 変数: `n`
- `Dynamic` - 動的: `?`

**LearnableStatus:**
- `Learnable` - 学習可能パラメータ
- `Frozen` - 固定パラメータ
- `Default` - 未指定

#### 2.2 EntityType

```rust
EntityType {
    Entity,           // 論理プログラミングのエンティティ
    Concept,          // 概念
    Tensor(TensorType) // テンソル型
}
```

### 3. 宣言ノード

#### 3.1 TensorDecl

```rust
TensorDecl {
    name: Identifier,
    tensor_type: TensorType,
    init_expr: Option<TensorExpr>
}
```

例: `tensor w: float32[10, 20] learnable = zeros([10, 20])`

#### 3.2 RelationDecl

```rust
RelationDecl {
    name: Identifier,
    params: Vec<Param>,
    embedding_spec: Option<TensorType>
}

Param {
    name: Identifier,
    entity_type: EntityType
}
```

例: `relation Parent(x: entity, y: entity) embed float32[64]`

#### 3.3 RuleDecl

```rust
RuleDecl {
    head: RuleHead,     // Ancestor(x, z)
    body: Vec<BodyTerm> // Parent(x, y), Ancestor(y, z)
}

RuleHead = Atom | TensorEquation
BodyTerm = Atom | TensorEquation | Constraint
```

例: `Ancestor(x, z) <- Parent(x, y), Ancestor(y, z)`

#### 3.4 EmbeddingDecl

```rust
EmbeddingDecl {
    name: Identifier,
    entities: EntitySet,     // {alice, bob} | auto
    dimension: usize,        // 64
    init_method: InitMethod  // xavier, he, random, ...
}
```

例:
```tensorlogic
embedding person {
    entities: {alice, bob, charlie}
    dimension: 64
    init: xavier
}
```

#### 3.5 FunctionDecl

```rust
FunctionDecl {
    name: Identifier,
    params: Vec<Param>,
    return_type: ReturnType,  // Tensor(TensorType) | Void
    body: Vec<Statement>
}
```

### 4. テンソル式

#### 4.1 TensorExpr

```rust
enum TensorExpr {
    Variable(Identifier),        // x
    Literal(TensorLiteral),      // [1, 2, 3]

    BinaryOp {                   // x + y, x @ y
        op: BinaryOp,
        left: Box<TensorExpr>,
        right: Box<TensorExpr>,
    },

    UnaryOp {                    // -x, transpose(x)
        op: UnaryOp,
        operand: Box<TensorExpr>,
    },

    EinSum {                     // einsum("ij,jk->ik", a, b)
        spec: String,
        tensors: Vec<TensorExpr>,
    },

    FunctionCall {               // sigmoid(x)
        name: Identifier,
        args: Vec<TensorExpr>,
    },

    EmbeddingLookup {            // person[alice]
        embedding: Identifier,
        entity: EntityRef,
    },
}
```

#### 4.2 演算子

**BinaryOp:**
- `Add` (+), `Sub` (-), `Mul` (*), `Div` (/)
- `MatMul` (@) - 行列積
- `Power` (**) - 累乗
- `TensorProd` (⊗) - テンソル積
- `Hadamard` (⊙) - アダマール積

**UnaryOp:**
- `Neg` (-) - 符号反転
- `Not` (!) - 論理否定
- `Transpose` - 転置
- `Inverse` - 逆行列
- `Determinant` - 行列式

#### 4.3 リテラル

```rust
TensorLiteral {
    Scalar(ScalarLiteral),  // 1.0, true, 1+2i
    Array(Vec<TensorLiteral>) // [[1, 2], [3, 4]]
}

ScalarLiteral {
    Integer(i64),        // 42
    Float(f64),          // 3.14
    Boolean(bool),       // true
    Complex { real, imag } // 1+2i
}
```

### 5. 論理式と制約

#### 5.1 Atom (論理述語)

```rust
Atom {
    predicate: Identifier,  // Parent
    terms: Vec<Term>        // (alice, bob)
}

Term = Variable(Identifier) | Constant | Tensor(TensorExpr)
```

例: `Parent(alice, bob)`, `Ancestor(x, y)`

#### 5.2 Constraint

```rust
enum Constraint {
    Comparison {              // x > 0, y == z
        op: CompOp,
        left: TensorExpr,
        right: TensorExpr,
    },

    Shape {                   // shape(x) == [10, 20]
        tensor: TensorExpr,
        shape: Vec<Dimension>,
    },

    Rank {                    // rank(x) == 2
        tensor: TensorExpr,
        rank: usize,
    },

    Norm {                    // norm(x) < 1.0
        tensor: TensorExpr,
        op: CompOp,
        value: f64,
    },

    Not(Box<Constraint>),     // not C
    And(Box<Constraint>, Box<Constraint>), // C1 and C2
    Or(Box<Constraint>, Box<Constraint>),  // C1 or C2
}
```

**CompOp:**
- `Eq` (==), `Ne` (!=)
- `Lt` (<), `Gt` (>), `Le` (<=), `Ge` (>=)
- `Approx` (≈) - 近似等価

#### 5.3 TensorEquation

```rust
TensorEquation {
    left: TensorExpr,
    right: TensorExpr,
    eq_type: EquationType    // Exact (=) | Approx (~) | Assign (:=)
}
```

### 6. 文

#### 6.1 Statement

```rust
enum Statement {
    Assignment {              // x := expr
        target: Identifier,
        value: TensorExpr,
    },

    Equation(TensorEquation), // x = y + z

    Query {                   // query Parent(x, y) where x > 0
        atom: Atom,
        constraints: Vec<Constraint>,
    },

    Inference {               // infer forward query ...
        method: InferenceMethod,
        query: Box<Statement>,
    },

    Learning(LearningSpec),   // learn { ... }

    ControlFlow(ControlFlow), // if, for, while
}
```

#### 6.2 推論メソッド

```rust
enum InferenceMethod {
    Forward,   // 前向き推論
    Backward,  // 後向き推論
    Gradient,  // 勾配ベース
    Symbolic,  // 記号的推論
}
```

#### 6.3 学習仕様

```rust
LearningSpec {
    objective: TensorExpr,      // 目的関数
    optimizer: OptimizerSpec,   // adam(lr=0.001)
    epochs: usize               // エポック数
}

OptimizerSpec {
    name: String,               // "adam", "sgd"
    params: Vec<(String, f64)>  // [("lr", 0.001), ("momentum", 0.9)]
}
```

### 7. 制御フロー

```rust
enum ControlFlow {
    If {
        condition: Condition,
        then_block: Vec<Statement>,
        else_block: Option<Vec<Statement>>,
    },

    For {
        variable: Identifier,
        iterable: Iterable,
        body: Vec<Statement>,
    },

    While {
        condition: Condition,
        body: Vec<Statement>,
    },
}

Condition = Constraint(Constraint) | Tensor(TensorExpr)
Iterable = Tensor(TensorExpr) | EntitySet(EntitySet) | Range(usize)
```

## ビジターパターン

### 8.1 Visitor (不変トラバーサル)

```rust
trait Visitor {
    type Error;

    fn visit_program(&mut self, program: &Program) -> Result<(), Self::Error>;
    fn visit_declaration(&mut self, decl: &Declaration) -> Result<(), Self::Error>;
    fn visit_tensor_expr(&mut self, expr: &TensorExpr) -> Result<(), Self::Error>;
    fn visit_statement(&mut self, stmt: &Statement) -> Result<(), Self::Error>;
    // ... その他のノード型
}
```

**用途:**
- 型チェック
- コード生成
- 静的解析
- プリティプリント

### 8.2 VisitorMut (可変トラバーサル)

```rust
trait VisitorMut {
    type Error;

    fn visit_program_mut(&mut self, program: &mut Program) -> Result<(), Self::Error>;
    fn visit_tensor_expr_mut(&mut self, expr: &mut TensorExpr) -> Result<(), Self::Error>;
    // ...
}
```

**用途:**
- AST変換
- 最適化パス
- デシュガリング

## ソース位置情報

### 9.1 Span

```rust
Position {
    line: usize,
    column: usize,
    offset: usize
}

Span {
    start: Position,
    end: Position
}

Spanned<T> {
    node: T,
    span: Span
}
```

**用途:**
- エラーメッセージの改善
- デバッグ情報
- IDE統合

## 使用例

### 型チェッカー実装例

```rust
struct TypeChecker {
    env: HashMap<Identifier, TensorType>,
}

impl Visitor for TypeChecker {
    type Error = TypeError;

    fn visit_tensor_decl(&mut self, decl: &TensorDecl) -> Result<(), TypeError> {
        // 型情報を環境に追加
        self.env.insert(decl.name.clone(), decl.tensor_type.clone());

        // 初期化式の型チェック
        if let Some(expr) = &decl.init_expr {
            let expr_type = self.infer_type(expr)?;
            self.check_type_compatible(&decl.tensor_type, &expr_type)?;
        }

        Ok(())
    }

    fn visit_binary_op(&mut self, op: BinaryOp, left: &TensorExpr, right: &TensorExpr)
        -> Result<TensorType, TypeError>
    {
        let left_ty = self.infer_type(left)?;
        let right_ty = self.infer_type(right)?;

        match op {
            BinaryOp::Add | BinaryOp::Sub => {
                self.check_broadcast_compatible(&left_ty, &right_ty)?;
                Ok(self.broadcast_result_type(&left_ty, &right_ty))
            }
            BinaryOp::MatMul => {
                self.check_matmul_compatible(&left_ty, &right_ty)?;
                Ok(self.matmul_result_type(&left_ty, &right_ty))
            }
            // ...
        }
    }
}
```

### AST構築例

```rust
use tensorlogic::ast::*;

// Ancestor(x, z) <- Parent(x, y), Ancestor(y, z)
let rule = RuleDecl {
    head: RuleHead::Atom(Atom::new("Ancestor", vec![
        Term::Variable(Identifier::new("x")),
        Term::Variable(Identifier::new("z")),
    ])),
    body: vec![
        BodyTerm::Atom(Atom::new("Parent", vec![
            Term::Variable(Identifier::new("x")),
            Term::Variable(Identifier::new("y")),
        ])),
        BodyTerm::Atom(Atom::new("Ancestor", vec![
            Term::Variable(Identifier::new("y")),
            Term::Variable(Identifier::new("z")),
        ])),
    ],
};

// tensor w: float32[10, 20] learnable
let tensor_decl = TensorDecl {
    name: Identifier::new("w"),
    tensor_type: TensorType::learnable_float32(vec![10, 20]),
    init_expr: None,
};

// x + y
let expr = TensorExpr::binary(
    BinaryOp::Add,
    TensorExpr::var("x"),
    TensorExpr::var("y"),
);
```

## 設計の利点

### 1. 型安全性
- Rustの型システムを活用
- コンパイル時の静的チェック
- パターンマッチングによる網羅性チェック

### 2. 拡張性
- 新しいノード型の追加が容易
- ビジターパターンで処理の分離
- モジュール構造が明確

### 3. 保守性
- ドキュメント化された型定義
- 明確な責務分離
- テストが書きやすい

### 4. パフォーマンス
- ゼロコストアブストラクション
- 効率的なメモリレイアウト
- Clone/Debugの自動導出

## 次のステップ

1. **パーサー実装** - Pestを使用したBNF → AST変換
2. **型チェッカー** - 静的型推論と検証
3. **インタープリター** - AST実行エンジン
4. **コード生成** - 最適化されたバイトコード/LLVM IR

## 参考資料

- [tensorlogic_grammar.md](./tensorlogic_grammar.md) - 言語文法定義
- [Rust AST設計パターン](https://rust-unofficial.github.io/patterns/patterns/structural/newtype.html)
- [Visitor Pattern in Rust](https://rust-unofficial.github.io/patterns/patterns/behavioural/visitor.html)
