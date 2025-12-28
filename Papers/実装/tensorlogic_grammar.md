
# テンソルロジック プログラミング言語 文法定義

## 概要

テンソルロジックは、テンソル代数と論理プログラミングを統一した新しいプログラミング言語です。この言語では、テンソル方程式が唯一の構成要素となり、ニューラルネットワークと記号的AIを根本的なレベルで統合します。

## 1. 基本構文 (BNF記法)

### 1.1 プログラム構造

```bnf
<program> ::= <declaration>* <main_block>?

<declaration> ::= <tensor_decl>
                | <relation_decl>
                | <rule_decl>
                | <embedding_decl>
                | <function_decl>

<main_block> ::= "main" "{" <statement>* "}"
```

### 1.2 テンソル宣言

```bnf
<tensor_decl> ::= "tensor" <identifier> ":" <tensor_type> ("=" <tensor_expr>)?

<tensor_type> ::= <base_type> "[" <dimension_list> "]" <learnable>?

<base_type> ::= "float32" | "float64" | "int32" | "int64" | "bool" | "complex64"

<dimension_list> ::= <dimension> ("," <dimension>)*

<dimension> ::= <integer> | <identifier> | "?"

<learnable> ::= "learnable" | "frozen"
```

### 1.3 関係宣言

```bnf
<relation_decl> ::= "relation" <identifier> "(" <param_list>? ")" <embedding_spec>?

<param_list> ::= <param> ("," <param>)*

<param> ::= <identifier> ":" <entity_type>

<entity_type> ::= "entity" | "concept" | <tensor_type>

<embedding_spec> ::= "embed" <tensor_type>
```

### 1.4 ルール宣言

```bnf
<rule_decl> ::= <rule_head> "<-" <rule_body>

<rule_head> ::= <atom> | <tensor_equation>

<rule_body> ::= <body_term> ("," <body_term>)*

<body_term> ::= <atom> | <tensor_equation> | <constraint>

<atom> ::= <identifier> "(" <term_list>? ")"

<term_list> ::= <term> ("," <term>)*

<term> ::= <identifier> | <constant> | <tensor_expr>
```

### 1.5 埋め込み宣言

```bnf
<embedding_decl> ::= "embedding" <identifier> "{" 
                     "entities" ":" <entity_set>
                     "dimension" ":" <integer>
                     "init" ":" <init_method>?
                     "}"

<entity_set> ::= "{" <identifier_list> "}" | "auto"

<identifier_list> ::= <identifier> ("," <identifier>)*

<init_method> ::= "random" | "xavier" | "he" | "zeros" | "ones"
```

## 2. テンソル式

### 2.1 テンソル演算

```bnf
<tensor_expr> ::= <tensor_term>
                | <tensor_expr> <binary_op> <tensor_expr>
                | <unary_op> <tensor_expr>
                | "(" <tensor_expr> ")"

<tensor_term> ::= <identifier>
                | <tensor_literal>
                | <einstein_sum>
                | <tensor_function_call>
                | <embedding_lookup>

<binary_op> ::= "+" | "-" | "*" | "/" | "@" | "**" | "⊗" | "⊙"

<unary_op> ::= "-" | "!" | "transpose" | "inv" | "det"
```

### 2.2 アインシュタイン和

```bnf
<einstein_sum> ::= "einsum" "(" <einsum_spec> "," <tensor_list> ")"

<einsum_spec> ::= <string_literal>

<tensor_list> ::= <tensor_expr> ("," <tensor_expr>)*
```

### 2.3 埋め込み参照

```bnf
<embedding_lookup> ::= <identifier> "[" <entity_ref> "]"

<entity_ref> ::= <identifier> | <string_literal>
```

## 3. 論理式

### 3.1 制約

```bnf
<constraint> ::= <comparison>
               | <tensor_constraint>
               | <logical_constraint>

<comparison> ::= <tensor_expr> <comp_op> <tensor_expr>

<comp_op> ::= "==" | "!=" | "<" | ">" | "<=" | ">=" | "≈"

<tensor_constraint> ::= "shape" "(" <tensor_expr> ")" "==" <shape_spec>
                      | "rank" "(" <tensor_expr> ")" "==" <integer>
                      | "norm" "(" <tensor_expr> ")" <comp_op> <number>

<logical_constraint> ::= "not" <constraint>
                       | <constraint> "and" <constraint>
                       | <constraint> "or" <constraint>
```

### 3.2 テンソル方程式

```bnf
<tensor_equation> ::= <tensor_expr> "=" <tensor_expr>
                    | <tensor_expr> "~" <tensor_expr>  // 近似等価
                    | <tensor_expr> ":=" <tensor_expr> // 代入
```

## 4. 関数定義

```bnf
<function_decl> ::= "function" <identifier> "(" <param_list>? ")" 
                    "->" <return_type> "{" <statement>* "}"

<return_type> ::= <tensor_type> | "void"
```

## 5. 文

```bnf
<statement> ::= <assignment>
              | <tensor_equation>
              | <query>
              | <inference_call>
              | <learning_call>
              | <control_flow>

<assignment> ::= <identifier> ":=" <tensor_expr>

<query> ::= "query" <atom> ("where" <constraint_list>)?

<constraint_list> ::= <constraint> ("," <constraint>)*

<inference_call> ::= "infer" <inference_method> <query>

<inference_method> ::= "forward" | "backward" | "gradient" | "symbolic"

<learning_call> ::= "learn" "{" <learning_spec> "}"

<learning_spec> ::= "objective" ":" <tensor_expr>
                   "optimizer" ":" <optimizer_spec>
                   "epochs" ":" <integer>
```

## 6. 制御フロー

```bnf
<control_flow> ::= <if_statement>
                 | <for_statement>
                 | <while_statement>

<if_statement> ::= "if" <condition> "{" <statement>* "}" 
                   ("else" "{" <statement>* "}")?

<condition> ::= <constraint> | <tensor_expr>

<for_statement> ::= "for" <identifier> "in" <iterable> "{" <statement>* "}"

<iterable> ::= <tensor_expr> | <entity_set> | "range" "(" <integer> ")"

<while_statement> ::= "while" <condition> "{" <statement>* "}"
```

## 7. リテラル

```bnf
<tensor_literal> ::= "[" <tensor_elements> "]"
                   | <scalar_literal>

<tensor_elements> ::= <tensor_element> ("," <tensor_element>)*

<tensor_element> ::= <tensor_literal> | <number>

<scalar_literal> ::= <number> | <boolean> | <complex_number>

<constant> ::= <number> | <string_literal> | <boolean>

<number> ::= <integer> | <float>

<integer> ::= <digit>+

<float> ::= <digit>+ "." <digit>+ (<exponent>)?

<exponent> ::= ("e" | "E") ("+" | "-")? <digit>+

<boolean> ::= "true" | "false"

<complex_number> ::= <number> ("+" | "-") <number> "i"

<string_literal> ::= "\"" <char>* "\""

<char> ::= <any_unicode_char_except_quote_and_backslash> | <escape_sequence>

<escape_sequence> ::= "\\" ("\"" | "\\" | "n" | "t" | "r")
```

## 8. 識別子とキーワード

```bnf
<identifier> ::= <letter> (<letter> | <digit> | "_")*

<letter> ::= "a"..."z" | "A"..."Z" | <unicode_letter>

<digit> ::= "0"..."9"
```

### 8.1 予約キーワード

```
tensor, relation, rule, embedding, function, main, learnable, frozen,
entity, concept, embed, einsum, query, infer, learn, forward, backward,
gradient, symbolic, if, else, for, while, in, range, true, false,
not, and, or, shape, rank, norm, transpose, inv, det, objective,
optimizer, epochs, auto, random, xavier, he, zeros, ones
```

## 9. 演算子優先順位

```
1. 関数呼び出し, 配列アクセス: f(), a[i]
2. 単項演算子: -, !, transpose, inv, det
3. 冪乗: **
4. 乗除: *, /, @, ⊗, ⊙
5. 加減: +, -
6. 比較: ==, !=, <, >, <=, >=, ≈
7. 論理否定: not
8. 論理積: and
9. 論理和: or
10. 代入: :=, =, ~
```

## 10. 型システム

### 10.1 型推論規則

- テンソル演算の結果型は入力テンソルの形状から自動推論
- アインシュタイン和の結果型は添字記法から決定
- 埋め込み参照の型は埋め込み宣言から決定
- 学習可能パラメータは勾配計算グラフに自動登録

### 10.2 型制約

- テンソル演算は互換性のある形状でのみ実行可能
- 論理演算子は真偽値テンソルまたはスカラーに適用
- 埋め込み参照は宣言されたエンティティセット内でのみ有効

## 11. 意味論

### 11.1 実行モデル

1. **宣言フェーズ**: すべての宣言を処理し、型チェックを実行
2. **推論フェーズ**: ルールに基づく前向き/後向き推論
3. **学習フェーズ**: 勾配ベースの最適化
4. **クエリフェーズ**: ユーザークエリの処理

### 11.2 メモリモデル

- テンソルは自動的にGPU/CPUメモリで管理
- 埋め込みは効率的なルックアップテーブルとして実装
- 学習可能パラメータは自動微分グラフで追跡

## 12. 使用例

```tensorlogic
// 埋め込み定義
embedding person_embed {
    entities: {alice, bob, charlie}
    dimension: 64
    init: xavier
}

embedding relation_embed {
    entities: {parent, ancestor}
    dimension: 64
    init: xavier
}

// 関係宣言
relation Parent(x: entity, y: entity) embed float32[64]
relation Ancestor(x: entity, y: entity) embed float32[64]

// ルール定義
Ancestor(x, y) <- Parent(x, y)
Ancestor(x, z) <- Ancestor(x, y), Parent(y, z)

// テンソル方程式による実装
Parent(x, y) = sigmoid(person_embed[x] @ relation_embed[parent] @ person_embed[y])

// 学習
learn {
    objective: cross_entropy(Parent(alice, bob), 1.0) + 
               cross_entropy(Parent(bob, charlie), 1.0)
    optimizer: adam(lr=0.001)
    epochs: 1000
}

// クエリ
query Ancestor(alice, charlie)
```

この文法定義により、テンソルロジックの理論的概念を実際のプログラミング言語として実装することが可能になります。
