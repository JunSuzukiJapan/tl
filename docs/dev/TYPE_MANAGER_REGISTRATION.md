# TypeManager メソッド登録ガイド

## 概要

ビルトイン型のメソッドは**TypeManager**に登録することで、セマンティクスチェックとコード生成の両方で自動的に認識されます。

---

## メソッド追加の流れ

### 1. 登録ファイルを選択

| 型カテゴリ | ファイル |
|---|---|
| プリミティブ（F32, I64, String等） | `codegen/builtin_types/non_generic/primitives.rs` |
| IO関連（File, Path, Env, Map, Http） | `codegen/builtin_types/non_generic/io.rs` |
| LLM関連（Tokenizer, KVCache, VarBuilder） | `codegen/builtin_types/non_generic/llm.rs` |
| Tensor | `codegen/builtin_types/non_generic/tensor.rs` |
| System | `codegen/builtin_types/non_generic/system.rs` |
| Param | `codegen/builtin_types/non_generic/param.rs` |

### 2. メソッド登録

#### シグネチャのみ（実装はランタイム）
```rust
type_info.register_instance_signature(
    "method_name",           // メソッド名
    vec![Type::I64],         // 引数型（selfを除く）
    Type::Bool               // 戻り値型
);
```

#### 実装付き（コンパイル時評価）
```rust
type_info.register_evaluated_instance_method(
    "method_name",
    compile_method_fn,       // fn(&mut CodeGenerator, ...) -> Result<...>
    vec![Type::I64],
    Type::Bool
);
```

---

## 例

### 例1: F32にsqrt追加（シグネチャのみ）
```rust
// primitives.rs
f32_type.register_instance_signature("sqrt", vec![], Type::F32);
```

### 例2: 静的メソッド追加
```rust
// io.rs
path.register_evaluated_static_method(
    "exists",
    compile_path_exists,
    vec![Type::String("String".to_string())],
    Type::Bool
);
```

### 例3: オーバーロード
```rust
// tensor.rs - sumは0引数と1引数の両方をサポート
tensor.register_instance_signature("sum", vec![], tensor_type.clone());
tensor.register_instance_signature("sum", vec![Type::I64], tensor_type.clone());
```

---

## 重要な注意点

1. **SemanticAnalyzerとCodeGeneratorで同じ登録関数が呼ばれる**
   - `semantics.rs`の`SemanticAnalyzer::new()`
   - `codegen/builtin_types/mod.rs`の`load_all_builtins()`

2. **新しい型を追加する場合**は両方の初期化箇所に登録関数呼び出しを追加

3. **SignatureOnlyメソッド**はランタイム関数で実装される
   - 命名規則: `tl_<Type>_<method>` または `tl_<type>_<method>`
