# フェーズ2: エラーハンドリング強化 — 実装チェックリスト

**目的**: パニックの可能性を排除し、エラー伝播を `TlError` 型に統一する
**前提**: フェーズ1（構造リファクタリング）完了済み、全テスト通過済み

---

## ステップ 1: `CodegenError` 型の拡張と `From` トレイト実装
**対象**: `src/compiler/error.rs`
**リスク**: 低（型定義追加のみ）

- [x] `CodegenErrorKind` にバリアント追加 (`LlvmBuilder`, `LlvmContext`, `Internal`)
- [x] `impl From<String> for TlError` を実装
- [x] `impl From<CodegenErrorKind> for TlError` を実装
- [x] `cargo build` 通過確認

---

## ステップ 2: LLVM ビルダーヘルパー関数の導入
**対象**: `src/compiler/codegen/mod.rs`
**リスク**: 低（ヘルパー追加のみ）

- [x] `current_block()` ヘルパー追加
- [x] `current_function()` ヘルパー追加
- [x] `get_fn(name)` ヘルパー追加
- [x] `cargo build` 通過確認

---

## ステップ 3: `codegen/mod.rs` の unwrap 置換 (91箇所)
**対象**: `src/compiler/codegen/mod.rs`
**リスク**: 中

- [x] `get_insert_block().unwrap()` → `self.current_block()?`
- [x] `get_parent().unwrap()` → `self.current_function()?` またはチェイン
- [x] `get_function("x").unwrap()` → `self.get_fn("x")?`
- [x] `build_call().unwrap()` → `.map_err(|e| e.to_string())?`
- [x] `build_return().unwrap()` → `.map_err(|e| e.to_string())?`
- [x] `build_store().unwrap()` → `.map_err(|e| e.to_string())?`
- [x] 残りの `build_*().unwrap()` を置換
- [x] `cargo build` 通過確認
- [x] `python3 scripts/verify_tl_files.py` 通過確認 (スモークテスト通過)

---

## ステップ 4: `stmt.rs` の unwrap 置換 (152箇所)
**対象**: `src/compiler/codegen/stmt.rs`
**リスク**: 中

- [x] `get_insert_block().unwrap()` → `self.current_block()?`
- [x] `get_parent().unwrap()` → `self.current_function()?`
- [x] `build_pointer_cast().unwrap()` → `.map_err(...)?`
- [x] `build_call().unwrap()` → `.map_err(...)?`
- [x] `build_unconditional_branch().unwrap()` → `.map_err(...)?`
- [x] `build_load().unwrap()` → `.map_err(...)?`
- [x] 残りの `build_*().unwrap()` を置換 (152→0、コメント内1残)
- [x] `cargo build` 通過確認
- [x] `python3 scripts/verify_tl_files.py` 通過確認 (262/262 成功)

---

## ステップ 5: `expr/` サブモジュール群の unwrap 置換 (~843箇所)
**対象**: `src/compiler/codegen/expr/`
**リスク**: 高（最大規模）

### 5-1: `expr/types.rs`
- [x] unwrap 置換 (元から0箇所)
- [x] `cargo build` 通過確認

### 5-2: `expr/method_registry.rs`
- [x] unwrap 置換 (3→0)
- [x] `cargo build` 通過確認

### 5-3: `expr/struct_ops.rs`
- [x] unwrap 置換 (2→0)
- [x] `cargo build` 通過確認

### 5-4: `expr/tensor_methods.rs` (~23箇所)
- [x] unwrap 置換 (23→0)
- [x] `cargo build` 通過確認

### 5-5: `expr/builtin_fns.rs` (~34箇所)
- [x] unwrap 置換 (34→0)
- [x] `cargo build` 通過確認

### 5-6: `expr/mod.rs` (~693箇所) — パターン別バッチ処理
- [x] `build_load().unwrap()` (80箇所)
- [x] `build_struct_gep().unwrap()` (52箇所)
- [x] `build_call().unwrap()` (34箇所)
- [x] `get_insert_block().unwrap()` (36箇所)
- [x] `build_pointer_cast().unwrap()` (28箇所)
- [x] 残りの `build_*().unwrap()` 全置換
- [x] `cargo build` 通過確認
- [x] スモークテスト通過確認

---

## ステップ 6: `Result<_, String>` → `Result<_, TlError>` シグネチャ統一
**対象**: codegen 全体 (541箇所)
**リスク**: 高（広範囲のシグネチャ変更）

- [ ] `codegen/mod.rs` — 18関数のシグネチャ変更
- [ ] `codegen/stmt.rs` — 18関数のシグネチャ変更
- [ ] `codegen/expr/mod.rs` — 49関数のシグネチャ変更
- [ ] `codegen/expr/*.rs` — 残りサブモジュール
- [ ] `builtin_types/**/*.rs` — 残りファイル群
- [ ] その他 (`tensor.rs`, `mono.rs`, `kb.rs` 等)
- [ ] `main.rs` の手動変換コード削除
- [ ] `cargo build` 通過確認
- [ ] `python3 scripts/verify_tl_files.py` 通過確認

---

## ステップ 7: `semantics.rs` + `main.rs` の unwrap 置換
**対象**: semantics.rs (22箇所), main.rs (2箇所)
**リスク**: 低

- [ ] `semantics.rs` の unwrap 置換 (22箇所)
- [ ] `main.rs` の unwrap 置換 (2箇所)
- [ ] `cargo build` 通過確認
- [ ] `python3 scripts/verify_tl_files.py` 通過確認

---

## 最終検証

- [ ] `grep -rn '\.unwrap()' src/compiler/codegen/` — 残存 unwrap の確認と正当性評価
- [ ] `grep -rn 'Result<.*String>' src/compiler/codegen/` — 残存 String エラーの確認
- [ ] 全 `examples/` スクリプトの実行テスト
- [ ] コミット
