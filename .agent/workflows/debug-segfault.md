---
description: セグメンテーションフォールトが発生した際のデバッグ手順
---

1. **再現確認**: 最小限の再現コードを作成する。
2. **JIT関数マッピングの確認**:
   - `src/compiler/codegen/builtins.rs` を確認する。
   - セグフォが発生している箇所で呼び出されているランタイム関数（例: `tl_tensor_to_f32`）が `execution_engine.add_global_mapping` で正しくマッピングされているか確認する。
   - **重要**: `add_fn` や `fn_return_types.insert` だけでは不十分。必ず `add_global_mapping` が必要。マッピングがないとJIT実行時に不正なアドレスにジャンプして即座にクラッシュする（エラーメッセージが出ないことが多い）。
3. **戻り値の型定義確認**:
   - `builtins.rs` の `fn_return_types` に登録されている型が正しいか確認する。特にポインタを返す関数が `Type::F32` などになっていないか。
4. **二重登録/二重解放**:
   - `src/compiler/codegen/expr.rs` で、ランタイム関数が既に登録済みテンソルを返しているのに、コンパイラ側で `emit_register_tensor` を呼んでいないか確認する。
5. **LLVM IR ダンプ**:
   - `src/compiler/codegen/mod.rs` に `self.module.print_to_file("debug.ll").ok();` を追加してIRを確認する。
