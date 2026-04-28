# LLMモデルファイル形式 組み込み型実装計画

**目的**: TLコンパイラのバグ発見と不足機能の特定  
**方針**: バグ修正を不足機能追加より優先する  
**位置づけ**: すべてのフォーマットを **TL標準ライブラリの組み込み型** として実装する

---

## 進捗サマリー

| 優先度 | 形式 | ブランチ | 状態 | 発見したバグ |
|:---|:---|:---|:---|:---|
| 1 | **GGUF** | `feature/gguf-builtin` | ✅ マージ済み | Vec要素メモリ管理、SRET動作 |
| 2 | **SafeTensors** | `feature/safetensors-builtin` | ✅ マージ済み | アロケータ不整合、String ARC未登録 |
| 3 | **NumPy .npy** | `feature/npy-builtin` | ✅ マージ済み | なし（バグ発見なし） |
| 4 | **NumPy .npz** | `feature/npz-builtin` | ✅ マージ済み | impl内SRET検出バグ(修正済み)、XOR/OR未実装 |

---

## SafeTensors で発見・修正したバグ

### バグ1: アロケータ不整合 (`tl_string_to_bytes`)

`tl_string_to_bytes` が Rust の `Vec::to_vec()` (jemalloc) でバッファを確保し、TL の `Vec::push` が C の `realloc` でそのバッファを拡張しようとして SIGABRT。

**修正**: `libc::malloc` + `memcpy` に変更。

### バグ2: String の ARC テーブル未登録

全ての String 生成関数 (`tl_string_new`, `tl_string_concat`, `tl_string_clone` 等) が `REF_COUNTS` テーブルに登録していなかった。`inc_ref` / `dec_ref` が全て no-op になり、ARC が機能していなかった。

**修正**: 全 String 生成パスに `tl_mem_register_struct` 呼び出しを追加。

### 設計変更: Deep Clone の廃止

`MEMORY_MANAGEMENT_STRATEGY.md` §5 "Deep Copy" セクションを削除。§2 の ARC ライフサイクルペアが正しく実装されていれば不要であり、バグの温床になると判断。

---

## NPZ で発見したバグ・不足機能

### バグ3: impl ブロック内での Vec<構造体>.get().unwrap() の型推論 (修正済み)

`impl NpzFile` 内のメソッドで `Vec<NpzEntry>.get(i).unwrap()` の返り値型が `Void` に推論され、フィールドアクセス (`entry.name`) で `Field access on non-struct type Void` エラーが発生。

**根本原因**: `Option<T>.unwrap()` が `T=構造体` に特殊化された場合、LLVMの関数はSRET規約を使うが、呼び出し側のSRET検出が最初のパラメータ名だけに依存していた。特殊化関数ではパラメータ名が設定されないため、SRET未検出→void戻り値→Type::Voidとなっていた。

**修正**: `ret_ty == Struct && LLVM returns void` の条件でもSRETを検出するように改善。

### 不足機能1: ビット演算 `^` (XOR) と `|` (OR)

`&`, `<<`, `>>` は正常動作するが、`^` と `|` が未実装。CRC-32計算（PKZIPで必要）が実装できない。

---

## 残りの作業

### ~~ステップ1: SafeTensors ブランチのマージ~~ ✅ 完了

### ~~ステップ2: NumPy .npy 実装~~ ✅ 完了

### ~~ステップ3: NumPy .npz 実装~~ ✅ 完了

### ステップ4: ビット演算 `^` (XOR) と `|` (OR) の実装

NPZ実装で発見した不足機能。CRC-32計算（PKZIP仕様で必要）を実装するために必要。

#### 対象

- `^` (XOR): レキサー、パーサー、コードジェン (LLVM `build_xor`)
- `|` (OR): レキサー、パーサー、コードジェン (LLVM `build_or`)

#### 検証

```tl
fn main() {
    println("0xFF ^ 0x55 = {}", 255 ^ 85);  // 170
    println("0x0F | 0xF0 = {}", 15 | 240);  // 255
}
```

---

## 共通パターン (全フォーマットで踏襲)

```tl
// パターン1: LE読み取り
fn read_u32_le(data: Vec<u8>, offset: i64) -> i64 { ... }
fn read_u64_le(data: Vec<u8>, offset: i64) -> i64 { ... }

// パターン2: LE書き込み (Vec<u8>を返す)
fn write_u32_le(buf: Vec<u8>, val: i64) -> Vec<u8> { ... }
fn write_u64_le(buf: Vec<u8>, val: i64) -> Vec<u8> { ... }

// パターン3: ファイルI/Oは既存のFFIを使う
let res = File::read_binary(path);      // Result<Vec<u8>, String>
let ok = File::write_binary(path, buf); // bool
```

## バグ発見時の対応フロー

1. **最小再現コードを特定** → `examples/<format>/debug_*.tl` に保存
2. **分類**:
   - コンパイラクラッシュ (panic) → `src/compiler/` 修正
   - 誤ったコード生成 → `src/compiler/codegen/` 修正
   - ランタイムエラー (segfault/ARC不整合) → `crates/tl_runtime/` 修正
3. **修正コミット**: `fix: <具体的な内容>` 形式
4. **リグレッションテスト**: `python3 scripts/verify_tl_files.py` で確認

---

## 既知の制限事項と対策

| 制限 | 対策 |
|:---|:---|
| f32のビット操作 (`transmute` 相当) がTLにない | i64で4バイトビットパターンとして扱う |
| 大きなファイルの全量読み込み | 現フェーズでは許容 |
| Deflate圧縮 (.npz) | STORED のみサポートとして回避 |
| Rust/C アロケータ不整合 | FFIで返すバッファは必ず `libc::malloc` で確保する |
