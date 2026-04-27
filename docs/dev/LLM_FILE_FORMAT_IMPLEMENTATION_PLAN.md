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
| 4 | **NumPy .npz** | `feature/npz-builtin` | ✅ 実装完了（マージ待ち） | impl内Vec型推論バグ、XOR/OR未実装 |

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

### バグ3: impl ブロック内での Vec<構造体>.get().unwrap() の型推論

`impl NpzFile` 内のメソッドで `Vec<NpzEntry>.get(i).unwrap()` の返り値型が `Void` に推論され、フィールドアクセス (`entry.name`) で `Field access on non-struct type Void` エラーが発生。

**回避策**: impl メソッドをフリー関数に変更。
**根本修正**: コンパイラの型推論エンジン要調査。

### 不足機能1: ビット演算 `^` (XOR) と `|` (OR)

`&`, `<<`, `>>` は正常動作するが、`^` と `|` が未実装。CRC-32計算（PKZIPで必要）が実装できない。

---

## 残りの作業

### ステップ1: SafeTensors ブランチのマージ

```bash
git checkout main
git merge feature/safetensors-builtin
git push
```

### ステップ2: NumPy .npy 実装 (`feature/npy-builtin`)

#### フォーマット仕様

```
[6 bytes]  magic: \x93NUMPY
[1 byte]   major version (1 or 2)
[1 byte]   minor version
[2 bytes]  header_len: u16 LE  (v1) / [4 bytes] u32 LE (v2)
[N bytes]  header: Python dict文字列 (ASCII)
[D bytes]  data: Cオーダーの連続配列
```

ヘッダーの例:
```python
{'descr': '<f4', 'fortran_order': False, 'shape': (4, 3), }
```

#### 公開API設計

```tl
struct NpyHeader {
    major_version: i64,
    minor_version: i64,
    descr: String,       // 型記述: "<f4", "<f8", "<i4" etc.
    fortran_order: bool,
    shape: Vec<i64>
}

struct NpyFile {
    header: NpyHeader,
    data: Vec<u8>
}

impl NpyFile {
    fn load(path: String) -> Result<NpyFile, String>;
    fn save(self, path: String) -> Result<bool, String>;
    fn element_count(self) -> i64;
    fn dtype_name(self) -> String;
}
```

#### 実装ファイル

```
src/compiler/codegen/builtin_types/non_generic/
├── npy.tl   ← TLコードで型定義とロジック
└── npy.rs   ← BuiltinLoader ローダー

examples/npy/
├── test.npy               ← テスト用データ
└── test_npy_builtin.tl    ← 組み込み型APIテスト
```

#### テスト用 .npy ファイルの生成

```python
import numpy as np
a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
np.save("examples/npy/test.npy", a)
```

#### 想定される発見事項

| カテゴリ | 内容 |
|:---|:---|
| バグ候補 | `bool` フィールドを持つ構造体の初期化とSRET返却 |
| バグ候補 | ASCII文字列から数値への変換のループ処理 |
| バグ候補 | パディング計算 (64バイト境界アライン) のオフバイワン |
| 不足機能 | `String::find(pattern: String) -> Option<i64>` の存在確認 |

#### Python dict パース戦略

完全なPythonパーサーは作らず、.npy 固有の構造のみを対象とする簡易パーサーをTLで実装:

```tl
// 'descr': '<f4' → "<f4"
// 'fortran_order': False → false
// 'shape': (4, 3) → Vec<i64> [4, 3]
fn parse_npy_header(data: Vec<u8>, offset: i64, size: i64) -> NpyHeader {
    // シングルクォートで囲まれた文字列を検索
    // キーワードマッチで各フィールドを抽出
}
```

### ステップ3: NumPy .npz 実装 (`feature/npz-builtin`) [Phase 3]

PKZIPヘッダーを手動パース。`compression_method == 0` (STORED) のみサポート。

#### 公開API設計

```tl
struct NpzFile {
    entries: Vec<NpzEntry>
}

struct NpzEntry {
    name: String,
    data: Vec<u8>
}

impl NpzFile {
    fn load(path: String) -> Result<NpzFile, String>;  // STOREDのみ
    fn get(self, name: String) -> Option<NpyFile>;     // エントリをNpyFileとして解析
    fn entry_count(self) -> i64;
}
```

#### 想定される発見事項

| カテゴリ | 内容 |
|:---|:---|
| バグ候補 | ビット演算 `&`, `|`, `<<`, `>>` の動作確認 |
| バグ候補 | u32範囲の値を i64 で扱う際のオーバーフロー挙動 |
| 不足機能 | ビット演算子の未実装またはバグ |

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
