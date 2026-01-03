# MemoryManager 使用ドキュメント

## 概要

MemoryManager は、TensorLogic コンパイラが生成するコードで使用される、スコープベースの自動メモリ管理システムです。

## 基本概念

### スコープとは

スコープは、変数のライフタイムを表す論理的な境界です。C/Rust の `{}` ブロックに相当します。

- 関数の本体
- ループの本体（while, for）
- 条件分岐の各ブランチ（if/else）

### MemoryManager の役割

スコープ内で作成されたテンソルと構造体を自動的に追跡し、スコープ終了時に解放します。

```
enter_scope()
  ↓
  変数の作成・登録
  ↓
exit_scope() → すべての登録済みポインタを解放
```

---

## C-ABI 関数

### スコープ管理

#### `tl_mem_enter_scope()`

新しいスコープに入る際に呼び出します。

**呼び出しタイミング:**
- 関数の先頭（引数の処理前）
- ループの本体の先頭
- if/else の各ブランチの先頭

**LLVM IR 例:**
```llvm
define void @main() {
entry:
  call void @tl_mem_enter_scope()
  ; ... 関数本体 ...
  call void @tl_mem_exit_scope()
  ret void
}
```

#### `tl_mem_exit_scope()`

現在のスコープから出る際に呼び出します。

**呼び出しタイミング:**
- 関数からの return 直前
- ループの本体の終わり
- if/else の各ブランチの終わり

**重要:** 
- すべての実行パスで `exit_scope` が呼ばれることを保証する必要があります
- return 文がある場合、その直前に呼び出す
- 例外的終了（panic など）では呼ばれない（現在未対応）

---

### 登録

#### `tl_mem_register_tensor(ptr: *mut OpaqueTensor)`

新しく作成されたテンソルを現在のスコープに登録します。

**呼び出しタイミング:**
- `tl_tensor_new` の直後
- `tl_tensor_randn` の直後  
- テンソルを返す任意の関数の直後

**注意:**
- `make_tensor` (Rust内部関数) は自動的に登録を行います
- ユーザーコード（LLVM IR）からは通常呼び出し不要ですが、外部からテンソルを受け取る場合は明示的に登録が必要

#### `tl_mem_register_struct(ptr: *mut c_void)`

新しく作成された構造体を現在のスコープに登録します。

**呼び出しタイミング:**
- `malloc` で構造体を作成した直後

**現在の問題:**
- `Expr::Struct`（構造体リテラル）で構造体を作成する際、登録が行われていない
- これが malloc checksum エラーの原因の一つ

---

### 登録解除

#### `tl_mem_unregister(ptr: *mut c_void)`

ポインタをMemoryManagerの追跡から除外します。スコープ終了時に解放されなくなります。

**使用ケース:**

1. **return 文**
   ```rust
   // 関数から値を返す場合、呼び出し元にポインタの所有権を移譲
   let result = tensor_add(a, b);
   tl_mem_unregister(result); // スコープ終了時に解放されない
   return result;
   ```

2. **フィールドへの代入**
   ```rust
   // 構造体フィールドに代入する場合、構造体が所有権を持つ
   struct.field = tensor;
   tl_mem_unregister(tensor); // スコープ終了時に解放されない
   ```

3. **変数のシャドーイング**
   ```rust
   let x = tensor_a; // A を登録
   let x = tensor_b; // B を登録、A は？
   // A を unregister しないと、スコープ終了時に A が解放される
   // しかし A は既にアクセス不可能（シャドーイングされた）
   ```

---

## 正しい使用パターン

### 関数の基本構造

```llvm
define void @my_function() {
entry:
  ; スコープ開始
  call void @tl_mem_enter_scope()
  
  ; 変数作成・処理
  %tensor = call ptr @tl_tensor_randn(...)
  ; %tensor は自動的に登録される (make_tensor 内で)
  
  ; スコープ終了（すべてのパスで）
  call void @tl_mem_exit_scope()
  ret void
}
```

### return 文を含む関数

```llvm
define ptr @create_tensor() {
entry:
  call void @tl_mem_enter_scope()
  
  %result = call ptr @tl_tensor_randn(...)
  
  ; 返り値を登録解除（所有権を呼び出し元に移譲）
  call void @tl_mem_unregister(ptr %result)
  
  ; スコープクリーンアップ
  call void @tl_mem_exit_scope()
  
  ret ptr %result
}
```

### 構造体フィールドへの代入

```llvm
; struct.field = tensor の処理
%tensor = call ptr @tl_tensor_randn(...)
%field_ptr = getelementptr %Struct, ptr %struct, i32 0, i32 0
store ptr %tensor, ptr %field_ptr

; 構造体が所有権を持つため、テンソルを登録解除
call void @tl_mem_unregister(ptr %tensor)
```

---

## 間違った使用パターン

### ❌ スコープの不一致

```llvm
define void @bad_function() {
entry:
  call void @tl_mem_enter_scope()
  
  br i1 %cond, label %then, label %else

then:
  ; exit_scope を呼ばずに return
  ret void  ; ← メモリリーク！

else:
  call void @tl_mem_exit_scope()
  ret void
}
```

**修正:**
```llvm
define void @good_function() {
entry:
  call void @tl_mem_enter_scope()
  
  br i1 %cond, label %then, label %else

then:
  call void @tl_mem_exit_scope()
  ret void

else:
  call void @tl_mem_exit_scope()
  ret void
}
```

### ❌ 二重登録

```llvm
%tensor = call ptr @tl_tensor_randn(...)
; make_tensor で既に登録済み

call void @tl_mem_register_tensor(ptr %tensor)
; ← 二重登録！スコープ終了時に二重解放
```

### ❌ unregister 忘れ

```llvm
define ptr @bad_return() {
entry:
  call void @tl_mem_enter_scope()
  
  %result = call ptr @tl_tensor_randn(...)
  
  ; unregister を忘れた！
  call void @tl_mem_exit_scope()
  ; ← ここで %result が解放される
  
  ret ptr %result  ; ← ダングリングポインタを返す！
}
```

---

## デバッグ

### 問題の兆候

1. **`malloc: Incorrect checksum`**
   - 二重解放または解放済みメモリへの書き込み

2. **Segmentation Fault**
   - ダングリングポインタへのアクセス

3. **メモリリーク**
   - `exit_scope` の呼び出し忘れ、または登録のみで解放がない

### デバッグ手法

1. **MemoryManager のログを有効化**
   ```rust
   // memory_manager.rs の該当行のコメントを外す
   eprintln!("DEBUG: freeing tensor at {:?}", tensor_ptr);
   ```

2. **Valgrind でのメモリチェック**
   ```bash
   valgrind --leak-check=full ./target/debug/tl run test.tl
   ```

3. **AddressSanitizer の使用**
   ```bash
   RUSTFLAGS="-Z sanitizer=address" cargo build
   ./target/debug/tl run test.tl
   ```

---

## 実装のベストプラクティス

1. **すべてのスコープで enter/exit のペア**
   - 関数、ループ、条件分岐のすべてで

2. **return は特殊処理**
   - 必ず `unregister` してから `exit_scope`

3. **フィールド代入も特殊処理**
   - 代入後に `unregister`

4. **シャドーイングに注意**
   - 古い値の扱いを明確に（unregister または free）

5. **構造体作成時は登録**
   - `malloc` 直後に `register_struct`
