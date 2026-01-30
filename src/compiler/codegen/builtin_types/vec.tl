// src/compiler/codegen/builtin_types/vec.tl

// Vec構造体の定義
struct Vec<T> {
    ptr: i64, // 内部ポインタ (ランタイムの Opaque Pointer)
    cap: i64,
    len: i64,
}

// 組み込み型 Vec の実装
// メソッドはすべて `extern` として宣言され、コンパイラの Monomorphization 時に
// ランタイム関数 (例: `tl_vec_i64_push` や `tl_vec_ptr_push`) にリンクされます。
impl<T> Vec<T> {
    // コンストラクタ
    extern fn new() -> Vec<T>;

    // メソッド
    extern fn push(self, item: T);
    extern fn pop(self) -> T;
    extern fn len(self) -> i64;
    extern fn get(self, index: i64) -> T;
    extern fn is_empty(self) -> bool;
    extern fn clear(self);
    extern fn remove(self, index: i64) -> T;
    extern fn insert(self, index: i64, item: T);
    extern fn contains(self, item: T) -> bool;
}
