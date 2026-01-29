// src/compiler/codegen/inner_type/vec.tl

// Vec構造体の定義
struct Vec<T> {
    ptr: ptr<T>, // 内部ポインタ (ランタイムの Opaque Pointer)
    cap: I64,
    len: I64,
}

// 組み込み型 Vec の実装
// メソッドはすべて `extern` として宣言され、コンパイラの Monomorphization 時に
// ランタイム関数 (例: `tl_vec_i64_push` や `tl_vec_ptr_push`) にリンクされます。
impl Vec<T> {
    // コンストラクタ
    extern fn new() -> Vec<T>;

    // メソッド
    extern fn push(self, item: T);
    extern fn pop(self) -> T;
    extern fn len(self) -> I64;
    extern fn get(self, index: I64) -> T;
    extern fn is_empty(self) -> Bool;
    extern fn clear(self);
    extern fn remove(self, index: I64) -> T;
}
