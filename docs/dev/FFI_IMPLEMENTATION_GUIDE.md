# TL コンパイラ FFI 実装ガイド

※ 本ドキュメントは、TLの基礎的なメモリ管理戦略（ヒープの扱い・所有権等）を解説した [MEMORY_MANAGEMENT_STRATEGY.md](./MEMORY_MANAGEMENT_STRATEGY.md) を補完するものです。事前または併せてご一読ください。

このドキュメントは、TensorLogic (TL) 言語から Rust で記述されたネイティブ関数パラダイム (FFI: Foreign Function Interface) を呼び出すための、C ABI を介した安全な実装レイヤーの設計指針を定めます。

## 1. コア原則: FFI境界ではプリミティブへ単純化する

TLとRust (C ABI) 間でデータをやり取りする場合、最も重要かつ厳守すべき掟は **「FFI 境界では `bool`、`i64`、`f32` などのプリミティブ型、または単純な生ポインタのみを扱う」** ことです。

TL は独自のヒープアロケーションと参照カウントの仕様（Enumが暗黙のヒープポインタ参照として扱われる等）を持っているため、**Rust の `enum` や大きめの `struct` を FFI で単純に値渡し・値戻ししてはいけません。** これに違反すると、ABIのレジスタ渡し規則の不整合により、`Null Pointer Returned` によるパニックや `Segmentation fault` を引き起こします。

## 2. 具体的な型マッピング

Rust (FFI) 側の `extern "C"` 関数を定義する場合、以下の型に対応させてください。

| TL側の型定義 | Rustの FFI シグネチャ (`extern "C"`) | 備考 |
| :--- | :--- | :--- |
| `i64` / `f32` / `bool` | `i64` / `f32` / `bool` | サイズとマッピングが完全に一致 |
| `String` | `*mut StringStruct` | `ptr` と `len` を持つポインタ。呼び出し時は必ず Null チェックを行うこと。 |
| `Vec<T>` | `*mut VecStruct` / `*mut TlVecU8` 等 | コレクションもヒープのポインタとして渡される。 |
| `Result<T, E>` | **絶対に使わない**（後述） | 代わりに `i64` 等で状態を返す。 |
| デバイスリソース | `i64` | `Channel`, `TcpListener` 等のような固有の重いリソースは `HashMap<i64, Arc<T>>` で Rust 側で管理し、`i64` (リソースID) だけを返す。 |

## 3. エラー処理と `Result` のデザインパターン

C ABI 境界越しに `Result` 等の複合的な列挙型（Enum）をそのまま伝えようとしないでください。代わりに、「C 側は状態コードやポインタを返し、TL 側のラッパーが `Result` を組み立てる」というレイヤー分離のパターンを使います。

### ❌ 悪い例（ABI 不整合でクラッシュする）

**Rust側**
```rust
#[repr(C)]
pub struct ResultTcpListener { tag: i32, payload: i64 } // 16バイトの値として返却しようとする

#[unsafe(no_mangle)]
pub extern "C" fn tl_net_listener_bind(addr: *mut StringStruct) -> ResultTcpListener {
    // 戻り値のメモリレイアウトがTLのEnum（ヒープ上のポインタ参照渡し）と一致せず Segfault する
}
```

### ⭕ 良い例（安全なプリミティブラッピング）

**Rust側 (`net_ffi.rs`)**
リソースが作成できた場合は 1 以上の `i64` IDを、失敗した場合は `0` または `-1` を返します。
```rust
#[unsafe(no_mangle)]
pub extern "C" fn tl_net_listener_bind(addr: *mut StringStruct) -> i64 {
    unsafe {
        if addr.is_null() || (*addr).ptr.is_null() { return 0; }
        // 成功時には内部レジストリ（HashMap）に実体を保存し、IDのみを返す
        match TcpListener::bind(...) {
            Ok(listener) => {
                let id = get_next_id();
                registry.insert(id, listener);
                id // 成功
            }
            Err(_) => 0 // 失敗
        }
    }
}
```

**TL側 (`net.tl`)**
TL本体の中で、安全に `Result::Ok` や `Result::Err` を構築します。
```tl
extern fn tl_net_listener_bind(addr: String) -> i64; // FFIの宣言はプリミティブ(i64)

impl TcpListener {
    fn bind(addr: String) -> Result<TcpListener, String> {
        let id = tl_net_listener_bind(addr);
        if id > 0 {
            // ここで型注釈をつけて Result の型推論（Bidirectional Inference漏れ）を防ぐ
            let res: Result<TcpListener, String> = Result::Ok(TcpListener { id: id });
            res
        } else {
            let res: Result<TcpListener, String> = Result::Err("Failed to bind");
            res
        }
    }
}
```

## 4. `Result` 生成時の型推論についての注意
上の例でも示されている通り、TL言語のコンパイラではコンストラクタ（`Result::Ok(val)`）が直接 `return` されるケースなどで、反対側のジェネリクス `E`（エラーの型）が確定せずに `Struct def E not found` となるケース（モノモーフィゼーションの漏れ）があります。

**対策:**
以下のように一度ローカル変数に落とし、明示的な型注釈（Type Annotation）を記述してから返すようにしてください。
```tl
let res: Result<MyStruct, String> = Result::Ok(MyStruct { ... });
return res;
```

## 5. 実装時のチェックリスト
- [ ] `extern "C"` な FFI 関数の引数と戻り値に `Result` やユーザー定義の `Struct` を値のまま記述していないか？
- [ ] 文字列（`StringStruct`）やポインタを受け取る場合、`is_null()` チェックを適切に行っているか？
- [ ] 巨大なオブジェクトを扱う際は、Rust内で `Arc` / `Mutex` と `HashMap` を用いた「IDレジストリ（Handle）パターン」で管理し、`i64` だけを受け渡ししているか？
