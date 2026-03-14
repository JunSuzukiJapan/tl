extern fn malloc(size: i64) -> ptr<u8>;
extern fn realloc(ptr: ptr<u8>, size: i64) -> ptr<u8>;
extern fn free(ptr: ptr<u8>);

struct Vec<T> {
    ptr: ptr<T>,
    cap: i64,
    len: i64,
}

impl<T> Vec<T> {
    fn new() -> Vec<T> {
        Vec<T> {
            ptr: 0 as ptr<T>,
            cap: 0,
            len: 0,
        }
    }

    fn with_capacity(cap: i64) -> Vec<T> {
        if cap == 0 {
             return Vec::new();
        }
        let size = cap * T::sizeof();
        let p = malloc(size);
        Vec<T> {
            ptr: p as ptr<T>,
            cap: cap,
            len: 0,
        }
    }

    fn len(self) -> i64 {
        self.len
    }

    fn capacity(self) -> i64 {
        self.cap
    }

    fn is_empty(self) -> bool {
        self.len == 0
    }

    fn push(mut self, item: T) {
        if self.len == self.cap {
            let new_cap = if self.cap == 0 { 4 } else { self.cap * 2 };
            let item_size = T::sizeof();
            let new_size = new_cap * item_size;
            
            let old_ptr = self.ptr as ptr<u8>;
            let new_ptr_u8 = realloc(old_ptr, new_size);
            
            self.ptr = new_ptr_u8 as ptr<T>;
            self.cap = new_cap;
        }

        let mut p = self.ptr;
        p[self.len] = item;
        self.len = self.len + 1;
    }

    fn pop(mut self) -> Option<T> {
        if self.len == 0 {
            return Option::None;
        }
        self.len = self.len - 1;
        let item = self.ptr[self.len];
        Option::Some(item)
    }

    fn get(self, index: i64) -> Option<T> {
        if index < 0 || index >= self.len {
            return Option::None;
        }
        let item = self.ptr[index];
        Option::Some(item)
    }

    fn set(self, index: i64, item: T) {
        if index < 0 || index >= self.len {
            return;
        }
        let p = self.ptr;
        p[index] = item;
    }

    fn insert(mut self, index: i64, item: T) {
        if index < 0 || index > self.len {
            return;
        }
        // Ensure capacity
        if self.len == self.cap {
            let new_cap = if self.cap == 0 { 4 } else { self.cap * 2 };
            let item_size = T::sizeof();
            let new_size = new_cap * item_size;
            let old_ptr = self.ptr as ptr<u8>;
            let new_ptr_u8 = realloc(old_ptr, new_size);
            self.ptr = new_ptr_u8 as ptr<T>;
            self.cap = new_cap;
        }
        // Shift elements right
        let mut i = self.len;
        while i > index {
            let p = self.ptr;
            p[i] = p[i - 1];
            i = i - 1;
        }
        let p = self.ptr;
        p[index] = item;
        self.len = self.len + 1;
    }

    fn remove(mut self, index: i64) -> T {
        let item = self.ptr[index];
        // Shift elements left
        let mut i = index;
        while i < self.len - 1 {
            let p = self.ptr;
            p[i] = p[i + 1];
            i = i + 1;
        }
        self.len = self.len - 1;
        item
    }


    fn clear(mut self) {
        self.len = 0;
    }

    fn contains(self, item: T) -> bool where T: PartialEq {
        let mut i: i64 = 0;
        while i < self.len {
            if self.ptr[i] == item {
                return true;
            }
            i = i + 1;
        }
        false
    }

    fn sort(mut self) where T: PartialOrd {
        let mut i: i64 = 1;
        while i < self.len {
            let key = self.ptr[i];
            let mut j = i - 1;
            while j >= 0 {
                if self.ptr[j] > key {
                    let p = self.ptr;
                    p[j + 1] = p[j];
                    j = j - 1;
                } else {
                    break;
                }
            }
            let p = self.ptr;
            p[j + 1] = key;
            i = i + 1;
        }
    }

    fn index_of(self, item: T) -> i64 where T: PartialEq {
        let mut i: i64 = 0;
        while i < self.len {
            if self.ptr[i] == item {
                return i;
            }
            i = i + 1;
        }
        -1
    }

    fn reverse(mut self) {
        let mut lo: i64 = 0;
        let mut hi: i64 = self.len - 1;
        while lo < hi {
            let p = self.ptr;
            let tmp = p[lo];
            p[lo] = p[hi];
            p[hi] = tmp;
            lo = lo + 1;
            hi = hi - 1;
        }
    }

    // TODO: extend, slice, clone は get().unwrap() → push() チェーンが
    // monomorphize 時に Option<T> の unwrap 戻り値型が ptr として生成され、
    // push(item: T) との型不一致を引き起こす問題がある。
    // コンパイラの型伝播修正後に有効化する。
    //
    // fn extend(mut self, other: Vec<T>) { ... }
    // fn slice(self, start: i64, end: i64) -> Vec<T> { ... }
    // fn clone(self) -> Vec<T> { ... }

    // NOTE: map/filter/any/all はメソッドレベルの型パラメータ U を必要とするため、
    // .tl ファイルで定義すると monomorphize 時に U が解決できずエラーになる。
    // 代わりに codegen の compile_vec_closure_method (expr.rs) で
    // クロージャ構文から直接インラインIR を生成している。
    // 使用例: v.map(|x: i64| x * 2), v.filter(|x: i64| x % 2 == 0)

    // NOTE: join(sep: String) -> String は T = String の場合のみ有効だが、
    // ジェネリック型で特定の T に対する特殊化ができないため、
    // compile_vec_closure_method と同様に expr.rs でインライン実装する必要がある。

    // NOTE: reduce(fn) -> T も同様に expr.rs で実装する必要がある。
}
