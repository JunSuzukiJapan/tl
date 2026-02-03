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
        Vec {
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
        Vec {
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
}
