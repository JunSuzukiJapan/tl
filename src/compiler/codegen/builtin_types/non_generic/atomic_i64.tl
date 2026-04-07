extern fn tl_atomic_i64_new(val: i64) -> i64;
extern fn tl_atomic_i64_load(ptr: i64) -> i64;
extern fn tl_atomic_i64_store(ptr: i64, val: i64);
extern fn tl_atomic_i64_add(ptr: i64, val: i64) -> i64;
extern fn tl_atomic_i64_sub(ptr: i64, val: i64) -> i64;
extern fn tl_atomic_i64_swap(ptr: i64, val: i64) -> i64;
extern fn tl_atomic_i64_compare_exchange(ptr: i64, current: i64, new_val: i64) -> bool;

struct AtomicI64 {
    id: i64,
}

impl AtomicI64 {
    fn new(val: i64) -> AtomicI64 {
        AtomicI64 { id: tl_atomic_i64_new(val) }
    }

    fn load(self) -> i64 {
        tl_atomic_i64_load(self.id)
    }

    fn store(self, val: i64) {
        tl_atomic_i64_store(self.id, val);
    }

    fn add(self, val: i64) -> i64 {
        tl_atomic_i64_add(self.id, val)
    }

    fn sub(self, val: i64) -> i64 {
        tl_atomic_i64_sub(self.id, val)
    }

    fn swap(self, val: i64) -> i64 {
        tl_atomic_i64_swap(self.id, val)
    }

    fn compare_exchange(self, current: i64, new_val: i64) -> bool {
        tl_atomic_i64_compare_exchange(self.id, current, new_val)
    }
}
