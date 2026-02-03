// src/compiler/codegen/builtin_types/generic/vec.tl

// Helper for type retention (Zero Sized)
struct PhantomData<T> {}

// Global Non-Generic Externs (Unified API)
extern fn tl_vec_unified_new() -> i64;
extern fn tl_vec_unified_push(handle: i64, val: i64, is_ref: bool);
extern fn tl_vec_unified_pop(handle: i64, is_ref: bool) -> i64;
extern fn tl_vec_unified_get(handle: i64, idx: i64, is_ref: bool) -> i64;
extern fn tl_vec_unified_len(handle: i64) -> i64;
extern fn tl_vec_unified_set(handle: i64, idx: i64, val: i64, is_ref: bool);
extern fn tl_vec_unified_free(handle: i64, is_ref: bool);

extern fn tl_ptr_inc_ref_i64(addr: i64);

// Intrinsics (Declared for Semantics, intercepted by Codegen)
extern fn __builtin_unsafe_to_i64<T>(val: T) -> i64;
extern fn __builtin_unsafe_from_i64<T>(val: i64, marker: PhantomData<T>) -> T;
extern fn __builtin_is_ref<T>(marker: PhantomData<T>) -> bool;

// Vec Struct
struct Vec<T> {
    handle: i64,
    marker: PhantomData<T>,
}

impl<T> Vec<T> {
    fn new() -> Vec<T> {
        let h = tl_vec_unified_new();
        return Vec {
            handle: h,
            marker: PhantomData {},
        };
    }

    fn push(self, item: T) {
        let is_ref = __builtin_is_ref(self.marker);
        let raw = __builtin_unsafe_to_i64(item);
        tl_vec_unified_push(self.handle, raw, is_ref);
    }
    
    fn len(self) -> i64 {
        tl_vec_unified_len(self.handle)
    }

    fn get(self, index: i64) -> T {
        let is_ref = __builtin_is_ref(self.marker);
        let raw = tl_vec_unified_get(self.handle, index, is_ref);
        __builtin_unsafe_from_i64(raw, self.marker)
    }
    
    fn pop(self) -> T {
         let is_ref = __builtin_is_ref(self.marker);
         // Pop returns the value. 
         // Runtime pop should NOT decrement the value itself if it returns it?
         // If runtime pop removes from array, it gives ownership to us.
         // We return T.
         // Calling pop(): T
         // If we don't handle RefCount here, ownership is transferred blindly.
         // Assuming runtime unified_pop just returns the value and forgets it from array.
         let raw = tl_vec_unified_pop(self.handle, is_ref);
         __builtin_unsafe_from_i64(raw, self.marker)
    }

    fn is_empty(self) -> bool {
        self.len() == 0
    }

    fn free(self) {
        let is_ref = __builtin_is_ref(self.marker);
        tl_vec_unified_free(self.handle, is_ref);
    }

    fn set(self, index: i64, item: T) {
        let is_ref = __builtin_is_ref(self.marker);
        let raw = __builtin_unsafe_to_i64(item);
        tl_vec_unified_set(self.handle, index, raw, is_ref);
    }

    // Unimplemented / TODO
    // These require more "unified" helpers
    /*
    fn clear(self);
    fn set(self, index: i64, item: T);
    fn last(self) -> T;
    fn remove(self, index: i64) -> T;
    fn insert(self, index: i64, item: T);
    fn contains(self, item: T) -> bool;
    */
    
    fn dummy_implement(self) {
        // Just keep compiling
    }
}
