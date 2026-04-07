extern fn tl_channel_new(capacity: i64) -> i64;
extern fn tl_channel_send(id: i64, val: i64) -> bool;
extern fn tl_channel_recv(id: i64) -> i64;
extern fn tl_channel_try_recv(id: i64, out_success_ptr: i64) -> i64;

struct Channel<T> {
    id: i64,
}

impl<T> Channel<T> {
    extern fn new(capacity: i64) -> Channel<T>;
    extern fn send(self, val: T);
    extern fn recv(self) -> T;
    extern fn try_recv(self) -> Option<T>;
}
