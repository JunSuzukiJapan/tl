extern fn tl_rt_instant_now() -> i64;
extern fn tl_rt_instant_elapsed(start_nanos: i64) -> i64;

struct Instant {
    nanos: i64,
}

impl Instant {
    fn now() -> Instant {
        Instant { nanos: tl_rt_instant_now() }
    }

    fn elapsed(self) -> Duration {
        Duration { nanos: tl_rt_instant_elapsed(self.nanos) }
    }
}
