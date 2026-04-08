extern fn tl_rt_duration_from_secs(secs: i64) -> i64;
extern fn tl_rt_duration_from_millis(millis: i64) -> i64;
extern fn tl_rt_duration_from_nanos(nanos: i64) -> i64;
extern fn tl_rt_duration_as_secs(nanos: i64) -> i64;
extern fn tl_rt_duration_as_millis(nanos: i64) -> i64;

struct Duration {
    nanos: i64,
}

impl Duration {
    fn from_secs(secs: i64) -> Duration {
        Duration { nanos: tl_rt_duration_from_secs(secs) }
    }

    fn from_millis(millis: i64) -> Duration {
        Duration { nanos: tl_rt_duration_from_millis(millis) }
    }

    fn from_nanos(nanos: i64) -> Duration {
        Duration { nanos: tl_rt_duration_from_nanos(nanos) }
    }

    fn as_secs(self) -> i64 {
        tl_rt_duration_as_secs(self.nanos)
    }

    fn as_millis(self) -> i64 {
        tl_rt_duration_as_millis(self.nanos)
    }

    fn as_nanos(self) -> i64 {
        self.nanos
    }

    fn add(self, other: Duration) -> Duration {
        Duration { nanos: self.nanos + other.nanos }
    }

    fn sub(self, other: Duration) -> Duration {
        Duration { nanos: self.nanos - other.nanos }
    }
}
