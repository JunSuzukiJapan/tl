extern fn tl_rt_datetime_now(out_offset: i64) -> i64;
extern fn tl_rt_datetime_utc_now() -> i64;
extern fn tl_rt_datetime_local_offset() -> i64;
extern fn tl_rt_datetime_year(ts: i64, offset: i64) -> i64;
extern fn tl_rt_datetime_month(ts: i64, offset: i64) -> i64;
extern fn tl_rt_datetime_day(ts: i64, offset: i64) -> i64;
extern fn tl_rt_datetime_hour(ts: i64, offset: i64) -> i64;
extern fn tl_rt_datetime_minute(ts: i64, offset: i64) -> i64;
extern fn tl_rt_datetime_second(ts: i64, offset: i64) -> i64;
extern fn tl_rt_datetime_format(ts: i64, offset: i64, fmt: String) -> String;

struct DateTime {
    timestamp: i64,
    offset: i64,
}

impl DateTime {
    fn now() -> DateTime {
        let offset = tl_rt_datetime_local_offset();
        let ts = tl_rt_datetime_utc_now();
        DateTime { timestamp: ts, offset: offset }
    }

    fn utc_now() -> DateTime {
        DateTime { timestamp: tl_rt_datetime_utc_now(), offset: 0 }
    }

    fn from_timestamp(ts: i64) -> DateTime {
        DateTime { timestamp: ts, offset: 0 }
    }

    fn from_timestamp_with_offset(ts: i64, offset: i64) -> DateTime {
        DateTime { timestamp: ts, offset: offset }
    }

    fn year(self) -> i64 {
        tl_rt_datetime_year(self.timestamp, self.offset)
    }

    fn month(self) -> i64 {
        tl_rt_datetime_month(self.timestamp, self.offset)
    }

    fn day(self) -> i64 {
        tl_rt_datetime_day(self.timestamp, self.offset)
    }

    fn hour(self) -> i64 {
        tl_rt_datetime_hour(self.timestamp, self.offset)
    }

    fn minute(self) -> i64 {
        tl_rt_datetime_minute(self.timestamp, self.offset)
    }

    fn second(self) -> i64 {
        tl_rt_datetime_second(self.timestamp, self.offset)
    }

    fn timestamp(self) -> i64 {
        self.timestamp
    }

    fn offset_hours(self) -> i64 {
        self.offset / 3600
    }

    fn offset_seconds(self) -> i64 {
        self.offset
    }

    fn to_utc(self) -> DateTime {
        DateTime { timestamp: self.timestamp, offset: 0 }
    }

    fn format(self, fmt: String) -> String {
        tl_rt_datetime_format(self.timestamp, self.offset, fmt)
    }
}
