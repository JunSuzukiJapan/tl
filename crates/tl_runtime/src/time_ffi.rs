//! 時間・日付関連の FFI 関数

use crate::string_ffi::StringStruct;
use std::ffi::{c_char, CStr, CString};

// ========== Duration ==========

/// Duration::from_secs(s) -> ナノ秒
#[unsafe(no_mangle)]
pub extern "C" fn tl_rt_duration_from_secs(secs: i64) -> i64 {
    secs.saturating_mul(1_000_000_000)
}

/// Duration::from_millis(ms) -> ナノ秒
#[unsafe(no_mangle)]
pub extern "C" fn tl_rt_duration_from_millis(millis: i64) -> i64 {
    millis.saturating_mul(1_000_000)
}

/// Duration::from_nanos(ns) -> ナノ秒 (identity)
#[unsafe(no_mangle)]
pub extern "C" fn tl_rt_duration_from_nanos(nanos: i64) -> i64 {
    nanos
}

/// Duration::as_secs(nanos) -> 秒
#[unsafe(no_mangle)]
pub extern "C" fn tl_rt_duration_as_secs(nanos: i64) -> i64 {
    nanos / 1_000_000_000
}

/// Duration::as_millis(nanos) -> ミリ秒
#[unsafe(no_mangle)]
pub extern "C" fn tl_rt_duration_as_millis(nanos: i64) -> i64 {
    nanos / 1_000_000
}

// ========== Instant ==========

use std::sync::OnceLock;
use std::time::Instant;

static EPOCH: OnceLock<Instant> = OnceLock::new();

fn get_epoch() -> &'static Instant {
    EPOCH.get_or_init(Instant::now)
}

/// Instant::now() -> 起動時からのナノ秒
#[unsafe(no_mangle)]
pub extern "C" fn tl_rt_instant_now() -> i64 {
    get_epoch().elapsed().as_nanos() as i64
}

/// Instant::elapsed(start_nanos) -> Duration ナノ秒
#[unsafe(no_mangle)]
pub extern "C" fn tl_rt_instant_elapsed(start_nanos: i64) -> i64 {
    let now = get_epoch().elapsed().as_nanos() as i64;
    now.saturating_sub(start_nanos)
}

// ========== DateTime (タイムゾーン対応) ==========

/// ローカル時刻の Unix timestamp を返し、UTCオフセット(秒)を out_offset に書き込む
#[unsafe(no_mangle)]
pub extern "C" fn tl_rt_datetime_now(out_offset: *mut i64) -> i64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);

    // ローカルタイムゾーンのオフセットを計算
    let offset = get_local_utc_offset(now);
    if !out_offset.is_null() {
        unsafe { *out_offset = offset; }
    }
    now
}

/// UTC の Unix timestamp を返す
#[unsafe(no_mangle)]
pub extern "C" fn tl_rt_datetime_utc_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// from_timestamp(ts, offset) - パススルー (validation 用)
#[unsafe(no_mangle)]
pub extern "C" fn tl_rt_datetime_from_timestamp(ts: i64) -> i64 {
    ts
}

/// ローカルUTCオフセット取得
#[unsafe(no_mangle)]
pub extern "C" fn tl_rt_datetime_local_offset() -> i64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    get_local_utc_offset(now)
}

// libc localtime_r を使用してオフセットを取得
fn get_local_utc_offset(unix_ts: i64) -> i64 {
    unsafe {
        let time_t = unix_ts as libc::time_t;
        let mut tm: libc::tm = std::mem::zeroed();
        libc::localtime_r(&time_t, &mut tm);
        tm.tm_gmtoff as i64
    }
}

// timestamp + offset からカレンダー要素を計算するヘルパー
fn decompose(ts: i64, offset: i64) -> (i64, i64, i64, i64, i64, i64) {
    let local_ts = ts + offset;
    // Unix timestamp -> calendar components
    // gmtime_r を使用してローカル調整済み timestamp を分解
    unsafe {
        let time_t = local_ts as libc::time_t;
        let mut tm: libc::tm = std::mem::zeroed();
        libc::gmtime_r(&time_t, &mut tm);
        (
            (tm.tm_year + 1900) as i64,
            (tm.tm_mon + 1) as i64,
            tm.tm_mday as i64,
            tm.tm_hour as i64,
            tm.tm_min as i64,
            tm.tm_sec as i64,
        )
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_rt_datetime_year(ts: i64, offset: i64) -> i64 {
    decompose(ts, offset).0
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_rt_datetime_month(ts: i64, offset: i64) -> i64 {
    decompose(ts, offset).1
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_rt_datetime_day(ts: i64, offset: i64) -> i64 {
    decompose(ts, offset).2
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_rt_datetime_hour(ts: i64, offset: i64) -> i64 {
    decompose(ts, offset).3
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_rt_datetime_minute(ts: i64, offset: i64) -> i64 {
    decompose(ts, offset).4
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_rt_datetime_second(ts: i64, offset: i64) -> i64 {
    decompose(ts, offset).5
}

/// DateTime::format(ts, offset, fmt_str) -> String
/// フォーマット文字列: %Y=年, %m=月, %d=日, %H=時, %M=分, %S=秒, %z=オフセット
#[unsafe(no_mangle)]
pub extern "C" fn tl_rt_datetime_format(ts: i64, offset: i64, fmt: *const c_char) -> *mut StringStruct {
    let fmt_str = if fmt.is_null() {
        "%Y-%m-%d %H:%M:%S".to_string()
    } else {
        unsafe { CStr::from_ptr(fmt).to_string_lossy().into_owned() }
    };

    let (year, month, day, hour, minute, second) = decompose(ts, offset);
    let offset_hours = offset / 3600;
    let offset_mins = (offset.abs() % 3600) / 60;
    let offset_sign = if offset >= 0 { '+' } else { '-' };

    let result = fmt_str
        .replace("%Y", &format!("{:04}", year))
        .replace("%m", &format!("{:02}", month))
        .replace("%d", &format!("{:02}", day))
        .replace("%H", &format!("{:02}", hour))
        .replace("%M", &format!("{:02}", minute))
        .replace("%S", &format!("{:02}", second))
        .replace("%z", &format!("{}{:02}:{:02}", offset_sign, offset_hours.abs(), offset_mins));

    let c_str = CString::new(result).unwrap_or_else(|_| CString::new("").unwrap());
    let ptr = c_str.into_raw();
    unsafe {
        let len = libc::strlen(ptr) as i64;
        let layout = std::alloc::Layout::new::<StringStruct>();
        let struct_ptr = std::alloc::alloc(layout) as *mut StringStruct;
        (*struct_ptr).ptr = ptr;
        (*struct_ptr).len = len;
        struct_ptr
    }
}
