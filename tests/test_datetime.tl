// Duration テスト
let d1 = Duration::from_secs(5);
let d2 = Duration::from_millis(2500);

println(d1.as_secs());        // 5
println(d1.as_millis());      // 5000
println(d2.as_secs());        // 2
println(d2.as_millis());      // 2500

let d3 = d1.add(d2);
println(d3.as_millis());      // 7500

let d4 = d1.sub(d2);
println(d4.as_millis());      // 2500

// Instant テスト
let start = Instant::now();
System::sleep(50.0);  // 50ms スリープ
let elapsed = start.elapsed();
let ms = elapsed.as_millis();
if ms >= 40 {
    println("Instant elapsed OK!");
} else {
    println("Instant elapsed FAIL!");
}

// DateTime テスト
let now = DateTime::now();
let year = now.year();
if year >= 2026 {
    println("DateTime year OK!");
} else {
    println("DateTime year FAIL!");
}

let month = now.month();
if month >= 1 {
    if month <= 12 {
        println("DateTime month OK!");
    }
}

let utc = DateTime::utc_now();
println(utc.offset_hours()); // 0

let local = DateTime::now();
let offset_h = local.offset_hours();
if offset_h != 0 {
    println("DateTime timezone OK!");
} else {
    println("DateTime timezone is UTC (may be correct)");
}

// format テスト
let formatted = now.format("%Y-%m-%d");
println(formatted);

let full = now.format("%Y-%m-%d %H:%M:%S %z");
println(full);

// to_utc テスト
let utc_from_local = local.to_utc();
println(utc_from_local.offset_hours()); // 0

// from_timestamp テスト (2026-01-01 00:00:00 UTC)
let ts = DateTime::from_timestamp(1767225600);
println(ts.year());   // 2026
println(ts.month());  // 1
println(ts.day());    // 1
