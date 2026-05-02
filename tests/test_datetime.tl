// STDOUT: Instant elapsed OK!
// STDOUT: DateTime year OK!
// STDOUT: DateTime month OK!

// Duration テスト
fn main() {
    let d1 = Duration::from_secs(5);
    let d2 = Duration::from_millis(2500);

    let d3 = d1.add(d2);
    let d4 = d1.sub(d2);

    // Instant テスト
    let start = Instant::now();
    System::sleep(0.05);  // 50ms スリープ (引数は秒)
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

    let local = DateTime::now();

    // format テスト
    let formatted = now.format("%Y-%m-%d");

    // to_utc テスト
    let utc_from_local = local.to_utc();

    // from_timestamp テスト (2026-01-01 00:00:00 UTC)
    let ts = DateTime::from_timestamp(1767225600);
}
