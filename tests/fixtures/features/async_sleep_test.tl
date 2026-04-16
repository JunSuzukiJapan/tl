// Phase 5 検証: sleep() 組み込み非同期プリミティブのテスト

async fn wait_and_compute(ms: i64, x: i64) -> i64 {
    let elapsed = sleep(ms).await;
    x * 2
}

fn main() {
    // sleep(1) は 1ms スリープ後に i64(0) を返す SleepFuture
    let result = wait_and_compute(1, 21);
    println(result);
}
