// Phase 1 検証: async fn と .await の構文パーステスト
// NOTE: .await は Phase 4 まで実行できないため、async fn の定義のみ確認する

async fn add_async(x: i64, y: i64) -> i64 {
    x + y
}

async fn greet_async(name: String) -> String {
    "Hello, " + name
}

fn main() {
    // async fn は定義できる（まだ await で呼び出せないが構文エラーにならない）
    println("async fn syntax OK");
}
