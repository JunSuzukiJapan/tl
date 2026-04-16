// Phase 4 検証: .await 式の実行テスト
//
// Phase 4 MVP: async fn はコルーチン変換なしで動作する。
// .await は inner.poll() を呼び出し、Poll::Ready(v) から v を抽出する。

struct ReadyFuture {
    value: i64,
}

impl Future for ReadyFuture {
    type Output = i64;
    fn poll(self) -> Poll<i64> {
        Poll::Ready(self.value)
    }
}

async fn add_ten(f: ReadyFuture) -> i64 {
    let v = f.await;
    v + 10
}

fn main() {
    let f = ReadyFuture { value: 32 };
    let result = add_ten(f);
    println(result);
}
