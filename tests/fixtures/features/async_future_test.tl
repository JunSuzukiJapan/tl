// Phase 2 検証: Future トレイトの実装と型システム統合テスト

// シンプルな即座完了 Future の実装例
struct ImmediateFuture {
    value: i64,
}

impl Future for ImmediateFuture {
    type Output = i64;

    fn poll(self) -> Poll<i64> {
        Poll::Ready(self.value)
    }
}

// async fn 定義 (Phase 4 まで呼び出し不可だが、定義は可能)
async fn async_compute() -> i64 {
    42
}

fn main() {
    // Future trait を実装した struct のインスタンスを作成して poll できる
    let f = ImmediateFuture { value: 100 };
    let result = f.poll();
    println("Future impl OK");

    // Poll::Ready / Poll::Pending も直接使える
    let p1: Poll<i64> = Poll::Ready(42);
    let p2: Poll<i64> = Poll::Pending;
    println("Poll variants OK");
}
