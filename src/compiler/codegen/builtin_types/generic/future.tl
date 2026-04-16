// TL 標準ライブラリ: 非同期プリミティブ
//
// Future トレイトと Poll 型は async/await の基盤となる。
// Rust の std::future::Future に相当するが、以下を簡略化している:
//   - Pin は不要 (TL の構造体はポインタ渡しが基本)
//   - Context/Waker は Phase 3 以降で追加

/// Future のポーリング結果。
/// Ready(T) : 値が確定した
/// Pending  : まだ準備できていない（再度ポーリングが必要）
enum Poll<T> {
    Ready(T),
    Pending,
}

impl<T> Poll<T> {
    fn is_ready(self) -> bool {
        match self {
            Poll::Ready(x) => true,
            Poll::Pending => false,
        }
    }

    fn is_pending(self) -> bool {
        match self {
            Poll::Ready(x) => false,
            Poll::Pending => true,
        }
    }

    fn unwrap(self) -> T {
        match self {
            Poll::Ready(x) => x,
            Poll::Pending => {
                panic("called unwrap() on Poll::Pending");
            },
        }
    }
}

/// 非同期計算を表すトレイト。
/// 実装者は poll() を提供し、計算が完了したら Poll::Ready(output) を返す。
/// Phase 4 ではコンパイラが async fn に対して自動的に impl Future を生成する。
trait Future {
    type Output;
    fn poll(self) -> Poll<Self::Output>;
}
