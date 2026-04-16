// async / await / Future トレイト 網羅的テスト
//
// 検証項目:
//   1.  Poll::Ready.unwrap()         → 42
//   2.  Poll::Pending.is_pending()   → true
//   3.  Poll::Ready.is_ready()       → true
//   4.  Poll::Ready.is_pending()     → false
//   5.  直接 .poll() + unwrap()      → 100
//   6.  async fn + .await (i64)      → 32
//   7.  3つの連続 .await             → 42  (10+20+12)
//   8.  String Output .await         → Hello, world
//   9.  sleep().await + 計算         → 99
//   10. ネストした async fn          → 43  (21*2+1)
//   11. 別の Future 実装 (×3倍)     → 21  (7*3)
//   12. Poll::Pending.is_ready()     → false

// STDOUT: 42
// STDOUT: true
// STDOUT: false
// STDOUT: 100
// STDOUT: 32
// STDOUT: Hello, world
// STDOUT: 99
// STDOUT: 43
// STDOUT: 21

// ── Future 実装 ───────────────────────────────────────────────────────

struct ImmFut { val: i64 }

impl Future for ImmFut {
    type Output = i64;
    fn poll(self) -> Poll<i64> {
        Poll::Ready(self.val)
    }
}

struct ReadyFut { value: i64 }

impl Future for ReadyFut {
    type Output = i64;
    fn poll(self) -> Poll<i64> {
        Poll::Ready(self.value)
    }
}

struct SeqF { v: i64 }

impl Future for SeqF {
    type Output = i64;
    fn poll(self) -> Poll<i64> { Poll::Ready(self.v) }
}

struct Greeting { name: String }

impl Future for Greeting {
    type Output = String;
    fn poll(self) -> Poll<String> {
        Poll::Ready("Hello, " + self.name)
    }
}

struct TripleFut { base: i64 }

impl Future for TripleFut {
    type Output = i64;
    fn poll(self) -> Poll<i64> {
        Poll::Ready(self.base * 3)
    }
}

struct NestedF { v: i64 }

impl Future for NestedF {
    type Output = i64;
    fn poll(self) -> Poll<i64> { Poll::Ready(self.v) }
}

// ── async fn 定義 ─────────────────────────────────────────────────────

async fn await_i64(f: ReadyFut) -> i64 {
    f.await
}

async fn seq_await(a: SeqF, b: SeqF, c: SeqF) -> i64 {
    let x = a.await;
    let y = b.await;
    let z = c.await;
    x + y + z
}

async fn greet(g: Greeting) -> String {
    g.await
}

async fn compute_after_sleep(ms: i64, x: i64) -> i64 {
    let elapsed = sleep(ms).await;
    x
}

async fn nested_inner(f: NestedF) -> i64 {
    f.await * 2
}

async fn nested_outer(f: NestedF) -> i64 {
    let doubled = nested_inner(f);
    doubled + 1
}

async fn triple_await(t: TripleFut) -> i64 {
    t.await
}

// ── main ──────────────────────────────────────────────────────────────

fn main() {
    // test1: Poll::Ready.unwrap() で値を取り出せる
    let p_ready: Poll<i64> = Poll::Ready(42);
    println(p_ready.unwrap());

    // test2: Poll::Pending.is_pending() → true
    let p_pend: Poll<i64> = Poll::Pending;
    println(p_pend.is_pending());

    // test3: Poll::Ready.is_ready() → true
    println(p_ready.is_ready());

    // test4: Poll::Ready.is_pending() → false
    println(p_ready.is_pending());

    // test5: 直接 .poll() を呼んで Poll::Ready から値を取り出す
    let f5 = ImmFut { val: 100 };
    let poll5 = f5.poll();
    println(poll5.unwrap());

    // test6: async fn + .await (i64 Output)
    let rf = ReadyFut { value: 32 };
    let v6 = await_i64(rf);
    println(v6);

    // test7: 3つの連続 .await → 10+20+12 = 42
    let v7 = seq_await(SeqF{v:10}, SeqF{v:20}, SeqF{v:12});
    println(v7);

    // test8: String Output .await
    let g = Greeting { name: "world" };
    let v8 = greet(g);
    println(v8);

    // test9: sleep().await の後に計算
    let v9 = compute_after_sleep(1, 99);
    println(v9);

    // test10: ネストした async fn → 21*2 + 1 = 43
    let v10 = nested_outer(NestedF{v:21});
    println(v10);

    // test11: 別の Future 実装 (×3倍) → 7*3 = 21
    let t = TripleFut { base: 7 };
    let v11 = triple_await(t);
    println(v11);

    // test12: Poll::Pending.is_ready() → false
    println(p_pend.is_ready());
}
