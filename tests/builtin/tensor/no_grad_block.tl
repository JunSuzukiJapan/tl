// no_grad ブロックのテスト
fn main() {
    let t = Tensor::randn([3, 3], true);

    // no_grad ブロック内でテンソル演算を行う
    no_grad {
        let a = t + t;
        let b = a * t;
        println(b);
    }

    // no_grad ブロック外では勾配計算が有効に戻っている
    let c = t * t;
    println(c);
    println("no_grad block test passed!");
}
