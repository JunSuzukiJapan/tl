// Linear Regression Task (Pure Manual Implementation)
// 注意: このタスクはTLコンパイラの基本演算（四則演算や勾配の自動微分トラッキング）の正当性を証明・バグ出しするためのテストです。
// あえて組み込みの高度なオプティマイザ(Adam)やLayerNorm、cross_entropy等を使用せず、
// すべて「手書きの Vanilla SGD」および「手計算の MSE Loss」を用いて実装しています。

// モデル設計: 単純な 1変数線形関数 (y = W * x + b)
struct LinearModel {
    W: Tensor<f32, 2>,
    b: Tensor<f32, 1>
}

impl LinearModel {
    fn new() -> LinearModel {
        // 重みは乱数で初期化、バイアスは0からスタート
        LinearModel(
            (Tensor::randn([1, 1], true) * 0.1).detach(true),
            (Tensor::randn([1], true) * 0.0).detach(true)
        )
    }

    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> {
        // [Batch, 1] x [1, 1] + [1]
        x.matmul(self.W) + self.b
    }

    fn step(self, lr: f32) -> LinearModel {
        let mut s = self;
        // 完全手書きの Vanilla SGD (純粋な引算と掛け算による勾配降下法)
        s.W = (s.W - s.W.grad() * lr).detach(true);
        s.b = (s.b - s.b.grad() * lr).detach(true);
        s
    }
}

// データジェネレータ
// 関数: y = 3.0 * x + 2.0
fn get_target(x: f32) -> f32 {
    x * 3.0 + 2.0
}

fn main() {
    print("Starting Manual SGD Linear Regression Training...");
    print("Task: Learn the function y = 3.0 * x + 2.0");

    let mut model = LinearModel::new();
    let lr = 0.01;
    let epochs = 500; // 単純なSGDのため多めに回す

    for epoch in range(0, epochs) {
        // バッチサイズ 4 で実行
        let r_t = Tensor::randn([4], true);
        let x1 = r_t[0] * 5.0;
        let x2 = r_t[1] * 5.0;
        let x3 = r_t[2] * 5.0;
        let x4 = r_t[3] * 5.0;

        let y1 = get_target(x1);
        let y2 = get_target(x2);
        let y3 = get_target(x3);
        let y4 = get_target(x4);

        let X_arr = [x1, x2, x3, x4];
        let X = X_arr.reshape([4, 1]);

        let Y_arr = [y1, y2, y3, y4];
        let Y = Y_arr.reshape([4, 1]);

        let pred = model.forward(X);

        // 手書きの MSE (Mean Squared Error) 
        // 組み込み関数を使わず、コンパイラの乗算と減算・sum の挙動をテストする
        let diff = pred - Y;
        let sq_err = diff * diff;
        let mse = sq_err.sum() * 0.25; // 4で割る(平均)

        mse.backward();
        model = model.step(lr);
        Tensor::clear_grads();

        if epoch - ((epoch / 50) * 50) == 0 {
            print("Epoch:"); print(epoch);
            print("MSE Loss:"); mse.print();
        }
    }

    print("Training Complete.");
    
    // 最終的に学習されたパラメータの値を確認
    let w_val = model.W.item();
    let b_val = model.b.item();
    print("Learned W (Expected: ~3.0):"); print(w_val);
    print("Learned b (Expected: ~2.0):"); print(b_val);

    print("Saving model weights to linear_model.safetensors...");
    Param::save(model, "linear_model.safetensors");
    print("Saved.");
}
