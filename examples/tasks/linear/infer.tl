// Linear Regression Task - Inference
// 学習済みパラメータを使用して推論テストを行います

struct LinearModel {
    W: Tensor<f32, 2>,
    b: Tensor<f32, 1>
}

impl LinearModel {
    fn new() -> LinearModel {
        LinearModel(
            Tensor::zeros([1, 1], false),
            Tensor::zeros([1], false)
        )
    }

    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> {
        x.matmul(self.W) + self.b
    }
}

fn inference(model: LinearModel, x_val: f32) {
    let expected = x_val * 3.0 + 2.0;

    let X_arr = [x_val];
    let X = X_arr.reshape([1, 1]);

    let pred_t = model.forward(X);
    let pred = pred_t.item();

    print("--------");
    print("Input x:"); print(x_val);
    print("Expected y:"); print(expected);
    print("Predicted y:"); print(pred);
    
    // 手書きのマージンテスト (誤差絶対値の計算)
    let mut diff = pred - expected;
    if diff < 0.0 {
        diff = diff * -1.0;
    }

    // 誤差が0.1未満ならHIT
    if diff < 0.1 {
        print("Result: HIT");
    } else {
        print("Result: MISS (Error >= 0.1)");
    }
}

fn main() {
    print("Loading manually trained linear_model...");
    let model = LinearModel::new();
    Param::load(model, "linear_model.safetensors");
    print("Model loaded.");

    // 未知のテストデータ (大きめの値などで検証)
    inference(model, 10.0);
    inference(model, -5.5);
    inference(model, 100.0);

    print("Inference Complete.");
}
