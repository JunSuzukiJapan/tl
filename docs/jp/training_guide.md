# サポートされるモデル学習 (Training) の仕組み

TensorLanguage (TL) は強力なテンソル演算と共に、ニューラルネットワークのモデル学習（トレーニング）をサポートしています。このドキュメントでは、TLを用いたモデルの定義から学習ループの実装、さらに学習済みモデルの保存までの流れを解説します。

## 1. 学習の基本コンセプト

TLでの学習は、以下のステップで実行されます。

1. **モデルの定義**: 構造体 (`struct`) を用いてパラメータと状態を持つ層 (Layer) やモデルを定義します
2. **順伝播 (Forward)**: 入力テンソルに対して計算を行い、予測スコアやロジットを取得します
3. **損失計算と逆伝播 (Backward)**: 損失関数（例: `cross_entropy`）から `loss.backward()` を呼び出し、各パラメータの勾配を計算します
4. **最適化 (Optimizer)**: パラメータごとに組み込みの最適化関数（例: `adam_step`）を呼び出してパラメータを更新し、`Tensor::clear_grads()` で勾配をリセットします

## 2. Tensor と GradTensor の違いと使い分け

TLには数値計算や学習に用いる2つの主要なテンソル型が存在します。

- **`Tensor<T, R>`**: 通常の多次元配列データです。勾配（計算履歴）を追跡しないため、メモリ消費量が少なく高速に動作します。主に**推論時のデータ処理**や、オプティマイザの**内部状態（モーメンタムや分散など）**の保存に使用します。
- **`GradTensor<T, R>`**: 学習用の勾配追跡機能付きテンソルです。計算の過程を記録（計算グラフを構築）し、`backward()` を呼び出した際に自動微分を行って勾配を計算します。最適化アルゴリズムによる**学習（更新）対象となるパラメータ（重みやバイアスなど）**には、必ず `GradTensor` を使用する必要があります。

## 3. 定義と初期化

モデルの各層は `struct` で定義します。例えば、Adamオプティマイザで学習を行うLinear層は、重みとバイアスの他に、モーメンタム (`m`, `v`) のための状態を保持する必要があります。学習パラメータには `GradTensor` を、オプティマイザの状態には `Tensor` を割り当てます。

```rust
struct Linear { 
    W: GradTensor<f32, 2>, b: GradTensor<f32, 1>, // 学習対象パラメータ
    mW: Tensor<f32, 2>, vW: Tensor<f32, 2>,       // オプティマイザの状態（勾配不要）
    mb: Tensor<f32, 1>, vb: Tensor<f32, 1>
}

impl Linear { 
    fn new(i: i64, o: i64) -> Linear { 
        Linear(
            (GradTensor::randn([i, o], true) * 0.1).detach(true), // W: 勾配計算の対象
            (GradTensor::randn([o], true) * 0.0).detach(true),    // b: 勾配計算の対象
            Tensor::zeros([i, o], false),                         // mW: オプティマイザ状態
            Tensor::zeros([i, o], false),                         // vW
            Tensor::zeros([o], false),                            // mb
            Tensor::zeros([o], false)                             // vb
        )
    } 
    
    // 順伝播
    fn forward(self, x: GradTensor<f32, 3>) -> GradTensor<f32, 3> { 
        x.matmul(self.W) + self.b 
    } 
}
```

*注意点*: パラメータの初期化時に `detach(true)` を呼び出すことで、このテンソルが勾配計算の対象であることを明示します。

## 4. 最適化ステップの実装

各層に `step` 関数を追加し、最適化アルゴリズム（例: Adam）を実行して自身の状態を更新します。TLの `step` メソッドは通常、更新後の新たな構造体を返すイミュータブルな設計になります。

```rust
impl Linear {
    // オプティマイザの更新処理
    fn step(self, step_n: i64, lr: f32) -> Linear { 
        let mut s = self; 
        
        // 組み込みの `adam_step` を呼び出す。勾配(grad)と現在の状態(m, v)を渡す
        s.W.adam_step(s.W.grad(), s.mW, s.vW, step_n, lr, 0.9, 0.999, 1e-8, 0.0);
        s.b.adam_step(s.b.grad(), s.mb, s.vb, step_n, lr, 0.9, 0.999, 1e-8, 0.0);
        
        s // 更新済みの自身を返す
    }
}
```

## 5. 学習ループと逆伝播

メインの学習ループでは、損失を計算し、`backward()` を呼び出した後に、モデルの `step` 更新と勾配のクリアを行います。

```rust
// 学習ステップの例
fn train_step(model: GPT, global_step: i64, lr: f32, X: GradTensor<f32, 2>, Y: GradTensor<f32, 1>) -> GPT {
    let mut m = model;
    
    // 順伝播
    let logits = m.forward(X);
    
    // 損失の計算
    let loss = logits.cross_entropy(Y);
    
    // 逆伝播
    loss.backward();
    
    // ログ表示
    print("Loss:"); loss.print();
    
    // オプティマイザ関数による更新
    m = m.step(global_step, lr);
    
    // 計算グラフと勾配のリセット
    Tensor::clear_grads();
    
    return m;
}
```

## 6. モデルの保存 (Safetensors)

学習したモデルのパラメータは、`Param::save` 関数を用いることで `.safetensors` 形式で保存することができます。保存されたデータは、推論用などに再利用することが可能です。

```rust
fn main() {
    let mut model = GPT::new(vocab_size, d_model);
    
    // 学習ループ処理...
    // model = train_step(model, ...);
    
    // モデのパラメータを保存
    Param::save(model, "model_output.safetensors");
    print("学習が完了し、モデルを保存しました！");
}
```
