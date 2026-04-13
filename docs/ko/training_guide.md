# 지원되는 모델 훈련 메커니즘

TensorLanguage(TL)는 강력한 텐서 작업 외에도 신경망 모델 교육을 지원합니다. 이 문서에서는 모델 정의부터 훈련 루프 구현, TL을 사용하여 훈련된 모델 저장까지의 워크플로를 설명합니다.

## 1. 기본 교육 개념

TL 교육은 다음 단계로 수행됩니다.

1. **모델 정의**: `struct`를 사용하여 매개변수와 상태를 보유하는 레이어와 모델을 정의합니다.
2. **정방향 전달**: 입력 텐서의 출력을 계산하여 예측 점수 또는 로짓을 얻습니다.
3. **손실 계산 및 역방향 전달**: 손실 함수(예: 'cross_entropy')에서 'loss.backward()'를 호출하여 각 매개변수의 기울기를 계산합니다.
4. **최적화**: 각 매개변수에 대해 내장된 최적화 함수(예: `adam_step`)를 호출하여 업데이트하고 `Tensor::clear_grads()`로 그라데이션을 재설정합니다.

## 2. Tensor와 GradTensor의 차이점

TL에는 수치 계산 및 훈련에 사용되는 두 가지 기본 정적 텐서 유형이 있습니다.

- **`Tensor<T, R>`**: 표준 다차원 배열 데이터. 그라데이션(계산 기록)을 추적하지 않으므로 빠르고 메모리 효율적입니다. 주로 **추론 중 데이터 처리** 및 **옵티마이저 내부 상태(예: 운동량 및 분산)** 저장에 사용됩니다.
- **`GradTensor<T, R>`**: 훈련용 경사 추적 텐서. 계산 프로세스를 기록하고(계산 그래프 작성) 'backward()'가 호출되면 자동 미분을 수행하여 기울기를 계산합니다. 최적화 알고리즘을 통해 **매개변수(가중치, 편향 등)를 학습/업데이트**하려면 항상 `GradTensor`를 사용해야 합니다.

## 3. 정의 및 초기화

모델의 각 레이어는 `struct`로 정의됩니다. 예를 들어 Adam 최적화 프로그램으로 훈련된 선형 계층은 가중치와 편향 외에도 운동량 상태(`m`, `v`)를 유지해야 합니다. 훈련 매개변수에 'GradTensor'를 할당하고 최적화 상태에 'Tensor'를 할당합니다.


```rust
struct Linear { 
    W: GradTensor<f32, 2>, b: GradTensor<f32, 1>, // Training parameters
    mW: Tensor<f32, 2>, vW: Tensor<f32, 2>,       // Optimizer state (no gradient needed)
    mb: Tensor<f32, 1>, vb: Tensor<f32, 1>
}

impl Linear { 
    fn new(i: i64, o: i64) -> Linear { 
        Linear(
            (GradTensor::randn([i, o], true) * 0.1).detach(true), // W: targeted by gradient computation
            (GradTensor::randn([o], true) * 0.0).detach(true),    // b: targeted by gradient computation
            Tensor::zeros([i, o], false),                         // mW: optimizer state
            Tensor::zeros([i, o], false),                         // vW
            Tensor::zeros([o], false),                            // mb
            Tensor::zeros([o], false)                             // vb
        )
    } 
    
    // Forward pass
    fn forward(self, x: GradTensor<f32, 3>) -> GradTensor<f32, 3> { 
        x.matmul(self.W) + self.b 
    } 
}
```


*참고*: 매개변수 초기화 중에 `detach(true)`를 호출하면 이 텐서를 명시적으로 경사 계산 대상으로 표시합니다.

## 4. 최적화 단계 구현

최적화 알고리즘(예: Adam)을 실행하고 상태를 업데이트하려면 각 레이어에 '단계' 함수를 추가하세요. TL `단계` 방법은 일반적으로 업데이트 후 새 구조를 반환하는 불변 설계를 사용합니다.


```rust
impl Linear {
    // Optimizer update processing
    fn step(self, step_n: i64, lr: f32) -> Linear { 
        let mut s = self; 
        
        // Call the built-in `adam_step`. Pass the gradient and current state (m, v)
        s.W.adam_step(s.W.grad(), s.mW, s.vW, step_n, lr, 0.9, 0.999, 1e-8, 0.0);
        s.b.adam_step(s.b.grad(), s.mb, s.vb, step_n, lr, 0.9, 0.999, 1e-8, 0.0);
        
        s // Return the updated self
    }
}
```


## 5. 훈련 루프 및 역방향 패스

기본 훈련 루프에서 손실을 계산하고 'backward()'를 호출하고 'step'을 통해 모델을 업데이트한 다음 기울기를 지웁니다.


```rust
// Example of a training step
fn train_step(model: GPT, global_step: i64, lr: f32, X: GradTensor<f32, 2>, Y: GradTensor<f32, 1>) -> GPT {
    let mut m = model;
    
    // Forward pass
    let logits = m.forward(X);
    
    // Compute loss
    let loss = logits.cross_entropy(Y);
    
    // Backward pass
    loss.backward();
    
    // Display log
    print("Loss:"); loss.print();
    
    // Update using the optimizer function
    m = m.step(global_step, lr);
    
    // Reset computation graph and gradients
    Tensor::clear_grads();
    
    return m;
}
```


## 6. 모델 저장(Safetensor)

학습된 모델 매개변수는 `Param::save` 함수를 사용하여 `.safetensors` 형식으로 저장할 수 있습니다. 저장된 데이터는 추론에 재사용할 수 있습니다.


```rust
fn main() {
    let mut model = GPT::new(vocab_size, d_model);
    
    // Training loop processing...
    // model = train_step(model, ...);
    
    // Save model parameters
    Param::save(model, "model_output.safetensors");
    print("Training is complete, model saved!");
}
```