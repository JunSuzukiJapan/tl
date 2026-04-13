# Поддерживаемые механизмы обучения моделей

TensorLanguage (TL) поддерживает обучение модели нейронной сети в дополнение к мощным тензорным операциям. В этом документе объясняется рабочий процесс от определения моделей до реализации цикла обучения и сохранения обученных моделей с помощью TL.

## 1. Основные концепции обучения

Обучение TL осуществляется в следующие этапы:

1. **Определение модели**. Используйте структуру для определения слоев и моделей, которые содержат параметры и состояние.
2. **Проход вперед**: вычислите выходные данные на основе входных тензоров, чтобы получить оценки прогнозирования или логиты.
3. **Вычисление потерь и обратный проход**: вызовите `loss.backward()` для функции потерь (например, `cross_entropy`), чтобы вычислить градиенты каждого параметра.
4. **Оптимизация**: вызовите встроенные функции оптимизации (например, `adam_step`) для каждого параметра, чтобы обновить их, и сбросьте градиенты с помощью `Tensor::clear_grads()`.

## 2. Различия между Tensor и GradTensor

TL имеет два основных типа статических тензоров, используемых для численных вычислений и обучения:

- **`Tensor<T, R>`**: стандартные данные многомерного массива. Он не отслеживает градиенты (историю вычислений), поэтому работает быстро и эффективно использует память. В основном он используется для **обработки данных во время вывода** и хранения **внутреннего состояния оптимизатора (например, импульса и дисперсии)**.
- **`GradTensor<T, R>`**: тензор отслеживания градиента для обучения. Он записывает процесс вычислений (строит граф вычислений) и выполняет автоматическое дифференцирование для вычисления градиентов при вызове функции backward(). Вы всегда должны использовать GradTensor для **параметров (весов, смещений и т. д.), которые будут изучаться/обновляться** алгоритмом оптимизации.

## 3. Определение и инициализация

Каждый уровень модели определяется как «структура». Например, линейный слой, обученный с помощью оптимизатора Адама, должен сохранять состояние импульса (m, v) в дополнение к весам и смещениям. Мы назначаем GradTensor параметрам обучения, а Tensor — состоянию оптимизатора.


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


*Примечание*: вызов `detach(true)` во время инициализации параметра явно помечает этот тензор как цель для вычисления градиента.

## 4. Реализация этапа оптимизации

Добавьте функцию «шаг» к каждому слою, чтобы выполнить алгоритм оптимизации (например, Адама) и обновить его состояние. Метод «шага» TL обычно использует неизменяемый дизайн, который возвращает новую структуру после обновления.


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


## 5. Тренировочный цикл и обратный проход

В основном цикле обучения вычислите потери, вызовите функцию backward(), обновите модель с помощью шага, а затем очистите градиенты.


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


## 6. Сохранение модели (защитные датчики)

Изученные параметры модели можно сохранить в формате .safetensors с помощью функции Param::save. Сохраненные данные можно повторно использовать для вывода.


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