# Підтримувані механізми навчання моделей

TensorLanguage (TL) підтримує навчання моделі нейронної мережі на додаток до потужних тензорних операцій. У цьому документі пояснюється робочий процес від визначення моделей до впровадження циклу навчання та збереження навчених моделей за допомогою TL.

## 1. Основні навчальні концепції

Навчання TL здійснюється за такими етапами:

1. **Визначення моделі**: використовуйте `struct`, щоб визначити шари та моделі, які містять параметри та стан.
2. **Forward Pass**: обчислюйте вихідні дані від вхідних тензорів, щоб отримати результати прогнозів або логіти.
3. **Обчислення втрат і зворотний перехід**: виклик `loss.backward()` у функції втрат (наприклад, `cross_entropy`) для обчислення градієнтів кожного параметра.
4. **Оптимізація**: викликайте вбудовані функції оптимізації (наприклад, `adam_step`) для кожного параметра, щоб оновити їх, і скидайте градієнти за допомогою `Tensor::clear_grads()`.

## 2. Відмінності між Tensor і GradTensor

TL має два основних типи статичних тензорів, які використовуються для чисельних обчислень і навчання:

- **`Tensor<T, R>`**: стандартний багатовимірний масив даних. Він не відстежує градієнти (історію обчислень), тому працює швидко та ефективно використовує пам’ять. Він в основному використовується для **обробки даних під час висновку** та зберігання **внутрішнього стану оптимізатора (наприклад, імпульсу та дисперсії)**.
- **`GradTensor<T, R>`**: тензор відстеження градієнта для навчання. Він записує процес обчислень (будує графік обчислень) і виконує автоматичне диференціювання для обчислення градієнтів під час виклику `backward()`. Ви завжди повинні використовувати `GradTensor` для **параметрів (ваги, зміщення тощо), які вивчаються/оновлюються** алгоритмом оптимізації.

## 3. Визначення та ініціалізація

Кожен шар моделі визначається як "структура". Наприклад, лінійний рівень, навчений за допомогою оптимізатора Адама, повинен утримувати стан імпульсу (`m`, `v`) на додаток до ваг і зміщень. Ми призначаємо `GradTensor` до параметрів навчання, а `Tensor` до стану оптимізатора.


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


*Примітка*: виклик `detach(true)` під час ініціалізації параметра явно позначає цей тензор як ціль для обчислення градієнта.

## 4. Реалізація кроку оптимізації

Додайте функцію `step` до кожного шару, щоб виконати алгоритм оптимізації (наприклад, Адам) і оновити його стан. Метод «кроку» TL зазвичай використовує незмінний дизайн, який повертає нову структуру після оновлення.


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


## 5. Навчальна петля та перехід назад

У основному навчальному циклі обчисліть втрати, викличте `backward()`, оновіть модель за допомогою `step`, а потім очистіть градієнти.


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


## 6. Збереження моделі (Safetensors)

Вивчені параметри моделі можна зберегти у форматі `.safetensors` за допомогою функції `Param::save`. Збережені дані можна повторно використовувати для висновків.


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