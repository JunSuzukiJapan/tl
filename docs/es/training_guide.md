# Mecanismos de capacitación modelo admitidos

TensorLanguage (TL) admite el entrenamiento de modelos de redes neuronales además de potentes operaciones tensoriales. Este documento explica el flujo de trabajo desde la definición de modelos hasta la implementación del ciclo de capacitación y el guardado de modelos entrenados mediante TL.

## 1. Conceptos básicos de formación

La formación en TL se realiza en los siguientes pasos:

1. **Definición de modelo**: utilice `struct` para definir capas y modelos que contienen parámetros y estado.
2. **Pase hacia adelante**: calcula las salidas de los tensores de entrada para obtener puntuaciones de predicción o logits.
3. **Cálculo de pérdida y paso hacia atrás**: Llame a `loss.backward()` en una función de pérdida (por ejemplo, `cross_entropy`) para calcular los gradientes de cada parámetro.
4. **Optimización**: llame a las funciones de optimización integradas (por ejemplo, `adam_step`) en cada parámetro para actualizarlos y restablezca los gradientes con `Tensor::clear_grads()`.

## 2. Diferencias entre Tensor y GradTensor

TL tiene dos tipos principales de tensores estáticos que se utilizan para el entrenamiento y el cálculo numérico:

- **`Tensor<T, R>`**: datos de matriz multidimensional estándar. No rastrea gradientes (historial de cálculo), por lo que es rápido y eficiente en cuanto a memoria. Se utiliza principalmente para **procesar datos durante la inferencia** y almacenar **el estado interno del optimizador (por ejemplo, impulso y varianza)**.
- **`GradTensor<T, R>`**: un tensor de seguimiento de gradiente para entrenamiento. Registra el proceso de cálculo (construye un gráfico de cálculo) y realiza una diferenciación automática para calcular gradientes cuando se llama a `hacia atrás()`. Siempre debe usar `GradTensor` para que **los parámetros (pesos, sesgos, etc.) sean aprendidos/actualizados** por el algoritmo de optimización.

## 3. Definición e inicialización

Cada capa del modelo se define como una "estructura". Por ejemplo, una capa lineal entrenada con el optimizador Adam necesita mantener el estado de impulso (`m`, `v`) además de pesos y sesgos. Asignamos `GradTensor` a los parámetros de entrenamiento y `Tensor` al estado del optimizador.


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


*Nota*: Llamar a `detach(true)` durante la inicialización de parámetros marca explícitamente este tensor como un objetivo para el cálculo del gradiente.

## 4. Implementación del paso de optimización

Agregue una función de "paso" a cada capa para ejecutar el algoritmo de optimización (por ejemplo, Adam) y actualizar su estado. Un método "paso" de TL normalmente utiliza un diseño inmutable que devuelve una nueva estructura después de la actualización.


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


## 5. Bucle de entrenamiento y pase hacia atrás

En el ciclo de entrenamiento principal, calcule la pérdida, llame a "hacia atrás ()", actualice el modelo mediante "paso" y luego borre los gradientes.


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


## 6. Guardar el modelo (tensores de seguridad)

Los parámetros del modelo aprendido se pueden guardar en formato `.safetensors` usando la función `Param::save`. Los datos guardados se pueden reutilizar para realizar inferencias.


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