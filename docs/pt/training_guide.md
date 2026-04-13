# Mecanismos de treinamento de modelo suportados

TensorLanguage (TL) oferece suporte ao treinamento de modelos de redes neurais, além de operações poderosas de tensores. Este documento explica o fluxo de trabalho desde a definição de modelos até a implementação do ciclo de treinamento e salvamento de modelos treinados usando TL.

## 1. Conceitos Básicos de Treinamento

O treinamento em TL é realizado nas seguintes etapas:

1. **Definição de modelo**: Use `struct` para definir camadas e modelos que contêm parâmetros e estado.
2. **Forward Pass**: calcula saídas de tensores de entrada para obter pontuações de previsão ou logits.
3. **Cálculo de perda e passagem para trás**: Chame `loss.backward()` em uma função de perda (por exemplo, `cross_entropy`) para calcular os gradientes de cada parâmetro.
4. **Otimização**: Chame funções de otimização integradas (por exemplo, `adam_step`) em cada parâmetro para atualizá-los e redefina gradientes com `Tensor::clear_grads()`.

## 2. Diferenças entre Tensor e GradTensor

TL tem dois tipos principais de tensores estáticos usados para cálculo numérico e treinamento:

- **`Tensor<T, R>`**: dados de array multidimensional padrão. Ele não rastreia gradientes (histórico de computação), por isso é rápido e eficiente em termos de memória. É usado principalmente para **processamento de dados durante inferência** e armazenamento de **estado interno do otimizador (por exemplo, momento e variação)**.
- **`GradTensor<T, R>`**: Um tensor de rastreamento de gradiente para treinamento. Ele registra o processo de computação (cria um gráfico de computação) e realiza diferenciação automática para calcular gradientes quando `backward()` é chamado. Você deve sempre usar `GradTensor` para **parâmetros (pesos, vieses, etc.) a serem aprendidos/atualizados** pelo algoritmo de otimização.

## 3. Definição e inicialização

Cada camada do modelo é definida como uma `struct`. Por exemplo, uma camada linear treinada com o otimizador Adam precisa manter o estado de momento (`m`, `v`), além de pesos e tendências. Atribuímos `GradTensor` aos parâmetros de treinamento e `Tensor` ao estado do otimizador.


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


*Nota*: Chamar `detach(true)` durante a inicialização do parâmetro marca explicitamente este tensor como um alvo para cálculo de gradiente.

## 4. Implementando a etapa de otimização

Adicione uma função `step` a cada camada para executar o algoritmo de otimização (por exemplo, Adam) e atualizar seu estado. Um método TL `step` normalmente usa um design imutável que retorna uma nova estrutura após a atualização.


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


## 5. Loop de treinamento e passe para trás

No loop de treinamento principal, calcule a perda, chame `backward()`, atualize o modelo via `step` e, em seguida, limpe os gradientes.


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


## 6. Salvando o modelo (safetensores)

Os parâmetros do modelo aprendido podem ser salvos no formato `.safetensors` usando a função `Param::save`. Os dados salvos podem ser reutilizados para inferência.


__CODE_BLOCO_3__