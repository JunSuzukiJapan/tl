# Supported Model Training Mechanisms

TensorLanguage (TL) supports neural network model training in addition to powerful tensor operations. This document explains the workflow from defining models to implementing the training loop and saving trained models using TL.

## 1. Basic Training Concepts

Training in TL is performed in the following steps:

1. **Model Definition**: Use `struct` to define layers and models that hold parameters and state.
2. **Forward Pass**: Compute outputs from input tensors to get prediction scores or logits.
3. **Loss Computation and Backward Pass**: Call `loss.backward()` on a loss function (e.g., `cross_entropy`) to compute the gradients of each parameter.
4. **Optimization**: Call built-in optimization functions (e.g., `adam_step`) on each parameter to update them, and reset gradients with `Tensor::clear_grads()`.

## 2. Differences Between Tensor and GradTensor

TL has two primary static tensor types used for numerical computation and training:

- **`Tensor<T, R>`**: Standard multi-dimensional array data. It does not track gradients (computation history), so it is fast and memory-efficient. It is mainly used for **data processing during inference** and storing **optimizer internal state (e.g. momentum and variance)**.
- **`GradTensor<T, R>`**: A gradient-tracking tensor for training. It records the computation process (builds a computation graph) and performs automatic differentiation to compute gradients when `backward()` is called. You must always use `GradTensor` for **parameters (weights, biases, etc.) to be learned/updated** by the optimization algorithm.

## 3. Definition and Initialization

Each layer of the model is defined as a `struct`. For example, a Linear layer trained with the Adam optimizer needs to hold momentum state (`m`, `v`) in addition to weights and biases. We assign `GradTensor` to the training parameters, and `Tensor` to the optimizer state.

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

*Note*: Calling `detach(true)` during parameter initialization explicitly marks this tensor as a target for gradient computation.

## 4. Implementing the Optimization Step

Add a `step` function to each layer to execute the optimization algorithm (e.g., Adam) and update its state. A TL `step` method typically uses an immutable design that returns a new structure after the update.

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

## 5. Training Loop and Backward Pass

In the main training loop, compute the loss, call `backward()`, update the model via `step`, and then clear the gradients.

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

## 6. Saving the Model (Safetensors)

Learned model parameters can be saved in `.safetensors` format using the `Param::save` function. Saved data can be reused for inference.

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
