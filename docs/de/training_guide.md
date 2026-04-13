# Unterstützte Modelltrainingsmechanismen

TensorLanguage (TL) unterstützt neben leistungsstarken Tensoroperationen auch das Training neuronaler Netzwerkmodelle. In diesem Dokument wird der Arbeitsablauf von der Modelldefinition über die Implementierung der Trainingsschleife bis hin zum Speichern trainierter Modelle mithilfe von TL erläutert.

## 1. Grundlegende Schulungskonzepte

Die Ausbildung in TL erfolgt in folgenden Schritten:

1. **Modelldefinition**: Verwenden Sie „struct“, um Ebenen und Modelle zu definieren, die Parameter und Zustände enthalten.
2. **Forward Pass**: Berechnen Sie die Ausgaben von Eingabetensoren, um Vorhersagewerte oder Logits zu erhalten.
3. **Verlustberechnung und Rückwärtsdurchlauf**: Rufen Sie „loss.backward()“ für eine Verlustfunktion (z. B. „cross_entropy“) auf, um die Gradienten jedes Parameters zu berechnen.
4. **Optimierung**: Rufen Sie integrierte Optimierungsfunktionen (z. B. „adam_step“) für jeden Parameter auf, um sie zu aktualisieren, und setzen Sie Farbverläufe mit „Tensor::clear_grads()“ zurück.

## 2. Unterschiede zwischen Tensor und GradTensor

TL verfügt über zwei primäre statische Tensortypen, die für numerische Berechnungen und Schulungen verwendet werden:

- **`Tensor<T, R>`**: Standardmäßige mehrdimensionale Array-Daten. Es verfolgt keine Verläufe (Berechnungsverlauf) und ist daher schnell und speichereffizient. Es wird hauptsächlich zur **Datenverarbeitung während der Inferenz** und zum Speichern des **internen Zustands des Optimierers (z. B. Impuls und Varianz)** verwendet.
- **`GradTensor<T, R>`**: Ein Gradienten-Tracking-Tensor für das Training. Es zeichnet den Berechnungsprozess auf (erstellt ein Berechnungsdiagramm) und führt eine automatische Differenzierung durch, um Gradienten zu berechnen, wenn „backward()“ aufgerufen wird. Sie müssen immer „GradTensor“ verwenden, damit **Parameter (Gewichte, Bias usw.) vom Optimierungsalgorithmus gelernt/aktualisiert werden**.

## 3. Definition und Initialisierung

Jede Ebene des Modells wird als „Struktur“ definiert. Beispielsweise muss eine mit dem Adam-Optimierer trainierte lineare Ebene zusätzlich zu Gewichten und Vorspannungen auch den Impulszustand („m“, „v“) enthalten. Wir weisen den Trainingsparametern „GradTensor“ und dem Optimiererzustand „Tensor“ zu.


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


*Hinweis*: Der Aufruf von „detach(true)“ während der Parameterinitialisierung markiert diesen Tensor explizit als Ziel für die Gradientenberechnung.

## 4. Implementierung des Optimierungsschritts

Fügen Sie jeder Ebene eine „Schritt“-Funktion hinzu, um den Optimierungsalgorithmus auszuführen (z. B. Adam) und seinen Zustand zu aktualisieren. Eine TL-„Schritt“-Methode verwendet normalerweise ein unveränderliches Design, das nach der Aktualisierung eine neue Struktur zurückgibt.


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


## 5. Trainingsschleife und Rückwärtspass

Berechnen Sie in der Haupttrainingsschleife den Verlust, rufen Sie „backward()“ auf, aktualisieren Sie das Modell über „step“ und löschen Sie dann die Farbverläufe.


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


## 6. Speichern des Modells (Safetensoren)

Gelernte Modellparameter können mit der Funktion „Param::save“ im Format „.safetensors“ gespeichert werden. Gespeicherte Daten können für Rückschlüsse wiederverwendet werden.


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