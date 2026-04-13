# Ondersteunde modeltrainingsmechanismen

TensorLanguage (TL) ondersteunt training van neurale netwerkmodellen naast krachtige tensorbewerkingen. In dit document wordt de workflow uitgelegd vanaf het definiëren van modellen tot het implementeren van de trainingslus en het opslaan van getrainde modellen met behulp van TL.

## 1. Basistrainingconcepten

Training in TL wordt uitgevoerd in de volgende stappen:

1. **Modeldefinitie**: Gebruik `struct` om lagen en modellen te definiëren die parameters en status bevatten.
2. **Forward Pass**: bereken de uitvoer van invoertensoren om voorspellingsscores of logits te verkrijgen.
3. **Verliesberekening en achterwaartse doorgang**: Roep `loss.backward()` aan op een verliesfunctie (bijvoorbeeld `cross_entropy`) om de gradiënten van elke parameter te berekenen.
4. **Optimalisatie**: Roep ingebouwde optimalisatiefuncties aan (bijvoorbeeld `adam_step`) voor elke parameter om ze bij te werken, en reset gradiënten met `Tensor::clear_grads()`.

## 2. Verschillen tussen Tensor en GradTensor

TL heeft twee primaire statische tensortypen die worden gebruikt voor numerieke berekeningen en training:

- **`Tensor<T, R>`**: Standaard multidimensionale arraygegevens. Het houdt geen gradiënten bij (berekeningsgeschiedenis), dus het is snel en geheugenefficiënt. Het wordt voornamelijk gebruikt voor **gegevensverwerking tijdens gevolgtrekking** en het opslaan van **de interne status van de optimalisatie (bijvoorbeeld momentum en variantie)**.
- **`GradTensor<T, R>`**: een gradiënt-tracking-tensor voor training. Het registreert het rekenproces (bouwt een berekeningsgrafiek op) en voert automatische differentiatie uit om gradiënten te berekenen wanneer `backward()` wordt aangeroepen. U moet altijd `GradTensor` gebruiken voor **parameters (gewichten, biases, enz.) die door het optimalisatiealgoritme moeten worden geleerd/bijgewerkt**.

## 3. Definitie en initialisatie

Elke laag van het model wordt gedefinieerd als een `struct`. Een lineaire laag die is getraind met de Adam-optimizer moet bijvoorbeeld de momentumstatus (`m`, `v`) vasthouden naast gewichten en biases. We wijzen 'GradTensor' toe aan de trainingsparameters, en 'Tensor' aan de optimalisatiestatus.


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


*Opmerking*: Het aanroepen van `detach(true)` tijdens parameterinitialisatie markeert deze tensor expliciet als een doel voor gradiëntberekening.

## 4. Implementatie van de optimalisatiestap

Voeg een 'step'-functie toe aan elke laag om het optimalisatie-algoritme (bijvoorbeeld Adam) uit te voeren en de status ervan bij te werken. Een TL `step`-methode maakt doorgaans gebruik van een onveranderlijk ontwerp dat na de update een nieuwe structuur retourneert.


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


## 5. Trainingslus en achterwaartse pas

Bereken in de hoofdtrainingslus het verlies, roep `backward()` aan, update het model via `step` en wis vervolgens de gradiënten.


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


## 6. Het model opslaan (Safetensors)

Geleerde modelparameters kunnen worden opgeslagen in het `.safetensors`-formaat met behulp van de `Param::save`-functie. Opgeslagen gegevens kunnen worden hergebruikt voor gevolgtrekkingen.


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