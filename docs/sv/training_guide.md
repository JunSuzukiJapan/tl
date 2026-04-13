# Understödda modellträningsmekanismer

TensorLanguage (TL) stöder utbildning av neurala nätverksmodeller förutom kraftfulla tensoroperationer. Detta dokument förklarar arbetsflödet från att definiera modeller till att implementera träningsslingan och spara tränade modeller med TL.

## 1. Grundläggande träningskoncept

Utbildning i TL utförs i följande steg:

1. **Modelldefinition**: Använd `struct` för att definiera lager och modeller som innehåller parametrar och tillstånd.
2. **Forward Pass**: Beräkna utdata från ingångstensorer för att få prediktionsresultat eller logiter.
3. **Förlustberäkning och bakåtpass**: Anropa `loss.backward()` på en förlustfunktion (t.ex. `cross_entropy`) för att beräkna gradienterna för varje parameter.
4. **Optimering**: Anropa inbyggda optimeringsfunktioner (t.ex. `adam_step`) på varje parameter för att uppdatera dem, och återställ gradienter med `Tensor::clear_grads()`.

## 2. Skillnader mellan Tensor och GradTensor

TL har två primära statiska tensortyper som används för numerisk beräkning och träning:

- **`Tensor<T, R>`**: Standard flerdimensionell matrisdata. Den spårar inte gradienter (beräkningshistorik), så den är snabb och minneseffektiv. Den används främst för **databehandling under slutledning** och lagring av **optimerarens interna tillstånd (t.ex. momentum och varians)**.
- **`GradTensor<T, R>`**: En gradientspårande tensor för träning. Den registrerar beräkningsprocessen (bygger en beräkningsgraf) och utför automatisk differentiering för att beräkna gradienter när `backward()` anropas. Du måste alltid använda `GradTensor` för att **parametrar (vikter, fördomar, etc.) ska läras in/uppdateras** av optimeringsalgoritmen.

## 3. Definition och initiering

Varje lager i modellen definieras som en `struktur`. Till exempel måste ett linjärt lager tränat med Adam-optimeraren hålla momentumtillståndet ('m', 'v') förutom vikter och förspänningar. Vi tilldelar "GradTensor" till träningsparametrarna och "Tensor" till optimerartillståndet.


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


*Notera*: Att anropa `detach(true)` under parameterinitiering markerar uttryckligen denna tensor som ett mål för gradientberäkning.

## 4. Implementera optimeringssteget

Lägg till en "steg"-funktion till varje lager för att exekvera optimeringsalgoritmen (t.ex. Adam) och uppdatera dess tillstånd. En TL `steg`-metod använder vanligtvis en oföränderlig design som returnerar en ny struktur efter uppdateringen.


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


## 5. Träningsslinga och bakåtpass

I huvudträningsslingan, beräkna förlusten, anrop `backward()`, uppdatera modellen via `step` och rensa sedan gradienterna.


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


## 6. Spara modellen (safetensors)

Inlärda modellparametrar kan sparas i `.safetensors`-format med hjälp av funktionen `Param::save`. Sparad data kan återanvändas för slutledning.


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