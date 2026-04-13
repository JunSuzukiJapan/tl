# Meccanismi di formazione del modello supportati

TensorLanguage (TL) supporta l'addestramento del modello di rete neurale oltre a potenti operazioni tensoriali. Questo documento spiega il flusso di lavoro dalla definizione dei modelli all'implementazione del ciclo di addestramento e al salvataggio dei modelli addestrati utilizzando TL.

##1. Concetti base della formazione

La formazione in TL viene eseguita nei seguenti passaggi:

1. **Definizione del modello**: utilizza `struct` per definire livelli e modelli che contengono parametri e stato.
2. **Forward Pass**: calcola gli output dai tensori di input per ottenere punteggi di previsione o logit.
3. **Calcolo delle perdite e passaggio all'indietro**: chiama `loss.backward()` su una funzione di perdita (ad esempio, `cross_entropy`) per calcolare i gradienti di ciascun parametro.
4. **Ottimizzazione**: chiama le funzioni di ottimizzazione integrate (ad esempio, `adam_step`) su ciascun parametro per aggiornarli e reimposta i gradienti con `Tensor::clear_grads()`.

## 2. Differenze tra Tensore e GradTensor

TL ha due tipi di tensori statici primari utilizzati per il calcolo numerico e l'addestramento:

- **`Tensor<T, R>`**: dati di array multidimensionali standard. Non tiene traccia dei gradienti (cronologia dei calcoli), quindi è veloce ed efficiente in termini di memoria. Viene utilizzato principalmente per l'**elaborazione dei dati durante l'inferenza** e la memorizzazione dello **stato interno dell'ottimizzatore (ad esempio slancio e varianza)**.
- **`GradTensor<T, R>`**: un tensore di tracciamento del gradiente per l'addestramento. Registra il processo di calcolo (costruisce un grafico di calcolo) ed esegue la differenziazione automatica per calcolare i gradienti quando viene chiamato `backward()`. È necessario utilizzare sempre `GradTensor` affinché i **parametri (pesi, bias, ecc.) vengano appresi/aggiornati** dall'algoritmo di ottimizzazione.

## 3. Definizione e inizializzazione

Ogni livello del modello è definito come una "struttura". Ad esempio, un livello lineare addestrato con l'ottimizzatore Adam deve mantenere lo stato del momento (`m`, `v`) oltre a pesi e bias. Assegnamo "GradTensor" ai parametri di training e "Tensor" allo stato dell'ottimizzatore.


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


*Nota*: La chiamata a `detach(true)` durante l'inizializzazione dei parametri contrassegna esplicitamente questo tensore come destinazione per il calcolo del gradiente.

## 4. Implementazione della fase di ottimizzazione

Aggiungi una funzione `step` a ciascun livello per eseguire l'algoritmo di ottimizzazione (ad esempio Adam) e aggiornarne lo stato. Un metodo "step" TL utilizza in genere un design immutabile che restituisce una nuova struttura dopo l'aggiornamento.


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


## 5. Ciclo di allenamento e passaggio all'indietro

Nel ciclo di training principale, calcola la perdita, chiama `backward()`, aggiorna il modello tramite `step`, quindi cancella i gradienti.


__CODICE_BLOCCO_2__


## 6. Salvataggio del modello (safetensor)

I parametri del modello appresi possono essere salvati nel formato ".safetensors" utilizzando la funzione "Param::save". I dati salvati possono essere riutilizzati per l'inferenza.


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