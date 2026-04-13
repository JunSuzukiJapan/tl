# Mécanismes de formation de modèles pris en charge

TensorLanguage (TL) prend en charge la formation de modèles de réseaux neuronaux en plus de puissantes opérations tensorielles. Ce document explique le flux de travail depuis la définition des modèles jusqu'à la mise en œuvre de la boucle de formation et la sauvegarde des modèles formés à l'aide de TL.

## 1. Concepts de formation de base

La formation en TL se déroule selon les étapes suivantes :

1. **Définition du modèle** : utilisez `struct` pour définir les couches et les modèles qui contiennent les paramètres et l'état.
2. **Forward Pass** : calculez les sorties des tenseurs d'entrée pour obtenir des scores de prédiction ou des logits.
3. **Calcul de perte et passage en arrière** : appelez `loss.backward()` sur une fonction de perte (par exemple, `cross_entropy`) pour calculer les gradients de chaque paramètre.
4. **Optimisation** : appelez les fonctions d'optimisation intégrées (par exemple, `adam_step`) sur chaque paramètre pour les mettre à jour et réinitialisez les dégradés avec `Tensor::clear_grads()`.

## 2. Différences entre Tensor et GradTensor

TL dispose de deux principaux types de tenseurs statiques utilisés pour le calcul numérique et la formation :

- **`Tensor<T, R>`** : données de tableau multidimensionnel standard. Il ne suit pas les gradients (historique des calculs), il est donc rapide et économe en mémoire. Il est principalement utilisé pour le **traitement des données pendant l'inférence** et pour le stockage de **l'état interne de l'optimiseur (par exemple, élan et variance)**.
- **`GradTensor<T, R>`** : Un tenseur de suivi de gradient pour l'entraînement. Il enregistre le processus de calcul (construit un graphe de calcul) et effectue une différenciation automatique pour calculer les gradients lorsque « backward() » est appelé. Vous devez toujours utiliser `GradTensor` pour que les **paramètres (poids, biais, etc.) soient appris/mis à jour** par l'algorithme d'optimisation.

## 3. Définition et initialisation

Chaque couche du modèle est définie comme une « struct ». Par exemple, une couche linéaire entraînée avec l'optimiseur Adam doit conserver l'état de dynamique (`m`, `v`) en plus des poids et des biais. Nous attribuons « GradTensor » aux paramètres d'entraînement et « Tensor » à l'état de l'optimiseur.


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


*Remarque* : L'appel de « detach(true) » lors de l'initialisation des paramètres marque explicitement ce tenseur comme cible pour le calcul du gradient.

## 4. Mise en œuvre de l'étape d'optimisation

Ajoutez une fonction « step » à chaque couche pour exécuter l'algorithme d'optimisation (par exemple, Adam) et mettre à jour son état. Une méthode `step` TL utilise généralement une conception immuable qui renvoie une nouvelle structure après la mise à jour.


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


## 5. Boucle d'entraînement et passe arrière

Dans la boucle d'entraînement principale, calculez la perte, appelez « backward() », mettez à jour le modèle via « step », puis effacez les dégradés.


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


## 6. Sauvegarde du modèle (Safetensors)

Les paramètres du modèle appris peuvent être enregistrés au format `.safetensors` à l'aide de la fonction `Param::save`. Les données enregistrées peuvent être réutilisées pour l'inférence.


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