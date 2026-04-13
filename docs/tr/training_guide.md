# Desteklenen Model Eğitim Mekanizmaları

TensorLanguage (TL), güçlü tensör işlemlerine ek olarak sinir ağı modeli eğitimini de destekler. Bu belge, modellerin tanımlanmasından eğitim döngüsünün uygulanmasına ve eğitilen modellerin TL kullanılarak kaydedilmesine kadar iş akışını açıklamaktadır.

## 1. Temel Eğitim Kavramları

TL'de eğitim aşağıdaki adımlarda gerçekleştirilir:

1. **Model Tanımı**: Parametreleri ve durumu tutan katmanları ve modelleri tanımlamak için "yapı"yı kullanın.
2. **İleri Geçiş**: Tahmin puanlarını veya logitleri almak için giriş tensörlerinden gelen çıktıları hesaplayın.
3. **Kayıp Hesaplaması ve Geriye Geçiş**: Her parametrenin gradyanlarını hesaplamak için bir kayıp fonksiyonunda (örneğin, `cross_entropy`) `loss.backward()` çağrısını yapın.
4. **Optimizasyon**: Her parametreyi güncellemek için yerleşik optimizasyon işlevlerini (örneğin, `adam_step`) çağırın ve `Tensor::clear_grads()` ile degradeleri sıfırlayın.

## 2. Tensör ve GradTensor Arasındaki Farklar

TL'nin sayısal hesaplama ve eğitim için kullanılan iki temel statik tensör türü vardır:

- **`Tensor<T, R>`**: Standart çok boyutlu dizi verileri. Degradeleri (hesaplama geçmişi) izlemez, dolayısıyla hızlıdır ve bellek açısından verimlidir. Esas olarak **çıkarım sırasında veri işleme** ve **optimizer dahili durumunu (örn. momentum ve varyans)** depolamak için kullanılır.
- **`GradTensor<T, R>`**: Eğitim için bir degrade izleme tensörü. Hesaplama sürecini kaydeder (bir hesaplama grafiği oluşturur) ve 'backward()' çağrıldığında gradyanları hesaplamak için otomatik farklılaşma gerçekleştirir. Optimizasyon algoritması tarafından **öğrenilecek/güncellenecek** parametreler (ağırlıklar, sapmalar vb.) için her zaman 'GradTensor'u kullanmalısınız.

## 3. Tanım ve Başlatma

Modelin her katmanı bir “yapı” olarak tanımlanır. Örneğin, Adam optimizer ile eğitilmiş bir Doğrusal katmanın, ağırlıklara ve önyargılara ek olarak momentum durumunu ("m", "v") tutması gerekir. Eğitim parametrelerine 'GradTensor'u ve optimize edici durumuna 'Tensor'u atarız.


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


*Not*: Parametre başlatma sırasında `detach(true)` çağrılması, bu tensörü açıkça gradyan hesaplaması için bir hedef olarak işaretler.

## 4. Optimizasyon Adımını Uygulamak

Optimizasyon algoritmasını (örneğin Adam) yürütmek ve durumunu güncellemek için her katmana bir "adım" işlevi ekleyin. TL "adım" yöntemi genellikle güncellemeden sonra yeni bir yapı döndüren değişmez bir tasarım kullanır.


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


## 5. Eğitim Döngüsü ve Geriye Geçiş

Ana eğitim döngüsünde kaybı hesaplayın, 'backward()'ı çağırın, modeli 'step' aracılığıyla güncelleyin ve ardından degradeleri temizleyin.


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


## 6. Modeli Kaydetme (Safetensörler)

Öğrenilen model parametreleri 'Param::save' fonksiyonu kullanılarak '.safetensors' formatında kaydedilebilir. Kaydedilen veriler çıkarım için yeniden kullanılabilir.


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