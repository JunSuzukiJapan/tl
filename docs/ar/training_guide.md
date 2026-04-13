# آليات التدريب النموذجية المدعومة

يدعم TensorLanguage (TL) التدريب على نماذج الشبكة العصبية بالإضافة إلى عمليات الموتر القوية. يشرح هذا المستند سير العمل بدءًا من تحديد النماذج وحتى تنفيذ حلقة التدريب وحفظ النماذج المدربة باستخدام TL.

## 1. مفاهيم التدريب الأساسية

يتم تنفيذ التدريب في TL في الخطوات التالية:

1. **تعريف النموذج**: استخدم ``الهيكل`` لتحديد الطبقات والنماذج التي تحتوي على المعلمات والحالة.
2. **التمرير الأمامي**: حساب المخرجات من موترات الإدخال للحصول على نتائج أو سجلات التنبؤ.
3. **حساب الخسارة والتمرير إلى الخلف**: قم باستدعاء `loss.backward()` على دالة خسارة (على سبيل المثال، `cross_entropy`) لحساب تدرجات كل معلمة.
4. **التحسين**: استدعاء وظائف التحسين المضمنة (على سبيل المثال، `adam_step`) في كل معلمة لتحديثها، وإعادة تعيين التدرجات باستخدام `Tensor::clear_grads()`.

## 2. الاختلافات بين Tensor وGradTensor

يحتوي TL على نوعين أساسيين من الموتر الثابت المستخدم في الحساب الرقمي والتدريب:

- **`Tensor<T, R>`**: بيانات مصفوفة قياسية متعددة الأبعاد. فهو لا يتتبع التدرجات (سجل الحساب)، لذا فهو سريع وفعال في الذاكرة. يتم استخدامه بشكل أساسي في ** معالجة البيانات أثناء الاستدلال ** وتخزين ** الحالة الداخلية للمُحسِّن (مثل الزخم والتباين) **.
- **`GradTensor<T, R>`**: موتر لتتبع التدرج للتدريب. فهو يسجل عملية الحساب (ينشئ رسمًا بيانيًا حسابيًا) وينفذ التمايز التلقائي لحساب التدرجات عند استدعاء `backward()`. يجب عليك دائمًا استخدام `GradTensor` لـ **المعلمات (الأوزان، والتحيزات، وما إلى ذلك) التي سيتم تعلمها/تحديثها** بواسطة خوارزمية التحسين.

## 3. التعريف والتهيئة

يتم تعريف كل طبقة من النموذج على أنها "بنية". على سبيل المثال، تحتاج الطبقة الخطية التي تم تدريبها باستخدام مُحسِّن Adam إلى الحفاظ على حالة الزخم (`m`، `v`) بالإضافة إلى الأوزان والتحيزات. نقوم بتعيين "GradTensor" لمعلمات التدريب، و"Tensor" لحالة المحسن.


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


*ملاحظة*: يؤدي استدعاء `detach(true)` أثناء تهيئة المعلمة إلى وضع علامة واضحة على هذا الموتر كهدف لحساب التدرج.

## 4. تنفيذ خطوة التحسين

أضف وظيفة "خطوة" إلى كل طبقة لتنفيذ خوارزمية التحسين (على سبيل المثال، آدم) وتحديث حالتها. عادةً ما تستخدم طريقة TL `step` تصميمًا غير قابل للتغيير يُرجع بنية جديدة بعد التحديث.


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


## 5. حلقة التدريب والتمرير الخلفي

في حلقة التدريب الرئيسية، قم بحساب الخسارة، واستدعاء `backward()`، وتحديث النموذج عبر `step`، ثم مسح التدرجات.


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


## 6. حفظ النموذج (أدوات الأمان)

يمكن حفظ معلمات النموذج التي تم تعلمها بتنسيق `.safetensors` باستخدام وظيفة `Param::save`. يمكن إعادة استخدام البيانات المحفوظة للاستدلال.


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