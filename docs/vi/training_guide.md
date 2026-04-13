# Cơ chế đào tạo mô hình được hỗ trợ

TensorLanguage (TL) hỗ trợ đào tạo mô hình mạng thần kinh bên cạnh các hoạt động tensor mạnh mẽ. Tài liệu này giải thích quy trình làm việc từ việc xác định mô hình đến triển khai vòng lặp đào tạo và lưu các mô hình đã đào tạo bằng TL.

## 1. Khái niệm đào tạo cơ bản

Việc đào tạo TL được thực hiện theo các bước sau:

1. **Định nghĩa mô hình**: Sử dụng `struct` để xác định các lớp và mô hình chứa các tham số và trạng thái.
2. **Chuyển tiếp**: Tính toán kết quả đầu ra từ tensor đầu vào để nhận điểm dự đoán hoặc nhật ký.
3. **Tính toán tổn thất và truyền ngược**: Gọi `loss.backward()` trên hàm mất mát (ví dụ: `cross_entropy`) để tính toán độ dốc của từng tham số.
4. **Tối ưu hóa**: Gọi các hàm tối ưu hóa tích hợp sẵn (ví dụ: `adam_step`) trên mỗi tham số để cập nhật chúng và đặt lại độ dốc bằng `Tensor::clear_grads()`.

## 2. Sự khác biệt giữa Tensor và GradTensor

TL có hai loại tensor tĩnh chính được sử dụng để tính toán và huấn luyện số:

- **`Tensor<T, R>`**: Dữ liệu mảng đa chiều chuẩn. Nó không theo dõi độ dốc (lịch sử tính toán), vì vậy nó nhanh và tiết kiệm bộ nhớ. Nó chủ yếu được sử dụng để **xử lý dữ liệu trong quá trình suy luận** và lưu trữ **trạng thái bên trong của trình tối ưu hóa (ví dụ: động lượng và phương sai)**.
- **`GradTensor<T, R>`**: Một tensor theo dõi độ dốc dành cho đào tạo. Nó ghi lại quá trình tính toán (xây dựng biểu đồ tính toán) và thực hiện vi phân tự động để tính toán độ dốc khi `backward()` được gọi. Bạn phải luôn sử dụng `GradTensor` cho **các tham số (trọng số, độ lệch, v.v.) để thuật toán tối ưu hóa học/cập nhật**.

## 3. Định nghĩa và khởi tạo

Mỗi lớp của mô hình được định nghĩa là một `struct`. Ví dụ: một lớp Tuyến tính được đào tạo bằng trình tối ưu hóa Adam cần giữ trạng thái động lượng (`m`, `v`) bên cạnh các trọng số và độ lệch. Chúng tôi gán `GradTensor` cho các tham số huấn luyện và `Tensor` cho trạng thái tối ưu hóa.


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


*Lưu ý*: Việc gọi `detach(true)` trong quá trình khởi tạo tham số sẽ đánh dấu rõ ràng tensor này là mục tiêu để tính toán độ dốc.

## 4. Thực hiện bước tối ưu hóa

Thêm hàm `step` vào mỗi lớp để thực thi thuật toán tối ưu hóa (ví dụ: Adam) và cập nhật trạng thái của nó. Phương thức `step` TL thường sử dụng một thiết kế bất biến, trả về cấu trúc mới sau khi cập nhật.


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


## 5. Vòng huấn luyện và Đường chuyền ngược

Trong vòng huấn luyện chính, tính toán tổn thất, gọi `backward()`, cập nhật mô hình qua `step`, sau đó xóa độ dốc.


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


## 6. Lưu Model (Safetensors)

Các tham số mô hình đã học có thể được lưu ở định dạng `.safetensors` bằng cách sử dụng hàm `Param::save`. Dữ liệu đã lưu có thể được sử dụng lại để suy luận.


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