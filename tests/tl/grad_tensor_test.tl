fn main() {
    // 1. GradTensor factory method
    let w = GradTensor::zeros([3]);
    println("GradTensor::zeros OK");

    // 2. Tensor factory
    let x = Tensor::zeros([3]);
    println("Tensor::zeros OK");

    // 3. as GradTensor (Tensor -> GradTensor)
    let x_grad = x as GradTensor;
    println("as GradTensor OK");

    // 4. as Tensor (GradTensor -> Tensor)
    let w_frozen = w as Tensor;
    println("as Tensor OK");

    // 5. GradTensor method call (backward, grad)
    let a = GradTensor::ones([2]);
    a.backward();
    let g = a.grad();
    println("backward/grad OK");

    println("All GradTensor tests passed!");
}
