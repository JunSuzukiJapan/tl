struct MyTensorWrapper {
    t: Tensor<f32, 2>
}

struct MyContainer {
    inner: MyTensorWrapper
}

fn create_wrapper() -> MyTensorWrapper {
    let t = Tensor::randn([2, 2], false);
    // Should trigger deep clone / acquire on 't'
    return MyTensorWrapper { t: t }; 
}

fn main() {
    print("Start Test");
    let w = create_wrapper();
    print("Wrapper Created");
    
    // Test Nested
    // Should trigger deep clone on 'w' (struct copy + tensor acquire)
    let c = MyContainer { inner: w };
    print("Container Created");
}
