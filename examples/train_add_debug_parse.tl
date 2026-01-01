struct GPT { w: Tensor<f32, 1> }
impl GPT { fn new() -> GPT { return GPT(randn([1], true)); } }
fn main() {
    let a = [1.0, 2.0]; // OK
    print("a OK");
    let b = [10.0, 20.0]; // Check 2 digits
    print("b OK");
    let c = [1.0, 10.0, 100.0]; 
    print("c OK");
}
