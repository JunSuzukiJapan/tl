
fn get_memory() -> i64 {
    return System::memory_mb();
}

struct Linear { W: Tensor<f32, 2>, b: Tensor<f32, 1> }

impl Linear {
    fn new(in_dim: i64, out_dim: i64) -> Linear {
        return Linear {
            W: Tensor::randn([in_dim, out_dim], false) * 0.01,
            b: Tensor::zeros([out_dim]),
        };
    }

    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> {
        return x.matmul(self.W) + self.b;
    }
}

fn test_basic_tensor() {
    print("Test 1: Basic Tensor Allocation");
    let initial = get_memory();
    for i in range(0, 1000) {
        let x = Tensor::randn([100, 100], false);
        let y = Tensor::randn([100, 100], false);
    }
    let final = get_memory();
    print("Diff (MB):"); print(final - initial);
}

fn test_binop() {
    print("Test 2: Binary Operation");
    let initial = get_memory();
    for i in range(0, 1000) {
        let x = Tensor::randn([100, 100], false);
        let y = Tensor::randn([100, 100], false);
        let z = x + y; // Should free x and y if temporary? No, x and y are vars.
         // Wait, x and y are vars, so they are freed at end of scope.
         // z is freed at end of scope.
         // The intermediate result (if composed) would be interesting.
    }
    let final = get_memory();
    print("Diff (MB):"); print(final - initial);
}

fn test_method_chain() {
    print("Test 3: Method Chain (temp tensors)");
    let initial = get_memory();
    for i in range(0, 1000) {
        let x = Tensor::randn([100, 100], false);
        let y = Tensor::randn([100, 100], false);
        // (x * y) creates temp, .relu() creates temp from temp
        let z = (x * y).relu(); 
    }
    let final = get_memory();
    print("Diff (MB):"); print(final - initial);
}

fn test_linear_forward() {
    print("Test 4: Linear Forward");
    let layer = Linear::new(100, 100);
    let initial = get_memory();
    for i in range(0, 1000) {
        let x = Tensor::randn([1, 100], false);
        let y = layer.forward(x);
    }
    let final = get_memory();
    print("Diff (MB):"); print(final - initial);
}

fn main() {
    print("Starting Memory Leak Test...");
    print("Initial Memory:"); print(get_memory());
    
    test_basic_tensor();
    test_binop();
    test_method_chain();
    test_linear_forward();
    
    print("Final Memory:"); print(get_memory());
}
