
fn get_memory() -> i64 { return System::memory_mb(); }

fn main() {
    print("Test Nested Calls");
    let initial = get_memory();
    for i in range(0, 500) {
        let x = Tensor::randn([100, 100], false);
        // matmul(x, x) returns temp T1
        // relu(T1) returns T2
        // w = T2
        // T1 should be freed after relu returns?
        let w = x.matmul(x).relu();
        let z = x.matmul(x).relu();
    }
    let final = get_memory();
    print(final - initial);
}
