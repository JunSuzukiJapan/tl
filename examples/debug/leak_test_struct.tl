
fn get_memory() -> i64 {
    return tl_get_memory_mb();
}

struct Linear { W: Tensor<f32, 2>, b: Tensor<f32, 1> }

fn main() {
    print("Test Struct Allocation");
    let start = get_memory();
    for i in range(0, 1000) {
        let l = Linear {
            W: Tensor::randn([100, 100], false),
            b: Tensor::randn([100], false),
        };
    }
    let end = get_memory();
    print(end - start);
}
