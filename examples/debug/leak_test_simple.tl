
fn get_memory() -> i64 {
    return tl_get_memory_mb();
}

fn main() {
    print("Test Basic Allocation");
    let start = get_memory();
    for i in range(0, 100) {
        let x = Tensor::randn([100, 100], false);
    }
    let end = get_memory();
    print(end - start);
}
