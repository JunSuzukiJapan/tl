// Test: struct return from function

struct Container {
    t: Tensor<f32, 2>,
}

fn create_container(r: i64, c: i64) -> Container {
    print("create_container: creating tensor...");
    let tensor = Tensor::randn([r, c], false);
    print("create_container: tensor =");
    print(tensor);
    print("create_container: constructing Container...");
    let result = Container(tensor);
    print("create_container: result.t =");
    print(result.t);
    print("create_container: returning...");
    result
}

fn main() {
    print("=== Struct Return Test ===");
    let c = create_container(2, 3);
    print("main: After return, c.t =");
    print(c.t);
}
