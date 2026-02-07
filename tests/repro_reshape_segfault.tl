// Minimal reshape segfault repro
tensor A: Tensor<f32, 1> = [1.0, 2.0, 3.0, 4.0];

fn main() {
    print("before reshape\n");
    let M1 = A.reshape([2, 2]);
    print("after reshape\n");
    print(M1);
    print("\nDONE\n");
}
