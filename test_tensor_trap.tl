struct Linear {
    W: Tensor<f32, 2>,
}

fn make_linear() -> Linear {
    print("Making");
    Linear(Tensor::randn([10, 10], false))
}

fn main() {
    print("Start");
    let l = make_linear();
    print("End");
}
