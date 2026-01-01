struct Data {
    t: Tensor<f32, 2>,
}

fn main() {
    let x = randn([2, 2], false);
    if true {
        let y = randn([1, 1], false);
        // let z = y[0, 0]; // Index access verified in repro_index.tl
        print("Inner OK");
    }
    
    let d = Data {
        t: randn([3, 3], false),
    };
    print(d.t);
}
