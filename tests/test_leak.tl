struct MyModel {
    W: Tensor<f32, 2>,
}

impl MyModel {
    fn step(self) -> MyModel {
        let mut s = self;
        let tmp = (s.W - Tensor::zeros([1000, 1000]));
        s.W = tmp.detach(true);
        s
    }
}

fn main() {
    let mut model = MyModel {
        W: Tensor::zeros([1000, 1000]),
    };
    
    let mut i = 0;
    while i < 10000 {
        model = model.step();
        i = i + 1;
    }
}
