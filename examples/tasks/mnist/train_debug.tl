// Debug train parsing

struct Linear {
    W: Tensor<f32, 2>,
    b: Tensor<f32, 1>
}

impl Linear {
    fn new(i: i64, o: i64) -> Linear {
        return Linear(
            (Tensor::randn([i, o], true) * 0.1).detach(true), 
            (Tensor::randn([o], true) * 0.0).detach(true)
        );
    }
    
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> {
        return matmul(x, self.W) + self.b;
    }
}

fn main() {
    print("Debug");
    
    let input = Tensor::randn([1, 784], false);
    let target = Tensor::zeros([1], false).to_i64();
    let model = Linear::new(784, 10);
    let lr = 0.1;

    for epoch in 0..20 {
        let logits = model.forward(input);
        let loss = cross_entropy(logits, target);
        backward(loss);
        
        // Manual optimization step
        let s = model;
        let gW = s.W.grad();
        let gb = s.b.grad();
        s.W = (s.W - gW * lr).detach(true); // Is this line ok in main?
        s.b = (s.b - gb * lr).detach(true);
        
        if epoch - ((epoch / 5) * 5) == 0 {
           print(epoch);
        }
    }
}
