struct Linear { W: Tensor<f32, 2>, b: Tensor<f32, 1> }
impl Linear { 
    fn new(i: i64, o: i64) -> Linear { 
        return Linear((Tensor::randn([i, o], true)*0.1).detach(true), (Tensor::randn([o], true)*0.0).detach(true)); 
    } 
    fn forward(self, x: Tensor<f32, 2>) -> Tensor<f32, 2> { 
        return x.matmul(self.W) + self.b; 
    } 
    fn step(self, lr: f32) -> Linear { 
        let s = self; 
        let gW = s.W.grad(); 
        let gb = s.b.grad(); 
        s.W = (s.W - gW * lr).detach(true); 
        s.b = (s.b - gb * lr).detach(true); 
        return s; 
    } 
}

fn main() {
    print("Simple MLP training loop test - testing compiler fix");
    
    let l1 = Linear::new(2, 4);
    let l2 = Linear::new(4, 3);
    
    let lr = 0.01;
    
    for epoch in range(0, 5) {
        println("Epoch: {}", epoch);
        
        let x_data = [1.0, 2.0];
        let x = x_data.reshape(1, 2);
        
        let h = l1.forward(x).relu();
        let y = l2.forward(h);
        
        let target = [1.0, 0.0, 0.0];
        let t = target.reshape(1, 3);
        let diff = y - t;
        let diff_sq = diff * diff;
        let loss = diff_sq.sum();
        
        println("Loss: {}", loss);
        
        loss.backward();
        
        // Using original syntax WITHOUT capturing return value
        // This should now work with the compiler fix
        l1.step(lr);
        l2.step(lr);
        
        print("Step done");
    }
    print("Training complete!");
}
