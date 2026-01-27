// Simple struct field access test

struct Container {
    t: Tensor<f32, 2>,
}

struct TwoFields {
    W: Tensor<f32, 2>,
    b: Tensor<f32, 1>,
}

impl Container {
    fn new(r: i64, c: i64) -> Container {
        print("Creating tensor...");
        let tensor = Tensor::randn([r, c], false);
        print("Tensor created:");
        print(tensor);
        print("Creating Container...");
        let c = Container(tensor);
        print("Container created.");
        return c;
    }
    
    fn get_tensor(self) -> Tensor<f32, 2> {
        print("get_tensor: accessing self.t...");
        let t = self.t;
        print("get_tensor: self.t =");
        print(t);
        return t;
    }
}

impl TwoFields {
    fn new(i: i64, o: i64) -> TwoFields {
        print("Creating W (without detach)...");
        let w = Tensor::randn([i, o], true) * 0.001;
        print("W created:");
        print(w);
        print("Creating b (without detach)...");
        let b = Tensor::zeros([o], true);
        print("b created:");
        print(b);
        print("Constructing TwoFields...");
        let result = TwoFields(w, b);
        print("TwoFields constructed.");
        return result;
    }
    
    fn get_W(self) -> Tensor<f32, 2> {
        print("get_W: accessing self.W...");
        let w = self.W;
        print("get_W: self.W =");
        print(w);
        return w;
    }
}

fn main() {
    print("=== Struct Field Access Test ===");
    let c = Container::new(3, 4);
    print("Container created, calling get_tensor...");
    let t = c.get_tensor();
    print("Final tensor:");
    print(t);
    
    print("\n=== TwoFields Test (like Linear) ===");
    let tf = TwoFields::new(4, 3);
    print("TwoFields created, calling get_W...");
    let w = tf.get_W();
    print("Final W:");
    print(w);
}
