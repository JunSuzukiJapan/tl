// Two tensor (different ranks) fields test

struct MixedRanks {
    a: Tensor<f32, 2>,
    b: Tensor<f32, 1>,
}

impl MixedRanks {
    fn new_mixed(x: i64, y: i64) -> MixedRanks {
        print("Creating tensor a (rank 2, with grad, NO mul)...");
        let ta = Tensor::randn([x, y], true);  // NO * 0.001
        print("Tensor a created:");
        print(ta);
        print("Creating tensor b (rank 1, with grad)...");
        let tb = Tensor::zeros([y], true);
        print("Tensor b created:");
        print(tb);
        print("Constructing MixedRanks...");
        let result = MixedRanks(ta, tb);
        print("MixedRanks constructed.");
        result
    }
    
    fn get_a(self) -> Tensor<f32, 2> {
        print("get_a: accessing self.a...");
        let a = self.a;
        print("get_a: self.a =");
        print(a);
        a
    }
    
    fn get_b(self) -> Tensor<f32, 1> {
        print("get_b: accessing self.b...");
        let b = self.b;
        print("get_b: self.b =");
        print(b);
        b
    }
}

fn main() {
    print("=== Mixed Ranks Test ===");
    let t = MixedRanks::new_mixed(2, 3);
    print("MixedRanks created, calling get_a...");
    let a = t.get_a();
    print("Final a:");
    print(a);
    print("Calling get_b...");
    let b = t.get_b();
    print("Final b:");
    print(b);
}
