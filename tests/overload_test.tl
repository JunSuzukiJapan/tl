
fn main() {
    println("--- Testing Overload Resolution ---");

    // 1. Static Methods Overloading
    println("1. Static Methods (zeros/ones/randn)");

    // zeros(shape)
    let z1 = Tensor::zeros([2, 2]);
    println("zeros([2,2]): OK");
    // print_shape(z1);

    // zeros(shape, requires_grad)
    let z2 = Tensor::zeros([2, 2], true);
    println("zeros([2,2], true): OK");
    // Verify it works (no crash)
    
    // ones(shape)
    let o1 = Tensor::ones([2]);
    println("ones([2]): OK");

    // randn(shape)
    let r1 = Tensor::randn([2]);
    println("randn([2]): OK");

    println("----------------");

    // 2. Instance Methods Overloading
    println("2. Instance Methods (Reduction Ops)");
    
    // Data: [[1.0, 2.0], [3.0, 4.0]]
    // min global = 1.0
    // min(dim=0) = [1.0, 2.0]
    // min(dim=1) = [1.0, 3.0]
    let mut t = Tensor::ones([2, 2]);
    // construct data manually or use index assignment
    t[0, 0] = 1.0; t[0, 1] = 2.0;
    t[1, 0] = 3.0; t[1, 1] = 4.0;

    // min() - 0 args
    let m0 = t.min();
    print("t.min() (expected 1.0): ");
    println(m0.item());

    // min(dim) - 1 arg
    let m1 = t.min(0);
    println("t.min(0) (expected [1.0, 2.0]):");
    print(m1[0]); print(", "); println(m1[1]);

    // min(dim, keepdim) - 2 args
    let m2 = t.min(1, true);
    println("t.min(1, true) (expected shape [2, 1]):");
    // print_shape(m2); // shape() runtime not available
    // Print first element as proxy check
    println(m2[0]); // Rank 0 tensor (from semantic perspective), indexing gives scalar

    // sum()
    let s0 = t.sum();
    print("t.sum() (expected 10.0): ");
    println(s0.item());

    // sum(dim)
    let s1 = t.sum(0);
    println("t.sum(0) (expected [4.0, 6.0]):");
    print(s1[0]); print(", "); println(s1[1]);


    println("--- Overload Test Completed ---");
}

/*
fn print_shape(t: Tensor) {
    let s = t.shape(); // Vec<i64>
    print("Shape: [");
    let len = s.len();
    for i in 0..len {
        print(s[i]);
        if i < len - 1 {
            print(", ");
        }
    }
    println("]");
}
*/
