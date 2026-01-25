fn main() {
    // 1. Basic Tensor Allocation & Auto-free
    {
        let t1 = Tensor::randn([2, 2]);
        // t1 should be freed at end of scope
    }
    println("Test 1: Allocation & Scope-based Free - OK");

    // 2. Variable Reassignment (V2 feature)
    let mut t2 = Tensor::zeros([2, 2]);
    t2 = Tensor::ones([2, 2]); 
    // Old t2 should be freed here (or marked for free). 
    // New t2 (ones) replaces it.
    println("Test 2: Variable Reassignment - OK");

    // 3. Tensor Operations (Binary/Unary) & Temporary Cleanup
    let a = Tensor::randn([3]);
    let b = Tensor::randn([3]);
    let c = (a + b) * 2.0; 
    // (a+b) is a temporary. It should be freed immediately after usage or at statement end.
    // c is a new tensor.
    println("Test 3: Operations & Temporary Cleanup - OK");

    // 4. Return from Function (Moving ownership) - Simulating with Block
    let res = {
        let t = Tensor::zeros([3, 3]);
        t // Ownership moves to res
    };
    // res should be valid.
    println("Test 4: Return Ownership (Block) - OK");
}

