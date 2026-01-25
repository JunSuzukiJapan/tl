fn main() {
    let mut ok = 1;

    // 1. Integer Mod Assign
    let mut i = 10;
    i %= 3;
    if i != 1 {
        print("Failed int %=: expected 1, got ");
        println(i);
        ok = 0;
    }

    // 2. Float Mod Assign
    let mut f = 10.5;
    f %= 3.0;
    // 10.5 % 3.0 = 1.5
    if (f - 1.5).abs() > 0.0001 {
        print("Failed float %=: expected 1.5, got ");
        println(f);
        ok = 0;
    }

    // 3. Tensor Mod Assign
    let mut t = Tensor::ones([2, 2]) * 10.0; // [[10, 10], [10, 10]]
    t %= 3.0; // [[1, 1], [1, 1]]
    
    let sum = t.sum();
    // 1+1+1+1 = 4
    if (sum - 4.0).abs() > 0.0001 {
        print("Failed tensor %=: expected sum 4.0, got ");
        println(sum);
        ok = 0;
    }

    if ok == 1 {
        println("PERCENT_ASSIGN_OK");
    }
}
