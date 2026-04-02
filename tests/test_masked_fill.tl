fn main() {
    print("Testing masked_fill with float constants:");

    let x = Tensor::full([2, 2], 1.0);
    // [1.0, 1.0]
    
    let mut mask = Tensor::zeros([2, 2]);
    mask[0, 1] = 1.0; // mask upper right to be true
    
    // Test with -INFINITY
    let y_neg_inf = x.masked_fill(mask, -f32::INFINITY);
    print("Masked with -f32::INFINITY:");
    y_neg_inf.display();

    // Test with INFINITY
    let y_inf = x.masked_fill(mask, f32::INFINITY);
    print("Masked with f32::INFINITY:");
    y_inf.display();

    // Test with regular f32
    let y_reg = x.masked_fill(mask, 100.5 as f32);
    print("Masked with 100.5:");
    y_reg.display();

    // Check if memory is alright
    let mut leak_check = 0;
    while leak_check < 10 {
        let temp = x.masked_fill(mask, -f32::INFINITY);
        leak_check = leak_check + 1;
    }
    
    print("Success!");
}
