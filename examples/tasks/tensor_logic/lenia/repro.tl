fn main() {
    let H = 64;
    let W = 64;
    let state = Tensor::randn([1, 1, H, W], false); 
    let state = state.abs().clamp(0.0, 1.0);
    let K_size = 7;
    let kernel = Tensor::ones([1, 1, K_size, K_size], false);
    let pad = K_size / 2;
    
    let steps = 10;
    for i in 0..steps {
        let neighbors = state.conv2d(kernel, pad, 1);
        print(i);
    }
}
