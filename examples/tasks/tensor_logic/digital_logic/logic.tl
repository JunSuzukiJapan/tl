fn main() {
    print("Differentiable Digital Logic (XOR Learning)...\n");

    // Goal: Learn weights for a 2-layer network that implements XOR
    // XOR Truth Table:
    // A=0, B=0 -> 0
    // A=0, B=1 -> 1
    // A=1, B=0 -> 1
    // A=1, B=1 -> 0
    
    // Network: 2 inputs -> 2 hidden (NAND-like) -> 1 output (OR-like)
    // Actually, XOR = (A NAND (A NAND B)) NAND (B NAND (A NAND B))
    // Simplified: Use 2 hidden units with sigmoid, then 1 output unit
    
    // XOR can be computed by:
    // h1 = A AND NOT B
    // h2 = NOT A AND B  
    // out = h1 OR h2
    
    // Or using neurons:
    // h1 = sigmoid(w1*A + w2*B + b1) -- learns boundary
    // h2 = sigmoid(w3*A + w4*B + b2) -- learns another boundary
    // out = sigmoid(w5*h1 + w6*h2 + b3)
    
    // Training Data
    let mut inputs = Tensor::zeros([4, 2]);
    inputs[0, 0] = 0.0; inputs[0, 1] = 0.0;
    inputs[1, 0] = 0.0; inputs[1, 1] = 1.0;
    inputs[2, 0] = 1.0; inputs[2, 1] = 0.0;
    inputs[3, 0] = 1.0; inputs[3, 1] = 1.0;
    
    let mut targets = Tensor::zeros([4, 1]);
    targets[0, 0] = 0.0; // 0 XOR 0
    targets[1, 0] = 1.0; // 0 XOR 1
    targets[2, 0] = 1.0; // 1 XOR 0
    targets[3, 0] = 0.0; // 1 XOR 1
    
    // Weights (Learnable)
    let mut W1 = Tensor::randn([2, 2], true); // input -> hidden
    let mut b1 = Tensor::randn([2], true);
    let mut W2 = Tensor::randn([2, 1], true); // hidden -> output
    let mut b2 = Tensor::randn([1], true);
    
    let lr = 1.0;
    let mut iter = 0;
    
    while iter < 1000 {
        // Forward Pass
        // hidden = sigmoid(inputs @ W1 + b1)
        let z1 = inputs.matmul(W1) + b1;
        let hidden = z1.sigmoid();
        
        // output = sigmoid(hidden @ W2 + b2)
        let z2 = hidden.matmul(W2) + b2;
        let output = z2.sigmoid();
        
        // Loss: Binary Cross-Entropy or MSE
        // Using MSE for simplicity
        let diff = output - targets;
        let loss = diff.pow(2).sumall();
        
        if (iter / 100) * 100 == iter {
            println("Iter {} Loss: {}", iter, loss.item());
        }
        
        loss.backward();
        
        // Update weights
        let gW1 = W1.grad();
        let gb1 = b1.grad();
        let gW2 = W2.grad();
        let gb2 = b2.grad();
        
        W1 = W1 - gW1 * lr;
        b1 = b1 - gb1 * lr;
        W2 = W2 - gW2 * lr;
        b2 = b2 - gb2 * lr;
        
        W1 = W1.detach(); W1.enable_grad();
        b1 = b1.detach(); b1.enable_grad();
        W2 = W2.detach(); W2.enable_grad();
        b2 = b2.detach(); b2.enable_grad();
        
        iter = iter + 1;
    }
    
    print("Training Done.\n");
    
    // Evaluate
    print("Evaluation:\n");
    let z1 = inputs.matmul(W1) + b1;
    let hidden = z1.sigmoid();
    let z2 = hidden.matmul(W2) + b2;
    let output = z2.sigmoid();
    
    let mut i = 0;
    while i < 4 {
        let a = inputs[i, 0];
        let b = inputs[i, 1];
        let pred = output[i, 0];
        let tgt = targets[i, 0];
        println("Input: ({}, {}) -> Pred: {} Target: {}", a, b, pred, tgt);
        i = i + 1;
    }
}
