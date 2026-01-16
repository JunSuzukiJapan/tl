// Differentiable Sudoku Solver using Tensor Logic
// Usage: cargo run --release --bin tl -- run examples/tasks/tensor_logic/sudoku/sudoku.tl -- "PUZZLE_STRING"

fn main() {
    // Check command line arguments
    let argc = args_count();
    
    // For testing, use a default puzzle if no args provided
    let puzzle_str = "530070000600195000098000060800060003400803001700020006060000280000419005000080079";
    if argc >= 1 {
        let puzzle_str = args_get(0);
    }
    
    println("Sudoku Solver - Differentiable Programming Approach");
    print("Input: ");
    println(puzzle_str);
    
    // Constants
    let N = 9;
    let lr = 0.5;
    let epochs = 3000;
    
    // Create mask and target tensors
    // mask: [9, 9, 1] - 1.0 if hint exists, 0.0 otherwise
    // target_probs: [9, 9, 9] - One-hot encoding of hints
    
    let hint_mask_2d = Tensor::zeros([N, N]);
    let target_probs = Tensor::zeros([N, N, N]);
    
    let target_probs = Tensor::zeros([N, N, N]);
    
    println("Parsing puzzle...");
    // Parse puzzle string
    for i in 0..81 {
        let c = char_at(puzzle_str, i);
        let val = 0;
        
        // Manual char to digit conversion
        if c == "1" { let val = 1; }
        if c == "2" { let val = 2; }
        if c == "3" { let val = 3; }
        if c == "4" { let val = 4; }
        if c == "5" { let val = 5; }
        if c == "6" { let val = 6; }
        if c == "7" { let val = 7; }
        if c == "8" { let val = 8; }
        if c == "9" { let val = 9; }
        
        if val > 0 {
            // Found a hint
            let row = i / 9;
            let col = i - (row * 9); // Modulo replacement
            
            // Set mask
            hint_mask_2d.set(row, col, 1.0);
            
            // Set target probability (one-hot)
            // digit is val-1 (0-8)
            let digit_idx = val - 1;
            target_probs.set(row, col, digit_idx, 1.0);
        }
    }
    println("Parsing done.");
    
    let hint_mask = hint_mask_2d.reshape([N, N, 1]); // Broadcast ready [9, 9, 1]
    
    // Initialize board logits
    let mut board = Tensor::randn([N, N, N], true);
    
    // Optimization loop
    let epochs = 600;
    for epoch in 0..epochs {
        // Convert logits to probabilities
        let probs = board.softmax(2);
        
        // 1. Hint Loss: MSE between generated probs and target probs, masked
        // (probs - target) * mask
        let diff = probs - target_probs;
        // Broadcast hint_mask [9,9,1] against diff [9,9,9]
        // Manual multiplication supports broadcasting
        let masked_diff = diff * hint_mask;
        let hint_loss = (masked_diff * masked_diff).sum() * 5.0; // Weight hint loss higher
        
        // 2. Constraints
        let row_sums = probs.sum(1);
        let col_sums = probs.sum(0);
        let cell_sums = probs.sum(2);
        
        let row_loss = (row_sums - 1.0).pow(2.0).sum();
        let col_loss = (col_sums - 1.0).pow(2.0).sum();
        let cell_loss = (cell_sums - 1.0).pow(2.0).sum();
        
        let total_loss = row_loss + col_loss + cell_loss + hint_loss;
        
        // Print progress
        if (epoch / 100) * 100 == epoch {
            print("Epoch ");
            print(epoch);
            print(" Loss: ");
            print(total_loss.item());
            print(" (Hint: ");
            print(hint_loss.item());
            println(")");
        }
        
        // Backprop
        total_loss.backward();
        let g = board.grad();
        // Update board (gradient descent)
        // Use assignment to update outer variable 'board', not let (shadowing)
        board = board - (g * lr);
        board = board.detach();
        board.enable_grad();
    }
    
    // Solution output
    println("");
    println("=== Solution ===");
    let probs = board.softmax(2);
    
    for row in 0..N {
        if row > 0 {
            if (row / 3) * 3 == row {
                println("------+-------+------");
            }
        }
        for col in 0..N {
            if col > 0 {
                if (col / 3) * 3 == col {
                    print("| ");
                }
            }
            
            // Argmax
            let mut best_digit = 0;
            let mut best_prob = -1.0;
            for d in 0..N {
                let p = probs.get(row, col, d);
                if p > best_prob {
                    best_prob = p; // Assignment
                    best_digit = d + 1; // Assignment
                }
            }
            print(best_digit);
            print(" ");
        }
        println("");
    }
}
