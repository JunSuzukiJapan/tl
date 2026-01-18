# TL Programming Language

A tensor logic programming language with first-class tensor support, JIT-compiled to LLVM.

## Features
- **Tensor Operations**: `tensor<f32>[128, 128]`, `matmul`, `topk`, etc. via Candle.
- **Logic Programming**: Datalog-style rules with tensor integration.
- **Hybrid Execution**: Logic terms can access tensor data (`data[i]`).
- **JIT Compilation**: High-performance execution using LLVM (Inkwell).
- **GPU Support**: Metal (macOS) backend supported. CUDA support is planned for the future.
- **Optimization**: Aggressive JIT optimization and fast logic inference.

## Installation

Install from crates.io:

```bash
cargo install tl-lang
```

This installs the `tl` command.

## Prerequisites (macOS)
Before building, ensure you have the required dependencies installed via Homebrew.

1. **Install LLVM 18 and OpenSSL**:
   ```bash
   brew install llvm@18 openssl
   ```

2. **Configure Environment Variables**:
   Add the following to your shell configuration (e.g., `~/.zshrc`) to help the build system find LLVM 18:
   ```bash
   export LLVM_SYS_181_PREFIX=$(brew --prefix llvm@18)
   ```
   Reload your shell (`source ~/.zshrc`) before running cargo commands.

## Build & Run
```bash
# Run a specific example
cargo run -- examples/hybrid_test.tl

# Use GPU (Metal on macOS)
cargo run --features metal -- examples/gpu_test.tl
```


## Syntax

TL's syntax is very similar to Rust, but without lifetimes.

### Basic Syntax

```rust
fn main() {
    let x = 5;
    println("{}", x);
}
```

### Tensor Operations

```rust
fn main() {
    let x = [1.0, 2.0, 3.0];
    let y = [4.0, 5.0, 6.0];
    let z = x + y;
    println("{}", z);
}
```

### if Statement

```rust
fn main() {
    let x = 5;
    if x > 0 {
        println("{}", x);
    }
}
```

### while Statement

```rust
fn main() {
    let mut x = 5;
    while x > 0 {
        println("{}", x);
        x = x - 1;
    }
}
```

### for Statement

```rust
fn main() {
    let mut x = 5;
    for i in 0..x {
        println("{}", i);
    }
}
```

### Function Definition

```rust
fn main() {
    let x = 5;
    let y = add(x, 1);
    println("{}", y);
}

fn add(x: i64, y: i64) -> i64 {
    x + y
}
```

### Tensor Comprehension

```rust
fn main() {
    let t = [1.0, 2.0, 3.0, 4.0];
    // Keep elements > 2.0 and double them, others become 0.0 (masking)
    let res = [i | i <- 0..4, t[i] > 2.0 { t[i] * 2.0 }];
    println("{}", res);
}
```

For more details, see [Tensor Comprehension](docs/tensor_comprehension.md)

## VSCode Extension
A syntax highlighting extension is provided in `vscode-tl`.

### Installation
1. Open the `vscode-tl` directory in VSCode.
2. Press **F5** to verify syntax highlighting in a new window.
3. Or install manually:
   - `cd vscode-tl`
   - `npm install -g vsce` (if needed)
   - `vsce package`
   - `code --install-extension tensor-logic-0.1.0.vsix`

## Code Example: N-Queens Problem (Solved via Tensor Optimization)

TensorLogic can solve logical constraints as continuous optimization problems using tensor operations.
Below is an example program that solves the N-Queens problem using gradient descent.

```rust
fn main() {
    let N = 8; // Board size (8x8)
    let solutions_to_find = 5; // Number of solutions to find
    let mut found_count = 0;

    print("Finding "); print(solutions_to_find); println(" solutions for N-Queens...");

    while found_count < solutions_to_find {
        let lr = 0.5;
        let epochs = 2000;

        // 1. Initialize board probability distribution (random noise)
        let mut board = Tensor::randn([N, N], true);

        // Optimization loop
        for i in 0..epochs {
             let probs = board.softmax(1);

             // Constraint 1: Column constraints
             let col_sums = probs.sum(0);
             let col_loss = (col_sums - 1.0).pow(2).sum();

             // Constraint 2: Diagonal constraints (Tensor Comprehension)
             let anti_diag_sums = [k | k <- 0..(2 * N - 1), r <- 0..N, c <- 0..N, r + c == k { probs[r, c] }];
             let main_diag_sums = [k | k <- 0..(2 * N - 1), r <- 0..N, c <- 0..N, r - c + N - 1 == k { probs[r, c] }];

             let anti_diag_loss = (anti_diag_sums - 1.0).relu().pow(2).sum();
             let main_diag_loss = (main_diag_sums - 1.0).relu().pow(2).sum();
             
             let total_loss = col_loss + anti_diag_loss + main_diag_loss;

             // Early exit with break
             if total_loss.item() < 1e-4 {
                 break;
             }

             total_loss.backward();
             let g = board.grad();
             board = board - g * lr;
             board = board.detach();
             board.enable_grad();
        }

        // --- Result verification and display ---
        let probs = board.softmax(1);
        
        // Re-calculate loss to check
        let col_sums = probs.sum(0);
        let col_loss = (col_sums - 1.0).pow(2).sum();
        let anti_diag_sums = [k | k <- 0..(2 * N - 1), r <- 0..N, c <- 0..N, r + c == k { probs[r, c] }];
        let main_diag_sums = [k | k <- 0..(2 * N - 1), r <- 0..N, c <- 0..N, r - c + N - 1 == k { probs[r, c] }];
        let anti_diag_loss = (anti_diag_sums - 1.0).relu().pow(2).sum();
        let main_diag_loss = (main_diag_sums - 1.0).relu().pow(2).sum();
        let total_loss = col_loss + anti_diag_loss + main_diag_loss;

        if total_loss.item() < 1e-3 {
            found_count = found_count + 1;
            print("Solution #"); println(found_count);
            
            let mut rows = 0;
            while rows < N {
                let mut cols = 0;
                while cols < N {
                   if probs[rows, cols] > 0.5 {
                       print(" Q ");
                   } else {
                       print(" . ");
                   }
                   cols = cols + 1;
                }
                println("");
                rows = rows + 1;
            }
            println("----------------");
        }
    }
}
```

## Documentation

- [Advantages of TensorLogic](docs/tensor_logic_advantages.md)
- [Language Specification](docs/reference/language_spec.md)
- [Standard Library](docs/reference/standard_library.md)

# LICENSE

[MIT](LICENSE)

# References

- [Tensor Logic: The Language of AI](https://arxiv.org/abs/2510.12269)
- [Video: What is Tensor Logic? (Japanese)](https://www.youtube.com/watch?v=rkBLPYqPkP4)
