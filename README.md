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

> [!WARNING]
> **Metal users**: Long-running loops may cause RSS (memory) growth due to Metal driver behavior, not TL memory leaks. For stable memory usage, use `TL_DEVICE=cpu`. See [Metal RSS Growth Notes](docs/dev/METAL_RSS_GROWTH_NOTES.md) for details.


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

    println("Finding {} solutions for N-Queens...", solutions_to_find);

    while found_count < solutions_to_find {
        let lr = 0.5;
        let epochs = 1500; // Fast retry strategy

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

             // Check periodically to improve performance
             if i % 100 == 0 {
                  // Early exit: Stop when we have N queens (prob > 0.5)
                 let current_queens = [r, c | r <- 0..N, c <- 0..N { 
                     if probs[r, c] > 0.5 { 1.0 } else { 0.0 } 
                 }].sum().item();

                 if current_queens == 8.0 {
                     break;
                 }
             }

             total_loss.backward();
             let g = board.grad();
             board = board - g * lr;
             board = board.detach();
             board.enable_grad();
        }

        // --- Result verification and display ---
        let probs = board.softmax(1);
        
        // Count queens and validate
        let mut valid = true;
        let mut total_queens = 0;
        let mut rows = 0;
        while rows < N {
            let mut queen_count = 0;
            let mut cols = 0;
            while cols < N {
               if probs[rows, cols] > 0.5 {
                   queen_count = queen_count + 1;
                   total_queens = total_queens + 1;
               }
               cols = cols + 1;
            }
            if queen_count != 1 {
                valid = false;
            }
            rows = rows + 1;
        }

        if valid && total_queens == N {
            found_count = found_count + 1;
            println("Solution #{}", found_count);
            
            let mut rows2 = 0;
            while rows2 < N {
                let mut cols2 = 0;
                while cols2 < N {
                   if probs[rows2, cols2] > 0.5 {
                       print(" Q ");
                   } else {
                       print(" . ");
                   }
                   cols2 = cols2 + 1;
                }
                println("");
                rows2 = rows2 + 1;
            }
            println("----------------");
        }
    }
}
```

## Code Example: Neuro-Symbolic AI (Spatial Reasoning)

The following example demonstrates how to fuse **Vision** (Tensor) and **Reasoning** (Logic).
It detects spatial relationships from raw coordinates and infers hidden facts using logic rules (Transitivity).

```rust
// 1. Symbolic Knowledge Base (The "Mind")
// Define concepts (objects)
object(1, cup).
object(2, box).
object(3, table).

// Recursive Logic Rule: Transitivity
// If X is on Y, then X is stacked on Y.
// If X is on Z, and Z is stacked on Y, then X is stacked on Y.
stacked_on(top, bot) :- on_top_of(top, bot).
stacked_on(top, bot) :- on_top_of(top, mid), stacked_on(mid, bot).


fn main() {
    println("--- Neuro-Symbolic AI Demo: Spatial Reasoning ---");

    // 2. Perception (Tensor Data)
    // Simulated bounding boxes detected by a vision model [x, y, w, h]
    let cup_bbox   = [10.0, 20.0, 4.0, 4.0];  // Cup is high (y=20)
    let box_bbox   = [10.0, 10.0, 10.0, 10.0]; // Box is middle (y=10)
    let table_bbox = [10.0, 0.0, 50.0, 10.0];  // Table is low (y=0)

    println("\n[Visual Scene]");
    println("   [Cup]   (y=20)");
    println("     |     ");
    println("   [Box]   (y=10)");
    println("     |     ");
    println("[=========] (Table y=0)");
    println("");

    // 3. Neuro-Symbolic Fusion
    // Detect spatial relationships from coordinates and inject them as facts.

    // Detect: Cup on Box?
    if cup_bbox[1] > box_bbox[1] {
        on_top_of(1, 2). 
        println("Detected: Cup is on_top_of Box");
    }

    // Detect: Box on Table?
    if box_bbox[1] > table_bbox[1] {
        on_top_of(2, 3).
        println("Detected: Box is on_top_of Table");
    }

    // 4. Logical Inference
    println("\n[Logical Inference]");
    println("Querying: Is Cup stacked on Table? (?stacked_on(1, 3))");

    // Execute Logic Query
    // We never explicitly saw "Cup on Table", but logic infers it via "Box".
    let res = ?stacked_on(1, 3);

    if res.item() > 0.5 {
        println("Result: YES (Confidence: {:.4})", res.item());
        println("Reasoning: Cup -> Box -> Table (Transitivity)");
    } else {
        println("Result: NO");
    }
}
```

## Documentation

- [Advantages of TensorLogic](docs/tensor_logic_advantages.md)
- [Tensor Comprehension Design](docs/tensor_comprehension_design.md)
- [Language Specification](docs/reference/language_spec.md)
- [API Reference](docs/functions_and_types.md)
- [Logic Programming](docs/logic_programming.md)

# LICENSE

[MIT](LICENSE)

# References

- [Tensor Logic: The Language of AI](https://arxiv.org/abs/2510.12269)
- [Video: What is Tensor Logic? (Japanese)](https://www.youtube.com/watch?v=rkBLPYqPkP4)

## Development

- [Testing Guide](docs/dev/testing.md) - How to run tests and verification scripts.
