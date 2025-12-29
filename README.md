# TensorLogic (TL)

A probabilistic logic programming language with first-class tensor support, JIT-compiled to LLVM.

## Features
- **Tensor Operations**: `tensor<f32>[128, 128]`, `matmul`, `topk`, etc. via Candle.
- **Logic Programming**: Datalog-style rules with tensor integration.
- **Hybrid Execution**: Logic terms can access tensor data (`data[i]`).
- **JIT Compilation**: High-performance execution using LLVM (Inkwell).
- **GPU Support**: Metal (macOS) and CUDA backends.
- **Optimization**: Aggressive JIT optimization and fast logic inference.

## Build & Run
```bash
# Run a specific example
cargo run -- run examples/hybrid_test.tl

# Use GPU (Metal on macOS)
cargo run --features metal -- run examples/gpu_test.tl --device metal
```

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
