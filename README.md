# TensorLogic (TL)

A tensor logic programming language with first-class tensor support, JIT-compiled to LLVM.

## Features
- **Tensor Operations**: `tensor<f32>[128, 128]`, `matmul`, `topk`, etc. via Candle.
- **Logic Programming**: Datalog-style rules with tensor integration.
- **Hybrid Execution**: Logic terms can access tensor data (`data[i]`).
- **JIT Compilation**: High-performance execution using LLVM (Inkwell).
- **GPU Support**: Metal (macOS) backend supported. CUDA support is planned for the future.
- **Optimization**: Aggressive JIT optimization and fast logic inference.

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
