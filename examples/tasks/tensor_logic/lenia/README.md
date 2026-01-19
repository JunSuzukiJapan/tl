# Lenia / Continuous Cellular Automata

This example demonstrates a simple implementation of **Lenia** (Continuous Cellular Automata) using TensorLogic.

It uses:
*   **Tensor Operations**: `conv2d` for neighborhood sensing.
*   **Element-wise Math**: `exp`, `pow` for the growth function.
*   **Time Integration**: Euler method for updating the state.
*   **Clamping**: To maintain state values between 0.0 and 1.0.

## How to Run

```bash
cargo run --release -- examples/tasks/tensor_logic/lenia/lenia.tl
```

The script will run for 100 steps and print the total mass of the grid at intervals to show activity.
