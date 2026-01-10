# TensorLogic (TL) Examples: Chatbot

This directory contains an interactive chatbot application implemented in TensorLogic (TL).

## 1. TinyLlama Chatbot (`tinyllama/chatbot.tl`)

A fully-featured chatbot utilizing the TinyLlama-1.1B model. It implements modern LLM inference components like KV Cache and RoPE (Rotary Positional Embedding) directly in the TL language.

### Key Features

- **Token Streaming**: Real-time display of generated tokens using the `print` function.
- **KV Cache**: Stores previous Key/Value states to accelerate inference.
- **ChatML Format**: Optimized prompt templates using `<|user|>` and `<|assistant|>` tags.
- **GGUF Support**: Native loading of quantized GGUF model files.

### Setup and Usage

1.  **Prepare Model Files**:
    Ensure the following files are located in `~/.llm/models/`:
    - `tinyllama-1.1b-chat-q4_0.gguf`
    - `tokenizer.json`

2.  **Run the Chatbot**:
    ```bash
    # Standard CPU execution
    cargo run --release --bin tl -- run examples/apps/tinyllama/chatbot.tl

    # With Metal (Apple Silicon GPU) acceleration
    cargo run --release --features metal --bin tl -- run examples/apps/tinyllama/chatbot.tl
    ```

3.  **Interaction**:
    - Type your prompt at the `User>` prompt.
    - Type `exit` or `quit` to terminate the session.

## 2. Llama 3 Chatbot (`llama3/chatbot_llama3.tl`)

A chatbot using the Llama 3.1 8B Instruct model.

### Key Features
- **Llama 3 Prompting**: Uses standard Llama 3 instruct format.
- **Improved Sampling**: Supports Top-P sampling.
- **Memory Safety**: Manages KV cache explicitly to prevent leaks.

### Usage
```bash
cargo run --release --bin tl -- run examples/apps/llama3/chatbot_llama3.tl
```

### Technical Notes

- **Argmax Sampling**: The current implementation always selects the token with the highest probability.
- **FFI Integration**: Uses Rust-based runtime functions for high-speed matrix math and tokenizer operations.
- **Modular Design**: Uses custom structs like `Attention`, `Block`, and `MLP` for a clean Transformer implementation.
