# Llama 3 Chatbot (`chatbot_llama3.tl`)

A chatbot application utilizing the Llama 3.1 8B Instruct model.

## Key Features

- **Llama 3 Prompting**: Uses standard Llama 3 instruct format (`<|begin_of_text|><|start_header_id|>...`).
- **Improved Sampling**: Supports Top-P sampling.
- **Memory Safety**: Manages KV cache explicitly to prevent leaks.
- **Stop Conditions**: Detects Llama 3 specific stop tokens.

## Setup and Usage

1.  **Prepare Model Files**:
    Ensure the following files are located in `~/.llm/models/`:
    - `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`
    - `llama3_tokenizer.json`
    (Adjust paths in `chatbot_llama3.tl` if necessary)

2.  **Run the Chatbot**:
    ```bash
    cargo run --release --bin tl -- run examples/apps/llama3/chatbot_llama3.tl
    ```

3.  **Interaction**:
    - Type your prompt at the `User>` prompt.
    - Type `exit` or `quit` to terminate the session.
