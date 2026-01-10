# Task: Associative Recall

The Associative Recall task is a fundamental test for neural networks' ability to retrieve information based on context.

## Task Description

The model is presented with a sequence of key-value pairs followed by a query key:
`Input: K1 V1 K2 V2 ... Kn Vn K_query -> Output: V_query`

To succeed, the model must:
1.  Learn to associate keys with their subsequent values.
2.  Store these associations in its internal state or attention weights.
3.  Correctly "attend" back to the position where the query key was last seen and output the associated value.

## Implementation Details

- **Model**: Simplified GPT with 1 Transformer Block.
- **Hidden Dimension**: 64.
- **Vocabulary Size**: 30.
- **Training Data**: Generated on-the-fly using a pseudo-random seed to ensure variety.

## Usage

To train the model:
```bash
cargo run --release --bin tl -- run examples/tasks/recall/train_recall.tl
```

The weights will be saved to `recall_weights.safetensors`. You can verify retrieval accuracy using the corresponding inference script.
