# Markov Logic Network (MLN) in Tensor Logic

This example demonstrates a simplified implementation of a Markov Logic Network (MLN) using **Tensor Logic**. It showcases how logical rules can be expressed as tensor operations and how probabilistic inference can be performed using continuous relaxation and gradient descent.

## Overview

Markov Logic Networks combine first-order logic with probabilistic graphical models. In this implementation, we define a small knowledge base with the following rules:

1.  **Smokes(x) => Cancer(x)**: Smokers are more likely to get cancer.
2.  **Friends(x, y) => (Smokes(x) <=> Smokes(y))**: Friends tend to have similar smoking habits.

We treat the truth values of predicates (`Smokes`, `Cancer`) as continuous probabilities $[0, 1]$ (logits relaxed via sigmoid) and maximize the total weighted satisfaction (energy) of the rules using Gradient Descent.

## Key Concepts

-   **Predicates as Tensors**: `Smokes` and `Cancer` are tensors of shape `[N]`.
-   **Logic as Algebra**:
    -   Implication ($A \Rightarrow B$) is modeled as $1 - A + AB$ (Reichenbach implication).
    -   Equivalence ($A \Leftrightarrow B$) is modeled using similarity measures.
-   **Differentiable Inference**: Instead of discrete sampling (like MCMC), we use `requires_grad=true` on the predicate logits and optimize them directly to satisfy the logical constraints.

## How to Run

```bash
cargo run --release -- examples/tasks/tensor_logic/mln/mln.tl
```

## Expected Output

The program performs inference over `N=20` individuals. You will see the loss (negative energy) decreasing as the system converges to a state that satisfies the logical rules and evidence.

```text
Initializing MLN for N=20
Generating network... Done.
Starting inference...
Epoch 0 Loss: -91.32773
Epoch 50 Loss: -125.97
...
```
