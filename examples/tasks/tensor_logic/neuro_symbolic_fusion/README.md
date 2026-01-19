# Neuro-Symbolic Fusion Examples

This directory contains examples demonstrating the core strength of TensorLogic (TL): the seamless fusion of **Neural Networks** (Tensor-based numerical computation) and **Symbolic Logic** (Datalog-based rule reasoning).

By combining these two paradigms in a single language, you can build systems that leverage the learning capabilities of AI with the interpretability and strict constraints of logic.

## 1. Recommendation System (`recommendation.tl`)

A hybrid recommendation engine that filters items using symbolic business rules while scoring them using tensor-based collaborative filtering.

-   **Logic Component**: Defines hard constraints.
    -   User attributes (Regular, Premium, Student) determine allowed item categories.
    -   Example: "Students cannot buy Electronics".
-   **Tensor Component**: Calculates soft preferences.
    -   Computes dot-product similarity between User Embeddings and Item Embeddings using `matmul`.
-   **Fusion**: The script queries the logic engine (`?can_view(user, item)`) to filter candidate items, then uses tensor scores to rank the valid ones.

## 2. Spatial Reasoning (`spatial.tl`)

Demonstrates how to derive high-level semantic relationships from low-level perceptual data (coordinates).

-   **Perception (Procedural/Tensor)**: Simulates an AI vision system that detects object bounding boxes. It compares coordinates to determine basic geometric relations like `on_top_of(cup, box)`.
-   **Reasoning (Logic)**: Defines recursive rules to infer transitive relationships.
    -   Rule: If A is on B, and B is on C, then A is `stacked_on` C.
-   **Fusion**: Raw numerical data is processed into base facts, which are then used by the logic engine to infer complex structural knowledge about the scene.

## 3. Smart Access Control (`access_control.tl`)

A security system combining probabilistic identity verification with deterministic access policies.

-   **Neural Auth (Tensor)**: Simulates Facial Recognition by comparing a camera input vector against a database of face embeddings. Calculates similarity scores to identify a person.
-   **Policy Enforcement (Logic)**: Implements Role-Based Access Control (RBAC).
    -   Roles: Admin, Engineer, Guest.
    -   Rules: Who can enter specific zones (ServerRoom, Office, Lobby).
-   **Fusion**: The system uses Tensor operations to determine *Reference Identity* ("Who is this?") and Logic to determine *Authorization* ("Can they enter?").

## Running the Examples

You can run these examples using the `tl` compiler:

```bash
cargo run -- examples/tasks/tensor_logic/neuro_symbolic_fusion/recommendation.tl
cargo run -- examples/tasks/tensor_logic/neuro_symbolic_fusion/spatial.tl
cargo run -- examples/tasks/tensor_logic/neuro_symbolic_fusion/access_control.tl
```
