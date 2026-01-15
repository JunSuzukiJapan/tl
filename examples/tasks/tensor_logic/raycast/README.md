# Differentiable Ray Caster 2D

This example demonstrates a simple differentiable ray caster using soft sampling for differentiability.

## Concept

- **Scene**: A 2D grid `[N, N]` representing intensity values.
- **Camera**: Located at the center, casting rays in all directions.
- **Soft Sampling**: Uses Gaussian-weighted sampling instead of hard indexing to enable gradient flow.
- **Rendering**: Produces a 1D image by accumulating samples along each ray.

## Implementation Details

- **Gaussian Weights**: For each sample position `(px, py)`, compute weights for all grid cells using `exp(-dist^2 / sigma^2)`.
- **Weighted Sample**: `sum(weights * scene) / (sum(weights) + epsilon)`.
- **Optimization**: Gradient descent to minimize MSE between rendered and target images.

## Usage

```bash
tl run examples/tasks/tensor_logic/raycast/raycast.tl
```

## Notes

- The current implementation uses a small grid (8x8) and limited rays for faster iteration.
- Gradient flow through soft sampling enables scene reconstruction from rendered images.
