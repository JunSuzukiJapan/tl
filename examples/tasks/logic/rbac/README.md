# RBAC (Logic-only) Example

This example demonstrates role-based access control (RBAC) using TL's logic programming features.
Tensor usage is minimal: query results are returned as tensors for printing.

## Files
- `facts.tl`: Users, roles, inheritance, and permissions
- `main.tl`: Rules and queries

## Run
```bash
tl main.tl
```

## What to look for
- `role_effective` shows role inheritance.
- `allowed` combines role-based permissions and direct grants.
- Output prints a readable user -> permission table.
