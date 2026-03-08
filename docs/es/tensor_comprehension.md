# Guía de comprensiones tensoriales

## Sintaxis
```rust
[ <índices> | <cláusulas> { <cuerpo> } ]
```

## 1. Generadores
```rust
let A = [i | i <- 0..5 { i }];     // Explícito
let B = [i | { A[i] * 2 }];        // Inferencia implícita
```

## 2. Condiciones (Filtrado)
```rust
let Evens = [i | i <- 0..5, (i % 2) == 0 { i }];
```

## 3. Reducciones
Las variables definidas en `<cláusulas>` pero ausentes de `<índices>` son **variables de reducción**. El análisis de reducción implícita (convención de suma de Einstein) detecta automáticamente estas variables.

```rust
let dot = [i | i <- 0..1, k <- 0..N { A[k] * B[k] }];
```

## 4. Ejemplos

### Multiplicación de matrices
```rust
let C = [i, j | i <- 0..M, j <- 0..P, k <- 0..N { A[i, k] * B[k, j] }];
```

## 5. Cuerpo opcional
```rust
let A = [i | i <- 0..5];                    // [0, 1, 2, 3, 4]
let grid = [i, j | i <- 0..N, j <- 0..M];  // Tensor Shape(N, M, 2)
```
