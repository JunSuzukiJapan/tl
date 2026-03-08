# Guia de compreensões tensoriais

## Sintaxe
```rust
[ <índices> | <cláusulas> { <corpo> } ]
```

## 1. Geradores
```rust
let A = [i | i <- 0..5 { i }];     // Explícito
let B = [i | { A[i] * 2 }];        // Inferência implícita
```

## 2. Condições (Filtragem)
```rust
let Evens = [i | i <- 0..5, (i % 2) == 0 { i }];
```

## 3. Reduções
Variáveis definidas em `<cláusulas>` mas ausentes de `<índices>` são **variáveis de redução**. A análise de redução implícita (convenção de soma de Einstein) detecta automaticamente essas variáveis.

```rust
let dot = [i | i <- 0..1, k <- 0..N { A[k] * B[k] }];
```

## 4. Exemplos

### Multiplicação de matrizes
```rust
let C = [i, j | i <- 0..M, j <- 0..P, k <- 0..N { A[i, k] * B[k, j] }];
```

## 5. Corpo opcional
```rust
let A = [i | i <- 0..5];                    // [0, 1, 2, 3, 4]
let grid = [i, j | i <- 0..N, j <- 0..M];  // Tensor Shape(N, M, 2)
```
