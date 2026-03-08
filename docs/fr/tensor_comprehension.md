# Guide des compréhensions tensorielles

## Syntaxe

```rust
[ <indices> | <clauses> { <corps> } ]
```

## 1. Générateurs
```rust
let A = [i | i <- 0..5 { i }];     // Explicite
let B = [i | { A[i] * 2 }];        // Inférence implicite
```

## 2. Conditions (Filtrage)
```rust
let Evens = [i | i <- 0..5, (i % 2) == 0 { i }];
```

## 3. Réductions
Les variables définies dans `<clauses>` mais absentes de `<indices>` sont des **variables de réduction**. L'analyse de réduction implicite (convention de sommation d'Einstein) détecte automatiquement ces variables.

```rust
let dot = [i | i <- 0..1, k <- 0..N { A[k] * B[k] }];
```

## 4. Exemples

### Multiplication matricielle
```rust
let C = [i, j | i <- 0..M, j <- 0..P, k <- 0..N { A[i, k] * B[k, j] }];
```

### Extraction de la diagonale
```rust
let Diag = [i | i <- 0..N, j <- 0..N, i == j { Matrix[i, j] }];
```

## 5. Corps optionnel
```rust
let A = [i | i <- 0..5];                    // [0, 1, 2, 3, 4]
let grid = [i, j | i <- 0..N, j <- 0..M];  // Tenseur Shape(N, M, 2)
```
