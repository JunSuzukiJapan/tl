# Leitfaden für Tensor-Comprehensions

## Syntax

```rust
[ <Indizes> | <Klauseln> { <Körper> } ]
```

## 1. Generatoren
```rust
let A = [i | i <- 0..5 { i }];     // Explizit
let B = [i | { A[i] * 2 }];        // Implizite Inferenz
```

## 2. Bedingungen (Filterung)
```rust
let Evens = [i | i <- 0..5, (i % 2) == 0 { i }];
```

## 3. Reduktionen
Variablen in `<Klauseln>`, die nicht in `<Indizes>` enthalten sind, sind **Reduktionsvariablen**. Die implizite Reduktionsanalyse (Einsteinsche Summenkonvention) erkennt diese automatisch.

```rust
let dot = [i | i <- 0..1, k <- 0..N { A[k] * B[k] }];
```

## 4. Beispiele

### Matrixmultiplikation
```rust
let C = [i, j | i <- 0..M, j <- 0..P, k <- 0..N { A[i, k] * B[k, j] }];
```

## 5. Optionaler Körper
```rust
let A = [i | i <- 0..5];                    // [0, 1, 2, 3, 4]
let grid = [i, j | i <- 0..N, j <- 0..M];  // Tensor Shape(N, M, 2)
```
