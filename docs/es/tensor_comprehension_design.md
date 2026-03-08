# Objetivos de diseño de las comprensiones tensoriales

La motivación original fue la contracción de tensores — reducir ciertas dimensiones sumando sobre variables específicas.

## Ejemplo: Multiplicación de matrices

$C_{i,k} = \sum_j A_{i,j} \cdot B_{j,k}$

```
[ i, k | { A[i, j] * B[j, k] } ]
```

## Relación con la convención de suma de Einstein

| Operación | Comprensión tensorial |
|-----------|----------------------|
| Producto matricial | `[ i, k \| { A[i,j] * B[j,k] } ]` |
| Traza | `[ \| { A[i,i] } ]` |
| Producto exterior | `[ i, j \| { a[i] * b[j] } ]` |

## Influencia de las comprensiones de listas de Haskell

Evolución:
1. **Idea original**: Contracción de tensores existentes
2. **Influencia de Haskell**: Generación de tensores mediante generadores
3. **Forma actual**: Soporta ambos casos de uso
