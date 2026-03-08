# Designziele der Tensor-Comprehensions

Die ursprüngliche Motivation war die Tensor-Kontraktion — die Reduktion bestimmter Dimensionen durch Summierung über spezifische Variablen.

## Beispiel: Matrixmultiplikation

$C_{i,k} = \sum_j A_{i,j} \cdot B_{j,k}$

```
[ i, k | { A[i, j] * B[j, k] } ]
```

- `i`, `k`: beibehaltene Indizes (Ausgabedimensionen)
- `j`: kontrahierter Index (automatisch summiert)

## Beziehung zur Einsteinschen Summenkonvention

| Operation | Tensor-Comprehension |
|-----------|---------------------|
| Matrixprodukt | `[ i, k \| { A[i,j] * B[j,k] } ]` |
| Spur | `[ \| { A[i,i] } ]` |
| Äußeres Produkt | `[ i, j \| { a[i] * b[j] } ]` |

## Einfluss der Haskell-Listencomprehensions

Entwicklung:
1. **Ursprüngliche Idee**: Kontraktion bestehender Tensoren
2. **Haskell-Einfluss**: Tensorerzeugung über Generatoren
3. **Aktuelle Form**: Unterstützt beide Anwendungsfälle
