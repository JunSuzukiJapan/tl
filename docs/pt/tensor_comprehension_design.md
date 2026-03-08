# Objetivos de design das compreensões tensoriais

A motivação original foi a contração de tensores — reduzir certas dimensões somando sobre variáveis específicas.

## Exemplo: Multiplicação de matrizes
$C_{i,k} = \sum_j A_{i,j} \cdot B_{j,k}$

```
[ i, k | { A[i, j] * B[j, k] } ]
```

## Relação com a convenção de soma de Einstein

| Operação | Compreensão tensorial |
|----------|----------------------|
| Produto matricial | `[ i, k \| { A[i,j] * B[j,k] } ]` |
| Traço | `[ \| { A[i,i] } ]` |
| Produto externo | `[ i, j \| { a[i] * b[j] } ]` |

## Influência das compreensões de lista do Haskell

Evolução:
1. **Ideia original**: Contração de tensores existentes
2. **Influência do Haskell**: Geração de tensores via geradores
3. **Forma atual**: Suporta ambos os casos de uso
