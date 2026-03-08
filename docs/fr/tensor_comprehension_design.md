# Objectifs de conception des compréhensions tensorielles

La motivation initiale était la contraction de tenseurs — réduire certaines dimensions en sommant sur des variables spécifiques.

## Exemple : Multiplication matricielle

$C_{i,k} = \sum_j A_{i,j} \cdot B_{j,k}$

```
[ i, k | { A[i, j] * B[j, k] } ]
```

- `i`, `k` : indices conservés (dimensions de sortie)
- `j` : indice contracté (sommé automatiquement)

## Relation avec la convention de sommation d'Einstein

| Opération | Compréhension tensorielle |
|-----------|--------------------------|
| Produit matriciel | `[ i, k \| { A[i,j] * B[j,k] } ]` |
| Trace | `[ \| { A[i,i] } ]` |
| Produit extérieur | `[ i, j \| { a[i] * b[j] } ]` |

## Influence des compréhensions de liste Haskell

Évolution :
1. **Idée initiale** : Contraction de tenseurs existants
2. **Influence Haskell** : Génération de tenseurs via générateurs
3. **Forme actuelle** : Supporte les deux cas d'utilisation
