# Tensor Logic : Pourquoi il peut résoudre plus de problèmes que les tenseurs seuls

## Aperçu

**Tensor Logic** intègre le calcul tensoriel au raisonnement logique, ajoutant la **structure logique**, la **satisfaction de contraintes** et le **raisonnement symbolique** aux capacités numériques traditionnelles.

## Limitations des tenseurs traditionnels
- Optimisation discrète difficile
- Expression naturelle des contraintes impossible
- Raisonnement symbolique non supporté
- Problèmes inverses difficiles

## Extensions de Tensor Logic

| Op. logique | Expression tensorielle |
|------------|----------------------|
| NOT(A) | `1.0 - A` |
| AND(A,B) | `A * B` |
| OR(A,B) | `A + B - A * B` |

## Nouvelles classes de problèmes

- **Optimisation combinatoire** : N-Queens, TSP, coloration de graphes, SAT
- **Problèmes inverses** : Jeu de la vie inverse, rendu inverse, apprentissage de circuits
- **Raisonnement structuré** : Complétion de graphes de connaissances, synthèse de programmes

## Avantages résumés

| Aspect | Tenseurs traditionnels | Tensor Logic |
|--------|----------------------|--------------|
| Représentation | Purement numérique | Logique + numérique |
| Contraintes | Termes de régularisation | Support natif |
| Problèmes discrets | Approximation difficile | Relaxation continue |
| Raisonnement symbolique | Non supporté | Converti en opérations tensorielles |
