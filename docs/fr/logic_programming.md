# Programmation logique dans TensorLanguage (TL)

TL intègre un puissant moteur d'inférence logique de style Prolog directement dans son runtime. Cela permet de définir des bases de connaissances, d'effectuer des raisonnements logiques et de combiner les résultats avec des réseaux neuronaux.

## 1. Aperçu de la syntaxe

### Faits
```rust
father(alice, bob).       // "alice est le père de bob"
```

### Déclarations de relations
```rust
relation parent(entity, entity);
```

### Règles
```rust
grandparent(x, z) :- father(x, y), father(y, z).
ancestor(x, y) :- father(x, y).
ancestor(x, y) :- father(x, z), ancestor(z, y).
```

### Requêtes
```rust
let is_father = ?father(alice, bob); 
let children = ?father(alice, $child);
```

## 2. Sortie symbolique

TL mappe automatiquement les noms d'entités vers des ID entiers uniques. Les résultats affichent automatiquement les noms originaux.

## 3. Portée et organisation des fichiers

Les faits et règles doivent être définis à la **portée globale**. Le système de modules (`mod`, `use`) permet l'organisation en fichiers multiples.

## 4. Intégration avec les tenseurs

Les résultats de requêtes sont des tenseurs TL standard, utilisables directement dans les calculs mathématiques et neuronaux.

## 5. Négation et comparaisons arithmétiques

### Négation
```rust
manager(X) :- employee(X), not(has_boss(X)).
```

TL supporte la négation stratifiée. Les cycles négatifs sont détectés et causent une erreur de compilation.

### Comparaisons arithmétiques
```rust
adult(X) :- person(X, Age), Age >= 18.
```

Opérateurs supportés : `>`, `<`, `>=`, `<=`, `==`, `!=`
