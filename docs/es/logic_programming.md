# Programación lógica en TensorLanguage (TL)

TL integra un potente motor de inferencia lógica al estilo Prolog directamente en su runtime. Esto permite definir bases de conocimiento, realizar razonamiento lógico y combinar los resultados con redes neuronales.

## 1. Resumen de sintaxis

### Hechos
```rust
father(alice, bob).       // "alice es el padre de bob"
```

### Declaraciones de relaciones
```rust
relation parent(entity, entity);
```

### Reglas
```rust
grandparent(x, z) :- father(x, y), father(y, z).
ancestor(x, y) :- father(x, y).
ancestor(x, y) :- father(x, z), ancestor(z, y).
```

### Consultas
```rust
let is_father = ?father(alice, bob);
let children = ?father(alice, $child);
```

## 2. Salida simbólica

TL mapea automáticamente los nombres de entidades a IDs enteros únicos. Al mostrar resultados, los IDs se resuelven automáticamente a sus nombres originales.

## 3. Ámbito y organización de archivos

Los hechos y reglas deben definirse en el **ámbito global**. El sistema de módulos (`mod`, `use`) permite la organización en múltiples archivos.

## 4. Integración con tensores

Los resultados de consultas son tensores TL estándar, utilizables directamente en operaciones matemáticas y de redes neuronales.

## 5. Negación y comparaciones aritméticas

### Negación
```rust
manager(X) :- employee(X), not(has_boss(X)).
```

TL soporta negación estratificada. Los ciclos negativos se detectan y causan un error de compilación.

### Comparaciones aritméticas
```rust
adult(X) :- person(X, Age), Age >= 18.
```

Operadores soportados: `>`, `<`, `>=`, `<=`, `==`, `!=`
