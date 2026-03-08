# Logikprogrammierung in TensorLanguage (TL)

TL integriert eine leistungsstarke Prolog-ähnliche Logik-Inferenz-Engine direkt in seine Laufzeitumgebung. Dies ermöglicht die Definition von Wissensbasen, logische Schlussfolgerungen und die nahtlose Kombination mit neuronalen Netzen.

## 1. Syntaxüberblick

### Fakten
```rust
father(alice, bob).       // "alice ist der Vater von bob"
```

### Relationsdeklarationen
```rust
relation parent(entity, entity);
```

### Regeln
```rust
grandparent(x, z) :- father(x, y), father(y, z).
ancestor(x, y) :- father(x, y).
ancestor(x, y) :- father(x, z), ancestor(z, y).
```

### Abfragen
```rust
let is_father = ?father(alice, bob);
let children = ?father(alice, $child);
```

## 2. Symbolische Ausgabe

TL ordnet Entitätsnamen automatisch eindeutigen Integer-IDs zu. Bei der Anzeige werden IDs automatisch in die ursprünglichen Namen aufgelöst.

## 3. Geltungsbereich und Dateiorganisation

Fakten und Regeln müssen im **globalen Geltungsbereich** definiert werden. Das Modulsystem (`mod`, `use`) ermöglicht die Organisation in mehreren Dateien.

## 4. Integration mit Tensoren

Abfrageergebnisse sind Standard-TL-Tensoren und können direkt in mathematischen und neuronalen Netzwerkoperationen verwendet werden.

## 5. Negation und arithmetische Vergleiche

### Negation
```rust
manager(X) :- employee(X), not(has_boss(X)).
```

TL unterstützt stratifizierte Negation. Negative Zyklen werden erkannt und verursachen einen Kompilierungsfehler.

### Arithmetische Vergleiche
```rust
adult(X) :- person(X, Age), Age >= 18.
```

Unterstützte Vergleichsoperatoren: `>`, `<`, `>=`, `<=`, `==`, `!=`
