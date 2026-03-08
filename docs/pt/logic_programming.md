# Programação lógica em TensorLanguage (TL)

TL integra um poderoso motor de inferência lógica estilo Prolog diretamente em seu runtime. Isso permite definir bases de conhecimento, realizar raciocínio lógico e combinar os resultados com redes neurais.

## 1. Visão geral da sintaxe

### Fatos
```rust
father(alice, bob).       // "alice é pai de bob"
```

### Declarações de relações
```rust
relation parent(entity, entity);
```

### Regras
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

## 2. Saída simbólica

TL mapeia automaticamente nomes de entidades para IDs inteiros únicos. Ao exibir resultados, os IDs são resolvidos automaticamente para os nomes originais.

## 3. Escopo e organização de arquivos

Fatos e regras devem ser definidos no **escopo global**. O sistema de módulos (`mod`, `use`) permite a organização em múltiplos arquivos.

## 4. Integração com tensores

Os resultados de consultas são tensores TL padrão, utilizáveis diretamente em operações matemáticas e de redes neurais.

## 5. Negação e comparações aritméticas

### Negação
```rust
manager(X) :- employee(X), not(has_boss(X)).
```

TL suporta negação estratificada. Ciclos negativos são detectados e causam erro de compilação.

### Comparações aritméticas
```rust
adult(X) :- person(X, Age), Age >= 18.
```

Operadores suportados: `>`, `<`, `>=`, `<=`, `==`, `!=`
