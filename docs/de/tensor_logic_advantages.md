# Tensor Logic: Warum es mehr Probleme lösen kann als Tensoren allein

## Überblick

**Tensor Logic** integriert Tensorberechnung mit logischem Schlussfolgern und erweitert die Problemlösungsfähigkeiten um **logische Struktur**, **Constraint-Erfüllung** und **symbolisches Schlussfolgern**.

## Grenzen traditioneller Tensoren
- Diskrete Optimierung schwierig
- Logische Constraints nicht natürlich ausdrückbar
- Symbolisches Schlussfolgern nicht unterstützt

## Erweiterungen

| Logische Op. | Tensorausdruck |
|-------------|---------------|
| NOT(A) | `1.0 - A` |
| AND(A,B) | `A * B` |
| OR(A,B) | `A + B - A * B` |

## Neue Problemklassen
- **Kombinatorische Optimierung**: N-Queens, TSP, Graphfärbung, SAT
- **Inverse Probleme**: Inverses Game of Life, inverses Rendering
- **Strukturiertes Schlussfolgern**: Wissensgraph-Vervollständigung, Programmsynthese

## Vorteile

| Aspekt | Traditionelle Tensoren | Tensor Logic |
|--------|----------------------|--------------|
| Darstellung | Rein numerisch | Logik + numerisch |
| Constraints | Regularisierungsterme | Native Unterstützung |
| Diskrete Probleme | Schwer zu approximieren | Kontinuierliche Relaxation |
| Symbolik | Nicht unterstützt | In Tensoroperationen konvertiert |
