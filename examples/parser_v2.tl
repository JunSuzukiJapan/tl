tensor A: Tensor<f32, 2> = [[1.0, 2.0], [3.0, 4.0]]
tensor B: Tensor<f32, 2> = [[5.0, 6.0], [7.0, 8.0]]

relation Edge(i64, i64)
relation Path(i64, i64)

rule Path(x, y) <- Edge(x, y)
rule Path(x, y) <- Edge(x, z), Path(z, y)

query Path(1, 2)
