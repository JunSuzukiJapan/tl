fn main() {
    println("--- Shortest Path in TL (Tropical Matrix Logic) ---");
    
    // Nodes: 4 nodes
    // 0: A
    // 1: B
    // 2: C
    // 3: D
    let N = 4;
    
    // Adjacency Matrix with Weights (Distances)
    // We use a large number (999.0) to represent Infinity (No edge)
    // In min-plus algebra, "Infinity" is the identity element for 'min' (addition),
    // and "0" is the identity element for '+' (multiplication).
    
    let INF = 999.0;
    
    // Initialize distance matrix with Infinity
    let mut dist = Tensor::ones([N, N]) * INF;
    
    // Self-loops distance is 0
    // dist[i, i] = 0
    for i in 0..N {
        dist.set(i, i, 0.0);
    }
    
    // Define Edges (Directed Graph)
    // A -> B (cost 1.0)
    dist.set(0, 1, 1.0);
    
    // B -> C (cost 2.0)
    dist.set(1, 2, 2.0);
    
    // A -> C (cost 5.0) - Slower direct path
    dist.set(0, 2, 5.0);
    
    // C -> D (cost 1.0)
    dist.set(2, 3, 1.0);
    
    println("Initial Distance Matrix (INF=999.0):");
    // Print logic inlined
    for r in 0..N {
        for c in 0..N {
            let val = dist[r, c];
            if val > 900.0 {
                print(" INF ");
            } else {
                print(" ");
                print(val);
                print(" ");
            }
        }
        println("");
    }
    println("----------------");
    
    // Algorithm: Tropical Matrix Power (Floyd-Warshall equivalent)
    // Standard Matrix Mul: C[i, j] = Sum_k ( A[i, k] * B[k, j] )
    // Tropical Matrix Mul: C[i, j] = Min_k ( A[i, k] + B[k, j] )
    // We iterate N times (diameter of graph)
    
    for iter in 0..N {
        // We use Tensor Comprehension to define the custom matrix operation
        // NewDist[i, j] = min_k ( Dist[i, k] + Dist[k, j] )
        
        let new_dist = [i, j, k | i <- 0..N, j <- 0..N, k <- 0..N {
             dist[i, k] + dist[k, j]
        }].min(2); // Reduction along k axis (dim 2)
        
        dist = new_dist;
    }
    
    println("Final Shortest Path Distance Matrix:");
    // Print logic inlined
    for r in 0..N {
        for c in 0..N {
            let val = dist[r, c];
            if val > 900.0 {
                print(" INF ");
            } else {
                print(" ");
                print(val);
                print(" ");
            }
        }
        println("");
    }
    println("----------------");
    
    // Verify specific paths
    let d_ac = dist[0, 2];
    print("Shortest Path A->C (Expected 3.0): ");
    println(d_ac);
    
    let d_ad = dist[0, 3];
    print("Shortest Path A->D (Expected 4.0): ");
    println(d_ad);
}
