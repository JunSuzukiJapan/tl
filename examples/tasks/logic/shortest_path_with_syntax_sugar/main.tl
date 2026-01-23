mod facts;
use facts::*;

fn main() {
    println("--- Shortest Path in TL (Syntax Sugar Edition) ---");
    
    // Nodes: 4 nodes (A=0, B=1, C=2, D=3)
    let N = 4;
    let INF = 999.0;
    
    // Initialize distance matrix with Infinity
    let mut dist = Tensor::ones([N, N]) * INF;
    
    // Self-loops distance is 0
    for i in 0..N {
        dist[i, i] = 0.0;
    }
    
    // Define edges using tensor (weights are numeric, so we use tensor operations)
    dist[0, 1] = 1.0; // A -> B (cost 1.0)
    dist[1, 2] = 2.0; // B -> C (cost 2.0)
    dist[0, 2] = 5.0; // A -> C (cost 5.0)
    dist[2, 3] = 1.0; // C -> D (cost 1.0)
    
    // Query: Is there an edge from A to B?
    println("Query: edge(a, b)?");
    let res = ?edge(a, b);
    print("Result: ");
    println(res);
    
    println("");
    println("Initial Distance Matrix:");
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
    
    // Tropical Matrix Power (Floyd-Warshall equivalent)
    for iter in 0..N {
        let new_dist = [i, j, k | i <- 0..N, j <- 0..N, k <- 0..N {
             dist[i, k] + dist[k, j]
        }].min(2);
        
        dist = new_dist;
    }
    
    println("Final Shortest Path Distance Matrix:");
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
