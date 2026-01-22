fn main() {
    println("--- Prolog-like Logic in TL (Matrix Logic Approach) ---");
    
    // 1. Entity Definition (ID mapping)
    // 0: Alice, 1: Bob, 2: Charlie, 3: Diana
    let N = 4;
    
    // 2. Facts: Parent relationship
    // parent[i, j] = 1.0 means "i is parent of j"
    let mut parent = Tensor::zeros([N, N]);
    
    // Add Facts: Relations
    parent.set(0, 1, 1.0); // Alice -> Bob
    parent.set(1, 2, 1.0); // Bob -> Charlie
    parent.set(2, 3, 1.0); // Charlie -> Diana
    
    println("Parent Matrix:");
    // Print Matrix Logic Inlined
    for r in 0..N {
        for c in 0..N {
            if parent[r, c] > 0.0 {
                print(" 1 ");
            } else {
                print(" . ");
            }
        }
        println("");
    }
    println("----------------");
    
    // 3. Rules: GrandParent
    // Prolog: grandparent(X, Y) :- parent(X, Z), parent(Z, Y).
    // Logic:  Trace Parent relation twice
    // TL:     Matrix Multiplication
    let grandparent = parent.matmul(parent);
    
    println("GrandParent Matrix (calculated via matmul):");
    // Print Matrix Logic Inlined
    for r in 0..N {
        for c in 0..N {
            if grandparent[r, c] > 0.0 {
                print(" 1 ");
            } else {
                print(" . ");
            }
        }
        println("");
    }
    println("----------------");
    
    // 4. Recursive Rules: Ancestor
    // Prolog: ancestor(X, Y) :- parent(X, Y).
    //         ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
    // TL:     Transitive Closure = P + P^2 + P^3 + ...
    
    let mut ancestor = parent.clone();
    let mut current_path = parent.clone();
    
    // Search up to max depth (graph diameter)
    for i in 0..N {
        // Calculate next generation path
        current_path = current_path.matmul(parent);
        
        // Add to existing Ancestor relations (addition implies OR)
        ancestor = ancestor + current_path;
    }
    
    // Normalize: > 0 means True
    let ancestor_bool = [i, j | i <- 0..N, j <- 0..N, ancestor[i, j] > 0.0 { 1.0 }];

    println("Ancestor Matrix (Transitive Closure):");
    // Print Matrix Logic Inlined
    for r in 0..N {
        for c in 0..N {
            if ancestor_bool[r, c] > 0.0 {
                print(" 1 ");
            } else {
                print(" . ");
            }
        }
        println("");
    }
    println("----------------");
    
    // Query: Descendants of Alice (0)
    println("Descendants of Alice (0):");
    // slice(start, length) gets rows. We want row 0.
    let alice_descendants = ancestor_bool.slice(0, 1).reshape([N]); 
    
    for c in 0..N {
        if ancestor_bool[0, c] > 0.0 {
             print("Found descendant ID: ");
             println(c);
        }
    }
}
