// Neuro-Symbolic Spatial Reasoning
// Demonstrates deducing semantic relationships from raw coordinate tensors

// 2. Logic-based Spatial Reasoning
// --------------------------------

// Define object properties (Symbolic)
object(1, cup).
object(2, box).
object(3, table).
object(4, bird).

// Base Spatial Relations (Simulating output of Tensor/Geometry engine)
on_top_of(1, 2). // Cup on Box
on_top_of(2, 3). // Box on Table

// Recursive Rule: Stack
// Logic handles the transitive closure
stacked_on(top, bot) :- on_top_of(top, bot).
stacked_on(top, bot) :- on_top_of(top, mid), stacked_on(mid, bot).

fn main() {
    println("--- Neuro-Symbolic Spatial Reasoning ---");

    // 1. Scene Perception (Tensor Data)
    // --------------------------------
    let cup_bbox = [10.0, 20.0, 4.0, 4.0];
    let box_bbox = [10.0, 10.0, 10.0, 10.0];
    let table_bbox = [10.0, 0.0, 50.0, 10.0];

    println("\n[Analysing Spatial Relationships]");

    // Procedural Spatial Checking 
    let cup_y = 20.0; let box_y = 10.0;
    
    if cup_y > box_y { 
        println("Visually: Cup is above Box"); 
    }
    
    println("Logic inference (Transitivity):");
    let stacked_res = ?stacked_on(1, 3);
    if stacked_res.item() > 0.5 { // 1->2->3 defined via facts
         println("Cup is stacked on Table (derived via Box)");
    }
}
