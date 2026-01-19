// Neuro-Symbolic Recommendation System
// Demonstrates fusion of Tensor-based Collaborative Filtering and Logic-based Business Rules

// 1. Symbolic Logic: Category & Role Matching
// -------------------------------------------

// Item Definitions: id, category
item_cat(1, electronics).
item_cat(2, electronics).
item_cat(3, books).
item_cat(4, books).
item_cat(5, toys).

// User Definitions: id, type
user_type(1, regular).
user_type(2, premium).
user_type(3, student).

// Access Rules (Symbolic)
allowed_cat(premium, electronics).
allowed_cat(premium, books).
allowed_cat(premium, toys).

allowed_cat(regular, electronics).
allowed_cat(regular, books).
allowed_cat(regular, toys).

allowed_cat(student, books).
allowed_cat(student, toys).

// Main Access Rule
// Variables: u=User, i=Item, t=Type, c=Category
can_view(u, i) :-
    user_type(u, t),
    item_cat(i, c),
    allowed_cat(t, c).

// 2. Numeric/Tensor Logic (Procedural)
// ------------------------------------
fn main() {
    println("--- Neuro-Symbolic Recommendation System ---");

    // Static Data for Numeric Constraints
    let user_data = [
        25.0, 1000.0, // User 1
        35.0, 2000.0, // User 2
        16.0, 50.0    // User 3
    ].reshape([3, 2]);

    let item_data = [
        500.0, 0.0,   // Item 1
        1200.0, 18.0, // Item 2
        20.0, 0.0,    // Item 3
        45.0, 18.0,   // Item 4
        50.0, 0.0     // Item 5
    ].reshape([5, 2]);

    // Embeddings
    let user_embs = Tensor::randn([3, 16], false);
    let item_embs = Tensor::randn([5, 16], false);

    // Pre-compute similarity scores for all pairs
    let scores = user_embs.matmul(item_embs.transpose(0, 1));

    let mut user_id = 1;
    while user_id <= 3 {
        // Simple type mapping
        let mut u_type = "unknown";
        if user_id == 1 { u_type = "regular"; }
        if user_id == 2 { u_type = "premium"; }
        if user_id == 3 { u_type = "student"; }

        println("\nRecommendations for User {} ({}):", user_id, u_type);
        
        let u_idx = user_id - 1;
        let age = user_data.get(u_idx, 0);
        let budget = user_data.get(u_idx, 1);

        let mut item_id = 1;
        while item_id <= 5 {
            // A. Symbolic Check (Logic Engine)
            let view_res = ?can_view(user_id, item_id);
            if view_res.item() > 0.5 {
                println("DEBUG: Logic allowed User {} -> Item {}", user_id, item_id);
                
                // B. Numeric Check
                let i_idx = item_id - 1;
                let price = item_data.get(i_idx, 0);
                let min_age = item_data.get(i_idx, 1);

                if age >= min_age {
                    if budget >= price {
                         // C. Tensor Scoring
                         let score = scores.get(u_idx, i_idx);
                         
                         if score > -3.0 { 
                             println("User {} -> Item {} [Allowed] Score: {}", user_id, item_id, score);
                         }
                    }
                }
            }
            item_id = item_id + 1;
        }
        user_id = user_id + 1;
    }
}
