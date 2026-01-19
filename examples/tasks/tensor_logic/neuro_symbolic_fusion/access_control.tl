// Neuro-Symbolic Smart Access Control
// Fusion of Facial Recognition (Tensor) and Role-Based Access Control (Logic)

// 1. Logic-Based Access Policy
// ----------------------------

// Roles
role(1, admin).
role(2, engineer).
role(3, guest).

// Permissions
accessible(admin, server_room).
accessible(admin, office).
accessible(admin, lobby).

accessible(engineer, office).
accessible(engineer, lobby).

// strict_access(Role, Zone)
strict_access(engineer, server_room).

accessible(guest, lobby).

// Context
context(0, 0). // Weekday, Normal

// Rules (Vars: p=Person, r=Role, z=Zone)
can_enter(p, z) :-
    person_role(p, r),
    accessible(r, z).

// Bind Person to Role (Static DB)
person_role(101, admin).    // Alice
person_role(102, engineer). // Bob
person_role(103, guest).    // Eve

// 2. Tensor-Based Identity Verification (Face Maching)
// ---------------------------------------------------
fn main() {
    println("--- Smart Access Control System ---");

    // Simulated Database of Face Embeddings (3 people, 16 dim)
    let db_faces = Tensor::randn([3, 16], false); 
    
    // Simulated Camera Input (Bob - Index 1)
    let bob_vec = db_faces[1];
    let noise = Tensor::randn([16], false) * 0.1;
    let camera_input = bob_vec + noise;

    println("Camera detected a person...");
    
    // Step 1: Face Recognition (Tensor)
    // Batch compute similarity: db_faces [3, 16] @ camera_input [16] -> scores [3]
    // We treat camera_input as [16, 1] for matmul or just broadcast?
    // TL matmul usually is [M, K] x [K, N] -> [M, N].
    // Let's reshape camera to [16, 1].
    let cam_col = camera_input.reshape([16, 1]);
    let scores = db_faces.matmul(cam_col); // [3, 1]

    let mut best_match_idx = -1;
    let mut max_score = -999.0;
    
    let mut i = 0;
    while i < 3 {
        // Access score from result vector
        let score = scores.get(i, 0);
        
        if score > max_score {
            max_score = score;
            best_match_idx = i;
        }
        i = i + 1;
    }

    let mut person_id = 0;
    if best_match_idx == 0 { person_id = 101; }
    if best_match_idx == 1 { person_id = 102; }
    if best_match_idx == 2 { person_id = 103; }

    println("Identity Verified: PersonID {} (Score: {})", person_id, max_score);

    // Step 2: Access Control (Logic)
    // Inline checks
    
    // Check Lobby
    let r1 = ?can_enter(person_id, lobby);
    if r1.item() > 0.5 { println("  [ACCESS GRANTED] -> Lobby"); }
    else { println("  [ACCESS DENIED]  -> Lobby"); }

    // Check Office
    let r2 = ?can_enter(person_id, office);
    if r2.item() > 0.5 { println("  [ACCESS GRANTED] -> Office"); }
    else { println("  [ACCESS DENIED]  -> Office"); }

    // Check ServerRoom
    let r3 = ?can_enter(person_id, server_room);
    if r3.item() > 0.5 { println("  [ACCESS GRANTED] -> ServerRoom"); }
    else { println("  [ACCESS DENIED]  -> ServerRoom"); }
}
