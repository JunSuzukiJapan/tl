father(tarou, ichirou).
father(tarou, jirou).

grandparent(x, y) :- father(x, z), father(z, y).

fn main() {
    print("Running Logic Sugar Test");

    // Test query expression
    let res = ?father(tarou, $x);
    print("Father query result:");
    print(res);
    
    // Method chain check (assuming .print() or similar exists on result tensor)
    // For now just print result to verify it compiles and runs (even if result is empty until runtime logic supported)
}
