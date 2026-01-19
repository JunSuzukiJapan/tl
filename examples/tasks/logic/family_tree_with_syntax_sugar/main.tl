mod facts;
use facts::*;

fn main() {
    println("--- Prolog-like Logic in TL (Syntax Sugar Edition) ---");
    
    // Query: Is alice the father of bob?
    println("Query: father(alice, bob)?");
    let res1 = ?father(alice, bob);
    print("Result: ");
    println(res1);
    
    // Query: Is alice the grandfather of charlie?
    println("");
    println("Query: grandparent(alice, charlie)?");
    let res2 = ?grandparent(alice, charlie);
    print("Result: ");
    println(res2);
    
    // Query: Find all descendants of alice
    println("");
    println("Query: ancestor(alice, $x)?");
    let res3 = ?ancestor(alice, $x);
    print("Descendants of alice: ");
    println(res3);
}
