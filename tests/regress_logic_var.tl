// Logic Query Variable Reproduction
// Tests if passing a variable to a query works.

p(100).

fn main() {
    let x = 100;
    println("Querying with literal 100:");
    let r1 = ?p(100);
    if r1.item() > 0.5 { println("Literal OK"); } else { println("Literal FAIL"); }

    println("Querying with variable x=100:");
    let r2 = ?p(x);
    if r2.item() > 0.5 { println("Variable OK"); } else { println("Variable FAIL"); }
    test_zero();
}
p(0).
fn test_zero() { let r = ?p(0); if r.item() > 0.5 { println("Zero OK"); } else { println("Zero FAIL"); } }
