
mod facts;
mod logic;

use logic::*;

fn main() {
    // Attempt 2: With use
    // If this fails, user is correct.
    let res = @grandparent(alice, $x)?;
    println("Result: {}", res);
}
