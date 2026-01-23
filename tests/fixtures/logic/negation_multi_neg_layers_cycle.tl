// EXIT: 1
// STDERR: negation is not stratified

role(alice, editor).
role(bob, editor).

banned_by_role(u) :- role(u, editor), not risk(u).
risk(u) :- banned_by_role(u).

ok(u) :- role(u, editor), not risk(u).

fn main() {}
