// EXIT: 0

role(alice, editor).
role(bob, editor).
role(carol, editor).
role(dave, editor).

banned(carol).

clean(u) :- role(u, editor), not banned(u).
risk(u) :- role(u, editor), not clean(u).
ok(u) :- role(u, editor), not risk(u).

fn main() {
    println(?ok(alice));
    println(?ok(bob));
    println(?ok(carol));
}
