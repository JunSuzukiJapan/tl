// EXIT: 0

role(alice, editor).
role(bob, editor).
role(dave, intern).

perm(editor, write).
perm(intern, read).

requires_restriction(editor).
restricted_role(r) :- requires_restriction(r).
restricted_user(u) :- role(u, r), restricted_role(r).

allowed(u, p) :- role(u, r), perm(r, p), not restricted_user(u).

fn main() {
    println(?allowed(alice, write));
    println(?allowed(bob, write));
    println(?restricted_user(alice));
}
