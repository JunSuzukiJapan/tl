// EXIT: 0

role(alice, editor).
role(bob, editor).
perm(editor, write).
deny(bob, write).

score(alice, 12).
score(bob, 20).

allowed(u, p) :- role(u, r), perm(r, p), not deny(u, p).
high(u) :- score(u, s), ge(s, 15).
low(u) :- score(u, s), not high(u).

sum2(x, y, z) :- mul(y, 2, t1), add(x, t1, t2), sub(t2, 1, z).

ok(2).
ok(3).

test(x) :- ok(x), ok(x + 1).

fn main() {
    println(?allowed(alice, write));
    println(?allowed(bob, write));
    println(?low(alice));
    println(?low(bob));
    println(?sum2(3, 4, 10));
    println(?test(2));
}
