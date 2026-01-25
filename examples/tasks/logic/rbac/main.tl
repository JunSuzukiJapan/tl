mod facts;
use facts::*;

// Effective role: direct or inherited (recursive)
role_effective(user, role) :- role(user, role).
role_effective(user, parent) :- role_effective(user, role), inherits(role, parent).

// Permission via role or direct grant
allowed(user, perm) :- role_effective(user, role), perm(role, perm).
allowed(user, perm) :- grant(user, perm).

// Permission catalog (unique)
permission(perm) :- perm(_, perm).

fn print_entity(name_id: i64) {
    let t = [i | i <- 0..1 { name_id }].reshape([1, 1]).to_i64();
    print(t);
}

fn main() {
    println("--- RBAC (Logic-only) ---");

    // Query: all (user, permission) pairs
    let res = allowed($U, $P).to_i64();

    // Print by user
    let users = user($U).to_i64();
    println("By user");
    println("-------");
    for ui in 0..users.dim(0) {
        let u = users[ui, 0] as i64;
        print_entity(u);
        print(": ");
        let mut first = 1;
        for i in 0..res.dim(0) {
            if res[i, 0] as i64 == u {
                if first == 0 {
                    print(", ");
                }
                let p = res[i, 1] as i64;
                print_entity(p);
                first = 0;
            }
        }
        println("");
    }

    // Print by permission
    let perms = permission($P).to_i64();
    println("");
    println("By permission");
    println("-------------");
    for pi in 0..perms.dim(0) {
        let p = perms[pi, 0] as i64;
        print_entity(p);
        print(": ");
        let mut first = 1;
        for i in 0..res.dim(0) {
            if res[i, 1] as i64 == p {
                if first == 0 {
                    print(", ");
                }
                let u = res[i, 0] as i64;
                print_entity(u);
                first = 0;
            }
        }
        println("");
    }

    println("");
    println("Check a specific permission:");
    println("alice can deploy? ");
    println("alice can deploy? ");
    println((allowed(alice, deploy))[0] > 0.0);
    println("dave can merge_pr? ");
    println((allowed(dave, merge_pr))[0] > 0.0);
}
