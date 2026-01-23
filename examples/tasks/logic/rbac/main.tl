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
    let t = [i | i <- 0..1 { name_id }].reshape(1, 1);
    print(t);
}

fn print_entity_list(list: Tensor) {
    let n = list.dim(0);
    let t = list.reshape(n, 1);
    print(t);
}

fn main() {
    println("--- RBAC (Logic-only) ---");

    // Query: all (user, permission) pairs
    let res = ?allowed($U, $P);

    // Print by user
    let users = ?user($U);
    println("By user");
    println("-------");
    for ui in 0..users.dim(0) {
        let u = users[ui, 0] as i64;
        let perms = [i | i <- 0..res.dim(0), res[i, 0] == u { res[i, 1] }];
        print_entity(u);
        print(": ");
        print_entity_list(perms);
        println("");
    }

    // Print by permission
    let perms = ?permission($P);
    println("");
    println("By permission");
    println("-------------");
    for pi in 0..perms.dim(0) {
        let p = perms[pi, 0] as i64;
        let users_for_p = [i | i <- 0..res.dim(0), res[i, 1] == p { res[i, 0] }];
        print_entity(p);
        print(": ");
        print_entity_list(users_for_p);
        println("");
    }

    println("");
    println("Check a specific permission:");
    println("alice can deploy? ");
    println(?allowed(alice, deploy));
    println("dave can merge_pr? ");
    println(?allowed(dave, merge_pr));
}
