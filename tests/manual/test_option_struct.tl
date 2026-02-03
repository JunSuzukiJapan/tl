struct User {
    id: i64,
    name: String,
}

fn main() {
    print("Testing Option<User>...\n");

    let u = User { id: 1, name: "Alice" };
    let opt = Option::Some(u);

    if opt.is_none() {
        print("FAIL: Expected Some\n");
    } else {
        print("OK: is_none check\n");
    }

    let u2 = opt.unwrap();
    print("User: ");
    print(u2.name); 
    print("\n");

    if u2.id != 1 {
        print("FAIL: ID mismatch\n");
    } else {
        print("OK: ID match\n");
    }

    // None check
    let none: Option<User> = Option::None;
    if !none.is_none() {
        print("FAIL: Expected None\n");
    } else {
        print("OK: None check\n");
    }
}
