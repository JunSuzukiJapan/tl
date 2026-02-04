// Match scrutinee exhaustive test
// Testing match as expression vs statement

enum TestEnum {
    A,
    B { val: i64 },
}

fn main() {
    test_match_as_expr();
    test_match_as_stmt();
    
    println("All tests passed!");
}

// Match as expression (inside let)
fn test_match_as_expr() {
    print("Test 1: Match as expression... ");
    let x = TestEnum::A;
    let result = match x {
        TestEnum::A => 1,
        TestEnum::B { val } => val,
    };
    if result == 1 {
        println("OK");
    } else {
        println("FAIL");
    }
}

// Match as statement (standalone)
fn test_match_as_stmt() {
    print("Test 2: Match as statement... ");
    let x = TestEnum::A;
    match x {
        TestEnum::A => { println("OK"); }
        TestEnum::B { val } => { println("FAIL"); }
    }
}
