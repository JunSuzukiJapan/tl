// Comprehensive Control Flow Syntax Tests
// Tests: if, if-else, else if, if expression, nested if

// STDOUT: 1
// STDOUT: 2
// STDOUT: 3
// STDOUT: 4
// STDOUT: 5
// STDOUT: 6
// STDOUT: 7
// STDOUT: 8
// STDOUT: 9
// STDOUT: 10
// STDOUT: 11
// STDOUT: 12
// STDOUT: 13
// STDOUT: 14

fn main() {
    // 1. Basic if (true branch)
    if true {
        print(1);
    }

    // 2. Basic if-else (true branch)
    if true {
        print(2);
    } else {
        print(-1);
    }

    // 3. Basic if-else (false branch)
    if false {
        print(-1);
    } else {
        print(3);
    }

    // 4. else if chain (first condition)
    let x = 1;
    if x == 1 {
        print(4);
    } else if x == 2 {
        print(-1);
    } else {
        print(-1);
    }

    // 5. else if chain (second condition)
    let y = 2;
    if y == 1 {
        print(-1);
    } else if y == 2 {
        print(5);
    } else {
        print(-1);
    }

    // 6. else if chain (else branch)
    let z = 3;
    if z == 1 {
        print(-1);
    } else if z == 2 {
        print(-1);
    } else {
        print(6);
    }

    // 7. Nested if
    if true {
        if true {
            print(7);
        }
    }

    // 8. Deeply nested if-else
    if false {
        print(-1);
    } else {
        if true {
            if false {
                print(-1);
            } else {
                print(8);
            }
        }
    }

    // 9. Complex boolean condition
    let a = 5;
    let b = 10;
    if a < b && b > 0 {
        print(9);
    }

    // 10. Or condition
    if false || true {
        print(10);
    }

    // 11. Not condition
    if !false {
        print(11);
    }

    // 12. Comparison operators
    if 5 > 3 && 2 < 4 && 5 >= 5 && 3 <= 3 && 1 == 1 && 1 != 2 {
        print(12);
    }

    // 13. If with function call result
    if returns_true() {
        print(13);
    }

    // 14. Multiple statements in branches
    if true {
        let temp = 7;
        let temp2 = temp + 7;
        print(temp2);
    }
}

fn returns_true() -> bool {
    true
}
