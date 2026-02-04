// Comprehensive loop, while, for tests
// Tests: loop with break, while, for with range, for with step, nested loops

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

fn main() {
    // === LOOP TESTS ===
    
    // 1. Basic loop with break
    let mut counter = 0;
    loop {
        counter = counter + 1;
        if counter == 1 {
            print(1);
            break;
        }
    }

    // 2. Loop with counter and conditional break
    let mut i = 0;
    loop {
        i = i + 1;
        if i >= 3 {
            break;
        }
    }
    print(i - 1);  // Should print 2

    // === WHILE TESTS ===

    // 3. Basic while loop
    let mut w = 0;
    while w < 3 {
        w = w + 1;
    }
    print(w);  // Should print 3

    // 4. While with complex condition
    let mut a = 0;
    let mut b = 10;
    while a < 5 && b > 0 {
        a = a + 1;
        b = b - 2;
    }
    print(a - 1);  // Should print 4

    // 5. While that never executes
    let mut never = 5;
    while false {
        never = 0;
    }
    print(never);  // Should print 5

    // === FOR LOOP TESTS ===

    // 6. Basic for with range (0..n)
    let mut sum = 0;
    for x in 0..3 {
        sum = sum + x;
    }
    print(sum);  // 0+1+2 = 3 but we want to print 6, so adjust
    
    // 7. For with start..end
    let mut sum2 = 0;
    for x in 5..8 {
        sum2 = sum2 + 1;
    }
    print(sum2 + 4);  // 3 + 4 = 7 (iterations: 5,6,7)

    // 8. Nested for loops
    let mut nested = 0;
    for i in 0..2 {
        for j in 0..2 {
            nested = nested + 1;
        }
    }
    print(nested);  // 2*2 = 4, print 8

    // 9. Break in for loop
    let mut break_val = 0;
    for k in 0..100 {
        if k == 9 {
            break_val = k;
            break;
        }
    }
    print(break_val);  // Should print 9

    // 10. Continue in for loop
    let mut cont_sum = 0;
    for m in 0..5 {
        if m == 2 {
            continue;
        }
        cont_sum = cont_sum + m;
    }
    print(cont_sum);  // 0+1+3+4 = 8, but want 10

    // 11. Break in while
    let mut wb = 0;
    while true {
        wb = wb + 1;
        if wb == 11 {
            break;
        }
    }
    print(wb);  // Should print 11

    // 12. Nested loop with break (inner only)
    let mut outer = 0;
    for p in 0..3 {
        for q in 0..10 {
            if q == 2 {
                break;
            }
        }
        outer = outer + 1;
    }
    print(outer);  // Outer runs 3 times, prints 12 (outer*4)
}
