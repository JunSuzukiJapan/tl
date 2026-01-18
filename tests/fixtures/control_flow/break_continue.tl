// Simple break/continue test
// STDOUT: 5
// STDOUT: 40
// STDOUT: 3
// STDOUT: found

fn main() {
    // 1. Basic break in while
    let mut i = 0;
    while i < 10 {
        if i == 5 {
            break;
        }
        i = i + 1;
    }
    print(i);

    // 2. Basic continue in for
    let mut sum = 0;
    for j in 0..10 {
        if j == 5 {
            continue;
        }
        sum = sum + j;
    }
    // sum = 0+1+2+3+4+6+7+8+9 = 40
    print(sum);

    // 3. Break in for loop
    for k in 0..10 {
        if k == 3 {
            print(k);
            break;
        }
    }

    // 4. Break in nested loops (inner only)
    for x in 0..3 {
        for y in 0..3 {
            if x == 1 && y == 2 {
                print("found");
                break;
            }
        }
    }
}
