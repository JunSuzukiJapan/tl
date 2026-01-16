fn main() {
    // IF / ELSE
    let x = 5;
    if x > 10 {
        print("big");
    } else {
        print("small"); // STDOUT: small
    }

    if x == 5 {
        print("five"); // STDOUT: five
    }

    // WHILE
    let mut i = 0;
    while i < 3 {
        print(i); // STDOUT: 0
                  // STDOUT: 1
                  // STDOUT: 2
        i = i + 1;
    }

    // FOR (0..N)
    for j in 0..3 {
        print(j); // STDOUT: 0
                  // STDOUT: 1
                  // STDOUT: 2
    }
}
