fn main() {
    // 1. Sum reduction with condition (filter)
    // k ranges 0..10. Sum only even k.
    // 0 + 2 + 4 + 6 + 8 = 20.
    // i is dummy output dimension size 1.
    let S = [i | i <- 0..1, k <- 0..10, (k - (k/2)*2) == 0 { k }];
    print(S);
    
    // 2. Multiple conditions
    // Sum k where k > 5 AND k is even.
    // 6 + 8 = 14.
    let S2 = [i | i <- 0..1, k <- 0..10, k > 5, (k - (k/2)*2) == 0 { k }];
    print(S2);

    // 3. Condition using external variable?
    let threshold = 5;
    let S3 = [i | i <- 0..1, k <- 0..10, k > threshold { 1.0 }];
    // k = 6, 7, 8, 9. Count = 4.
    print(S3);
}
