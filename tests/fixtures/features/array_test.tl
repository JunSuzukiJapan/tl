// Fixed-size array test
let arr: [i64; 3] = [10, 20, 30];

// Read elements
print(arr[0]);  // expect: 10
print(arr[1]);  // expect: 20
print(arr[2]);  // expect: 30

// Write element
let mut arr2: [i64; 3] = [1, 2, 3];
arr2[1] = 99;
print(arr2[0]);  // expect: 1
print(arr2[1]);  // expect: 99
print(arr2[2]);  // expect: 3

// Sum
let sum = arr2[0] + arr2[1] + arr2[2];
print(sum);      // expect: 103 (1 + 99 + 3)
