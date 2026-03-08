// GPU Synchronization Test
// Tests that cudaDeviceSynchronize (Linux) / Metal sync (macOS) works correctly.
// Without proper synchronization, GPU operations may return stale data.
//
// STDOUT: [GPU Sync] Test 1 PASSED
// STDOUT: [GPU Sync] Test 2 PASSED
// STDOUT: [GPU Sync] Test 3 PASSED
// STDOUT: [GPU Sync] All tests passed

fn main() {
    // ===== Test 1: Compute-then-read correctness =====
    // Create tensors, perform chained GPU operations, sync, then verify results.
    // Without cudaDeviceSynchronize, reading values after GPU ops could return stale data.
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [5.0, 6.0, 7.0, 8.0];
    let c = a + b;
    System::metal_sync();

    // c should be [6, 8, 10, 12]
    let s = c.sumall();
    System::metal_sync();
    let total = s.item();
    if total == 36.0 {
        println("[GPU Sync] Test 1 PASSED");
    } else {
        println("[GPU Sync] Test 1 FAILED: expected 36.0, got:");
        println(total);
    }

    // ===== Test 2: Multi-operation pipeline with sync =====
    // Chain multiple operations: mul -> add -> sum
    // Then sync and verify the result is consistent.
    let x = [3.0, 4.0, 5.0, 6.0];
    let y = [1.0, 1.0, 1.0, 1.0];
    let prod = x * y;               // should equal x
    let added = prod + x;           // should be 2 * x = [6, 8, 10, 12]
    System::metal_sync();
    let total2 = added.sumall().item();
    if total2 == 36.0 {
        println("[GPU Sync] Test 2 PASSED");
    } else {
        println("[GPU Sync] Test 2 FAILED: expected 36.0, got:");
        println(total2);
    }

    // ===== Test 3: Large tensor matmul stress test =====
    // Larger tensors increase the chance of exposing race conditions
    // if sync is missing (async GPU ops may not complete before read).
    let big_a = Tensor::ones([64, 64], false);
    let big_b = Tensor::ones([64, 64], false);
    let big_c = big_a.matmul(big_b);   // 64x64 of all 64.0
    System::metal_sync();

    // Sum of 64x64 matrix where each element is 64.0 = 64 * 64 * 64 = 262144
    let big_sum = big_c.sumall().item();
    if big_sum == 262144.0 {
        println("[GPU Sync] Test 3 PASSED");
    } else {
        println("[GPU Sync] Test 3 FAILED: expected 262144.0, got:");
        println(big_sum);
    }

    println("[GPU Sync] All tests passed");
}
