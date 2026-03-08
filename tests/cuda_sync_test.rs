//! Tests for CUDA synchronization (cudaDeviceSynchronize) on Linux.
//!
//! Commit 7d95b21 replaced a TODO in runtime_gpu_sync() with an actual
//! cudaDeviceSynchronize() call. These tests verify that:
//! 1. GPU sync completes without crash
//! 2. Data read back after sync is correct (not stale)
//! 3. CPU fallback path also works

#[cfg(target_os = "linux")]
mod cuda_sync {
    use std::process::Command;

    /// Run a .tl file and return (exit_code, stdout, stderr)
    fn run_tl_file(path: &str) -> (i32, String, String) {
        let output = Command::new(env!("CARGO_BIN_EXE_tl"))
            .arg(path)
            .output()
            .expect("Failed to execute tl binary");

        let exit_code = output.status.code().unwrap_or(-1);
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        (exit_code, stdout, stderr)
    }

    /// Test: The GPU sync integration test (.tl) runs successfully on GPU.
    /// This exercises the full pipeline: tensor creation -> GPU ops -> sync -> read.
    #[test]
    fn test_gpu_sync_integration() {
        let (exit_code, stdout, stderr) = run_tl_file("tests/fixtures/tensor/test_gpu_sync.tl");

        if exit_code != 0 {
            eprintln!("stderr:\n{}", stderr);
            panic!(
                "test_gpu_sync.tl exited with code {} (expected 0)\nstdout:\n{}",
                exit_code, stdout
            );
        }

        assert!(
            stdout.contains("[GPU Sync] Test 1 PASSED"),
            "Test 1 (compute-then-read) failed.\nstdout:\n{}",
            stdout
        );
        assert!(
            stdout.contains("[GPU Sync] Test 2 PASSED"),
            "Test 2 (multi-op pipeline) failed.\nstdout:\n{}",
            stdout
        );
        assert!(
            stdout.contains("[GPU Sync] Test 3 PASSED"),
            "Test 3 (large tensor stress) failed.\nstdout:\n{}",
            stdout
        );
    }

    /// Test: GPU sync on CPU fallback path (TL_DEVICE=cpu).
    /// runtime_gpu_sync() should be a no-op when running on CPU.
    #[test]
    fn test_gpu_sync_cpu_fallback() {
        let output = Command::new(env!("CARGO_BIN_EXE_tl"))
            .arg("tests/fixtures/tensor/test_gpu_sync.tl")
            .env("TL_DEVICE", "cpu")
            .output()
            .expect("Failed to execute tl binary");

        let exit_code = output.status.code().unwrap_or(-1);
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        if exit_code != 0 {
            eprintln!("stderr:\n{}", stderr);
            panic!(
                "test_gpu_sync.tl (CPU mode) exited with code {} (expected 0)\nstdout:\n{}",
                exit_code, stdout
            );
        }

        assert!(
            stdout.contains("[GPU Sync] All tests passed"),
            "CPU fallback test failed.\nstdout:\n{}",
            stdout
        );
    }

    /// Test: Multiple sync calls in sequence should not crash or deadlock.
    /// This regression test ensures cudaDeviceSynchronize can be called repeatedly.
    #[test]
    fn test_gpu_sync_repeated_calls() {
        // Create a minimal .tl program that calls sync multiple times
        let test_program = r#"
fn main() {
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [1.0, 1.0, 1.0, 1.0];

    // Sync after first op
    let c = a + b;
    System::metal_sync();

    // Sync again after second op
    let d = c * a;
    System::metal_sync();

    // Sync again after third op  
    let e = d + b;
    System::metal_sync();

    // a=[1,2,3,4] b=[1,1,1,1]
    // c = a+b = [2,3,4,5]
    // d = c*a = [2,6,12,20]
    // e = d+b = [3,7,13,21] -> sum = 44
    let total = e.sumall().item();
    if total == 44.0 {
        println("REPEATED_SYNC_OK");
    } else {
        println("REPEATED_SYNC_FAIL");
        println(total);
    }
}
"#;
        let tmp_file = "/tmp/test_gpu_sync_repeated.tl";
        std::fs::write(tmp_file, test_program).expect("Failed to write temp .tl file");

        let (exit_code, stdout, stderr) = run_tl_file(tmp_file);

        // Cleanup
        let _ = std::fs::remove_file(tmp_file);

        if exit_code != 0 {
            eprintln!("stderr:\n{}", stderr);
            panic!(
                "Repeated sync test exited with code {} (expected 0)\nstdout:\n{}",
                exit_code, stdout
            );
        }

        assert!(
            stdout.contains("REPEATED_SYNC_OK"),
            "Repeated sync test failed.\nstdout:\n{}",
            stdout
        );
    }
}
