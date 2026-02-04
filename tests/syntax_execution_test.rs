use glob::glob;
use std::fs;
use std::path::Path;
use std::process::Command;

#[test]
fn run_syntax_fixtures() {
    let mut failures = Vec::new();

    // Find all .tl files in fixtures directory
    let entries = glob("tests/fixtures/**/*.tl").expect("Failed to read glob pattern");

    for entry in entries {
        match entry {
            Ok(path) => {
                println!("Running syntax test: {:?}", path);
                if let Err(e) = run_fixture(&path) {
                    let msg = format!("{:?}: {}", path, e);
                    eprintln!("{}", msg);
                    failures.push(msg);
                }
            }
            Err(e) => failures.push(format!("Glob error: {}", e)),
        }
    }

    if !failures.is_empty() {
        panic!(
            "{} syntax tests failed:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
}

fn run_fixture(path: &Path) -> Result<(), String> {
    let content = fs::read_to_string(path).map_err(|e| e.to_string())?;

    // Parse expectations
    let mut expected_exit_code = 0;
    let mut expected_stdout = Vec::new();
    let mut expected_stderr = Vec::new();

    for line in content.lines() {
        // Skip marker - if present, test is skipped
        if line.trim().starts_with("// SKIP:") || line.trim().starts_with("// SKIP") {
            return Ok(());  // Skip this test
        }
        if let Some(val) = line.trim().strip_prefix("// EXIT: ") {
            expected_exit_code = val.trim().parse().unwrap_or(0);
        }
        if let Some(val) = line.trim().strip_prefix("// STDOUT: ") {
            expected_stdout.push(val.trim().to_string());
        }
        if let Some(val) = line.trim().strip_prefix("// STDERR: ") {
            expected_stderr.push(val.trim().to_string());
        }
    }

    // Run the compiler
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_tl"));
    cmd.arg(path);

    // Timeout or kill logic if needed? For now rely on simple execution.
    // If it hangs, user can kill. Syntax tests should be fast.

    let output = cmd
        .output()
        .map_err(|e| format!("Failed to execute command: {}", e))?;

    // Check exit code
    if expected_exit_code == 0 {
        if !output.status.success() {
            return Err(format!(
                "Expected success, got failure code {:?}.\nStdout: {}\nStderr: {}",
                output.status.code(),
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ));
        }
    } else {
        match output.status.code() {
            Some(code) => {
                if code != expected_exit_code {
                    return Err(format!(
                        "Expected exit code {}, got {}.\nStdout: {}\nStderr: {}",
                        expected_exit_code,
                        code,
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    ));
                }
            }
            None => return Err("Process terminated by signal".to_string()),
        }
    }

    // Check Output
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    for expected in expected_stdout {
        if !stdout.contains(&expected) {
            return Err(format!(
                "Missing expected stdout: '{}'\nActual stdout:\n{}",
                expected, stdout
            ));
        }
    }

    for expected in expected_stderr {
        if !stderr.contains(&expected) {
            return Err(format!(
                "Missing expected stderr: '{}'\nActual stderr:\n{}",
                expected, stderr
            ));
        }
    }

    Ok(())
}
