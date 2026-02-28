/// build.rs for tl_cuda
/// CUDA Toolkit のパスを検出し、リンク設定とカーネルコンパイルを行う。

fn main() {
    // CUDA Toolkit のパスを検出
    let cuda_path = std::env::var("CUDA_PATH")
        .or_else(|_| std::env::var("CUDA_HOME"))
        .unwrap_or_else(|_| {
            // デフォルトパスを試行
            if std::path::Path::new("/usr/local/cuda/lib64/libcudart.so").exists() {
                "/usr/local/cuda".to_string()
            } else {
                // /usr/local/cuda-* を探す
                let mut candidates: Vec<_> = std::fs::read_dir("/usr/local/")
                    .into_iter()
                    .flatten()
                    .flatten()
                    .filter(|e| {
                        e.file_name()
                            .to_str()
                            .map_or(false, |n| n.starts_with("cuda-"))
                    })
                    .collect();
                candidates.sort_by(|a, b| b.file_name().cmp(&a.file_name()));
                candidates
                    .first()
                    .map(|e| e.path().to_string_lossy().to_string())
                    .expect("CUDA Toolkit not found. Set CUDA_PATH or CUDA_HOME.")
            }
        });

    let lib_path = format!("{}/lib64", cuda_path);
    let nvcc_path = format!("{}/bin/nvcc", cuda_path);

    // GPU の compute capability を自動検出
    let gpu_arch = detect_gpu_arch();

    // CUDA カーネルのコンパイル
    let nvcc_exists = std::path::Path::new(&nvcc_path).exists()
        || std::process::Command::new("nvcc")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);

    if nvcc_exists {
        std::env::set_var("NVCC", &nvcc_path);

        let mut build = cc::Build::new();
        build.cuda(true).cudart("shared");

        // 検出されたアーキテクチャ、またはフォールバック
        let arch = gpu_arch.unwrap_or_else(|| "75".to_string());
        let gencode = format!("arch=compute_{},code=sm_{}", arch, arch);
        build.flag("-gencode").flag(&gencode);

        // PTX も生成して forward compatibility を確保
        let gencode_ptx = format!("arch=compute_{},code=compute_{}", arch, arch);
        build.flag("-gencode").flag(&gencode_ptx);

        build
            .file("src/cuda_kernels/autograd.cu")
            .compile("tl_cuda_autograd_kernels");

        println!(
            "cargo:warning=CUDA kernels compiled for sm_{} with {}",
            arch, nvcc_path
        );
    } else {
        println!("cargo:warning=nvcc not found at {} or in PATH.", nvcc_path);
    }

    // リンクパスを設定
    println!("cargo:rustc-link-search=native={}", lib_path);
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");

    // 再ビルドトリガー
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-changed=src/cuda_kernels/autograd.cu");
}

/// nvidia-smi で GPU の compute capability を検出
fn detect_gpu_arch() -> Option<String> {
    // nvidia-smi で GPU 名を取得し、compute capability をマッピング
    let output = std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=compute_cap")
        .arg("--format=csv,noheader,nounits")
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let cap = String::from_utf8_lossy(&output.stdout);
    let cap = cap.trim();
    if cap.is_empty() {
        return None;
    }

    // "8.6" → "86", "7.5" → "75"
    let arch = cap.replace('.', "");
    println!(
        "cargo:warning=Detected GPU compute capability: {} (sm_{})",
        cap, arch
    );
    Some(arch)
}
