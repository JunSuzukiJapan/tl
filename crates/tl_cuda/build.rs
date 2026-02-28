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

    // CUDA カーネルのコンパイル
    let nvcc_exists = std::path::Path::new(&nvcc_path).exists()
        || std::process::Command::new("nvcc")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);

    if nvcc_exists {
        // nvcc のパスを環境変数で cc crate に伝える
        std::env::set_var("NVCC", &nvcc_path);

        cc::Build::new()
            .cuda(true)
            .cudart("shared")
            .flag("-gencode")
            .flag("arch=compute_75,code=sm_75")
            .file("src/cuda_kernels/autograd.cu")
            .compile("tl_cuda_autograd_kernels");

        println!(
            "cargo:warning=CUDA kernels compiled successfully with {}",
            nvcc_path
        );
    } else {
        println!(
            "cargo:warning=nvcc not found at {} or in PATH. CUDA kernels will not be compiled.",
            nvcc_path
        );
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
