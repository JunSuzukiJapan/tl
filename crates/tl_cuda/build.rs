/// build.rs for tl_cuda
/// CUDA Toolkit のパスを検出し、リンク設定を行う。

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

    // リンクパスを設定
    println!("cargo:rustc-link-search=native={}", lib_path);
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");

    // 再ビルドトリガー
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
}
