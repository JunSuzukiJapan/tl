fn main() {
    println!("cargo:rerun-if-changed=src/cuda_kernels/activation.cu");
    println!("cargo:rerun-if-changed=build.rs");

    // Only build CUDA kernels if the 'cuda' feature is enabled
    #[cfg(target_os = "linux")]
    {
        // check if nvcc is installed
        let output = std::process::Command::new("nvcc").arg("--version").output();

        if let Ok(output) = output {
            if output.status.success() {
                // nvcc is available, compile kernels
                let mut build = cc::Build::new();
                build.cuda(true).cudart("shared"); // Link against shared cuda runtime

                // Fat Binary: 複数アーキテクチャの SASS を生成（不特定多数の GPU に対応）
                // sm_75: RTX 20XX / T4
                // sm_80: A100 / RTX 30XX (GA100)
                // sm_86: RTX 30XX (GA102/GA104)
                // sm_89: RTX 40XX (Ada Lovelace)
                // sm_90: H100 (Hopper)
                for arch in &["75", "80", "86", "89", "90"] {
                    let gencode = format!("arch=compute_{},code=sm_{}", arch, arch);
                    build.flag("-gencode").flag(&gencode);
                }

                // PTX フォールバック: 将来の GPU (RTX 50XX 等) でも JIT コンパイルで動作
                build
                    .flag("-gencode")
                    .flag("arch=compute_75,code=compute_75");

                build
                    .file("src/cuda_kernels/activation.cu")
                    .compile("tl_cuda_kernels");

                // Link CUDA libraries
                println!("cargo:rustc-link-lib=dylib=cudart");
                println!("cargo:rustc-link-lib=dylib=cuda");
            } else {
                println!(
                    "cargo:warning=nvcc found but failed version check. Skipping CUDA kernel compilation."
                );
            }
        } else {
            panic!(
                "\n\n\
                ========================================\n\
                ERROR: nvcc not found!\n\
                ========================================\n\
                \n\
                CUDA feature is enabled but nvcc (NVIDIA CUDA Compiler) was not found in PATH.\n\
                \n\
                To fix this:\n\
                \n\
                1. Install CUDA Toolkit from:\n\
                   https://developer.nvidia.com/cuda-downloads\n\
                \n\
                2. Add nvcc to your PATH:\n\
                   export PATH=/usr/local/cuda/bin:$PATH\n\
                \n\
                3. Verify installation:\n\
                   nvcc --version\n\
                \n\
                ========================================\n"
            );
        }
    }
}
