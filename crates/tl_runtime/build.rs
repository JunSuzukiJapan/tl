fn main() {
    println!("cargo:rerun-if-changed=src/cuda_kernels/activation.cu");
    println!("cargo:rerun-if-changed=build.rs");

    // Only build CUDA kernels if the 'cuda' feature is enabled
    #[cfg(feature = "cuda")]
    {
        // check if nvcc is installed
        let output = std::process::Command::new("nvcc").arg("--version").output();

        if let Ok(output) = output {
            if output.status.success() {
                // nvcc is available, compile kernels
                cc::Build::new()
                    .cuda(true)
                    .cudart("shared") // Link against shared cuda runtime
                    .flag("-gencode")
                    .flag("arch=compute_75,code=sm_75") // Target T4/RTX20xx class, adjust as needed
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
