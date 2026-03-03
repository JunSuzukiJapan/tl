fn main() {
    // Prevent the linker from stripping #[no_mangle] extern "C" symbols
    // that are only referenced by LLVM JIT at runtime.
    // Without this, `cargo install` produces a binary that segfaults because
    // runtime symbols like tl_tensor_print, tl_tensor_acquire, etc. are
    // removed as "unused" by the linker.
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-arg=-rdynamic");

    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-arg=-rdynamic");
}
