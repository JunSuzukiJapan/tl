fn main() {
    let mut usage = std::mem::MaybeUninit::uninit();
    unsafe {
        if libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) == 0 {
            let usage = usage.assume_init();
            #[cfg(target_os = "macos")]
            println!("RSS: {} bytes", usage.ru_maxrss);
            #[cfg(target_os = "linux")]
            println!("RSS: {} KB", usage.ru_maxrss);
        } else {
            println!("getrusage failed");
        }
    }
}
