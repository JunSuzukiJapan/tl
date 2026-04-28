// examples/npy/test_npy_builtin.tl

fn main() {
    println("Testing NumPy .npy builtin...");
    let res = NpyFile::load("examples/npy/test.npy");
    match res {
        Result::Ok(npy) => {
            println("Successfully loaded .npy file!");
            println("Version: {}.{}", npy.header.major_version, npy.header.minor_version);
            println("Descr: {}", npy.header.descr);
            println("Fortran order: {}", npy.header.fortran_order);
            println("Shape: {}", npy.shape_str());
            println("Element count: {}", npy.element_count());
            println("Data bytes: {}", npy.data.len());
            println("Dtype: {}", npy.dtype_name());
            
            // Test save (roundtrip)
            let save_res = npy.save("examples/npy/test_out.npy");
            match save_res {
                Result::Ok(ok) => {
                    println("Successfully saved test_out.npy: {}", ok);
                    // Verify roundtrip by loading saved file
                    let res2 = NpyFile::load("examples/npy/test_out.npy");
                    match res2 {
                        Result::Ok(npy2) => {
                            println("Roundtrip verification:");
                            println("  Shape: {}", npy2.shape_str());
                            println("  Descr: {}", npy2.header.descr);
                            println("  Data bytes: {}", npy2.data.len());
                        }
                        Result::Err(err) => {
                            println("Failed to reload: {}", err);
                        }
                    }
                }
                Result::Err(err) => {
                    println("Failed to save: {}", err);
                }
            }
        }
        Result::Err(err) => {
            println("Failed to load .npy file: {}", err);
        }
    }
}
