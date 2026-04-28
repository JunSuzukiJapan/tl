// examples/npz/test_npz_builtin.tl

fn main() {
    println("Testing NumPy .npz builtin...");
    let res = NpzFile::load("examples/npz/test.npz");
    match res {
        Result::Ok(npz) => {
            println("Successfully loaded .npz file!");
            println("Entry count: {}", npz.entry_count());

            // List entries
            let entries = npz.entries;
            let mut i = 0;
            while i < entries.len() {
                let entry = entries.get(i).unwrap();
                println("Entry {}: name={}, data_bytes={}", i, entry.name, entry.data.len());
                i = i + 1;
            }

            // Test get_entry (now as impl method)
            let w = npz.get_entry("weights.npy");
            match w {
                Option::Some(e) => {
                    println("Found weights.npy: {} bytes", e.data.len());
                }
                Option::None => {
                    println("weights.npy not found");
                }
            }

            // Get as NpyFile
            let weights_opt = npz_get_npy(npz, "weights");
            match weights_opt {
                Option::Some(wn) => {
                    println("weights: shape={}, dtype={}, elements={}", wn.shape_str(), wn.dtype_name(), wn.element_count());
                }
                Option::None => {
                    println("weights not found");
                }
            }

            let biases_opt = npz_get_npy(npz, "biases");
            match biases_opt {
                Option::Some(b) => {
                    println("biases: shape={}, dtype={}, elements={}", b.shape_str(), b.dtype_name(), b.element_count());
                }
                Option::None => {
                    println("biases not found");
                }
            }

            // Test save (roundtrip)
            let save_res = save_npz(npz, "examples/npz/test_out.npz");
            match save_res {
                Result::Ok(ok) => {
                    println("Successfully saved test_out.npz: {}", ok);
                    let res2 = NpzFile::load("examples/npz/test_out.npz");
                    match res2 {
                        Result::Ok(npz2) => {
                            println("Roundtrip verification:");
                            println("  Entry count: {}", npz2.entry_count());
                            let w2 = npz_get_npy(npz2, "weights");
                            match w2 {
                                Option::Some(w3) => {
                                    println("  weights: shape={}, elements={}", w3.shape_str(), w3.element_count());
                                }
                                Option::None => {
                                    println("  weights not found in roundtrip");
                                }
                            }
                        }
                        Result::Err(err2) => {
                            println("Failed to reload: {}", err2);
                        }
                    }
                }
                Result::Err(err) => {
                    println("Failed to save: {}", err);
                }
            }
        }
        Result::Err(err) => {
            println("Failed to load .npz file: {}", err);
        }
    }
}
