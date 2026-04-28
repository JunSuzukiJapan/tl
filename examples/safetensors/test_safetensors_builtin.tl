// examples/safetensors/test_safetensors_builtin.tl

fn main() {
    println("Testing SafeTensors builtin...");
    let res = SafeTensorsFile::load("examples/safetensors/test.safetensors");
    match res {
        Result::Ok(stf) => {
            println("Successfully loaded SafeTensors file!");
            println("Tensor count: {}", stf.tensor_count());
            let tensors = stf.tensors;
            let mut i = 0;
            while i < tensors.len() {
                let info = tensors.get(i).unwrap();
                println("Tensor {}: name={}, dtype={}, data_begin={}, data_end={}", i, info.name, info.dtype, info.data_begin, info.data_end);
                i = i + 1;
            }
            
            // Test get
            let weight_opt = stf.get("weight");
            match weight_opt {
                Option::Some(w) => {
                    println("Got weight tensor");
                }
                Option::None => {
                    println("Weight tensor not found");
                }
            }
            
            let save_res = stf.save("examples/safetensors/test_out.safetensors");
            match save_res {
                Result::Ok(ok) => {
                    println("Successfully saved test_out.safetensors: {}", ok);
                }
                Result::Err(err) => {
                    println("Failed to save: {}", err);
                }
            }
        }
        Result::Err(err) => {
            println("Failed to load SafeTensors file: {}", err);
        }
    }
}
