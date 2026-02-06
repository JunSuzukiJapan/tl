//! Simple GPU shader test

use tl_metal::{MetalTensor, DType};
use tl_metal::GpuOps;

fn main() {
    println!("=== tl_metal GPU TEST ===\n");
    
    // Test 1: from_slice と to_vec
    let a = MetalTensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], DType::F32);
    let a_data = a.to_vec::<f32>();
    println!("Test 1 - from_slice/to_vec:");
    println!("  Input: {:?}", a_data);
    assert_eq!(a_data, vec![1.0, 2.0, 3.0], "from_slice/to_vec failed");
    println!("  PASSED\n");
    
    // Test 2: GPU add
    let b = MetalTensor::from_slice(&[4.0f32, 5.0, 6.0], &[3], DType::F32);
    let c = a.add(&b);
    let c_data = c.to_vec::<f32>();
    println!("Test 2 - GPU add:");
    println!("  a: {:?}", a.to_vec::<f32>());
    println!("  b: {:?}", b.to_vec::<f32>());
    println!("  c = a + b: {:?}", c_data);
    println!("  Expected:  [5.0, 7.0, 9.0]");
    
    if c_data == vec![5.0, 7.0, 9.0] {
        println!("  PASSED\n");
    } else {
        println!("  FAILED - GPU shader not working correctly!\n");
    }
    
    println!("=== DONE ===");
}
