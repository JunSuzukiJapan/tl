
fn main() {
    let a = Tensor::randn([2, 3], false);
    let b = Tensor::randn([4, 5], false); // Incompatible shape for matmul (2x3 * 4x5)
    
    // This should fail in backend, return NULL
    let c = a.matmul(b); 
    
    // This should cause segfault because c is NULL and compiler doesn't check
    print(c);
}
