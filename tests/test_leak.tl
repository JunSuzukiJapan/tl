extern fn tl_get_memory_mb() -> i64;
extern fn tl_print_i64(v: i64);

fn main() {
    print("Starting memory leak test...");
    
    let start_mem = tl_get_memory_mb();
    print("Initial memory (MB):");
    tl_print_i64(start_mem);

    // 100x100 Tensor ~ 40KB. 10000 iter = 400MB if leaked.
    // Loop enough times to see clear growth.
    for i in range(0, 5000) {

        let x = Tensor::randn([128, 128], true);
        let y = Tensor::randn([128, 128], true);
        let z = x + y;
        let w = z * x;
        let loss = w.sum(); 
        
        backward(loss);
        
        if i - ((i/250)*250) == 0 {
             let cur = tl_get_memory_mb();
             print("Iter:");
             print(i);
             print("Mem (MB):");
             tl_print_i64(cur);
        }
    }
    let end = tl_get_memory_mb();
    print("Final Mem (MB):");
    tl_print_i64(end);
    print("Done.");
}
