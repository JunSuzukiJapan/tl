
fn main() {
    print("Starting memory leak test...");
    
    let start_mem = System::memory_mb();
    print("Initial memory (MB):");
    print(start_mem);

    // 100x100 Tensor ~ 40KB. 10000 iter = 400MB if leaked.
    // Loop enough times to see clear growth.
    for i in range(0, 5000) {

        let x = Tensor::randn([128, 128], true);
        let y = Tensor::randn([128, 128], true);
        let z = x + y;
        let w = z * x;
        let loss = w.sum(); 
        
        loss.backward();
        
        if i - ((i/250)*250) == 0 {
             let cur = System::memory_mb();
             print("Iter:");
             print(i);
             print("Mem (MB):");
             print(cur);
        }
    }
    let end = System::memory_mb();
    print("Final Mem (MB):");
    print(end);
    print("Done.");
}
