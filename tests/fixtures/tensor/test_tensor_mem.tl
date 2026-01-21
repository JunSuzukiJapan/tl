fn main() {
    print("Testing tensor with memory manager...");
    
    let mem_start = System::memory_mb();
    print("Start memory:");
    print(mem_start);
    
    for i in range(0, 3) {
        let x = Tensor::randn([2, 3], false);
        print("Created tensor");
    }
    
    let mem_end = System::memory_mb();
    print("End memory:");
    print(mem_end);
}
