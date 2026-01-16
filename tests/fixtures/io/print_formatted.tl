fn main() {
    let x = 42;
    let y = 3.14;
    let name = "TensorLogic";
    
    println("--- Testing Formatted Print ---");
    // STDOUT: --- Testing Formatted Print ---
    
    println("Integer: {}", x);
    // STDOUT: Integer: 42
    
    println("Float: {}", y);
    // STDOUT: Float: 3.14
    
    println("String: {}", name);
    // STDOUT: String: TensorLogic
    
    println("Multiple: {} | {} | {}", x, y, name);
    // STDOUT: Multiple: 42 | 3.14 | TensorLogic
    
    print("No Newline: {} ", x);
    println("With Newline: {}", y);
    // STDOUT: No Newline: 42 With Newline: 3.14
    
    println("Adjacent: {}{}", "A", "B");
    // STDOUT: Adjacent: AB
    
    println("Surrounded: <{}>", "Center");
    // STDOUT: Surrounded: <Center>
    
    println("Done.");
    // STDOUT: Done.
}
