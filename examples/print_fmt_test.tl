fn main() {
    let x = 42;
    let y = 3.14;
    let name = "TensorLogic";
    
    println("--- Testing Formatted Print ---");
    println("Integer: {}", x);
    println("Float: {}", y);
    println("String: {}", name);
    println("Multiple: {} | {} | {}", x, y, name);
    
    print("No Newline: {} ", x);
    println("With Newline: {}", y);
    
    println("Adjacent: {}{}", "A", "B");
    println("Surrounded: <{}>", "Center");
    
    println("Done.");
}
