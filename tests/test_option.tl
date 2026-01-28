// Test for Option<T> type - simple version

fn main() {
    // Test Option::some
    let someVal = Option::some(42);
    print("is_some: ");
    if someVal.is_some() {
        println("true");
    } else {
        println("false");
    }
    
    print("is_none: ");
    if someVal.is_none() {
        println("true");
    } else {
        println("false");
    }
    
    // Test unwrap
    let val = someVal.unwrap();
    print("unwrap: ");
    println(val);
    
    // Test unwrap_or on Some
    let someDefault = someVal.unwrap_or(999);
    print("some unwrap_or: ");
    println(someDefault);
    
    // Test Option::none with explicit type
    let noneVal = Option<i64>::none();
    print("none is_some: ");
    if noneVal.is_some() {
        println("true");
    } else {
        println("false");
    }
    
    // Test unwrap_or
    let defaultVal = noneVal.unwrap_or(100);
    print("unwrap_or: ");
    println(defaultVal);
}
