fn main() {
    println("--- Testing String Comprehensive ---");

    // 1. Literals
    let s1 = "Hello";
    let s2 = "World";
    let empty = "";
    println("s1: {}", s1);
    println("s2: {}", s2);
    println("empty: '{}'", empty);

    // Escape sequences (basic check)
    println("Escape: \n\t\"quoted\"");

    // 2. Concatenation
    let s3 = s1 + " " + s2;
    println("Concat (+): {}", s3);
    if s3 == "Hello World" {
        println("Concat (+): OK");
    } else {
        println("Concat (+): FAIL");
    }

    let s4 = s1.concat(", ").concat(s2).concat("!");
    println("Concat (method): {}", s4);
    if s4 == "Hello, World!" {
        println("Concat (method): OK");
    } else {
        println("Concat (method): FAIL");
    }

    // 3. Equality
    if s1 == "Hello" {
        println("s1 == \"Hello\": OK");
    } else {
        println("s1 == \"Hello\": FAIL");
    }

    if s1 != s2 {
        println("s1 != s2: OK");
    } else {
        println("s1 != s2: FAIL");
    }

    // 4. Length
    println("len(s1): {}", s1.len());
    if s1.len() == 5 {
        println("len(s1): OK");
    } else {
        println("len(s1): FAIL");
    }

    if empty.len() == 0 {
        println("len(empty): OK");
    } else {
        println("len(empty): FAIL");
    }

    // 5. char_at
    let c = s1.char_at(0);
    println("char_at(0): {}", c);
    // Explicitly check against char literal if/when supported, or print comparison
    // For now printing is the main test, logic verification via logic ops if supported for Char
    // Assuming Char doesn't support '==' with string literals, testing via printing result for now.
    
    // 6. contains
    if s3.contains("World") {
        println("contains(\"World\"): OK");
    } else {
        println("contains(\"World\"): FAIL");
    }

    if !s3.contains("Universe") {
        println("!contains(\"Universe\"): OK");
    } else {
        println("!contains(\"Universe\"): FAIL");
    }

    // 7. to_i64
    let num_str = "12345";
    let num = num_str.to_i64();
    println("to_i64: {}", num);
    if num == 12345 {
        println("to_i64: OK");
    } else {
        println("to_i64: FAIL");
    }

    // 8. String::from_int
    let str_from_int = String::from_int(9876);
    println("from_int(9876): {}", str_from_int);
    if str_from_int == "9876" {
        println("from_int: OK");
    } else {
        println("from_int: FAIL");
    }

    println("--- String Tests Complete ---");
}
