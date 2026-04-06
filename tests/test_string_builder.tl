fn test_string_builder_basic() {
    let mut sb = StringBuilder::new();
    
    if sb.is_empty() {
        println("OK: StringBuilder is initially empty");
    } else {
        println("FAIL: StringBuilder is not empty initially");
    }
    
    sb.append("Hello");
    sb.append(", ");
    sb.append("World!");
    
    if sb.len() == 13 {
        println("OK: StringBuilder correctly appended text");
    } else {
        println("FAIL: StringBuilder len is {}, expected 13", sb.len());
    }
    
    let result = sb.build();
    if result == "Hello, World!" {
        println("OK: build() == \"Hello, World!\"");
    } else {
        println("FAIL: build() == \"{}\"", result);
    }
}

fn test_string_builder_numbers() {
    let mut sb = StringBuilder::with_capacity(20);
    
    sb.append("Count: ");
    sb.append_i64(1);
    sb.append(", ");
    sb.append_i64(2);
    sb.append(", ");
    sb.append_i64(3);
    
    let result = sb.build();
    if result == "Count: 1, 2, 3" {
        println("OK: append_i64 works correctly");
    } else {
        println("FAIL: append_i64 returned \"{}\"", result);
    }
    
    sb.clear();
    if sb.is_empty() {
        println("OK: clear() works");
    } else {
        println("FAIL: clear() did not empty builder");
    }
}

fn test_string_builder_multibyte() {
    let mut sb = StringBuilder::new();
    sb.append("こんにちは");
    sb.append("世界");
    
    let result = sb.build();
    if result == "こんにちは世界" {
        println("OK: multibyte characters work correctly");
    } else {
        println("FAIL: multibyte returned \"{}\"", result);
    }
}

fn main() {
    test_string_builder_basic();
    test_string_builder_numbers();
    test_string_builder_multibyte();
}
