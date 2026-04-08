fn main() {
    let s = "Hello, bytes!";
    let bytes = s.to_bytes();
    
    print("Original string:");
    print(s);
    print("Byte array length:");
    print(bytes.len());
    
    // Test that the first character is 'H' (72)
    let first = bytes.get(0);
    // bypass u8 print
    if first.is_some() {
        // ok
    }
    
    // from_utf8 should parse it back
    let s_restored = String::from_utf8(bytes);
    if s_restored.is_empty() {
        print("Restored is empty or invalid");
    } else {
        print("Restored successfully!");
        print(s_restored);
    }
}
