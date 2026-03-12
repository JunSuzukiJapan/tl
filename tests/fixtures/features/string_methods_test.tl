// String methods integration test
// Tests: trim, starts_with, ends_with, replace, substring, is_empty

// STDOUT: [trim] |hello|
// STDOUT: [starts_with] true
// STDOUT: [starts_with_neg] false
// STDOUT: [ends_with] true
// STDOUT: [ends_with_neg] false
// STDOUT: [replace] hello world
// STDOUT: [substring] ello
// STDOUT: [is_empty] true
// STDOUT: [is_empty_neg] false

fn main() {
    // trim
    let s = "  hello  ";
    let trimmed = s.trim();
    println("[trim] |{}|", trimmed);

    // starts_with
    let s2 = "hello world";
    let result = s2.starts_with("hello");
    println("[starts_with] {}", result);
    let result2 = s2.starts_with("world");
    println("[starts_with_neg] {}", result2);

    // ends_with
    let result3 = s2.ends_with("world");
    println("[ends_with] {}", result3);
    let result4 = s2.ends_with("hello");
    println("[ends_with_neg] {}", result4);

    // replace
    let s3 = "hello earth";
    let replaced = s3.replace("earth", "world");
    println("[replace] {}", replaced);

    // substring
    let s4 = "hello";
    let sub = s4.substring(1, 4);
    println("[substring] {}", sub);

    // is_empty
    let empty = "";
    let not_empty = "abc";
    println("[is_empty] {}", empty.is_empty());
    println("[is_empty_neg] {}", not_empty.is_empty());
}
