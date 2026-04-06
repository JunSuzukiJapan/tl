fn test_regex_basic() {
    let re = Regex::new("hello (worl)+d");
    if re.is_valid() {
        println("OK: Regex is valid");
        
        let match1 = re.is_match("hello world");
        if match1 {
            println("OK: Regex matches 'hello world'");
        } else {
            println("FAIL: Regex should match 'hello world'");
        }
        
        let match2 = re.is_match("hello worlworld");
        if match2 {
            println("OK: Regex matches 'hello worlworld'");
        } else {
            println("FAIL: Regex should match 'hello worlworld'");
        }
        
        let match3 = re.is_match("hello abc");
        if !match3 {
            println("OK: Regex correctly rejects 'hello abc'");
        } else {
            println("FAIL: Regex should not match 'hello abc'");
        }
        
        re.release();
    } else {
        println("FAIL: Failed to compile regex");
    }
}

fn test_regex_replace() {
    let re = Regex::new("[0-9]+");
    if re.is_valid() {
        let replaced = re.replace("There are 123 apples and 45 oranges.", "#");
        if replaced == "There are # apples and # oranges." {
            println("OK: Regex replacement works");
        } else {
            println("FAIL: Regex replacement returned '{}'", replaced);
        }
        
        re.release();
    } else {
        println("FAIL: Failed to compile replacement regex");
    }
}

fn main() {
    test_regex_basic();
    test_regex_replace();
}
