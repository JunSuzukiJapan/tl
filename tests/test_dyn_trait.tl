trait Animal {
    fn speak(self) -> String;
}

struct Dog {
    name: String
}

impl Animal for Dog {
    fn speak(self) -> String {
        return self.name + " says Woof!";
    }
}

struct Cat {
    name: String
}

impl Animal for Cat {
    fn speak(self) -> String {
        return self.name + " says Meow!";
    }
}

fn make_animal(is_dog: bool) -> dyn Animal {
    if is_dog {
        return Dog { name: "Rex" };
    } else {
        return Cat { name: "Luna" };
    }
}

fn test_animal(a: dyn Animal) -> String {
    return a.speak();
}

fn main() {
    let a1 = make_animal(true);
    let a2 = make_animal(false);

    let s1 = a1.speak();
    let s2 = a2.speak();

    println(s1);
    println(s2);

    assert(s1 == "Rex says Woof!", "Assertion failed");
    assert(s2 == "Luna says Meow!", "Assertion failed");

    // Pass as parameter
    assert(test_animal(a1) == "Rex says Woof!", "Assertion failed");
    assert(test_animal(a2) == "Luna says Meow!", "Assertion failed");
    
    println("dyn Trait test passed!");
}
