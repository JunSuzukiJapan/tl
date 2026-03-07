// Trait with default methods test

trait Describable {
    fn name(self) -> String;

    // Default method that uses the required method
    fn describe(self) -> String {
        "I am " + self.name()
    }
}

struct Cat {
    species: String,
}

impl Describable for Cat {
    fn name(self) -> String {
        self.species
    }
    // describe() uses the default implementation
}

struct Dog {
    breed: String,
}

impl Describable for Dog {
    fn name(self) -> String {
        self.breed
    }

    // Override default
    fn describe(self) -> String {
        "Woof! I am a " + self.name()
    }
}

fn main() {
    let c = Cat { species: "Siamese" };
    println(c.name());
    println(c.describe());

    let d = Dog { breed: "Labrador" };
    println(d.name());
    println(d.describe());
}
