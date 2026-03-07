// Basic trait definition and implementation test

trait Greet {
    fn greeting(self) -> String;
}

struct Person {
    name: String,
}

impl Greet for Person {
    fn greeting(self) -> String {
        "Hello, " + self.name
    }
}

struct Robot {
    id: i64,
}

impl Greet for Robot {
    fn greeting(self) -> String {
        "Beep boop " + String::from_int(self.id)
    }
}

fn main() {
    let p = Person { name: "Alice" };
    println(p.greeting());

    let r = Robot { id: 42 };
    println(r.greeting());
}
