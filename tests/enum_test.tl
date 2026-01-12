
// 1. Simple Enum (C-style)
enum Color {
    Red,
    Green,
    Blue
}

// 2. Complex Enum (Tagged Union)
enum Message {
    Quit,
    Move { x: i64, y: i64 },
    Write { content: String }, // String object
    ChangeColor { r: i64, g: i64, b: i64 }
}

enum Wrapper {
    WrappedMsg { msg: Message },
    None
}

fn get_color_name(c: Color) -> String {
    match c {
        Color::Red => "Red",
        Color::Green => "Green",
        Color::Blue => "Blue"
    }
}

fn main() {
    print("--- Test 1: Simple Enum ---\n");
    let c = Color::Green;
    print("Color Code: ");
    print(get_color_name(c));
    print("\n");

    print("\n--- Test 2: Pattern Matching with Binding ---\n");
    let msg1 = Message::Move { x: 10, y: 20 };
    match msg1 {
        Message::Quit => print("Quit\n"),
        Message::Move { x, y } => {
            print("Move to x: ");
            print(x);
            print(", y: ");
            print(y);
            print("\n");
        },
        Message::Write { content } => {
            print("Write: ");
            print(content);
            print("\n");
        },
        Message::ChangeColor { r, g, b } => print("ChangeColor\n")
    }

    print("\n--- Test 3: Mixed Variants ---\n");
    let msg2 = Message::Write { content: "Hello Enum!" };
    match msg2 {
        Message::Write { content } => {
            print("Message Content: ");
            print(content);
            print("\n");
        },
        _ => print("Other\n")
    }

    print("\n--- Test 4: Nested Enum (Wrapper) ---\n");
    let wrap = Wrapper::WrappedMsg { msg: msg1 };
    match wrap {
        Wrapper::WrappedMsg { msg } => {
            print("Unwrapped: ");
            match msg {
                 Message::Move { x, y } => {
                     print("Move x=");
                     print(x);
                     print(", y=");
                     print(y);
                     print("\n");
                 },
                 _ => print("Unknown inner\n")
            }
        },
        Wrapper::None => print("None\n")
    }

     print("\n--- Test 5: Wildcard Match ---\n");
     match c {
         Color::Red => print("Red\n"),
         _ => print("Not Red\n")
     }
}
