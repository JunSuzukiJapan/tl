enum Shape {
    Circle { radius: f32 },
    Rectangle { width: f32, height: f32 },
    Point
}

fn main() {
    let c = Shape::Circle { radius: 10.0 };
    let r = Shape::Rectangle { width: 5.0, height: 2.0 };
    let p = Shape::Point;

    let c_area = match c {
        Shape::Circle { radius } => 3.14159 * radius * radius,
        Shape::Rectangle { width, height } => width * height,
        Shape::Point => 0.0
    };

    let r_area = match r {
        Shape::Circle { radius: r } => 3.14159 * r * r,
        Shape::Rectangle { width: w, height: h } => w * h,
        Shape::Point => 0.0
    };

     let p_area = match p {
        Shape::Circle { radius: r } => 3.14159 * r * r,
        Shape::Rectangle { width: w, height: h } => w * h,
        Shape::Point => 0.0
    };

    print("Circle Area: ");
    print(c_area); // Exp: 314.159
    print("\n");
    // STDOUT: Circle Area: 314.159

    print("Rectangle Area: ");
    print(r_area); // Exp: 10.0
    print("\n");
    // STDOUT: Rectangle Area: 10
    
    print("Point Area: ");
    print(p_area); // Exp: 0.0
    print("\n");
    // STDOUT: Point Area: 0

    // Nested structures test (memory management)
    let c2 = Shape::Circle { radius: 5.0 };
    match c2 {
        Shape::Circle { radius } => {
            print("Nested Match Radius: ");
            print(radius);
            print("\n");
        },
        _ => {}
    }
    // STDOUT: Nested Match Radius: 5
}
