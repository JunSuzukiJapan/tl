mod test_mod_a;
use test_mod_a::foo;
use test_mod_a::Point;

fn main() {
    print("In main");
    test_mod_a::foo();
    foo(); // via use
    
    let p = Point::new(10, 20);
    p.print();
}
