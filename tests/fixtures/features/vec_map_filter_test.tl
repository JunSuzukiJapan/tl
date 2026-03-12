// Vec map/filter with for loop test
fn main() {
    let mut v: Vec<i64> = Vec::new();
    v.push(1);
    v.push(2);
    v.push(3);
    v.push(4);
    v.push(5);

    // Test for loop over normal Vec
    for val in v {
        println("v: {}", val);
    }

    // Test map + for loop
    let doubled = v.map(|x: i64| x * 2);
    for val in doubled {
        println("map: {}", val);
    }

    // Test filter + for loop
    let evens = v.filter(|x: i64| x % 2 == 0);
    for val in evens {
        println("filter: {}", val);
    }

    // Test any/all
    println("any even: {}", v.any(|x: i64| x % 2 == 0));
    println("all positive: {}", v.all(|x: i64| x > 0));
}
