// Vec map/filter basic test - verify len only
fn main() {
    let mut v: Vec<i64> = Vec::new();
    v.push(1);
    v.push(2);
    v.push(3);
    v.push(4);
    v.push(5);

    // Test map: double each element
    let doubled = v.map(|x: i64| x * 2);
    println("map len: {}", doubled.len());

    // Test filter: keep even numbers
    let evens = v.filter(|x: i64| x % 2 == 0);
    println("filter len: {}", evens.len());

    // Test any/all (already working)
    println("any even: {}", v.any(|x: i64| x % 2 == 0));
    println("all positive: {}", v.all(|x: i64| x > 0));
}
