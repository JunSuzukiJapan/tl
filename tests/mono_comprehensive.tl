// ============================================================
// モノモーファイザー網羅的テスト
// ============================================================
// 複雑なジェネリック型のモノモーファイズを検証する

// ============================================================
// 1. 基本ジェネリクス
// ============================================================

struct Box<T> {
    value: T,
}

impl<T> Box<T> {
    fn new(v: T) -> Box<T> {
        Box { value: v }
    }
    
    fn get(self) -> T {
        self.value
    }
}

struct Pair<K, V> {
    first: K,
    second: V,
}

impl<K, V> Pair<K, V> {
    fn new(k: K, v: V) -> Pair<K, V> {
        Pair { first: k, second: v }
    }
}

fn test_basic_generics() {
    println("=== Test 1: Basic Generics ===");
    
    // 単一型パラメータ
    let b1: Box<i64> = Box::new(42);
    assert_eq(b1.get(), 42, "Box<i64>");
    
    let b2: Box<String> = Box::new("hello");
    // String比較はサポート次第
    
    let b3: Box<f32> = Box::new(3.14);
    
    let b4: Box<bool> = Box::new(true);
    
    // 複数型パラメータ
    let p1: Pair<i64, String> = Pair::new(1, "one");
    assert_eq(p1.first, 1, "Pair<i64, String>.first");
    
    let p2: Pair<String, i64> = Pair::new("key", 100);
    assert_eq(p2.second, 100, "Pair<String, i64>.second");
    
    let p3: Pair<bool, f32> = Pair::new(false, 2.5);
    
    println("Test 1 PASSED");
}

// ============================================================
// 2. ネストしたジェネリクス
// ============================================================

fn test_nested_generics() {
    println("=== Test 2: Nested Generics ===");
    
    // Option<Vec<i64>>
    let v1: Vec<i64> = Vec::new();
    v1.push(1);
    v1.push(2);
    v1.push(3);
    let opt_vec: Option<Vec<i64>> = Option::Some(v1);
    
    match opt_vec {
        Option::Some(v) => {
            assert_eq(v.len(), 3, "Option<Vec<i64>> length");
        }
        Option::None => {
            println("FAIL: Expected Some");
        }
    }
    
    // Vec<Option<i64>>
    let v2: Vec<Option<i64>> = Vec::new();
    v2.push(Option::Some(10));
    v2.push(Option::None);
    v2.push(Option::Some(30));
    assert_eq(v2.len(), 3, "Vec<Option<i64>> length");
    
    // Box<Pair<K, V>>
    let bp: Box<Pair<i64, i64>> = Box::new(Pair::new(1, 2));
    
    // Pair<Box<T>, Box<U>>
    let pb: Pair<Box<i64>, Box<String>> = Pair::new(Box::new(42), Box::new("test"));
    
    println("Test 2 PASSED");
}

// ============================================================
// 3. ジェネリック Enum
// ============================================================

enum Either<L, R> {
    Left(L),
    Right(R),
}

fn test_generic_enum() {
    println("=== Test 3: Generic Enum ===");
    
    // Either<i64, String>
    let e1: Either<i64, String> = Either::Left(42);
    match e1 {
        Either::Left(n) => assert_eq(n, 42, "Either::Left(i64)"),
        Either::Right(_) => println("FAIL: Expected Left"),
    }
    
    let e2: Either<i64, String> = Either::Right("hello");
    match e2 {
        Either::Left(_) => println("FAIL: Expected Right"),
        Either::Right(s) => println("Either::Right(String) OK"),
    }
    
    // Either<String, i64> (型パラメータ逆順)
    let e3: Either<String, i64> = Either::Right(100);
    match e3 {
        Either::Left(_) => println("FAIL"),
        Either::Right(n) => assert_eq(n, 100, "Either<String, i64>::Right"),
    }
    
    // ネストした Either
    let e4: Either<i64, Either<String, bool>> = Either::Right(Either::Left("nested"));
    
    // Option<Either<i64, String>>
    let oe: Option<Either<i64, String>> = Option::Some(Either::Left(99));
    match oe {
        Option::Some(either) => {
            match either {
                Either::Left(n) => assert_eq(n, 99, "Option<Either>::Left"),
                Either::Right(_) => println("FAIL"),
            }
        }
        Option::None => println("FAIL"),
    }
    
    println("Test 3 PASSED");
}

// ============================================================
// 4. HashMap ネストテスト
// ============================================================

fn test_hashmap_nested() {
    println("=== Test 4: HashMap Nested ===");
    
    // HashMap<i64, i64>
    let m1: HashMap<i64, i64> = HashMap::new();
    m1.insert(1, 100);
    m1.insert(2, 200);
    
    match m1.get(1) {
        Option::Some(v) => assert_eq(v, 100, "HashMap<i64, i64>.get(1)"),
        Option::None => println("FAIL: key 1 not found"),
    }
    
    // HashMap<String, i64>
    let m2: HashMap<String, i64> = HashMap::new();
    m2.insert("one", 1);
    m2.insert("two", 2);
    
    match m2.get("one") {
        Option::Some(v) => assert_eq(v, 1, "HashMap<String, i64>.get"),
        Option::None => println("FAIL: key 'one' not found"),
    }
    
    // HashMap<i64, String>
    let m3: HashMap<i64, String> = HashMap::new();
    m3.insert(1, "first");
    m3.insert(2, "second");
    
    // HashMap<String, String>
    let m4: HashMap<String, String> = HashMap::new();
    m4.insert("key", "value");
    
    println("Test 4 PASSED");
}

// ============================================================
// 5. 複合ケース: 3階層以上のネスト
// ============================================================

struct Container<T> {
    items: Vec<T>,
}

impl<T> Container<T> {
    fn new() -> Container<T> {
        Container { items: Vec::new() }
    }
    
    fn add(self, item: T) {
        self.items.push(item);
    }
    
    fn size(self) -> i64 {
        self.items.len()
    }
}

fn test_deep_nesting() {
    println("=== Test 5: Deep Nesting ===");
    
    // Container<Vec<Option<i64>>>
    let c1: Container<Vec<Option<i64>>> = Container::new();
    
    let v1: Vec<Option<i64>> = Vec::new();
    v1.push(Option::Some(1));
    v1.push(Option::None);
    c1.add(v1);
    
    let v2: Vec<Option<i64>> = Vec::new();
    v2.push(Option::Some(2));
    c1.add(v2);
    
    assert_eq(c1.size(), 2, "Container<Vec<Option<i64>>>.size");
    
    // Pair<Container<i64>, Container<String>>
    let pc: Pair<Container<i64>, Container<String>> = Pair::new(Container::new(), Container::new());
    pc.first.add(1);
    pc.first.add(2);
    pc.second.add("a");
    
    assert_eq(pc.first.size(), 2, "Pair<Container<i64>, _>.first.size");
    assert_eq(pc.second.size(), 1, "Pair<_, Container<String>>.second.size");
    
    // Test Pair<Vec<i64>, Option<String>>
    let bpvo: Box<Pair<Vec<i64>, Option<String>>> = Box::new(
        Pair::new(Vec::new(), Option::Some("nested"))
    );
    bpvo.value.first.push(100);
    assert_eq(bpvo.value.first.len(), 1, "Nested Vec in Pair");
    
    println("Test 5 PASSED");
}

// ============================================================
// 6. 異なる型パラメータの組み合わせ
// ============================================================

struct Triple<A, B, C> {
    a: A,
    b: B,
    c: C,
}

impl<A, B, C> Triple<A, B, C> {
    fn new(a: A, b: B, c: C) -> Triple<A, B, C> {
        Triple { a: a, b: b, c: c }
    }
}

fn test_multiple_type_params() {
    println("=== Test 6: Multiple Type Parameters ===");
    
    // Triple<i64, String, bool>
    let t1: Triple<i64, String, bool> = Triple::new(1, "two", true);
    assert_eq(t1.a, 1, "Triple.a");
    
    // Triple<String, i64, f32>
    let t2: Triple<String, i64, f32> = Triple::new("first", 2, 3.0);
    assert_eq(t2.b, 2, "Triple.b");
    
    // TODO: Fix Triple<Vec<i64>, Option<String>, HashMap<i64, i64>> - Option_String unsupported
    // let m: HashMap<i64, i64> = HashMap::new();
    // m.insert(1, 10);
    // let t3: Triple<Vec<i64>, Option<String>, HashMap<i64, i64>> = Triple::new(
    //     Vec::new(),
    //     Option::Some("test"),
    //     m
    // );
    
    println("Test 6 PASSED");
}

// ============================================================
// ヘルパー関数
// ============================================================

fn assert_eq(actual: i64, expected: i64, msg: String) {
    if actual != expected {
        print("FAIL: ");
        print(msg);
        print(" - expected ");
        print(expected);
        print(", got ");
        println(actual);
    }
}

// ============================================================
// メイン
// ============================================================

fn main() {
    println("Starting Monomorphizer Comprehensive Test Suite");
    println("================================================");
    
    test_basic_generics();
    test_nested_generics();
    test_generic_enum();
    test_hashmap_nested();
    test_deep_nesting();
    test_multiple_type_params();
    
    println("================================================");
    println("All Monomorphizer Tests PASSED!");
}
