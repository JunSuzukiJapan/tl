// =============================================================================
// 網羅的な型推論テスト
// =============================================================================
// このファイルは、TLコンパイラの型推論機能をテストします。
// TLでは浮動小数点リテラルはf32として扱われます。
// =============================================================================

// -----------------------------------------------------------------------------
// Section 1: プリミティブ型の推論
// -----------------------------------------------------------------------------
fn test_primitive_inference() {
    // 整数リテラル → i64
    let a = 42;
    
    // 浮動小数点リテラル → f32 (TLのデフォルト)
    let b = 3.14;
    
    // 文字列リテラル → String
    let c = "hello";
    
    // ブーリアンリテラル → bool
    let d = true;
    let e = false;
    
    // 演算結果の推論
    let g = a + 10;          // i64 + i64 → i64
    let h = b * 2.0;         // f32 * f32 → f32
    let i = a > 0;           // 比較 → bool
    let j = d && e;          // bool && bool → bool
    
    print("Section 1: Primitive inference OK\n");
}

// -----------------------------------------------------------------------------
// Section 2: Tensor型の推論
// -----------------------------------------------------------------------------
fn test_tensor_inference() {
    // Tensor静的メソッドからの推論
    let F = Tensor::zeros([2, 3]);
    let G = Tensor::ones([3, 3]);
    
    // Tensor演算結果の推論
    let D = F + F;   // Tensor + Tensor → Tensor
    let E = F * 2.0; // Tensor * scalar → Tensor
    
    // Tensorインスタンスメソッドからの推論
    let sum_val = F.sum();  // → Tensor (0-d)
    
    print("Section 2: Tensor inference OK\n");
}

// -----------------------------------------------------------------------------
// Section 3: Vec型の推論
// -----------------------------------------------------------------------------
fn test_vec_inference() {
    // 型注釈付きVec
    let v1: Vec<i64> = Vec::new();
    v1.push(1);
    v1.push(2);
    
    // len()からの推論
    let length = v1.len();   // → i64
    
    // is_empty()
    let empty = v1.is_empty(); // → bool
    
    print("Section 3: Vec inference OK\n");
}

// -----------------------------------------------------------------------------
// Section 4: HashMap型の推論
// -----------------------------------------------------------------------------
fn test_hashmap_inference() {
    // 型注釈付きHashMap
    let m1: HashMap<String, i64> = HashMap::new();
    m1.insert("one", 1);
    m1.insert("two", 2);
    
    // len()
    let map_len = m1.len();  // → i64
    
    print("Section 4: HashMap inference OK\n");
}

// -----------------------------------------------------------------------------
// Section 5: 関数戻り値からの推論
// -----------------------------------------------------------------------------
fn returns_i64() -> i64 {
    return 42;
}

fn returns_string() -> String {
    return "hello";
}

fn returns_bool() -> bool {
    return true;
}

fn test_function_return_inference() {
    // 関数呼び出し結果からの推論
    let a = returns_i64();    // → i64
    let b = returns_string(); // → String
    let c = returns_bool();   // → bool
    
    // 関数呼び出し結果を使った演算
    let d = a + 10;           // i64
    let e = c && true;        // bool
    
    print("Section 5: Function return inference OK\n");
}

// -----------------------------------------------------------------------------
// Section 6: 構造体フィールドアクセスからの推論
// -----------------------------------------------------------------------------
struct Point {
    x: f32,
    y: f32,
}

struct Container {
    value: i64,
    name: String,
}

fn test_struct_field_inference() {
    let p = Point { x: 1.0, y: 2.0 };
    
    // フィールドアクセスからの型推論
    let px = p.x;             // → f32
    let py = p.y;             // → f32
    
    // 演算に使用
    let distance = px * px + py * py;  // f32
    
    let c = Container { value: 100, name: "test" };
    let v = c.value;          // → i64
    let n = c.name;           // → String
    
    print("Section 6: Struct field inference OK\n");
}

// -----------------------------------------------------------------------------
// Section 7: If式からの推論
// -----------------------------------------------------------------------------
fn test_if_expression_inference() {
    let cond = true;
    
    // if式の両ブランチから推論
    let a = if cond { 1 } else { 2 };       // → i64
    let b = if cond { "yes" } else { "no" }; // → String
    let c = if cond { 1.5 } else { 2.5 };   // → f32
    
    // ネストしたif式
    let d = if cond {
        if true { 10 } else { 20 }
    } else {
        30
    };  // → i64
    
    print("Section 7: If expression inference OK\n");
}

// -----------------------------------------------------------------------------
// Section 8: タプルからの推論
// -----------------------------------------------------------------------------
fn test_tuple_inference() {
    // タプルリテラル
    let t1 = (1, "hello", 3.14);  // → (i64, String, f32)
    
    // タプル要素アクセス
    let first = t1.0;             // → i64
    let second = t1.1;            // → String
    let third = t1.2;             // → f32
    
    print("Section 8: Tuple inference OK\n");
}

// -----------------------------------------------------------------------------
// Section 9: 静的メソッド呼び出しからの推論（非コンストラクタ）
// これは先ほど修正したバグのリグレッションテスト
// -----------------------------------------------------------------------------
fn test_static_method_inference() {
    // Arena静的メソッド（修正済みバグのテスト）
    let offset = Arena::get_offset();   // → i64
    let is_active = Arena::is_active(); // → bool
    
    // System静的メソッド
    let time = System::time();          // → i64
    
    // String静的メソッド
    let s = String::from_int(42);       // → String
    
    print("Section 9: Static method inference OK\n");
}

// -----------------------------------------------------------------------------
// Section 10: 複合式の推論
// -----------------------------------------------------------------------------
fn test_complex_expression_inference() {
    let a = 10;
    let b = 20;
    
    // 複合算術式
    let c = (a + b) * 2 - a / 2;  // → i64
    
    // 複合比較式
    let d = a < b && b > 0 || a == 10;  // → bool
    
    // チェーンメソッド呼び出し
    let v: Vec<i64> = Vec::new();
    v.push(1);
    let len = v.len() + 1;  // → i64
    
    print("Section 10: Complex expression inference OK\n");
}

// -----------------------------------------------------------------------------
// Section 11: ジェネリクスの具体化推論
// -----------------------------------------------------------------------------
fn test_generic_instantiation() {
    // ジェネリック構造体のインスタンス化
    let v1: Vec<i64> = Vec::new();
    let v2: Vec<String> = Vec::new();
    let v3: Vec<f32> = Vec::new();
    
    v1.push(1);
    v2.push("hello");
    v3.push(3.14);
    
    // 異なる型のHashMap
    let m1: HashMap<i64, String> = HashMap::new();
    let m2: HashMap<String, i64> = HashMap::new();
    
    m1.insert(1, "one");
    m2.insert("one", 1);
    
    print("Section 11: Generic instantiation OK\n");
}

// -----------------------------------------------------------------------------
// Section 12: Tensorリテラルからの推論（letで）
// -----------------------------------------------------------------------------
fn test_tensor_literal_inference() {
    // letでのTensorリテラル（型注釈なし）
    let t1 = [[1.0, 2.0], [3.0, 4.0]];  // → Tensor
    
    // letでのTensorリテラル（1D）
    let t2 = [1.0, 2.0, 3.0];  // → Tensor
    
    // Tensorリテラルの演算
    let t3 = t1 + t1;  // Tensor + Tensor → Tensor
    
    print("Section 12: Tensor literal inference OK\n");
}

// -----------------------------------------------------------------------------
// メイン関数：すべてのテストを実行
// -----------------------------------------------------------------------------
fn main() {
    print("=== 型推論テスト開始 ===\n\n");
    
    test_primitive_inference();
    test_tensor_inference();
    test_vec_inference();
    test_hashmap_inference();
    test_function_return_inference();
    test_struct_field_inference();
    test_if_expression_inference();
    test_tuple_inference();
    test_static_method_inference();
    test_complex_expression_inference();
    test_generic_instantiation();
    test_tensor_literal_inference();
    
    print("\n=== すべての型推論テストが成功しました ===\n");
}
