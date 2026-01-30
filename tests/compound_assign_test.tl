struct S { x: i64 }
struct ST { t: Tensor<f32, 2> }

fn test_i64() {
    println("Testing I64...");
    let mut a = 10;
    a += 5;
    if a != 15 { println("Error: I64 += failed"); }
    a -= 3;
    if a != 12 { println("Error: I64 -= failed"); }
    a *= 2;
    if a != 24 { println("Error: I64 *= failed"); }
    a /= 4;
    if a != 6 { println("Error: I64 /= failed"); }
    a %= 4;
    if a != 2 { println("Error: I64 %= failed"); }

    let mut s = S { x: 10 };
    s.x += 5;
    if s.x != 15 { println("Error: I64 field += failed"); }
    s.x -= 3;
    if s.x != 12 { println("Error: I64 field -= failed"); }
    s.x *= 2;
    if s.x != 24 { println("Error: I64 field *= failed"); }
    s.x /= 4;
    if s.x != 6 { println("Error: I64 field /= failed"); }
    s.x %= 4;
    if s.x != 2 { println("Error: I64 field %= failed"); }
}

fn test_f32() {
    println("Testing F32...");
    let mut a = 10.0;
    a += 5.0;
    if a != 15.0 { println("Error: F32 += failed"); }
    a -= 3.0;
    if a != 12.0 { println("Error: F32 -= failed"); }
    a *= 2.0;
    if a != 24.0 { println("Error: F32 *= failed"); }
    a /= 4.0;
    if a != 6.0 { println("Error: F32 /= failed"); }
}

fn test_tensor() {
    println("Testing Tensor...");
    let mut a = Tensor::ones([2, 2]);
    a += Tensor::ones([2, 2]);
    if a.sumall().item() != 8.0 { println("Error: Tensor += failed"); }
    
    a -= Tensor::ones([2, 2]);
    if a.sumall().item() != 4.0 { println("Error: Tensor -= failed"); }
    
    a *= 2.0; // Scalar multiply
    if a.sumall().item() != 8.0 { println("Error: Tensor *= scalar failed"); }

    let mut s = ST { t: Tensor::ones([2, 2]) };
    s.t += Tensor::ones([2, 2]);
    if s.t.sumall().item() != 8.0 { println("Error: Tensor field += failed"); }
    
    s.t -= Tensor::ones([2, 2]);
    if s.t.sumall().item() != 4.0 { println("Error: Tensor field -= failed"); }
    
    s.t *= 3.0;
    if s.t.sumall().item() != 12.0 { println("Error: Tensor field *= failed"); }
}

fn main() {
    test_i64();
    test_f32();
    test_tensor();
    println("All tests finished.");
}
