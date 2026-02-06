// Either<String, i64> 完全テスト

enum Either<L, R> {
    Left(L),
    Right(R),
}

fn main() {
    println("Test Either<String, i64>");
    
    // Left バリアント（String）
    let e1: Either<String, i64> = Either::Left("Hello");
    println("e1 (Left): created");
    
    // Right バリアント（i64）
    let e2: Either<String, i64> = Either::Right(42);
    println("e2 (Right): created");
    
    println("All tests passed!");
}
