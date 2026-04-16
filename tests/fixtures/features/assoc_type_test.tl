// Phase 0 検証: 関連型 (associated type) の基本動作テスト

// テスト 1: 関連型の基本登録と解決
trait Converter {
    type Output;
    fn convert(self) -> Self::Output;
}

struct IntWrapper {
    val: i64,
}

// impl 側で具象型を明示
impl Converter for IntWrapper {
    type Output = String;
    fn convert(self) -> String {
        String::from_int(self.val)
    }
}

struct Doubler {
    val: i64,
}

impl Converter for Doubler {
    type Output = i64;
    fn convert(self) -> i64 {
        self.val * 2
    }
}

// テスト 2: Self::Output をシグネチャに使う trait + impl
trait Mapper {
    type Item;
    fn map(self) -> Self::Item;
    fn identity(self) -> Self::Item;
}

struct Wrapper {
    x: i64,
}

impl Mapper for Wrapper {
    type Item = i64;
    // impl 側でも Self::Item を使用（substitute_self_types のテスト）
    fn map(self) -> Self::Item {
        self.x + 1
    }
    fn identity(self) -> Self::Item {
        self.x
    }
}

fn main() {
    // テスト 1
    let w = IntWrapper { val: 42 };
    let s = w.convert();
    println(s);

    let d = Doubler { val: 21 };
    let n = d.convert();
    println(String::from_int(n));

    // テスト 2
    let wr = Wrapper { x: 10 };
    let mapped = wr.map();
    println(String::from_int(mapped));

    let wr2 = Wrapper { x: 99 };
    let id = wr2.identity();
    println(String::from_int(id));
}
