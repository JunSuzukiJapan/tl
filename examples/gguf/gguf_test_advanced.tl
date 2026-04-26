// ============================================================
// GGUF Advanced Test — TLコンパイラのバグ探索
// ============================================================
// テスト項目:
//   1. Vec<u8> の push/get/len サイクル
//   2. i64 → u8 キャスト（境界値）
//   3. Vec<u8> を複数の関数に連続して渡す
//   4. u8 → i64 キャスト（境界値）
//   5. Vec<u8> に Vec<u8> をアペンドする
//   6. String::from_chars と Vec<i64>
//   7. String.to_bytes と Vec<u8>
//   8. struct を返す関数の連鎖
//   9. match Result<Vec<u8>, String>
//  10. GGUF ラウンドトリップ (全メタデータ型)
// ============================================================

// --- ヘルパー ---

fn read_u32_le(data: Vec<u8>, offset: i64) -> i64 {
    let b0 = data.get(offset).unwrap() as i64;
    let b1 = data.get(offset + 1).unwrap() as i64;
    let b2 = data.get(offset + 2).unwrap() as i64;
    let b3 = data.get(offset + 3).unwrap() as i64;
    b0 + b1 * 256 + b2 * 65536 + b3 * 16777216
}

fn write_u32_le(buf: Vec<u8>, val: i64) -> Vec<u8> {
    let mut b = buf;
    b.push((val % 256) as u8);
    b.push(((val / 256) % 256) as u8);
    b.push(((val / 65536) % 256) as u8);
    b.push(((val / 16777216) % 256) as u8);
    b
}

fn read_u64_le(data: Vec<u8>, offset: i64) -> i64 {
    let lo = read_u32_le(data, offset);
    let hi = read_u32_le(data, offset + 4);
    lo + hi * 4294967296
}

fn write_u64_le(buf: Vec<u8>, val: i64) -> Vec<u8> {
    let lo = val % 4294967296;
    let hi = val / 4294967296;
    let b = write_u32_le(buf, lo);
    write_u32_le(b, hi)
}

// --- テスト 1: Vec<u8> 基本操作 ---
fn test_vec_u8_basic() {
    println("--- Test 1: Vec<u8> basic ---");
    let mut v: Vec<u8> = Vec::new();
    v.push(0 as u8);
    v.push(127 as u8);
    v.push(255 as u8);
    v.push(1 as u8);

    println("len={}", v.len());
    println("v[0]={}", v.get(0).unwrap() as i64);
    println("v[1]={}", v.get(1).unwrap() as i64);
    println("v[2]={}", v.get(2).unwrap() as i64);
    println("v[3]={}", v.get(3).unwrap() as i64);
}

// --- テスト 2: i64→u8 境界値キャスト ---
fn test_cast_i64_to_u8() {
    println("--- Test 2: i64->u8 cast boundary ---");
    let mut v: Vec<u8> = Vec::new();

    // 0, 1, 127, 128, 254, 255
    let vals: Vec<i64> = Vec::new();
    let mut vals2: Vec<i64> = Vec::new();
    vals2.push(0);
    vals2.push(1);
    vals2.push(127);
    vals2.push(128);
    vals2.push(254);
    vals2.push(255);

    let mut i = 0;
    while i < vals2.len() {
        let n = vals2.get(i).unwrap();
        v.push(n as u8);
        i = i + 1;
    }

    i = 0;
    while i < v.len() {
        let byte = v.get(i).unwrap() as i64;
        println("  cast[{}]={}", i, byte);
        i = i + 1;
    }
}

// --- テスト 3: u32 エンコード・デコード ---
fn test_u32_roundtrip() {
    println("--- Test 3: u32 encode/decode ---");
    let test_values: Vec<i64> = Vec::new();
    let mut tvs: Vec<i64> = Vec::new();
    tvs.push(0);
    tvs.push(1);
    tvs.push(255);
    tvs.push(256);
    tvs.push(65535);
    tvs.push(65536);
    tvs.push(1179993927); // GGUF magic

    let mut i = 0;
    while i < tvs.len() {
        let val = tvs.get(i).unwrap();
        let mut buf: Vec<u8> = Vec::new();
        buf = write_u32_le(buf, val);
        let decoded = read_u32_le(buf, 0);
        if val == decoded {
            println("  u32 roundtrip OK: {}", val);
        } else {
            println("  u32 roundtrip FAIL: {} != {}", val, decoded);
        }
        i = i + 1;
    }
}

// --- テスト 4: u64 エンコード・デコード ---
fn test_u64_roundtrip() {
    println("--- Test 4: u64 encode/decode ---");
    let mut tvs: Vec<i64> = Vec::new();
    tvs.push(0);
    tvs.push(1);
    tvs.push(4294967295); // 0xFFFFFFFF
    tvs.push(4294967296); // 0x100000000
    tvs.push(8589934591); // 2^33 - 1

    let mut i = 0;
    while i < tvs.len() {
        let val = tvs.get(i).unwrap();
        let mut buf: Vec<u8> = Vec::new();
        buf = write_u64_le(buf, val);
        let decoded = read_u64_le(buf, 0);
        if val == decoded {
            println("  u64 roundtrip OK: {}", val);
        } else {
            println("  u64 roundtrip FAIL: {} != {}", val, decoded);
        }
        i = i + 1;
    }
}

// --- テスト 5: String.to_bytes / from_chars ラウンドトリップ ---
fn test_string_bytes_roundtrip() {
    println("--- Test 5: String bytes roundtrip ---");
    let s = "Hello, GGUF!";
    let bytes = s.to_bytes();
    println("  original: {}", s);
    println("  bytes len: {}", bytes.len());

    // バイトから文字列を再構築
    let mut chars: Vec<i64> = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes.get(i).unwrap() as i64;
        chars.push(b);
        i = i + 1;
    }
    let rebuilt = String::from_chars(chars);
    println("  rebuilt: {}", rebuilt);
    if s == rebuilt {
        println("  roundtrip OK");
    } else {
        println("  roundtrip FAIL");
    }
}

// --- テスト 6: Vec<u8> を連続して関数に渡す ---
fn append_byte(buf: Vec<u8>, b: u8) -> Vec<u8> {
    let mut v = buf;
    v.push(b);
    v
}

fn test_vec_chained_calls() {
    println("--- Test 6: Vec<u8> chained function calls ---");
    let mut buf: Vec<u8> = Vec::new();
    buf = append_byte(buf, 10 as u8);
    buf = append_byte(buf, 20 as u8);
    buf = append_byte(buf, 30 as u8);
    buf = append_byte(buf, 40 as u8);
    buf = append_byte(buf, 50 as u8);
    println("  len={}", buf.len());
    let mut i = 0;
    while i < buf.len() {
        println("  buf[{}]={}", i, buf.get(i).unwrap() as i64);
        i = i + 1;
    }
}

// --- テスト 7: ネストした Vec<u8> 構造 ---
struct BytePair {
    lo: u8,
    hi: u8
}

fn split_u16(val: i64) -> BytePair {
    BytePair {
        lo: (val % 256) as u8,
        hi: ((val / 256) % 256) as u8
    }
}

fn test_struct_with_u8() {
    println("--- Test 7: struct with u8 fields ---");
    let p = split_u16(1000); // 1000 = 0x03E8: lo=0xE8=232, hi=0x03=3
    println("  lo={}", p.lo as i64);
    println("  hi={}", p.hi as i64);
    let reconstructed = p.lo as i64 + p.hi as i64 * 256;
    if reconstructed == 1000 {
        println("  u16 split OK: {}", reconstructed);
    } else {
        println("  u16 split FAIL: {} != 1000", reconstructed);
    }
}

// --- テスト 8: GGUF GGUF 文字列 エンコード・デコード ---
struct ReadStringResult {
    value: String,
    next_offset: i64
}

fn write_gguf_string(buf: Vec<u8>, s: String) -> Vec<u8> {
    let bytes = s.to_bytes();
    let len = bytes.len();
    let mut b = write_u64_le(buf, len);
    let mut i = 0;
    while i < len {
        b.push(bytes.get(i).unwrap());
        i = i + 1;
    }
    b
}

fn read_gguf_string(data: Vec<u8>, offset: i64) -> ReadStringResult {
    let len = read_u64_le(data, offset);
    let str_start = offset + 8;
    let mut chars: Vec<i64> = Vec::new();
    let mut i = 0;
    while i < len {
        let byte = data.get(str_start + i).unwrap() as i64;
        chars.push(byte);
        i = i + 1;
    }
    let result = String::from_chars(chars);
    ReadStringResult {
        value: result,
        next_offset: str_start + len
    }
}

fn test_gguf_string_roundtrip() {
    println("--- Test 8: GGUF string encode/decode ---");
    let test_strings: Vec<i64> = Vec::new(); // dummy
    let mut strs: Vec<String> = Vec::new();
    strs.push("");
    strs.push("a");
    strs.push("hello");
    strs.push("general.architecture");
    strs.push("Meta-Llama-3.1-8B-Instruct");

    let mut i = 0;
    while i < strs.len() {
        let s = strs.get(i).unwrap();
        let mut buf: Vec<u8> = Vec::new();
        buf = write_gguf_string(buf, s);
        let result = read_gguf_string(buf, 0);
        if s == result.value {
            println("  gguf_string OK: '{}'", s);
        } else {
            println("  gguf_string FAIL: '{}' != '{}'", s, result.value);
        }
        i = i + 1;
    }
}

// --- テスト 9: 大きな Vec<u8> のパフォーマンス ---
fn test_large_vec_u8() {
    println("--- Test 9: large Vec<u8> ---");
    let mut buf: Vec<u8> = Vec::new();
    let n = 10000;
    let mut i = 0;
    while i < n {
        buf.push((i % 256) as u8);
        i = i + 1;
    }
    println("  created {} bytes", buf.len());

    // 一部の値を検証
    let check_idx = 9999;
    let expected = check_idx % 256;
    let got = buf.get(check_idx).unwrap() as i64;
    if expected == got {
        println("  large Vec<u8> check OK: buf[{}]={}", check_idx, got);
    } else {
        println("  large Vec<u8> check FAIL: expected={} got={}", expected, got);
    }
}

// --- テスト 10: Vec<u8> を File::write_binary で書いて read_binary で読む ---
fn test_file_binary_roundtrip() {
    println("--- Test 10: File binary roundtrip ---");
    let path = "/tmp/tl_gguf_test_binary.bin";

    let mut buf: Vec<u8> = Vec::new();
    // 既知のパターンを書く
    let mut i = 0;
    while i < 256 {
        buf.push(i as u8);
        i = i + 1;
    }
    println("  Writing {} bytes", buf.len());

    if File::write_binary(path, buf) {
        println("  Write OK");
    } else {
        println("  Write FAIL");
        return;
    }

    let res = File::read_binary(path);
    let data = match res {
        Result::Ok(v) => v,
        Result::Err(e) => {
            println("  Read FAIL: {}", e);
            return;
        }
    };
    println("  Read back {} bytes", data.len());

    // 全バイトを検証
    let mut errors = 0;
    i = 0;
    while i < 256 {
        let got = data.get(i).unwrap() as i64;
        if got != i {
            errors = errors + 1;
        }
        i = i + 1;
    }
    if errors == 0 {
        println("  All 256 bytes verified OK");
    } else {
        println("  {} bytes FAILED verification", errors);
    }
}

// --- メイン ---

fn main() {
    println("=== GGUF Advanced Tests ===");
    println("");
    test_vec_u8_basic();
    println("");
    test_cast_i64_to_u8();
    println("");
    test_u32_roundtrip();
    println("");
    test_u64_roundtrip();
    println("");
    test_string_bytes_roundtrip();
    println("");
    test_vec_chained_calls();
    println("");
    test_struct_with_u8();
    println("");
    test_gguf_string_roundtrip();
    println("");
    test_large_vec_u8();
    println("");
    test_file_binary_roundtrip();
    println("");
    println("=== All tests complete ===");
}
