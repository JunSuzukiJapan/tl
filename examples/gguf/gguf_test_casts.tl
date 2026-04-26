// ============================================================
// キャスト境界値テスト — TLコンパイラのバグ探索
// ============================================================
// テスト項目:
//   1. i64→u8 オーバーフローキャスト (256, 511, etc.)
//   2. u8→i64 キャスト (常に0-255の範囲)
//   3. i64→u8 で256以上の値
//   4. Vec<u8> の連続書き込みと読み返しの正確性
// ============================================================

fn write_u32_le(buf: Vec<u8>, val: i64) -> Vec<u8> {
    let mut b = buf;
    b.push((val % 256) as u8);
    b.push(((val / 256) % 256) as u8);
    b.push(((val / 65536) % 256) as u8);
    b.push(((val / 16777216) % 256) as u8);
    b
}

fn read_u32_le(data: Vec<u8>, offset: i64) -> i64 {
    let b0 = data.get(offset).unwrap() as i64;
    let b1 = data.get(offset + 1).unwrap() as i64;
    let b2 = data.get(offset + 2).unwrap() as i64;
    let b3 = data.get(offset + 3).unwrap() as i64;
    b0 + b1 * 256 + b2 * 65536 + b3 * 16777216
}

fn test_direct_u8_cast() {
    println("--- Test: direct i64 as u8 cast ---");
    // 256 as u8 → in TL, should truncate to 0
    let a: i64 = 256;
    let b: u8 = a as u8;
    println("256 as u8 = {}", b as i64);
    // 257 as u8 → should be 1
    let c: i64 = 257;
    let d: u8 = c as u8;
    println("257 as u8 = {}", d as i64);
    // 511 as u8 → should be 255
    let e: i64 = 511;
    let f: u8 = e as u8;
    println("511 as u8 = {}", f as i64);
    // 512 as u8 → should be 0
    let g: i64 = 512;
    let h: u8 = g as u8;
    println("512 as u8 = {}", h as i64);
}

fn test_u32_all_bytes() {
    println("--- Test: u32 all-byte patterns ---");
    // 0x01020304 = 16909060
    let val: i64 = 16909060;
    let mut buf: Vec<u8> = Vec::new();
    buf = write_u32_le(buf, val);
    println("  val=16909060 (0x01020304)");
    println("  buf[0]={} (expect 4)", buf.get(0).unwrap() as i64);
    println("  buf[1]={} (expect 3)", buf.get(1).unwrap() as i64);
    println("  buf[2]={} (expect 2)", buf.get(2).unwrap() as i64);
    println("  buf[3]={} (expect 1)", buf.get(3).unwrap() as i64);
    let decoded = read_u32_le(buf, 0);
    if decoded == val {
        println("  roundtrip OK: {}", val);
    } else {
        println("  roundtrip FAIL: decoded={} expected={}", decoded, val);
    }
}

fn test_sequential_writes() {
    println("--- Test: sequential writes and reads ---");
    let mut buf: Vec<u8> = Vec::new();
    // Write multiple u32 values back-to-back
    let mut vals: Vec<i64> = Vec::new();
    vals.push(1179993927); // GGUF magic
    vals.push(3);          // version
    vals.push(42);         // small value
    vals.push(16777215);   // 0xFFFFFF

    let mut i = 0;
    while i < vals.len() {
        buf = write_u32_le(buf, vals.get(i).unwrap());
        i = i + 1;
    }

    println("  total bytes: {}", buf.len());

    i = 0;
    while i < vals.len() {
        let offset = i * 4;
        let got = read_u32_le(buf, offset);
        let expected = vals.get(i).unwrap();
        if got == expected {
            println("  vals[{}] OK: {}", i, expected);
        } else {
            println("  vals[{}] FAIL: got={} expected={}", i, got, expected);
        }
        i = i + 1;
    }
}

fn test_vec_u8_get_type() {
    println("--- Test: Vec<u8> get returns correct type ---");
    let mut v: Vec<u8> = Vec::new();
    v.push(100 as u8);
    v.push(200 as u8);

    let a = v.get(0).unwrap();
    let b = v.get(1).unwrap();
    println("  v[0] as i64 = {}", a as i64);
    println("  v[1] as i64 = {}", b as i64);

    // Add them as u8 (might overflow)
    let sum_u8: u8 = (a as i64 + b as i64) as u8;  // 300 % 256 = 44
    println("  (100 + 200) as u8 = {}", sum_u8 as i64);
}

fn test_modulo_behavior() {
    println("--- Test: modulo with large values ---");
    // Test that % correctly extracts bytes
    let val: i64 = 4294967295; // 0xFFFFFFFF
    let b0 = val % 256;
    let b1 = (val / 256) % 256;
    let b2 = (val / 65536) % 256;
    let b3 = (val / 16777216) % 256;
    println("  0xFFFFFFFF bytes: {} {} {} {}", b0, b1, b2, b3);
    // All should be 255
    if b0 == 255 && b1 == 255 && b2 == 255 && b3 == 255 {
        println("  modulo bytes OK");
    } else {
        println("  modulo bytes FAIL");
    }
}

fn main() {
    println("=== GGUF Cast Tests ===");
    println("");
    test_direct_u8_cast();
    println("");
    test_u32_all_bytes();
    println("");
    test_sequential_writes();
    println("");
    test_vec_u8_get_type();
    println("");
    test_modulo_behavior();
    println("");
    println("=== Cast tests complete ===");
}
