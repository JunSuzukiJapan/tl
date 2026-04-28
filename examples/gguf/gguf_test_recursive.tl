// ============================================================
// 再帰・メモリ管理・文字列連結テスト
// ============================================================
// テスト項目:
//   1. 再帰関数（skip_metadata_value と同じパターン）
//   2. 連続した文字列連結（String + String）
//   3. Vec<u8>のループ内での反復アクセスと所有権
//   4. 構造体への値の書き込みと解放
// ============================================================

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

// write_u64_le は組み込み関数として提供済み

// --- テスト 1: skip_metadata_value 相当の再帰関数 ---

fn elem_size(vtype: i64) -> i64 {
    if vtype == 0 { 1 }       // UINT8
    else if vtype == 1 { 1 }  // INT8
    else if vtype == 2 { 2 }  // UINT16
    else if vtype == 3 { 2 }  // INT16
    else if vtype == 4 { 4 }  // UINT32
    else if vtype == 5 { 4 }  // INT32
    else if vtype == 6 { 4 }  // FLOAT32
    else if vtype == 7 { 1 }  // BOOL
    else if vtype == 10 { 8 } // UINT64
    else if vtype == 11 { 8 } // INT64
    else if vtype == 12 { 8 } // FLOAT64
    else { 0 }
}

fn skip_value(data: Vec<u8>, offset: i64, vtype: i64) -> i64 {
    if vtype == 8 {
        // STRING
        let len = read_u64_le(data, offset);
        offset + 8 + len
    } else if vtype == 9 {
        // ARRAY (再帰)
        let elem_type = read_u32_le(data, offset);
        let count = read_u64_le(data, offset + 4);
        let mut pos = offset + 12;
        let mut i = 0;
        while i < count {
            pos = skip_value(data, pos, elem_type);
            i = i + 1;
        }
        pos
    } else {
        offset + elem_size(vtype)
    }
}

fn test_recursive_skip() {
    println("--- Test: recursive skip_value ---");

    // Build a test payload:
    // UINT32 value: 42
    let mut buf: Vec<u8> = Vec::new();
    buf = write_u32_le(buf, 42);
    let next = skip_value(buf, 0, 4); // UINT32 = type 4
    println("  skip UINT32: next={} (expect 4)", next);

    // STRING value: "hello"
    let mut buf2: Vec<u8> = Vec::new();
    buf2 = write_u64_le(buf2, 5); // len=5
    buf2.push(104 as u8); // h
    buf2.push(101 as u8); // e
    buf2.push(108 as u8); // l
    buf2.push(108 as u8); // l
    buf2.push(111 as u8); // o
    let next2 = skip_value(buf2, 0, 8); // STRING = type 8
    println("  skip STRING 'hello': next={} (expect 13)", next2);

    // ARRAY of UINT32 [1, 2, 3]
    let mut buf3: Vec<u8> = Vec::new();
    buf3 = write_u32_le(buf3, 4); // elem_type = UINT32
    buf3 = write_u64_le(buf3, 3); // count = 3
    buf3 = write_u32_le(buf3, 1);
    buf3 = write_u32_le(buf3, 2);
    buf3 = write_u32_le(buf3, 3);
    let next3 = skip_value(buf3, 0, 9); // ARRAY = type 9
    println("  skip ARRAY<UINT32>[3]: next={} (expect 24)", next3);
    // 4 (elem_type u32) + 8 (count u64) + 3*4 = 4+8+12 = 24

    if next == 4 && next2 == 13 && next3 == 24 {
        println("  recursive skip OK");
    } else {
        println("  recursive skip FAIL");
    }
}

// --- テスト 2: String 連結チェーン ---
fn format_tensor_info(name: String, dtype: i64, dim0: i64, dim1: i64) -> String {
    let type_name = if dtype == 0 { "F32" } else if dtype == 8 { "Q8_0" } else { "UNKNOWN" };
    "[" + name + ": " + type_name + " shape=[" + String::from_int(dim0) + ", " + String::from_int(dim1) + "]]"
}

fn test_string_concat_chain() {
    println("--- Test: string concat chain ---");
    let s1 = format_tensor_info("token_embd.weight", 8, 384, 30522);
    println("  {}", s1);
    let s2 = format_tensor_info("output.weight", 0, 32000, 4096);
    println("  {}", s2);

    // Check non-empty
    if s1.len() > 0 && s2.len() > 0 {
        println("  string concat chain OK");
    } else {
        println("  string concat chain FAIL");
    }
}

// --- テスト 3: Vec<u8>のループ内反復アクセス ---
fn compute_checksum(data: Vec<u8>) -> i64 {
    let mut sum: i64 = 0;
    let mut i = 0;
    while i < data.len() {
        sum = sum + data.get(i).unwrap() as i64;
        i = i + 1;
    }
    sum
}

fn test_repeated_vec_access() {
    println("--- Test: repeated Vec<u8> access in loop ---");
    let mut data: Vec<u8> = Vec::new();
    let mut i = 0;
    while i < 100 {
        data.push((i % 256) as u8);
        i = i + 1;
    }

    // Call checksum multiple times with the same data
    let cs1 = compute_checksum(data);
    let cs2 = compute_checksum(data);
    let cs3 = compute_checksum(data);

    // 0+1+2+...+99 = 4950
    println("  checksum1={} (expect 4950)", cs1);
    println("  checksum2={} (expect 4950)", cs2);
    println("  checksum3={} (expect 4950)", cs3);
    if cs1 == 4950 && cs2 == 4950 && cs3 == 4950 {
        println("  repeated Vec<u8> access OK");
    } else {
        println("  repeated Vec<u8> access FAIL");
    }
}

// --- テスト 4: Vec<u8>のコピーと独立性 ---
fn test_vec_independence() {
    println("--- Test: Vec<u8> copy independence ---");
    let mut original: Vec<u8> = Vec::new();
    original.push(1 as u8);
    original.push(2 as u8);
    original.push(3 as u8);

    // copy (deep clone should happen)
    let mut copy = original.clone();
    copy.push(99 as u8);

    // original should still have 3 elements
    println("  original.len={} (expect 3)", original.len());
    println("  copy.len={} (expect 4)", copy.len());
    println("  original[2]={} (expect 3)", original.get(2).unwrap() as i64);
    println("  copy[3]={} (expect 99)", copy.get(3).unwrap() as i64);

    if original.len() == 3 && copy.len() == 4 {
        println("  Vec<u8> copy independence OK");
    } else {
        println("  Vec<u8> copy independence FAIL");
    }
}

// --- テスト 5: 深い再帰（ネストしたARRAY） ---
// ARRAY of ARRAY of UINT32
fn test_nested_array_skip() {
    println("--- Test: nested array skip (ARRAY of ARRAY) ---");

    // Inner array: ARRAY<UINT32>[2] = [10, 20]
    let mut inner: Vec<u8> = Vec::new();
    inner = write_u32_le(inner, 4); // elem_type = UINT32
    inner = write_u64_le(inner, 2); // count = 2
    inner = write_u32_le(inner, 10);
    inner = write_u32_le(inner, 20);
    // inner size = 4+8+2*4 = 20 bytes

    // Outer array: ARRAY<ARRAY>[2] = [inner, inner]
    // Note: in GGUF ARRAY type (9) contains elem_type and count
    // For simplicity, build the bytes manually
    let mut buf: Vec<u8> = Vec::new();
    buf = write_u32_le(buf, 9); // elem_type = ARRAY (9)
    buf = write_u64_le(buf, 2); // count = 2

    // First inner array element
    buf = write_u32_le(buf, 4); // elem_type = UINT32
    buf = write_u64_le(buf, 2); // count = 2
    buf = write_u32_le(buf, 10);
    buf = write_u32_le(buf, 20);

    // Second inner array element
    buf = write_u32_le(buf, 4); // elem_type = UINT32
    buf = write_u64_le(buf, 2); // count = 2
    buf = write_u32_le(buf, 30);
    buf = write_u32_le(buf, 40);

    // outer: 4+8 = 12 header + 2*(4+8+2*4) = 12 + 2*20 = 52 bytes total
    let next = skip_value(buf, 0, 9);
    println("  skip ARRAY<ARRAY<UINT32>[2]>[2]: next={} (expect 52)", next);
    if next == 52 {
        println("  nested array skip OK");
    } else {
        println("  nested array skip FAIL");
    }
}

fn main() {
    println("=== GGUF Recursive/Memory Tests ===");
    println("");
    test_recursive_skip();
    println("");
    test_string_concat_chain();
    println("");
    test_repeated_vec_access();
    println("");
    test_vec_independence();
    println("");
    test_nested_array_skip();
    println("");
    println("=== Recursive tests complete ===");
}
