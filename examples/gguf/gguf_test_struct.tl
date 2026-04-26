// ============================================================
// 構造体・高度パターンテスト
// ============================================================
// テスト項目:
//   1. Vec<u8>を保持する構造体の返却
//   2. 構造体フィールドのVec<u8>操作
//   3. mut変数の複数再代入
//   4. 関数戻り値の直接チェーン
//   5. Vec<u8>を含む構造体のネスト
//   6. GGUFライター全体の詳細検証
// ============================================================

struct Buffer {
    data: Vec<u8>,
    pos: i64
}

fn buffer_new() -> Buffer {
    Buffer {
        data: Vec::new(),
        pos: 0
    }
}

fn buffer_write_u8(buf: Buffer, val: u8) -> Buffer {
    let mut d = buf.data;
    d.push(val);
    Buffer { data: d, pos: buf.pos + 1 }
}

fn buffer_read_u8(buf: Buffer, offset: i64) -> i64 {
    buf.data.get(offset).unwrap() as i64
}

fn buffer_len(buf: Buffer) -> i64 {
    buf.data.len()
}

fn test_buffer_struct() {
    println("--- Test: Buffer struct ---");
    let mut buf = buffer_new();
    buf = buffer_write_u8(buf, 10 as u8);
    buf = buffer_write_u8(buf, 20 as u8);
    buf = buffer_write_u8(buf, 30 as u8);
    println("  len={}", buffer_len(buf));
    println("  buf[0]={}", buffer_read_u8(buf, 0));
    println("  buf[1]={}", buffer_read_u8(buf, 1));
    println("  buf[2]={}", buffer_read_u8(buf, 2));
    println("  pos={}", buf.pos);
}

// --- ネストした構造体 ---
struct GgufTestHeader {
    magic: i64,
    version: i64
}

struct GgufTestFile {
    header: GgufTestHeader,
    data: Vec<u8>
}

fn write_u32_le_v(buf: Vec<u8>, val: i64) -> Vec<u8> {
    let mut b = buf;
    b.push((val % 256) as u8);
    b.push(((val / 256) % 256) as u8);
    b.push(((val / 65536) % 256) as u8);
    b.push(((val / 16777216) % 256) as u8);
    b
}

fn read_u32_le_v(data: Vec<u8>, offset: i64) -> i64 {
    let b0 = data.get(offset).unwrap() as i64;
    let b1 = data.get(offset + 1).unwrap() as i64;
    let b2 = data.get(offset + 2).unwrap() as i64;
    let b3 = data.get(offset + 3).unwrap() as i64;
    b0 + b1 * 256 + b2 * 65536 + b3 * 16777216
}

fn build_test_file() -> GgufTestFile {
    let header = GgufTestHeader { magic: 1179993927, version: 3 };
    let mut data: Vec<u8> = Vec::new();
    data = write_u32_le_v(data, 1179993927);
    data = write_u32_le_v(data, 3);
    GgufTestFile { header: header, data: data }
}

fn test_nested_struct() {
    println("--- Test: nested struct with Vec<u8> ---");
    let f = build_test_file();
    println("  header.magic={}", f.header.magic);
    println("  header.version={}", f.header.version);
    println("  data.len={}", f.data.len());
    let magic_from_bytes = read_u32_le_v(f.data, 0);
    let version_from_bytes = read_u32_le_v(f.data, 4);
    println("  data magic={}", magic_from_bytes);
    println("  data version={}", version_from_bytes);
    if magic_from_bytes == f.header.magic && version_from_bytes == f.header.version {
        println("  nested struct OK");
    } else {
        println("  nested struct FAIL");
    }
}

// --- mut変数の複数再代入 ---
fn test_mut_reassign() {
    println("--- Test: mut variable multiple reassign ---");
    let mut v: Vec<u8> = Vec::new();
    v = write_u32_le_v(v, 1);
    v = write_u32_le_v(v, 2);
    v = write_u32_le_v(v, 3);
    v = write_u32_le_v(v, 4);
    println("  len={}", v.len());

    let mut sum: i64 = 0;
    let mut i = 0;
    while i < v.len() {
        sum = sum + v.get(i).unwrap() as i64;
        i = i + 1;
    }
    // 1,0,0,0, 2,0,0,0, 3,0,0,0, 4,0,0,0 → sum = 1+2+3+4 = 10
    println("  byte sum={} (expect 10)", sum);
    if sum == 10 {
        println("  mut reassign OK");
    } else {
        println("  mut reassign FAIL");
    }
}

// --- インライン関数チェーン ---
fn wrap_byte(val: i64) -> u8 {
    (val % 256) as u8
}

fn sum_bytes(a: u8, b: u8, c: u8) -> i64 {
    a as i64 + b as i64 + c as i64
}

fn test_function_chain() {
    println("--- Test: function chain ---");
    let result = sum_bytes(wrap_byte(100), wrap_byte(200), wrap_byte(300));
    // 300 % 256 = 44
    // 100 + 200 + 44 = 344
    println("  sum_bytes(100, 200, 300%256) = {} (expect 344)", result);
    if result == 344 {
        println("  function chain OK");
    } else {
        println("  function chain FAIL");
    }
}

// --- Vec<u8>を含む構造体の配列(Vec) ---
struct ByteChunk {
    id: i64,
    bytes: Vec<u8>
}

fn make_chunk(id: i64, size: i64) -> ByteChunk {
    let mut v: Vec<u8> = Vec::new();
    let mut i = 0;
    while i < size {
        v.push((id * 10 + i) as u8);
        i = i + 1;
    }
    ByteChunk { id: id, bytes: v }
}

fn test_vec_of_structs_with_vec() {
    println("--- Test: Vec of structs with Vec<u8> fields ---");
    let mut chunks: Vec<ByteChunk> = Vec::new();
    chunks.push(make_chunk(1, 3));
    chunks.push(make_chunk(2, 3));
    chunks.push(make_chunk(3, 3));

    println("  chunks.len={}", chunks.len());
    let mut ci = 0;
    while ci < chunks.len() {
        let c = chunks.get(ci).unwrap();
        print("  chunk[{}]: id={} bytes=[", ci, c.id);
        let mut bi = 0;
        while bi < c.bytes.len() {
            if bi > 0 {
                print(",");
            }
            print("{}", c.bytes.get(bi).unwrap() as i64);
            bi = bi + 1;
        }
        println("]");
        ci = ci + 1;
    }
}

// --- GGUFライター詳細検証 ---
fn write_u64_le_v(buf: Vec<u8>, val: i64) -> Vec<u8> {
    let lo = val % 4294967296;
    let hi = val / 4294967296;
    let b = write_u32_le_v(buf, lo);
    write_u32_le_v(b, hi)
}

fn read_u64_le_v(data: Vec<u8>, offset: i64) -> i64 {
    let lo = read_u32_le_v(data, offset);
    let hi = read_u32_le_v(data, offset + 4);
    lo + hi * 4294967296
}

fn test_gguf_header_detailed() {
    println("--- Test: GGUF header detailed verification ---");
    let mut buf: Vec<u8> = Vec::new();

    // Header: magic(u32), version(u32), tensor_count(u64), metadata_kv_count(u64)
    let magic: i64 = 1179993927;
    let version: i64 = 3;
    let tensor_count: i64 = 197;
    let metadata_kv_count: i64 = 27;

    buf = write_u32_le_v(buf, magic);
    buf = write_u32_le_v(buf, version);
    buf = write_u64_le_v(buf, tensor_count);
    buf = write_u64_le_v(buf, metadata_kv_count);

    println("  header bytes: {}", buf.len());

    let got_magic = read_u32_le_v(buf, 0);
    let got_version = read_u32_le_v(buf, 4);
    let got_tensor_count = read_u64_le_v(buf, 8);
    let got_metadata_kv_count = read_u64_le_v(buf, 16);

    if got_magic == magic {
        println("  magic OK: {}", magic);
    } else {
        println("  magic FAIL: got={} expected={}", got_magic, magic);
    }
    if got_version == version {
        println("  version OK: {}", version);
    } else {
        println("  version FAIL: got={} expected={}", got_version, version);
    }
    if got_tensor_count == tensor_count {
        println("  tensor_count OK: {}", tensor_count);
    } else {
        println("  tensor_count FAIL: got={} expected={}", got_tensor_count, tensor_count);
    }
    if got_metadata_kv_count == metadata_kv_count {
        println("  metadata_kv_count OK: {}", metadata_kv_count);
    } else {
        println("  metadata_kv_count FAIL: got={} expected={}", got_metadata_kv_count, metadata_kv_count);
    }
}

fn main() {
    println("=== GGUF Struct/Pattern Tests ===");
    println("");
    test_buffer_struct();
    println("");
    test_nested_struct();
    println("");
    test_mut_reassign();
    println("");
    test_function_chain();
    println("");
    test_vec_of_structs_with_vec();
    println("");
    test_gguf_header_detailed();
    println("");
    println("=== Struct tests complete ===");
}
