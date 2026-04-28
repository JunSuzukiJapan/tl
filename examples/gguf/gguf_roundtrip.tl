// ============================================================
// GGUF ラウンドトリップテスト
// ============================================================
// gguf_writer.tl が書き出したファイルを gguf_reader.tl の
// パーサー関数でフル解析し、整合性を検証する。
// ============================================================

// --- 共通ヘルパー ---

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

// GGUF文字列書き込み
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

// GGUF文字列読み取り
struct ReadStringResult {
    value: String,
    next_offset: i64
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

// パディング書き込み
fn write_padding(buf: Vec<u8>, alignment: i64) -> Vec<u8> {
    let mut b = buf;
    let remainder = b.len() % alignment;
    if remainder != 0 {
        let pad = alignment - remainder;
        let mut i = 0;
        while i < pad {
            b.push(0 as u8);
            i = i + 1;
        }
    }
    b
}

// --- GGUFファイル構造 ---

struct GgufHeader {
    magic: i64,
    version: i64,
    tensor_count: i64,
    metadata_kv_count: i64
}

struct GgufMetadataKV {
    key: String,
    value_type: i64,
    value_str: String,
    value_int: i64
}

struct GgufTensorInfo {
    name: String,
    n_dims: i64,
    dtype: i64,
    offset: i64
}

// --- ヘッダーパース ---
fn parse_header(data: Vec<u8>) -> GgufHeader {
    let magic = read_u32_le(data, 0);
    let version = read_u32_le(data, 4);
    let tensor_count = read_u64_le(data, 8);
    let metadata_kv_count = read_u64_le(data, 16);
    GgufHeader {
        magic: magic,
        version: version,
        tensor_count: tensor_count,
        metadata_kv_count: metadata_kv_count
    }
}

fn value_type_size(vt: i64) -> i64 {
    if vt == 0 { 1 }
    else if vt == 1 { 1 }
    else if vt == 2 { 2 }
    else if vt == 3 { 2 }
    else if vt == 4 { 4 }
    else if vt == 5 { 4 }
    else if vt == 6 { 4 }
    else if vt == 7 { 1 }
    else if vt == 10 { 8 }
    else if vt == 11 { 8 }
    else if vt == 12 { 8 }
    else { 0 }
}

fn skip_metadata_value(data: Vec<u8>, offset: i64, vtype: i64) -> i64 {
    if vtype == 8 {
        let len = read_u64_le(data, offset);
        offset + 8 + len
    } else if vtype == 9 {
        let elem_type = read_u32_le(data, offset);
        let count = read_u64_le(data, offset + 4);
        let mut pos = offset + 12;
        let mut i = 0;
        while i < count {
            pos = skip_metadata_value(data, pos, elem_type);
            i = i + 1;
        }
        pos
    } else {
        offset + value_type_size(vtype)
    }
}

fn parse_kv_string_value(data: Vec<u8>, offset: i64) -> ReadStringResult {
    let r = read_gguf_string(data, offset);
    r
}

// --- メイン ---

fn main() {
    println("=== GGUF Roundtrip Test ===");
    println("");

    let out_path = "/tmp/tl_gguf_roundtrip_test.gguf";

    // === ステップ1: GGUFファイルを書く ===
    println("Step 1: Writing GGUF file...");

    let mut buf: Vec<u8> = Vec::new();

    // Header
    buf = write_u32_le(buf, 1179993927); // magic "GGUF"
    buf = write_u32_le(buf, 3);           // version
    buf = write_u64_le(buf, 3);           // tensor_count = 3
    buf = write_u64_le(buf, 4);           // metadata_kv_count = 4

    // Metadata KV 1: general.architecture = "roundtrip-test" (STRING=8)
    buf = write_gguf_string(buf, "general.architecture");
    buf = write_u32_le(buf, 8);
    buf = write_gguf_string(buf, "roundtrip-test");

    // Metadata KV 2: general.name = "TL GGUF Test" (STRING=8)
    buf = write_gguf_string(buf, "general.name");
    buf = write_u32_le(buf, 8);
    buf = write_gguf_string(buf, "TL GGUF Test");

    // Metadata KV 3: model.layer_count = 4 (UINT32=4)
    buf = write_gguf_string(buf, "model.layer_count");
    buf = write_u32_le(buf, 4); // value_type = UINT32
    buf = write_u32_le(buf, 4); // value = 4

    // Metadata KV 4: model.hidden_size = 256 (UINT64=10)
    buf = write_gguf_string(buf, "model.hidden_size");
    buf = write_u32_le(buf, 10); // value_type = UINT64
    buf = write_u64_le(buf, 256); // value = 256

    // Tensor info 1: "weight" (F32=0, shape=[4, 3], offset=0)
    buf = write_gguf_string(buf, "weight");
    buf = write_u32_le(buf, 2);   // n_dims = 2
    buf = write_u64_le(buf, 4);   // dim[0] = 4
    buf = write_u64_le(buf, 3);   // dim[1] = 3
    buf = write_u32_le(buf, 0);   // type = F32
    buf = write_u64_le(buf, 0);   // offset = 0

    // Tensor info 2: "bias" (F32=0, shape=[4], offset=48)
    buf = write_gguf_string(buf, "bias");
    buf = write_u32_le(buf, 1);   // n_dims = 1
    buf = write_u64_le(buf, 4);   // dim[0] = 4
    buf = write_u32_le(buf, 0);   // type = F32
    buf = write_u64_le(buf, 48);  // offset = 48

    // Tensor info 3: "emb" (Q8_0=8, shape=[256, 100], offset=64)
    buf = write_gguf_string(buf, "emb");
    buf = write_u32_le(buf, 2);   // n_dims = 2
    buf = write_u64_le(buf, 256); // dim[0] = 256
    buf = write_u64_le(buf, 100); // dim[1] = 100
    buf = write_u32_le(buf, 8);   // type = Q8_0
    buf = write_u64_le(buf, 80);  // offset = 80

    // Padding to 32 bytes
    buf = write_padding(buf, 32);

    // Tensor data (zeros)
    let mut i = 0;
    while i < 200 {
        buf.push(0 as u8);
        i = i + 1;
    }

    println("  Written {} bytes", buf.len());

    if File::write_binary(out_path, buf) {
        println("  Write OK");
    } else {
        println("  Write FAILED");
        return;
    }

    // === ステップ2: 書いたファイルをパースして検証 ===
    println("");
    println("Step 2: Parsing and verifying...");

    let res = File::read_binary(out_path);
    let data = match res {
        Result::Ok(v) => v,
        Result::Err(e) => {
            println("  Read FAILED: {}", e);
            return;
        }
    };
    println("  Read back {} bytes", data.len());

    // Header検証
    let header = parse_header(data);

    if header.magic == 1179993927 {
        println("  magic: OK");
    } else {
        println("  magic: FAIL ({})", header.magic);
    }
    if header.version == 3 {
        println("  version: OK ({})", header.version);
    } else {
        println("  version: FAIL ({})", header.version);
    }
    if header.tensor_count == 3 {
        println("  tensor_count: OK ({})", header.tensor_count);
    } else {
        println("  tensor_count: FAIL ({})", header.tensor_count);
    }
    if header.metadata_kv_count == 4 {
        println("  metadata_kv_count: OK ({})", header.metadata_kv_count);
    } else {
        println("  metadata_kv_count: FAIL ({})", header.metadata_kv_count);
    }

    // Metadata KV1 検証
    let mut offset = 24; // header = 4+4+8+8 = 24 bytes

    let kv1_key = read_gguf_string(data, offset);
    println("  KV1 key: {}", kv1_key.value);
    offset = kv1_key.next_offset + 4; // skip value_type
    let kv1_val = read_gguf_string(data, offset);
    println("  KV1 val: {}", kv1_val.value);
    if kv1_key.value == "general.architecture" && kv1_val.value == "roundtrip-test" {
        println("  KV1: OK");
    } else {
        println("  KV1: FAIL");
    }
    offset = kv1_val.next_offset;

    // KV2
    let kv2_key = read_gguf_string(data, offset);
    println("  KV2 key: {}", kv2_key.value);
    offset = kv2_key.next_offset + 4;
    let kv2_val = read_gguf_string(data, offset);
    println("  KV2 val: {}", kv2_val.value);
    if kv2_key.value == "general.name" && kv2_val.value == "TL GGUF Test" {
        println("  KV2: OK");
    } else {
        println("  KV2: FAIL");
    }
    offset = kv2_val.next_offset;

    // KV3: model.layer_count = 4 (UINT32)
    let kv3_key = read_gguf_string(data, offset);
    println("  KV3 key: {}", kv3_key.value);
    offset = kv3_key.next_offset;
    let kv3_vtype = read_u32_le(data, offset);
    offset = offset + 4;
    let kv3_val = read_u32_le(data, offset);
    offset = offset + 4;
    println("  KV3 val (UINT32): {}", kv3_val);
    if kv3_key.value == "model.layer_count" && kv3_vtype == 4 && kv3_val == 4 {
        println("  KV3: OK");
    } else {
        println("  KV3: FAIL");
    }

    // KV4: model.hidden_size = 256 (UINT64)
    let kv4_key = read_gguf_string(data, offset);
    println("  KV4 key: {}", kv4_key.value);
    offset = kv4_key.next_offset;
    let kv4_vtype = read_u32_le(data, offset);
    offset = offset + 4;
    let kv4_val = read_u64_le(data, offset);
    offset = offset + 8;
    println("  KV4 val (UINT64): {}", kv4_val);
    if kv4_key.value == "model.hidden_size" && kv4_vtype == 10 && kv4_val == 256 {
        println("  KV4: OK");
    } else {
        println("  KV4: FAIL");
    }

    // Tensor info検証
    // Tensor 1: "weight"
    let t1_name = read_gguf_string(data, offset);
    offset = t1_name.next_offset;
    let t1_ndims = read_u32_le(data, offset);
    offset = offset + 4;
    let t1_dim0 = read_u64_le(data, offset);
    offset = offset + 8;
    let t1_dim1 = read_u64_le(data, offset);
    offset = offset + 8;
    let t1_dtype = read_u32_le(data, offset);
    offset = offset + 4;
    let t1_toffset = read_u64_le(data, offset);
    offset = offset + 8;
    println("  T1: {} ndims={} shape=[{},{}] dtype={} offset={}",
        t1_name.value, t1_ndims, t1_dim0, t1_dim1, t1_dtype, t1_toffset);
    if t1_name.value == "weight" && t1_ndims == 2 && t1_dim0 == 4 && t1_dim1 == 3 && t1_dtype == 0 && t1_toffset == 0 {
        println("  T1: OK");
    } else {
        println("  T1: FAIL");
    }

    // Tensor 2: "bias"
    let t2_name = read_gguf_string(data, offset);
    offset = t2_name.next_offset;
    let t2_ndims = read_u32_le(data, offset);
    offset = offset + 4;
    let t2_dim0 = read_u64_le(data, offset);
    offset = offset + 8;
    let t2_dtype = read_u32_le(data, offset);
    offset = offset + 4;
    let t2_toffset = read_u64_le(data, offset);
    offset = offset + 8;
    println("  T2: {} ndims={} shape=[{}] dtype={} offset={}",
        t2_name.value, t2_ndims, t2_dim0, t2_dtype, t2_toffset);
    if t2_name.value == "bias" && t2_ndims == 1 && t2_dim0 == 4 && t2_dtype == 0 && t2_toffset == 48 {
        println("  T2: OK");
    } else {
        println("  T2: FAIL");
    }

    // Tensor 3: "emb"
    let t3_name = read_gguf_string(data, offset);
    offset = t3_name.next_offset;
    let t3_ndims = read_u32_le(data, offset);
    offset = offset + 4;
    let t3_dim0 = read_u64_le(data, offset);
    offset = offset + 8;
    let t3_dim1 = read_u64_le(data, offset);
    offset = offset + 8;
    let t3_dtype = read_u32_le(data, offset);
    offset = offset + 4;
    let t3_toffset = read_u64_le(data, offset);
    println("  T3: {} ndims={} shape=[{},{}] dtype={} offset={}",
        t3_name.value, t3_ndims, t3_dim0, t3_dim1, t3_dtype, t3_toffset);
    if t3_name.value == "emb" && t3_ndims == 2 && t3_dim0 == 256 && t3_dim1 == 100 && t3_dtype == 8 && t3_toffset == 80 {
        println("  T3: OK");
    } else {
        println("  T3: FAIL");
    }

    println("");
    println("=== GGUF Roundtrip Test Complete ===");
}
