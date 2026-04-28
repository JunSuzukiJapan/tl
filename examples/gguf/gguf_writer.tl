// ============================================================
// GGUF Writer — TL言語で記述されたGGUFバイナリライター
// ============================================================
// 目的:
//   - TLコンパイラのバグを発見する
//   - TLに不足している機能を発見する
//
// GGUFフォーマット仕様:
//   Header:
//     magic: u32 ("GGUF" = 0x46554747 LE)
//     version: u32 (3)
//     tensor_count: u64
//     metadata_kv_count: u64
//
//   Metadata KV × metadata_kv_count:
//     key: gguf_string (u64 len + bytes)
//     value_type: u32
//     value: 型依存
//
//   Tensor Info × tensor_count:
//     name: gguf_string
//     n_dims: u32
//     dims[]: u64 × n_dims
//     type: u32
//     offset: u64
//
//   [Padding to 32-byte alignment]
//
//   Tensor Data
// ============================================================

// --- バイト書き込みヘルパー関数 ---

// リトルエンディアンでu32を書き込む
fn write_u32_le(buf: Vec<u8>, val: i64) -> Vec<u8> {
    let mut b = buf;
    let b0 = val % 256;
    let b1 = (val / 256) % 256;
    let b2 = (val / 65536) % 256;
    let b3 = (val / 16777216) % 256;
    b.push(b0 as u8);
    b.push(b1 as u8);
    b.push(b2 as u8);
    b.push(b3 as u8);
    b
}

fn write_u64_le(buf: Vec<u8>, val: i64) -> Vec<u8> {
    let lo = val % 4294967296;
    let hi = val / 4294967296;
    let b = write_u32_le(buf, lo);
    write_u32_le(b, hi)
}

// GGUF文字列を書き込む (u64 len + bytes)
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

// 0パディングを追加 (32バイトアラインメント)
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

// --- メイン ---

fn main() {
    println("=== GGUF Writer (written in TL) ===");
    println("");

    // テスト用の小さなGGUFファイルを作成
    // - 2つのメタデータ: general.architecture = "test", general.name = "tl-test-model"
    // - 2つのテンソル: "weight" (F32, [4, 3]), "bias" (F32, [4])

    let out_path = "~/.llm/models/gte-small/tl_test_output.gguf";

    let mut buf: Vec<u8> = Vec::new();

    // --- ヘッダー ---
    println("Writing header...");

    // Magic: "GGUF" = 0x46554747 = 1179993927 (LE)
    buf = write_u32_le(buf, 1179993927);
    // Version: 3
    buf = write_u32_le(buf, 3);
    // Tensor count: 2
    buf = write_u64_le(buf, 2);
    // Metadata KV count: 2
    buf = write_u64_le(buf, 2);

    // --- メタデータKVペア ---
    println("Writing metadata...");

    // KV 1: general.architecture = "test" (STRING = type 8)
    buf = write_gguf_string(buf, "general.architecture");
    buf = write_u32_le(buf, 8); // STRING type
    buf = write_gguf_string(buf, "test");

    // KV 2: general.name = "tl-test-model" (STRING = type 8)
    buf = write_gguf_string(buf, "general.name");
    buf = write_u32_le(buf, 8); // STRING type
    buf = write_gguf_string(buf, "tl-test-model");

    // --- テンソルインフォ ---
    println("Writing tensor info...");

    // Tensor 1: "weight" (F32=0, shape=[4, 3], offset=0)
    // Data size = 4 * 3 * 4 bytes = 48 bytes
    buf = write_gguf_string(buf, "weight");
    buf = write_u32_le(buf, 2);  // n_dims = 2
    buf = write_u64_le(buf, 4);  // dim[0] = 4
    buf = write_u64_le(buf, 3);  // dim[1] = 3
    buf = write_u32_le(buf, 0);  // type = F32
    buf = write_u64_le(buf, 0);  // offset = 0 (relative to data start)

    // Tensor 2: "bias" (F32=0, shape=[4], offset=48)
    // Data size = 4 * 4 bytes = 16 bytes
    buf = write_gguf_string(buf, "bias");
    buf = write_u32_le(buf, 1);  // n_dims = 1
    buf = write_u64_le(buf, 4);  // dim[0] = 4
    buf = write_u32_le(buf, 0);  // type = F32
    buf = write_u64_le(buf, 48); // offset = 48

    // --- パディング (32バイトアラインメント) ---
    buf = write_padding(buf, 32);

    // --- テンソルデータ ---
    println("Writing tensor data...");

    // weight: 12個のf32 (= [1.0, 2.0, ..., 12.0])
    // f32をバイト列に変換 — 簡易的に整数値のIEEE754表現を手動生成
    // 1.0f32 = 0x3F800000, 2.0f32 = 0x40000000, ...
    // 簡略化：各値を0で初期化（ゼロテンソル）
    let mut i = 0;
    while i < 48 {
        // 48 bytes = 12 x f32 (all zeros)
        buf.push(0 as u8);
        i = i + 1;
    }

    // bias: 4個のf32 (all zeros)
    i = 0;
    while i < 16 {
        buf.push(0 as u8);
        i = i + 1;
    }

    // --- ファイル書き込み ---
    println("Writing file: {}", out_path);
    println("Total size: {} bytes", buf.len());

    if File::write_binary(out_path, buf) {
        println("Write successful!");
    } else {
        println("ERROR: Write failed!");
        return;
    }

    // --- 検証: 書いたファイルを読み返す ---
    println("");
    println("=== Verification: Reading back written file ===");

    let verify_res = File::read_binary(out_path);
    let data = match verify_res {
        Result::Ok(v) => v,
        Result::Err(e) => {
            println("ERROR: Failed to read back: {}", e);
            return;
        }
    };

    println("Read back {} bytes", data.len());

    // ヘッダーの検証
    let magic = read_u32_le(data, 0);
    let version = read_u32_le(data, 4);
    let tensor_count = read_u64_le(data, 8);
    let metadata_kv_count = read_u64_le(data, 16);

    if magic == 1179993927 {
        println("Magic: GGUF (valid)");
    } else {
        println("Magic: INVALID ({})", magic);
    }
    println("Version: {}", version);
    println("Tensor count: {}", tensor_count);
    println("Metadata KV count: {}", metadata_kv_count);

    // メタデータの検証
    let mut offset = 24;
    let key1 = read_gguf_string(data, offset);
    println("Key 1: {}", key1.value);
    offset = key1.next_offset + 4; // skip value_type
    let val1 = read_gguf_string(data, offset);
    println("Val 1: {}", val1.value);
    offset = val1.next_offset;

    let key2 = read_gguf_string(data, offset);
    println("Key 2: {}", key2.value);
    offset = key2.next_offset + 4;
    let val2 = read_gguf_string(data, offset);
    println("Val 2: {}", val2.value);

    println("");
    println("=== GGUF Write + Verify complete ===");
}

// --- 読み取りヘルパー (gguf_reader.tl と同じ) ---

struct ReadStringResult {
    value: String,
    next_offset: i64
}

fn read_u32_le(data: Vec<u8>, offset: i64) -> i64 {
    let b0 = data.get(offset).unwrap() as i64;
    let b1 = data.get(offset + 1).unwrap() as i64;
    let b2 = data.get(offset + 2).unwrap() as i64;
    let b3 = data.get(offset + 3).unwrap() as i64;
    b0 + b1 * 256 + b2 * 65536 + b3 * 16777216
}

fn read_u64_le(data: Vec<u8>, offset: i64) -> i64 {
    let lo = read_u32_le(data, offset);
    let hi = read_u32_le(data, offset + 4);
    lo + hi * 4294967296
}

fn read_gguf_string(data: Vec<u8>, offset: i64) -> ReadStringResult {
    let len = read_u64_le(data, offset);
    let str_start = offset + 8;
    let mut chars = Vec::new();
    let mut i = 0;
    while i < len {
        let byte = data.get(str_start + i).unwrap() as i64;
        if byte >= 32 && byte <= 126 {
            chars.push(byte);
        } else {
            chars.push(63);
        }
        i = i + 1;
    }
    let result = String::from_chars(chars);
    ReadStringResult {
        value: result,
        next_offset: str_start + len
    }
}
