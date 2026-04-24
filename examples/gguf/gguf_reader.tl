// ============================================================
// GGUF Reader — TL言語で記述されたGGUFバイナリパーサー
// ============================================================
// 目的:
//   - TLコンパイラのバグを発見する
//   - TLに不足している機能を発見する
//
// GGUFフォーマット仕様:
//   Header:
//     magic: u32 ("GGUF" = 0x46554747 LE)
//     version: u32
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
// ============================================================

// --- 結果を保持する構造体 ---

struct ReadStringResult {
    value: String,
    next_offset: i64
}

struct GgufHeader {
    magic: i64,
    version: i64,
    tensor_count: i64,
    metadata_kv_count: i64
}

struct GgufTensorInfo {
    name: String,
    n_dims: i64,
    shape: Vec<i64>,
    dtype: i64,
    offset: i64
}

struct GgufMetadataKV {
    key: String,
    value_type: i64,
    value_str: String,
    value_int: i64
}

// --- バイト読み取りヘルパー関数 ---

// リトルエンディアンでu32を読み取る
fn read_u32_le(data: Vec<u8>, offset: i64) -> i64 {
    let b0 = data.get(offset).unwrap() as i64;
    let b1 = data.get(offset + 1).unwrap() as i64;
    let b2 = data.get(offset + 2).unwrap() as i64;
    let b3 = data.get(offset + 3).unwrap() as i64;
    b0 + b1 * 256 + b2 * 65536 + b3 * 16777216
}

// リトルエンディアンでu64を読み取る
fn read_u64_le(data: Vec<u8>, offset: i64) -> i64 {
    let lo = read_u32_le(data, offset);
    let hi = read_u32_le(data, offset + 4);
    lo + hi * 4294967296
}

// GGUF文字列を読み取る (u64 len + bytes)
fn read_gguf_string(data: Vec<u8>, offset: i64) -> ReadStringResult {
    let len = read_u64_le(data, offset);
    let str_start = offset + 8;

    // バイト列からASCII文字列を構築 — String::from_chars(Vec<i64>) を使用
    let mut chars = Vec::new();
    let mut i = 0;
    while i < len {
        let byte = data.get(str_start + i).unwrap() as i64;
        // 印字可能なASCII文字のみを追加
        if byte >= 32 && byte <= 126 {
            chars.push(byte);
        } else {
            chars.push(63); // '?' = 63
        }
        i = i + 1;
    }
    let result = String::from_chars(chars);

    ReadStringResult {
        value: result,
        next_offset: str_start + len
    }
}

// --- GGUFヘッダーのパース ---

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

// マジックバイトの検証 ("GGUF" = 0x46554747 = 1179993927)
fn verify_magic(magic: i64) -> bool {
    magic == 1179993927
}

// GGMLタイプ名を返す
fn ggml_type_name(dtype: i64) -> String {
    if dtype == 0 {
        "F32"
    } else if dtype == 1 {
        "F16"
    } else if dtype == 2 {
        "Q4_0"
    } else if dtype == 3 {
        "Q4_1"
    } else if dtype == 6 {
        "Q5_0"
    } else if dtype == 7 {
        "Q5_1"
    } else if dtype == 8 {
        "Q8_0"
    } else if dtype == 9 {
        "Q8_1"
    } else if dtype == 12 {
        "Q4_K"
    } else if dtype == 14 {
        "Q6_K"
    } else {
        "UNKNOWN(" + String::from_int(dtype) + ")"
    }
}

// GGUFメタデータ値タイプ名を返す
fn value_type_name(vt: i64) -> String {
    if vt == 0 {
        "UINT8"
    } else if vt == 1 {
        "INT8"
    } else if vt == 2 {
        "UINT16"
    } else if vt == 3 {
        "INT16"
    } else if vt == 4 {
        "UINT32"
    } else if vt == 5 {
        "INT32"
    } else if vt == 6 {
        "FLOAT32"
    } else if vt == 7 {
        "BOOL"
    } else if vt == 8 {
        "STRING"
    } else if vt == 9 {
        "ARRAY"
    } else if vt == 10 {
        "UINT64"
    } else if vt == 11 {
        "INT64"
    } else if vt == 12 {
        "FLOAT64"
    } else {
        "UNKNOWN"
    }
}

// メタデータ値のサイズを返す（配列・文字列以外の固定サイズ型）
fn value_type_size(vt: i64) -> i64 {
    if vt == 0 { 1 }       // UINT8
    else if vt == 1 { 1 }  // INT8
    else if vt == 2 { 2 }  // UINT16
    else if vt == 3 { 2 }  // INT16
    else if vt == 4 { 4 }  // UINT32
    else if vt == 5 { 4 }  // INT32
    else if vt == 6 { 4 }  // FLOAT32
    else if vt == 7 { 1 }  // BOOL
    else if vt == 10 { 8 } // UINT64
    else if vt == 11 { 8 } // INT64
    else if vt == 12 { 8 } // FLOAT64
    else { 0 }             // STRING, ARRAY は可変長
}

// --- メタデータKVペアのパース ---

fn skip_metadata_value(data: Vec<u8>, offset: i64, vtype: i64) -> i64 {
    if vtype == 8 {
        // STRING: u64 len + bytes
        let len = read_u64_le(data, offset);
        offset + 8 + len
    } else if vtype == 9 {
        // ARRAY: u32 elem_type + u64 count + elements
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
        // 固定サイズ型
        offset + value_type_size(vtype)
    }
}

fn parse_metadata_kv(data: Vec<u8>, offset: i64) -> GgufMetadataKV {
    // キーを読む
    let key_result = read_gguf_string(data, offset);
    let key = key_result.value;
    let mut pos = key_result.next_offset;

    // 値の型を読む
    let value_type = read_u32_le(data, pos);
    pos = pos + 4;

    // 値を読む（簡易版：文字列とi64のみ完全サポート）
    let mut value_str = "";
    let mut value_int = 0;

    if value_type == 8 {
        // STRING
        let str_result = read_gguf_string(data, pos);
        value_str = str_result.value;
        pos = str_result.next_offset;
    } else if value_type == 4 || value_type == 5 {
        // UINT32 / INT32
        value_int = read_u32_le(data, pos);
        value_str = String::from_int(value_int);
        pos = pos + 4;
    } else if value_type == 10 || value_type == 11 {
        // UINT64 / INT64
        value_int = read_u64_le(data, pos);
        value_str = String::from_int(value_int);
        pos = pos + 8;
    } else if value_type == 6 {
        // FLOAT32 — 4 bytes (読み取りだけスキップ)
        value_str = "<f32>";
        pos = pos + 4;
    } else if value_type == 7 {
        // BOOL — 1 byte
        let b = data.get(pos).unwrap() as i64;
        value_int = b;
        if b == 0 {
            value_str = "false";
        } else {
            value_str = "true";
        }
        pos = pos + 1;
    } else if value_type == 0 || value_type == 1 {
        // UINT8 / INT8
        value_int = data.get(pos).unwrap() as i64;
        value_str = String::from_int(value_int);
        pos = pos + 1;
    } else if value_type == 2 || value_type == 3 {
        // UINT16 / INT16
        let b0 = data.get(pos).unwrap() as i64;
        let b1 = data.get(pos + 1).unwrap() as i64;
        value_int = b0 + b1 * 256;
        value_str = String::from_int(value_int);
        pos = pos + 2;
    } else if value_type == 9 {
        // ARRAY — 型＋カウント＋要素をスキップ
        let elem_type = read_u32_le(data, pos);
        let count = read_u64_le(data, pos + 4);
        value_str = "[array: " + value_type_name(elem_type) + " x " + String::from_int(count) + "]";
        pos = pos + 12;
        // 要素をスキップ
        let mut i = 0;
        while i < count {
            pos = skip_metadata_value(data, pos, elem_type);
            i = i + 1;
        }
    } else {
        value_str = "<unknown type " + String::from_int(value_type) + ">";
        pos = skip_metadata_value(data, pos, value_type);
    }

    GgufMetadataKV {
        key: key,
        value_type: value_type,
        value_str: value_str,
        value_int: value_int
    }
}

// --- テンソルインフォのパース ---

fn parse_tensor_info(data: Vec<u8>, offset: i64) -> GgufTensorInfo {
    // 名前を読む
    let name_result = read_gguf_string(data, offset);
    let name = name_result.value;
    let mut pos = name_result.next_offset;

    // 次元数
    let n_dims = read_u32_le(data, pos);
    pos = pos + 4;

    // 各次元のサイズ
    let mut shape = Vec::new();
    let mut i = 0;
    while i < n_dims {
        let dim = read_u64_le(data, pos);
        shape.push(dim);
        pos = pos + 8;
        i = i + 1;
    }

    // 型
    let dtype = read_u32_le(data, pos);
    pos = pos + 4;

    // オフセット
    let tensor_offset = read_u64_le(data, pos);

    GgufTensorInfo {
        name: name,
        n_dims: n_dims,
        shape: shape,
        dtype: dtype,
        offset: tensor_offset
    }
}

// --- メイン ---

fn main() {
    println("=== GGUF Reader (written in TL) ===");
    println("");

    // ---- ファイルの準備 ----
    let model_path = "~/.llm/models/gte-small/gte-small-q8_0.gguf";

    if !File::exists(model_path) {
        println("Model file not found. Downloading...");
        File::create_dir("~/.llm/models/gte-small");
        let url = "https://huggingface.co/ggml-org/gte-small-Q8_0-GGUF/resolve/main/gte-small-q8_0.gguf";
        if !File::download(url, model_path) {
            println("ERROR: Download failed.");
            return;
        }
        println("Download complete.");
    }

    // ---- バイナリ読み込み ----
    println("Reading file: {}", model_path);
    let res = File::read_binary(model_path);
    let data = match res {
        Result::Ok(v) => v,
        Result::Err(e) => {
            println("ERROR: Failed to read file: {}", e);
            return;
        }
    };
    println("File size: {} bytes", data.len());
    println("");

    // ---- ヘッダーパース ----
    println("--- Header ---");
    let header = parse_header(data);

    if !verify_magic(header.magic) {
        println("ERROR: Invalid magic number: {}", header.magic);
        return;
    }
    println("Magic: GGUF (valid)");
    println("Version: {}", header.version);
    println("Tensor count: {}", header.tensor_count);
    println("Metadata KV count: {}", header.metadata_kv_count);
    println("");

    // ---- メタデータパース ----
    println("--- Metadata ---");
    let mut offset = 24;  // ヘッダーサイズ = 4 + 4 + 8 + 8 = 24 bytes
    let mut kv_idx = 0;
    while kv_idx < header.metadata_kv_count {
        let kv = parse_metadata_kv(data, offset);
        println("  [{}] {} ({}) = {}",
            kv_idx, kv.key, value_type_name(kv.value_type), kv.value_str);

        // 次のKVペアのオフセットを計算
        // parse_metadata_kv 内で消費したバイト数を再計算
        let key_result = read_gguf_string(data, offset);
        offset = key_result.next_offset + 4; // +4 for value_type
        offset = skip_metadata_value(data, offset, kv.value_type);

        kv_idx = kv_idx + 1;
    }
    println("");

    // ---- テンソルインフォパース ----
    println("--- Tensor Info ---");
    let mut tensor_idx = 0;
    let max_display = 20;
    while tensor_idx < header.tensor_count {
        let info = parse_tensor_info(data, offset);

        if tensor_idx < max_display {
            // shapeの表示
            let mut shape_str = "[";
            let mut d = 0;
            while d < info.n_dims {
                if d > 0 {
                    shape_str = shape_str + ", ";
                }
                shape_str = shape_str + String::from_int(info.shape.get(d).unwrap());
                d = d + 1;
            }
            shape_str = shape_str + "]";

            println("  [{}] {} : {} shape={} offset={}",
                tensor_idx, info.name, ggml_type_name(info.dtype),
                shape_str, info.offset);
        }

        // 次のテンソルインフォへ
        let name_result = read_gguf_string(data, offset);
        offset = name_result.next_offset;
        offset = offset + 4; // n_dims
        offset = offset + info.n_dims * 8; // dims
        offset = offset + 4; // type
        offset = offset + 8; // offset

        tensor_idx = tensor_idx + 1;
    }

    if header.tensor_count > max_display {
        println("  ... ({} more tensors)", header.tensor_count - max_display);
    }

    println("");
    println("=== GGUF parsing complete ===");
    println("Total tensors: {}", header.tensor_count);
}
