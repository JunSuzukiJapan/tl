// src/compiler/codegen/builtin_types/non_generic/npy.tl

struct NpyHeader {
    major_version: i64,
    minor_version: i64,
    descr: String,
    fortran_order: i64,
    shape: Vec<i64>
}

struct NpyFile {
    header: NpyHeader,
    data: Vec<u8>
}

// ヘルパー: LE読み取り
fn npy_read_u16_le(data: Vec<u8>, offset: i64) -> i64 {
    let b0 = data.get(offset).unwrap() as i64;
    let b1 = data.get(offset + 1).unwrap() as i64;
    b0 + b1 * 256
}

fn npy_read_u32_le(data: Vec<u8>, offset: i64) -> i64 {
    let b0 = data.get(offset).unwrap() as i64;
    let b1 = data.get(offset + 1).unwrap() as i64;
    let b2 = data.get(offset + 2).unwrap() as i64;
    let b3 = data.get(offset + 3).unwrap() as i64;
    b0 + b1 * 256 + b2 * 65536 + b3 * 16777216
}

// ヘルパー: LE書き込み
fn npy_write_u16_le(buf: Vec<u8>, val: i64) -> Vec<u8> {
    let mut res = buf;
    let b0 = val % 256;
    let b1 = (val / 256) % 256;
    res.push(b0 as u8);
    res.push(b1 as u8);
    res
}

// ヘルパー: ASCII空白スキップ
fn npy_skip_spaces(data: Vec<u8>, pos: i64) -> i64 {
    let mut i = pos;
    let len = data.len();
    while i < len {
        let b = data.get(i).unwrap() as i64;
        if b != 32 {
            if b != 10 {
                if b != 13 {
                    return i;
                }
            }
        }
        i = i + 1;
    }
    return i;
}

// ヘルパー: シングルクォート文字列を読み取る
// 例: 'descr' → "descr"
fn npy_read_quoted_string(data: Vec<u8>, pos: i64) -> (Option<String>, i64) {
    let p = npy_skip_spaces(data, pos);
    if p >= data.len() {
        let none: Option<String> = Option::None;
        let res = (none, p);
        return res;
    }
    let ch = data.get(p).unwrap() as i64;
    if ch != 39 { // not '
        let none: Option<String> = Option::None;
        let res = (none, p);
        return res;
    }
    let mut i = p + 1;
    let mut str_bytes: Vec<u8> = Vec::new();
    while i < data.len() {
        let b = data.get(i).unwrap() as i64;
        if b == 39 { // closing '
            let s = String::from_utf8(str_bytes);
            let some: Option<String> = Option::Some(s);
            let res = (some, i + 1);
            return res;
        }
        str_bytes.push(b as u8);
        i = i + 1;
    }
    let none: Option<String> = Option::None;
    let res = (none, i);
    return res;
}

// ヘルパー: 数値を読み取る (shape内の整数)
fn npy_read_int(data: Vec<u8>, pos: i64) -> (Option<i64>, i64) {
    let mut i = npy_skip_spaces(data, pos);
    let mut str_bytes: Vec<u8> = Vec::new();
    while i < data.len() {
        let b = data.get(i).unwrap() as i64;
        let mut is_digit = false;
        if b >= 48 {
            if b <= 57 {
                is_digit = true;
            }
        }
        if b == 45 { // minus sign
            is_digit = true;
        }
        if is_digit {
            str_bytes.push(b as u8);
            i = i + 1;
        } else {
            break;
        }
    }
    if str_bytes.len() == 0 {
        let none: Option<i64> = Option::None;
        let res = (none, i);
        return res;
    }
    let s = String::from_utf8(str_bytes);
    let val = s.to_i64();
    let some: Option<i64> = Option::Some(val);
    let res = (some, i);
    return res;
}

// ヘルパー: shape タプルをパース  (2, 3) → Vec<i64>
fn npy_parse_shape(data: Vec<u8>, pos: i64) -> (Vec<i64>, i64) {
    let mut shape: Vec<i64> = Vec::new();
    let p = npy_skip_spaces(data, pos);
    if p >= data.len() {
        let res = (shape, p);
        return res;
    }
    if (data.get(p).unwrap() as i64) != 40 { // not (
        let res = (shape, p);
        return res;
    }
    let mut i = p + 1;
    while i < data.len() {
        let p2 = npy_skip_spaces(data, i);
        if p2 < data.len() {
            if (data.get(p2).unwrap() as i64) == 41 { // )
                i = p2 + 1;
                break;
            }
        }
        let r = npy_read_int(data, i);
        match r.0 {
            Option::Some(v) => { shape.push(v); }
            Option::None => {
                // skip non-digit characters (like trailing comma)
                i = r.1;
                if i < data.len() {
                    i = i + 1;
                }
            }
        }
        i = r.1;
        // skip comma or spaces
        let p3 = npy_skip_spaces(data, i);
        if p3 < data.len() {
            if (data.get(p3).unwrap() as i64) == 44 { // ,
                i = p3 + 1;
            } else {
                i = p3;
            }
        } else {
            i = p3;
        }
    }
    let res = (shape, i);
    return res;
}

// Python dict ヘッダーをパース
// {'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }
fn npy_parse_header(header_bytes: Vec<u8>) -> NpyHeader {
    let mut descr = "";
    let mut fortran_order = 0;
    let mut shape: Vec<i64> = Vec::new();

    let mut pos = 0;
    // skip opening {
    let p = npy_skip_spaces(header_bytes, pos);
    if p < header_bytes.len() {
        if (header_bytes.get(p).unwrap() as i64) == 123 { // {
            pos = p + 1;
        }
    }

    while pos < header_bytes.len() {
        // skip spaces
        let p2 = npy_skip_spaces(header_bytes, pos);
        if p2 >= header_bytes.len() {
            break;
        }
        // check for closing }
        if (header_bytes.get(p2).unwrap() as i64) == 125 { // }
            break;
        }

        // read key
        let key_result = npy_read_quoted_string(header_bytes, p2);
        pos = key_result.1;

        // skip : 
        let p3 = npy_skip_spaces(header_bytes, pos);
        if p3 < header_bytes.len() {
            if (header_bytes.get(p3).unwrap() as i64) == 58 { // :
                pos = p3 + 1;
            }
        }

        match key_result.0 {
            Option::Some(key) => {
                if key == "descr" {
                    let val_result = npy_read_quoted_string(header_bytes, pos);
                    match val_result.0 {
                        Option::Some(v) => { descr = v; }
                        Option::None => {}
                    }
                    pos = val_result.1;
                } else {
                    if key == "fortran_order" {
                        // True or False
                        let p4 = npy_skip_spaces(header_bytes, pos);
                        if p4 + 3 < header_bytes.len() {
                            let b0 = header_bytes.get(p4).unwrap() as i64;
                            if b0 == 84 { // 'T' for True
                                fortran_order = 1;
                                pos = p4 + 4; // skip "True"
                            } else {
                                fortran_order = 0;
                                pos = p4 + 5; // skip "False"
                            }
                        }
                    } else {
                        if key == "shape" {
                            let shape_result = npy_parse_shape(header_bytes, pos);
                            shape = shape_result.0;
                            pos = shape_result.1;
                        } else {
                            // unknown key, skip value until comma or }
                            while pos < header_bytes.len() {
                                let b = header_bytes.get(pos).unwrap() as i64;
                                if b == 44 { // ,
                                    break;
                                }
                                if b == 125 { // }
                                    break;
                                }
                                pos = pos + 1;
                            }
                        }
                    }
                }
            }
            Option::None => {
                break;
            }
        }

        // skip comma
        let p5 = npy_skip_spaces(header_bytes, pos);
        if p5 < header_bytes.len() {
            if (header_bytes.get(p5).unwrap() as i64) == 44 { // ,
                pos = p5 + 1;
            } else {
                pos = p5;
            }
        } else {
            pos = p5;
        }
    }

    let header = NpyHeader {
        major_version: 0,
        minor_version: 0,
        descr: descr,
        fortran_order: fortran_order,
        shape: shape
    };
    header
}

impl NpyFile {
    fn load(path: String) -> Result<NpyFile, String> {
        let res = File::read_binary(path);
        let data = match res {
            Result::Ok(v) => v,
            Result::Err(e) => {
                let err: Result<NpyFile, String> = Result::Err(e);
                return err;
            }
        };

        // Check magic: \x93NUMPY
        if data.len() < 10 {
            let err: Result<NpyFile, String> = Result::Err("File too short");
            return err;
        }
        let m0 = data.get(0).unwrap() as i64;
        let m1 = data.get(1).unwrap() as i64;
        let m2 = data.get(2).unwrap() as i64;
        let m3 = data.get(3).unwrap() as i64;
        let m4 = data.get(4).unwrap() as i64;
        let m5 = data.get(5).unwrap() as i64;
        // \x93 = 147, N=78, U=85, M=77, P=80, Y=89
        if m0 != 147 {
            let err: Result<NpyFile, String> = Result::Err("Invalid magic");
            return err;
        }
        if m1 != 78 {
            let err: Result<NpyFile, String> = Result::Err("Invalid magic");
            return err;
        }

        let major = data.get(6).unwrap() as i64;
        let minor = data.get(7).unwrap() as i64;

        // Read header length
        let mut header_len = 0;
        let mut data_offset = 0;
        if major == 1 {
            header_len = npy_read_u16_le(data, 8);
            data_offset = 10 + header_len;
        } else {
            header_len = npy_read_u32_le(data, 8);
            data_offset = 12 + header_len;
        }

        // Extract header bytes
        let mut header_bytes: Vec<u8> = Vec::new();
        let header_start = if major == 1 { 10 } else { 12 };
        let mut i = 0;
        while i < header_len {
            header_bytes.push(data.get(header_start + i).unwrap());
            i = i + 1;
        }

        // Parse header
        let mut header = npy_parse_header(header_bytes);
        header.major_version = major;
        header.minor_version = minor;

        // Extract data bytes
        let mut npy_data: Vec<u8> = Vec::new();
        let mut j = data_offset;
        while j < data.len() {
            npy_data.push(data.get(j).unwrap());
            j = j + 1;
        }

        let npy = NpyFile {
            header: header,
            data: npy_data
        };
        let ok: Result<NpyFile, String> = Result::Ok(npy);
        ok
    }

    fn element_count(self) -> i64 {
        let mut count = 1;
        let shape = self.header.shape;
        let mut i = 0;
        while i < shape.len() {
            count = count * shape.get(i).unwrap();
            i = i + 1;
        }
        count
    }

    fn dtype_name(self) -> String {
        self.header.descr
    }

    fn shape_str(self) -> String {
        let shape = self.header.shape;
        let mut sb = StringBuilder::new();
        sb.append("(");
        let mut i = 0;
        while i < shape.len() {
            if i > 0 {
                sb.append(", ");
            }
            sb.append_i64(shape.get(i).unwrap());
            i = i + 1;
        }
        sb.append(")");
        sb.build()
    }
}

fn save_npy(npy: NpyFile, path: String) -> Result<bool, String> {
    // Build header string: {'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }
    let mut sb = StringBuilder::new();
    sb.append("{'descr': '");
    sb.append(npy.header.descr);
    sb.append("', 'fortran_order': ");
    if npy.header.fortran_order != 0 {
        sb.append("True");
    } else {
        sb.append("False");
    }
    sb.append(", 'shape': (");
    let shape = npy.header.shape;
    let mut i = 0;
    while i < shape.len() {
        if i > 0 {
            sb.append(", ");
        }
        sb.append_i64(shape.get(i).unwrap());
        i = i + 1;
    }
    sb.append("), }");

    let header_str = sb.build();
    let header_bytes = header_str.to_bytes();
    let header_len = header_bytes.len();

    // Pad to 64-byte alignment (magic(6) + version(2) + header_len(2) + header + padding = multiple of 64)
    let preamble_size = 10; // v1: magic(6) + version(2) + header_len(2)
    let total_unpadded = preamble_size + header_len;
    let mut padded_header_len = header_len;
    let rem = total_unpadded % 64;
    if rem != 0 {
        padded_header_len = header_len + (64 - rem);
    }
    // +1 for trailing newline
    padded_header_len = padded_header_len;

    // Build file data
    let mut file_data: Vec<u8> = Vec::new();
    // Magic
    file_data.push(147 as u8); // \x93
    file_data.push(78 as u8);  // N
    file_data.push(85 as u8);  // U
    file_data.push(77 as u8);  // M
    file_data.push(80 as u8);  // P
    file_data.push(89 as u8);  // Y
    // Version 1.0
    file_data.push(1 as u8);
    file_data.push(0 as u8);
    // Header length (u16 LE)
    file_data = npy_write_u16_le(file_data, padded_header_len);

    // Header bytes
    let mut k = 0;
    while k < header_bytes.len() {
        file_data.push(header_bytes.get(k).unwrap());
        k = k + 1;
    }

    // Padding (spaces)
    let mut p = header_bytes.len();
    while p < padded_header_len - 1 {
        file_data.push(32 as u8); // space
        p = p + 1;
    }
    // Trailing newline
    file_data.push(10 as u8); // \n

    // Data
    let mut d = 0;
    while d < npy.data.len() {
        file_data.push(npy.data.get(d).unwrap());
        d = d + 1;
    }

    let ok = File::write_binary(path, file_data);
    if ok {
        let res: Result<bool, String> = Result::Ok(true);
        return res;
    }
    let err: Result<bool, String> = Result::Err("Failed to write file");
    err
}
