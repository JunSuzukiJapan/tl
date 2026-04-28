// src/compiler/codegen/builtin_types/non_generic/npz.tl

struct NpzEntry {
    name: String,
    data: Vec<u8>
}

struct NpzFile {
    entries: Vec<NpzEntry>
}

// ヘルパー: LE読み取り
fn npz_read_u16_le(data: Vec<u8>, offset: i64) -> i64 {
    let b0 = data.get(offset).unwrap() as i64;
    let b1 = data.get(offset + 1).unwrap() as i64;
    b0 + b1 * 256
}

fn npz_read_u32_le(data: Vec<u8>, offset: i64) -> i64 {
    let b0 = data.get(offset).unwrap() as i64;
    let b1 = data.get(offset + 1).unwrap() as i64;
    let b2 = data.get(offset + 2).unwrap() as i64;
    let b3 = data.get(offset + 3).unwrap() as i64;
    b0 + b1 * 256 + b2 * 65536 + b3 * 16777216
}

// ヘルパー: LE書き込み
fn npz_write_u16_le(buf: Vec<u8>, val: i64) -> Vec<u8> {
    let mut res = buf;
    res.push((val % 256) as u8);
    res.push(((val / 256) % 256) as u8);
    res
}

fn npz_write_u32_le(buf: Vec<u8>, val: i64) -> Vec<u8> {
    let mut res = buf;
    res.push((val % 256) as u8);
    res.push(((val / 256) % 256) as u8);
    res.push(((val / 65536) % 256) as u8);
    res.push(((val / 16777216) % 256) as u8);
    res
}

// ヘルパー: バイト列からファイル名を抽出
fn npz_read_filename(data: Vec<u8>, offset: i64, length: i64) -> String {
    let mut bytes: Vec<u8> = Vec::new();
    let mut i = 0;
    while i < length {
        bytes.push(data.get(offset + i).unwrap());
        i = i + 1;
    }
    String::from_utf8(bytes)
}

// PKZIP Local File Header パーサー
// Signature: PK\x03\x04 (0x04034b50)
// Offset  Size  Field
// 0       4     signature
// 4       2     version needed
// 6       2     general purpose bit flag
// 8       2     compression method (0=STORED)
// 10      2     last mod time
// 12      2     last mod date
// 14      4     CRC-32
// 18      4     compressed size
// 22      4     uncompressed size
// 26      2     filename length
// 28      2     extra field length
// 30      N     filename
// 30+N    M     extra field
// 30+N+M  data  file data

impl NpzFile {
    fn load(path: String) -> Result<NpzFile, String> {
        let res = File::read_binary(path);
        let data = match res {
            Result::Ok(v) => v,
            Result::Err(e) => {
                let err: Result<NpzFile, String> = Result::Err(e);
                return err;
            }
        };

        let mut entries: Vec<NpzEntry> = Vec::new();
        let mut offset = 0;

        while offset + 30 <= data.len() {
            // Check PK\x03\x04 signature
            let sig = npz_read_u32_le(data, offset);
            if sig != 67324752 { // 0x04034b50
                // Not a local file header, stop
                break;
            }

            let compression = npz_read_u16_le(data, offset + 8);
            if compression != 0 {
                let err: Result<NpzFile, String> = Result::Err("Only STORED (uncompressed) entries are supported");
                return err;
            }

            let compressed_size = npz_read_u32_le(data, offset + 18);
            let uncompressed_size = npz_read_u32_le(data, offset + 22);
            let filename_len = npz_read_u16_le(data, offset + 26);
            let extra_len = npz_read_u16_le(data, offset + 28);

            // Read filename
            let name = npz_read_filename(data, offset + 30, filename_len);

            // Data starts after header + filename + extra
            let data_start = offset + 30 + filename_len + extra_len;

            // Extract data
            let mut entry_data: Vec<u8> = Vec::new();
            let mut i = 0;
            while i < uncompressed_size {
                entry_data.push(data.get(data_start + i).unwrap());
                i = i + 1;
            }

            let entry = NpzEntry {
                name: name,
                data: entry_data
            };
            entries.push(entry);

            // Move to next entry
            offset = data_start + compressed_size;
        }

        let npz = NpzFile {
            entries: entries
        };
        let ok: Result<NpzFile, String> = Result::Ok(npz);
        ok
    }

    fn entry_count(self) -> i64 {
        self.entries.len()
    }

    fn get_entry(self, name: String) -> Option<NpzEntry> {
        let mut i = 0;
        let entries = self.entries;
        while i < entries.len() {
            let entry = entries.get(i).unwrap();
            if entry.name == name {
                let some: Option<NpzEntry> = Option::Some(entry);
                return some;
            }
            i = i + 1;
        }
        let none: Option<NpzEntry> = Option::None;
        none
    }
}

fn npz_get_npy(npz: NpzFile, name: String) -> Option<NpyFile> {
    // Try exact name first
    let entry_opt = npz.get_entry(name);
    match entry_opt {
        Option::Some(entry) => {
            let npy_result = npz_parse_npy_from_bytes(entry.data);
            match npy_result {
                Result::Ok(npy) => {
                    let some: Option<NpyFile> = Option::Some(npy);
                    return some;
                }
                Result::Err(err) => {
                    let none: Option<NpyFile> = Option::None;
                    return none;
                }
            }
        }
        Option::None => {
            // Try appending .npy
            let mut sb = StringBuilder::new();
            sb.append(name);
            sb.append(".npy");
            let lookup2 = sb.build();
            let entry_opt2 = npz.get_entry(lookup2);
            match entry_opt2 {
                Option::Some(entry2) => {
                    let npy_result2 = npz_parse_npy_from_bytes(entry2.data);
                    match npy_result2 {
                        Result::Ok(npy2) => {
                            let some2: Option<NpyFile> = Option::Some(npy2);
                            return some2;
                        }
                        Result::Err(err2) => {
                            let none2: Option<NpyFile> = Option::None;
                            return none2;
                        }
                    }
                }
                Option::None => {
                    let none3: Option<NpyFile> = Option::None;
                    return none3;
                }
            }
        }
    }
}

// NpyFile パーサー (bytes から直接)
fn npz_parse_npy_from_bytes(data: Vec<u8>) -> Result<NpyFile, String> {
    if data.len() < 10 {
        let err: Result<NpyFile, String> = Result::Err("Data too short for .npy");
        return err;
    }
    let m0 = data.get(0).unwrap() as i64;
    if m0 != 147 {
        let err: Result<NpyFile, String> = Result::Err("Invalid .npy magic");
        return err;
    }

    let major = data.get(6).unwrap() as i64;
    let minor = data.get(7).unwrap() as i64;

    let mut header_len = 0;
    let mut data_offset = 0;
    if major == 1 {
        header_len = npy_read_u16_le(data, 8);
        data_offset = 10 + header_len;
    } else {
        header_len = npy_read_u32_le(data, 8);
        data_offset = 12 + header_len;
    }

    let header_start = if major == 1 { 10 } else { 12 };
    let mut header_bytes: Vec<u8> = Vec::new();
    let mut i = 0;
    while i < header_len {
        header_bytes.push(data.get(header_start + i).unwrap());
        i = i + 1;
    }

    let mut header = npy_parse_header(header_bytes);
    header.major_version = major;
    header.minor_version = minor;

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

// NPZ ファイルの書き出し (STORED のみ)
fn save_npz(npz: NpzFile, path: String) -> Result<bool, String> {
    let mut file_data: Vec<u8> = Vec::new();
    let mut cd_offsets: Vec<i64> = Vec::new();

    let entries = npz.entries;
    let mut i = 0;
    while i < entries.len() {
        let entry = entries.get(i).unwrap();
        let name_bytes = entry.name.to_bytes();
        let data_size = entry.data.len();

        cd_offsets.push(file_data.len());

        // Local file header: PK\x03\x04
        file_data.push(80 as u8);
        file_data.push(75 as u8);
        file_data.push(3 as u8);
        file_data.push(4 as u8);
        file_data = npz_write_u16_le(file_data, 20);  // version
        file_data = npz_write_u16_le(file_data, 0);   // flags
        file_data = npz_write_u16_le(file_data, 0);   // compression: STORED
        file_data = npz_write_u16_le(file_data, 0);   // mod time
        file_data = npz_write_u16_le(file_data, 0);   // mod date
        file_data = npz_write_u32_le(file_data, 0);   // CRC-32
        file_data = npz_write_u32_le(file_data, data_size);  // compressed
        file_data = npz_write_u32_le(file_data, data_size);  // uncompressed
        file_data = npz_write_u16_le(file_data, name_bytes.len());  // name len
        file_data = npz_write_u16_le(file_data, 0);   // extra len

        let mut k = 0;
        while k < name_bytes.len() {
            file_data.push(name_bytes.get(k).unwrap());
            k = k + 1;
        }
        let mut d = 0;
        while d < entry.data.len() {
            file_data.push(entry.data.get(d).unwrap());
            d = d + 1;
        }
        i = i + 1;
    }

    // Central directory
    let cd_start = file_data.len();
    let mut ci = 0;
    while ci < entries.len() {
        let entry2 = entries.get(ci).unwrap();
        let name_bytes2 = entry2.name.to_bytes();
        let data_size2 = entry2.data.len();
        let local_offset = cd_offsets.get(ci).unwrap();

        // PK\x01\x02
        file_data.push(80 as u8);
        file_data.push(75 as u8);
        file_data.push(1 as u8);
        file_data.push(2 as u8);
        file_data = npz_write_u16_le(file_data, 20);  // version made by
        file_data = npz_write_u16_le(file_data, 20);  // version needed
        file_data = npz_write_u16_le(file_data, 0);   // flags
        file_data = npz_write_u16_le(file_data, 0);   // compression
        file_data = npz_write_u16_le(file_data, 0);   // mod time
        file_data = npz_write_u16_le(file_data, 0);   // mod date
        file_data = npz_write_u32_le(file_data, 0);   // CRC-32
        file_data = npz_write_u32_le(file_data, data_size2);
        file_data = npz_write_u32_le(file_data, data_size2);
        file_data = npz_write_u16_le(file_data, name_bytes2.len());
        file_data = npz_write_u16_le(file_data, 0);   // extra len
        file_data = npz_write_u16_le(file_data, 0);   // comment len
        file_data = npz_write_u16_le(file_data, 0);   // disk start
        file_data = npz_write_u16_le(file_data, 0);   // internal attrs
        file_data = npz_write_u32_le(file_data, 0);   // external attrs
        file_data = npz_write_u32_le(file_data, local_offset);

        let mut k2 = 0;
        while k2 < name_bytes2.len() {
            file_data.push(name_bytes2.get(k2).unwrap());
            k2 = k2 + 1;
        }
        ci = ci + 1;
    }
    let cd_end = file_data.len();
    let cd_size = cd_end - cd_start;

    // End of central directory: PK\x05\x06
    file_data.push(80 as u8);
    file_data.push(75 as u8);
    file_data.push(5 as u8);
    file_data.push(6 as u8);
    file_data = npz_write_u16_le(file_data, 0);  // disk num
    file_data = npz_write_u16_le(file_data, 0);  // cd disk
    file_data = npz_write_u16_le(file_data, entries.len());  // cd records this disk
    file_data = npz_write_u16_le(file_data, entries.len());  // cd records total
    file_data = npz_write_u32_le(file_data, cd_size);
    file_data = npz_write_u32_le(file_data, cd_start);
    file_data = npz_write_u16_le(file_data, 0);  // comment len

    let ok = File::write_binary(path, file_data);
    if ok {
        let res: Result<bool, String> = Result::Ok(true);
        return res;
    }
    let err: Result<bool, String> = Result::Err("Failed to write file");
    err
}
