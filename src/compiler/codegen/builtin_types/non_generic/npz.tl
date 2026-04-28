// src/compiler/codegen/builtin_types/non_generic/npz.tl

struct NpzEntry {
    name: String,
    data: Vec<u8>
}

struct NpzFile {
    entries: Vec<NpzEntry>
}

impl NpzFile {
    // --- LE helpers (static methods) ---

    fn read_u16_le(data: Vec<u8>, offset: i64) -> i64 {
        let b0 = data.get(offset).unwrap() as i64;
        let b1 = data.get(offset + 1).unwrap() as i64;
        b0 + b1 * 256
    }

    fn read_u32_le(data: Vec<u8>, offset: i64) -> i64 {
        let b0 = data.get(offset).unwrap() as i64;
        let b1 = data.get(offset + 1).unwrap() as i64;
        let b2 = data.get(offset + 2).unwrap() as i64;
        let b3 = data.get(offset + 3).unwrap() as i64;
        b0 + b1 * 256 + b2 * 65536 + b3 * 16777216
    }

    fn write_u16_le(buf: Vec<u8>, val: i64) -> Vec<u8> {
        let mut res = buf;
        res.push((val % 256) as u8);
        res.push(((val / 256) % 256) as u8);
        res
    }

    fn write_u32_le(buf: Vec<u8>, val: i64) -> Vec<u8> {
        let mut res = buf;
        res.push((val % 256) as u8);
        res.push(((val / 256) % 256) as u8);
        res.push(((val / 65536) % 256) as u8);
        res.push(((val / 16777216) % 256) as u8);
        res
    }

    fn read_filename(data: Vec<u8>, offset: i64, length: i64) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        let mut i = 0;
        while i < length {
            bytes.push(data.get(offset + i).unwrap());
            i = i + 1;
        }
        String::from_utf8(bytes)
    }

    // --- Public API ---

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
            let sig = NpzFile::read_u32_le(data, offset);
            if sig != 67324752 {
                break;
            }

            let compression = NpzFile::read_u16_le(data, offset + 8);
            if compression != 0 {
                let err: Result<NpzFile, String> = Result::Err("Only STORED (uncompressed) entries are supported");
                return err;
            }

            let compressed_size = NpzFile::read_u32_le(data, offset + 18);
            let uncompressed_size = NpzFile::read_u32_le(data, offset + 22);
            let filename_len = NpzFile::read_u16_le(data, offset + 26);
            let extra_len = NpzFile::read_u16_le(data, offset + 28);

            let name = NpzFile::read_filename(data, offset + 30, filename_len);

            let data_start = offset + 30 + filename_len + extra_len;

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

    fn save(self, path: String) -> Result<bool, String> {
        let mut file_data: Vec<u8> = Vec::new();
        let mut cd_offsets: Vec<i64> = Vec::new();

        let entries = self.entries;
        let mut i = 0;
        while i < entries.len() {
            let entry = entries.get(i).unwrap();
            let name_bytes = entry.name.to_bytes();
            let data_size = entry.data.len();

            cd_offsets.push(file_data.len());

            file_data.push(80 as u8);
            file_data.push(75 as u8);
            file_data.push(3 as u8);
            file_data.push(4 as u8);
            file_data = NpzFile::write_u16_le(file_data, 20);
            file_data = NpzFile::write_u16_le(file_data, 0);
            file_data = NpzFile::write_u16_le(file_data, 0);
            file_data = NpzFile::write_u16_le(file_data, 0);
            file_data = NpzFile::write_u16_le(file_data, 0);
            file_data = NpzFile::write_u32_le(file_data, 0);
            file_data = NpzFile::write_u32_le(file_data, data_size);
            file_data = NpzFile::write_u32_le(file_data, data_size);
            file_data = NpzFile::write_u16_le(file_data, name_bytes.len());
            file_data = NpzFile::write_u16_le(file_data, 0);

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

        let cd_start = file_data.len();
        let mut ci = 0;
        while ci < entries.len() {
            let entry2 = entries.get(ci).unwrap();
            let name_bytes2 = entry2.name.to_bytes();
            let data_size2 = entry2.data.len();
            let local_offset = cd_offsets.get(ci).unwrap();

            file_data.push(80 as u8);
            file_data.push(75 as u8);
            file_data.push(1 as u8);
            file_data.push(2 as u8);
            file_data = NpzFile::write_u16_le(file_data, 20);
            file_data = NpzFile::write_u16_le(file_data, 20);
            file_data = NpzFile::write_u16_le(file_data, 0);
            file_data = NpzFile::write_u16_le(file_data, 0);
            file_data = NpzFile::write_u16_le(file_data, 0);
            file_data = NpzFile::write_u16_le(file_data, 0);
            file_data = NpzFile::write_u32_le(file_data, 0);
            file_data = NpzFile::write_u32_le(file_data, data_size2);
            file_data = NpzFile::write_u32_le(file_data, data_size2);
            file_data = NpzFile::write_u16_le(file_data, name_bytes2.len());
            file_data = NpzFile::write_u16_le(file_data, 0);
            file_data = NpzFile::write_u16_le(file_data, 0);
            file_data = NpzFile::write_u16_le(file_data, 0);
            file_data = NpzFile::write_u16_le(file_data, 0);
            file_data = NpzFile::write_u32_le(file_data, 0);
            file_data = NpzFile::write_u32_le(file_data, local_offset);

            let mut k2 = 0;
            while k2 < name_bytes2.len() {
                file_data.push(name_bytes2.get(k2).unwrap());
                k2 = k2 + 1;
            }
            ci = ci + 1;
        }
        let cd_end = file_data.len();
        let cd_size = cd_end - cd_start;

        file_data.push(80 as u8);
        file_data.push(75 as u8);
        file_data.push(5 as u8);
        file_data.push(6 as u8);
        file_data = NpzFile::write_u16_le(file_data, 0);
        file_data = NpzFile::write_u16_le(file_data, 0);
        file_data = NpzFile::write_u16_le(file_data, entries.len());
        file_data = NpzFile::write_u16_le(file_data, entries.len());
        file_data = NpzFile::write_u32_le(file_data, cd_size);
        file_data = NpzFile::write_u32_le(file_data, cd_start);
        file_data = NpzFile::write_u16_le(file_data, 0);

        let ok = File::write_binary(path, file_data);
        if ok {
            let res: Result<bool, String> = Result::Ok(true);
            return res;
        }
        let err: Result<bool, String> = Result::Err("Failed to write file");
        err
    }
}

// get_npy と parse_npy_from_bytes は impl 外のフリー関数として残す
// (TLコンパイラの制約: Option<NpyFile> を返す NpzFile impl メソッドの IR 検証エラー)
// 名前衝突リスクが低い _npz_ プレフィックス付き

fn _npz_parse_npy_from_bytes(data: Vec<u8>) -> Result<NpyFile, String> {
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
        header_len = NpyFile::read_u16_le(data, 8);
        data_offset = 10 + header_len;
    } else {
        header_len = NpyFile::read_u32_le(data, 8);
        data_offset = 12 + header_len;
    }

    let header_start = if major == 1 { 10 } else { 12 };
    let mut header_bytes: Vec<u8> = Vec::new();
    let mut i = 0;
    while i < header_len {
        header_bytes.push(data.get(header_start + i).unwrap());
        i = i + 1;
    }

    let mut header = NpyFile::parse_header(header_bytes);
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

fn _npz_get_npy(npz: NpzFile, name: String) -> Option<NpyFile> {
    let entry_opt = npz.get_entry(name);
    match entry_opt {
        Option::Some(entry) => {
            let npy_result = _npz_parse_npy_from_bytes(entry.data);
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
                    let npy_result2 = _npz_parse_npy_from_bytes(entry2.data);
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
