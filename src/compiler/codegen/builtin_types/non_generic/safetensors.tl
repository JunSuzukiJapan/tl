// src/compiler/codegen/builtin_types/non_generic/safetensors.tl

struct SafeTensorInfo {
    name: String,
    dtype: String,
    shape: Vec<i64>,
    data_begin: i64,
    data_end: i64
}

struct SafeTensorsFile {
    tensor_count: i64,
    tensors: Vec<SafeTensorInfo>
}

// JSON parsing helpers
fn skip_whitespace(data: Vec<u8>, pos: i64) -> i64 {
    let mut i = pos;
    let len = data.len();
    while i < len {
        let b = data.get(i).unwrap();
        let b_val = b as i64;
        if b_val != 32 {
            if b_val != 10 {
                if b_val != 13 {
                    if b_val != 9 {
                        return i;
                    }
                }
            }
        }
        i = i + 1;
    }
    return i;
}

fn expect_char(data: Vec<u8>, pos: i64, expected: i64) -> (bool, i64) {
    let p = skip_whitespace(data, pos);
    if p < data.len() {
        if (data.get(p).unwrap() as i64) == expected {
            let res = (true, p + 1);
            return res;
        }
    }
    let res2 = (false, p);
    return res2;
}

fn read_string(data: Vec<u8>, pos: i64) -> (Option<String>, i64) {
    let r1 = expect_char(data, pos, 34); // "
    if !r1.0 {
        let none: Option<String> = Option::None;
        let res = (none, r1.1);
        return res;
    }
    let mut i = r1.1;
    let mut str_bytes: Vec<u8> = Vec::new();
    let len = data.len();
    while i < len {
        let b = data.get(i).unwrap();
        let b_val = b as i64;
        if b_val == 34 { // "
            let s = String::from_utf8(str_bytes);
            let some: Option<String> = Option::Some(s);
            let res = (some, i + 1);
            return res;
        }
        str_bytes.push(b);
        i = i + 1;
    }
    let none: Option<String> = Option::None;
    let res = (none, i);
    return res;
}

fn read_number(data: Vec<u8>, pos: i64) -> (Option<i64>, i64) {
    let mut i = skip_whitespace(data, pos);
    let mut str_bytes: Vec<u8> = Vec::new();
    let len = data.len();
    while i < len {
        let b = data.get(i).unwrap();
        let b_val = b as i64;
        let mut is_num = false;
        if b_val >= 48 {
            if b_val <= 57 {
                is_num = true;
            }
        }
        if b_val == 45 {
            is_num = true;
        }
        if is_num {
            str_bytes.push(b);
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

fn read_i64_array(data: Vec<u8>, pos: i64) -> (Option<Vec<i64>>, i64) {
    let r1 = expect_char(data, pos, 91); // [
    if !r1.0 {
        let none: Option<Vec<i64>> = Option::None;
        let res = (none, r1.1);
        return res;
    }
    let mut i = r1.1;
    let mut arr: Vec<i64> = Vec::new();
    
    while i < data.len() {
        let p_check = skip_whitespace(data, i);
        if p_check < data.len() {
            if (data.get(p_check).unwrap() as i64) == 93 { // ]
                i = p_check;
                break;
            }
        }
        
        let r2 = read_number(data, i);
        match r2.0 {
            Option::Some(v) => { arr.push(v); }
            Option::None => {
                let none: Option<Vec<i64>> = Option::None;
                let res = (none, r2.1);
                return res;
            }
        }
        i = r2.1;
        let r3 = expect_char(data, i, 44); // ,
        if r3.0 {
            i = r3.1;
        } else {
            break;
        }
    }
    
    let r4 = expect_char(data, i, 93); // ]
    if !r4.0 {
        let none: Option<Vec<i64>> = Option::None;
        let res = (none, r4.1);
        return res;
    }
    
    let some: Option<Vec<i64>> = Option::Some(arr);
    let res = (some, r4.1);
    return res;
}

impl SafeTensorsFile {
    fn read_u64_le(data: Vec<u8>, offset: i64) -> i64 {
        let b0 = data.get(offset).unwrap() as i64;
        let b1 = data.get(offset + 1).unwrap() as i64;
        let b2 = data.get(offset + 2).unwrap() as i64;
        let b3 = data.get(offset + 3).unwrap() as i64;
        let b4 = data.get(offset + 4).unwrap() as i64;
        let b5 = data.get(offset + 5).unwrap() as i64;
        let b6 = data.get(offset + 6).unwrap() as i64;
        let b7 = data.get(offset + 7).unwrap() as i64;
        let lo = b0 + b1 * 256 + b2 * 65536 + b3 * 16777216;
        let hi = b4 + b5 * 256 + b6 * 65536 + b7 * 16777216;
        lo + hi * 4294967296
    }

    fn tensor_count(self) -> i64 {
        self.tensor_count
    }

    fn load(path: String) -> Result<SafeTensorsFile, String> {
        let res = File::read_binary(path);
        let data = match res {
            Result::Ok(v) => v,
            Result::Err(e) => {
                let err: Result<SafeTensorsFile, String> = Result::Err(e);
                return err;
            }
        };

        if data.len() < 8 {
            let err: Result<SafeTensorsFile, String> = Result::Err("File too short");
            return err;
        }

        let header_size = SafeTensorsFile::read_u64_le(data, 0);
        if data.len() < 8 + header_size {
            let err: Result<SafeTensorsFile, String> = Result::Err("File truncated");
            return err;
        }

        let mut json_bytes: Vec<u8> = Vec::new();
        let mut i = 0;
        while i < header_size {
            json_bytes.push(data.get(8 + i).unwrap());
            i = i + 1;
        }

        // ここで簡易JSONパースを行う
        let mut tensors: Vec<SafeTensorInfo> = Vec::new();
        
        let mut pos = 0;
        let r_obj_start = expect_char(json_bytes.clone(), pos, 123); // {
        if r_obj_start.0 {
            pos = r_obj_start.1;
            while pos < json_bytes.len() {
                let p_check = skip_whitespace(json_bytes.clone(), pos);
                if p_check < json_bytes.len() {
                    if (json_bytes.get(p_check).unwrap() as i64) == 125 { // }
                        pos = p_check;
                        break;
                    }
                }
                
                // Read tensor name (or __metadata__)
                let r_key = read_string(json_bytes.clone(), pos);
                pos = r_key.1;
                
                let r_colon = expect_char(json_bytes.clone(), pos, 58); // :
                pos = r_colon.1;
                
                let r_val_start = expect_char(json_bytes.clone(), pos, 123); // {
                pos = r_val_start.1;
                
                match r_key.0 {
                    Option::Some(key_str) => {
                        if key_str != "__metadata__" {
                            let mut dtype_str = String::from_utf8(Vec::new());
                            let mut shape_arr: Vec<i64> = Vec::new();
                            let mut offsets_arr: Vec<i64> = Vec::new();
                            
                            // parse tensor info object
                            while pos < json_bytes.len() {
                                let p_check2 = skip_whitespace(json_bytes.clone(), pos);
                                if p_check2 < json_bytes.len() {
                                    if (json_bytes.get(p_check2).unwrap() as i64) == 125 { // }
                                        pos = p_check2;
                                        break;
                                    }
                                }
                                
                                let r_inner_key = read_string(json_bytes.clone(), pos);
                                pos = r_inner_key.1;
                                
                                let r_inner_colon = expect_char(json_bytes.clone(), pos, 58); // :
                                pos = r_inner_colon.1;
                                
                                match r_inner_key.0 {
                                    Option::Some(inner_key) => {
                                        if inner_key == "dtype" {
                                            let r_dtype = read_string(json_bytes.clone(), pos);
                                            match r_dtype.0 {
                                                Option::Some(dt) => { dtype_str = dt; }
                                                Option::None => {}
                                            }
                                            pos = r_dtype.1;
                                        } else {
                                            if inner_key == "shape" {
                                                let r_shape = read_i64_array(json_bytes.clone(), pos);
                                                match r_shape.0 {
                                                    Option::Some(arr) => { shape_arr = arr; }
                                                    Option::None => {}
                                                }
                                                pos = r_shape.1;
                                            } else {
                                                if inner_key == "data_offsets" {
                                                    let r_offsets = read_i64_array(json_bytes.clone(), pos);
                                                    match r_offsets.0 {
                                                        Option::Some(arr) => { offsets_arr = arr; }
                                                        Option::None => {}
                                                    }
                                                    pos = r_offsets.1;
                                                } else {
                                                    // ignore other keys
                                                    let r_ignore = read_string(json_bytes.clone(), pos);
                                                    pos = r_ignore.1;
                                                }
                                            }
                                        }
                                    }
                                    Option::None => {}
                                }
                                
                                let r_inner_comma = expect_char(json_bytes.clone(), pos, 44); // ,
                                if r_inner_comma.0 {
                                    pos = r_inner_comma.1;
                                } else {
                                    break;
                                }
                            }
                            let r_val_end = expect_char(json_bytes.clone(), pos, 125); // }
                            pos = r_val_end.1;
                            
                            let begin = if offsets_arr.len() > 0 { offsets_arr.get(0).unwrap() } else { 0 };
                            let end = if offsets_arr.len() > 1 { offsets_arr.get(1).unwrap() } else { 0 };
                            
                            let info = SafeTensorInfo {
                                name: key_str,
                                dtype: dtype_str,
                                shape: shape_arr,
                                data_begin: begin,
                                data_end: end
                            };
                            tensors.push(info);
                            
                        } else {
                            // ignore __metadata__ object
                            while pos < json_bytes.len() {
                                let p_check2 = skip_whitespace(json_bytes.clone(), pos);
                                if p_check2 < json_bytes.len() {
                                    if (json_bytes.get(p_check2).unwrap() as i64) == 125 { // }
                                        pos = p_check2;
                                        break;
                                    }
                                }
                                
                                let r_inner_key = read_string(json_bytes.clone(), pos);
                                pos = r_inner_key.1;
                                
                                let r_inner_colon = expect_char(json_bytes.clone(), pos, 58); // :
                                pos = r_inner_colon.1;
                                
                                let r_inner_val = read_string(json_bytes.clone(), pos);
                                pos = r_inner_val.1;
                                
                                let r_inner_comma = expect_char(json_bytes.clone(), pos, 44); // ,
                                if r_inner_comma.0 {
                                    pos = r_inner_comma.1;
                                } else {
                                    break;
                                }
                            }
                            let r_val_end = expect_char(json_bytes.clone(), pos, 125); // }
                            pos = r_val_end.1;
                        }
                    }
                    Option::None => {}
                }
                
                let r_comma = expect_char(json_bytes.clone(), pos, 44); // ,
                if r_comma.0 {
                    pos = r_comma.1;
                } else {
                    break;
                }
            }
        }
        
        let stf = SafeTensorsFile {
            tensor_count: tensors.len(),
            tensors: tensors
        };
        let ok: Result<SafeTensorsFile, String> = Result::Ok(stf);
        ok
    }
    
}

fn get_safetensor(stf: SafeTensorsFile, name: String) -> Option<SafeTensorInfo> {
    let mut i = 0;
    let tensors = stf.tensors;
    while i < tensors.len() {
        let info = tensors.get(i).unwrap();
        if info.name == name {
            let some: Option<SafeTensorInfo> = Option::Some(info);
            return some;
        }
        i = i + 1;
    }
    let none: Option<SafeTensorInfo> = Option::None;
    return none;
}

fn write_u64_le(buf: Vec<u8>, val: i64) -> Vec<u8> {
    let mut res = buf;
    let mut v = val;
    let mut i = 0;
    while i < 8 {
        let b = v % 256;
        res.push(b as u8);
        v = v / 256;
        i = i + 1;
    }
    res
}

fn save_safetensors(stf: SafeTensorsFile, path: String) -> Result<bool, String> {
    let mut sb = StringBuilder::new();
    sb.append("{");
    sb.append("\"__metadata__\":{\"format\":\"pt\"}");
    
    let mut i = 0;
    let tensors = stf.tensors;
    while i < tensors.len() {
        sb.append(",");
        let info = tensors.get(i).unwrap();
        sb.append("\"");
        sb.append(info.name);
        sb.append("\":{\"dtype\":\"");
        sb.append(info.dtype);
        sb.append("\",\"shape\":[");
        let mut j = 0;
        let shape = info.shape;
        while j < shape.len() {
            if j > 0 {
                sb.append(",");
            }
            sb.append_i64(shape.get(j).unwrap());
            j = j + 1;
        }
        sb.append("],\"data_offsets\":[");
        sb.append_i64(info.data_begin);
        sb.append(",");
        sb.append_i64(info.data_end);
        sb.append("]}");
        i = i + 1;
    }
    sb.append("}");
    
    let mut json_str = sb.build();
    let mut json_bytes = json_str.to_bytes();
    let mut header_size = json_bytes.len();
    let rem = header_size % 8;
    if rem != 0 {
        let pad = 8 - rem;
        let mut p = 0;
        while p < pad {
            json_bytes.push(32 as u8); // space
            p = p + 1;
        }
        header_size = json_bytes.len();
    }
    
    let mut file_data: Vec<u8> = Vec::new();
    file_data = write_u64_le(file_data, header_size as i64);
    
    let mut k = 0;
    while k < json_bytes.len() {
        file_data.push(json_bytes.get(k).unwrap());
        k = k + 1;
    }
    
    let ok = File::write_binary(path, file_data);
    if ok {
        let res: Result<bool, String> = Result::Ok(true);
        return res;
    }
    let err: Result<bool, String> = Result::Err("Failed to write file");
    err
}
