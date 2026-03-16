//! Tokenizer 関連の FFI 関数

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::sync::Arc;

/// Tokenizer ラッパー
pub struct OpaqueTokenizer {
    pub inner: Arc<tokenizers::Tokenizer>,
}

/// @ffi_sig (i8*) -> i64
/// 新しい Tokenizer を作成
/// codegen ABI: (path: *const c_char) -> i64 (handle)
#[unsafe(no_mangle)]
pub extern "C" fn tl_tokenizer_new(path: *const c_char) -> i64 {
    unsafe {
        if path.is_null() {
            return 0;
        }
        let path_str = CStr::from_ptr(path).to_string_lossy();
        let path_buf = crate::file_io::expand_path(&path_str);
        
        match tokenizers::Tokenizer::from_file(&path_buf) {
            Ok(tokenizer) => {
                println!("Loaded tokenizer from {:?}", path_buf);
                let tok = Box::new(OpaqueTokenizer {
                    inner: Arc::new(tokenizer),
                });
                Box::into_raw(tok) as i64
            }
            Err(e) => {
                eprintln!("Failed to load tokenizer: {}", e);
                0
            }
        }
    }
}

/// @ffi_sig (i64, i8*) -> Tensor*
/// テキストをエンコード
/// codegen ABI: (tok_handle: i64, text: *const c_char) -> *mut OpaqueTensor
#[unsafe(no_mangle)]
pub extern "C" fn tl_tokenizer_encode(
    tokenizer_handle: i64,
    text: *const c_char,
) -> *mut crate::OpaqueTensor {
    unsafe {
        if tokenizer_handle == 0 || text.is_null() {
            return std::ptr::null_mut();
        }
        let tokenizer = &*(tokenizer_handle as *const OpaqueTokenizer);
        let tok = &tokenizer.inner;
        let text_str = CStr::from_ptr(text).to_string_lossy();
        
        match tok.encode(text_str.as_ref(), false) {
            Ok(encoding) => {
                let mut ids: Vec<f32> = vec![1.0]; // Prepend BOS token (ID=1)
                ids.extend(encoding.get_ids().iter().map(|&id| id as f32));
                let shape = vec![ids.len()];
                crate::device_ffi::create_runtime_tensor_f32(&ids, &shape) as *mut crate::OpaqueTensor
            }
            Err(e) => {
                eprintln!("Tokenizer encode error: {}", e);
                std::ptr::null_mut()
            }
        }
    }
}

/// @ffi_sig (i64, Tensor*) -> i8*
/// トークン ID をデコード
/// codegen ABI: (tok_handle: i64, ids: *mut OpaqueTensor) -> *const c_char
#[unsafe(no_mangle)]
pub extern "C" fn tl_tokenizer_decode(
    tokenizer_handle: i64,
    ids: *mut crate::OpaqueTensor,
) -> *const c_char {
    unsafe {
        if tokenizer_handle == 0 || ids.is_null() {
            let empty = CString::new("").unwrap();
            return empty.into_raw();
        }
        let tokenizer = &*(tokenizer_handle as *const OpaqueTokenizer);
        let tok = &tokenizer.inner;
        let data: Vec<f32> = crate::device_ffi::read_runtime_tensor_to_f32_vec(ids as *mut std::ffi::c_void);
        let token_ids: Vec<u32> = data.iter().map(|&f| f as u32).collect();
        
        let text = if token_ids.len() == 1 {
            // 1 トークンの場合: id_to_token で直接変換
            // decode() は post-processing で先頭スペースを消すため
            match tok.id_to_token(token_ids[0]) {
                Some(token_str) => {
                    // <0xHH> バイトトークンを実際のバイト値に変換
                    if token_str.starts_with("<0x") && token_str.ends_with('>') && token_str.len() == 6 {
                        if let Ok(byte_val) = u8::from_str_radix(&token_str[3..5], 16) {
                            return CString::new(vec![byte_val])
                                .unwrap_or_else(|_| CString::new("").unwrap())
                                .into_raw();
                        }
                    }
                    // 特殊トークン (<|...|>) はスキップ
                    if token_str.starts_with("<|") && token_str.ends_with("|>") {
                        String::new()
                    } else {
                        // SentencePiece: ▁ (U+2581) → スペース
                        // BPE (GPT-2): Ġ (U+0120) → スペース, Ċ (U+010A) → 改行
                        token_str
                            .replace('\u{2581}', " ")
                            .replace('\u{0120}', " ")
                            .replace('\u{010A}', "\n")
                    }
                },
                None => String::new(),
            }
        } else {
            // 複数トークンの場合: 通常の decode
            match tok.decode(&token_ids, true) {
                Ok(text) => text,
                Err(e) => {
                    eprintln!("Tokenizer decode error: {}", e);
                    String::new()
                }
            }
        };
        
        let c_str = CString::new(text).unwrap_or_else(|_| CString::new("").unwrap());
        c_str.into_raw()
    }
}

/// @ffi_sig (i64, i8*) -> Tensor*
/// Llama 3 チャットテンプレートに準拠したトークン列を生成
///
/// codegen ABI: (tok_handle: i64, user_text: *const c_char) -> *mut OpaqueTensor
#[unsafe(no_mangle)]
pub extern "C" fn tl_tokenizer_encode_chat(
    tokenizer_handle: i64,
    user_text: *const c_char,
) -> *mut crate::OpaqueTensor {
    unsafe {
        if tokenizer_handle == 0 || user_text.is_null() {
            return std::ptr::null_mut();
        }
        let tokenizer = &*(tokenizer_handle as *const OpaqueTokenizer);
        let tok = &tokenizer.inner;
        let user_str = CStr::from_ptr(user_text).to_string_lossy();

        // Llama 3 特殊トークンID
        const BOS: u32        = 128000; // <|begin_of_text|>
        const START_HDR: u32  = 128006; // <|start_header_id|>
        const END_HDR: u32    = 128007; // <|end_header_id|>
        const EOT: u32        = 128009; // <|eot_id|>

        let mut ids: Vec<u32> = Vec::new();

        // <|begin_of_text|>
        ids.push(BOS);

        // System message
        let system_msg = "You are a helpful assistant.";
        ids.push(START_HDR);
        if let Ok(enc) = tok.encode("system", false) {
            ids.extend(enc.get_ids());
        }
        ids.push(END_HDR);
        if let Ok(enc) = tok.encode("\n\n", false) {
            ids.extend(enc.get_ids());
        }
        if let Ok(enc) = tok.encode(system_msg, false) {
            ids.extend(enc.get_ids());
        }
        ids.push(EOT);

        // User message
        ids.push(START_HDR);
        if let Ok(enc) = tok.encode("user", false) {
            ids.extend(enc.get_ids());
        }
        ids.push(END_HDR);
        if let Ok(enc) = tok.encode("\n\n", false) {
            ids.extend(enc.get_ids());
        }
        if let Ok(enc) = tok.encode(user_str.as_ref(), false) {
            ids.extend(enc.get_ids());
        }
        ids.push(EOT);

        // Assistant generation prompt
        ids.push(START_HDR);
        if let Ok(enc) = tok.encode("assistant", false) {
            ids.extend(enc.get_ids());
        }
        ids.push(END_HDR);
        if let Ok(enc) = tok.encode("\n\n", false) {
            ids.extend(enc.get_ids());
        }

        // f32 テンソルとして返す
        let f32_ids: Vec<f32> = ids.iter().map(|&id| id as f32).collect();
        let shape = vec![f32_ids.len()];
        crate::device_ffi::create_runtime_tensor_f32(&f32_ids, &shape) as *mut crate::OpaqueTensor
    }
}

/// @ffi_sig (i64, i8*) -> Tensor*
/// TinyLlama Zephyr chat template encode
/// Full template is encoded as a single string so SentencePiece handles it correctly.
/// The tokenizer recognizes </s> as special token (ID=2) from added_tokens.
#[unsafe(no_mangle)]
pub extern "C" fn tl_tokenizer_encode_chat_zephyr(
    tokenizer_handle: i64,
    user_text: *const c_char,
) -> *mut crate::OpaqueTensor {
    unsafe {
        if tokenizer_handle == 0 || user_text.is_null() {
            return std::ptr::null_mut();
        }
        let tokenizer = &*(tokenizer_handle as *const OpaqueTokenizer);
        let tok = &tokenizer.inner;
        let user_str = CStr::from_ptr(user_text).to_string_lossy();

        // Build the complete template as a single string
        // Format: <|system|>\nYou are a helpful assistant.</s>\n<|user|>\n{user_input}</s>\n<|assistant|>\n
        let full_text = format!(
            "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n{}</s>\n<|assistant|>\n",
            user_str
        );

        match tok.encode(full_text.as_str(), false) {
            Ok(encoding) => {
                // Prepend BOS token (ID=1) manually
                // LLaMA architectures typically require BOS token as an Attention Sink
                // tokenizers doesn't always automatically inject BOS, so we add it here
                let mut ids: Vec<f32> = vec![1.0];
                ids.extend(encoding.get_ids().iter().map(|&id| id as f32));
                let shape = vec![ids.len()];
                crate::device_ffi::create_runtime_tensor_f32(&ids, &shape) as *mut crate::OpaqueTensor
            }
            Err(e) => {
                eprintln!("Tokenizer encode_chat_zephyr error: {}", e);
                std::ptr::null_mut()
            }
        }
    }
}
