//! stdlib スタブモジュール
//! 
//! 各種モジュールからの re-export でコンパイラの互換性を維持。

// string_ffi からの re-export
pub use crate::string_ffi::{
    tl_string_len,
    tl_string_concat,
    tl_string_contains,
    tl_string_from_int,
    tl_string_eq,
    tl_string_new,
    tl_string_to_i64,
    tl_string_char_at,
    tl_string_from_char,
};

// system からの re-export
pub use crate::system::{
    tl_system_sleep,
    tl_system_time,
    tl_read_line,
    tl_prompt,
};

// path 関連は file_io.rs から re-export
pub use crate::file_io::{
    tl_path_new,
    tl_path_free,
    tl_path_join,
    tl_path_to_string,
    tl_path_exists,
    tl_path_is_file,
    tl_path_is_dir,
};

/// Tensor stride を取得（スタブ）
pub fn get_strides(_shape: &[usize]) -> Vec<usize> {
    vec![]
}

// image 関連は tensor_ops_ext から re-export
pub use crate::tensor_ops_ext::{
    tl_image_load_grayscale,
    tl_image_width,
    tl_image_height,
};

// file_io 関連は file_io.rs から re-export
pub use crate::file_io::{
    tl_file_open,
    tl_file_close,
    tl_file_read_string,
    tl_file_write_string,
    tl_file_read_binary,
    tl_file_write_binary,
    tl_http_get,
    tl_http_download,
    tl_env_get,
    tl_env_set,
};

// string_ffi から hash_string を re-export
pub use crate::string_ffi::tl_hash_string;
