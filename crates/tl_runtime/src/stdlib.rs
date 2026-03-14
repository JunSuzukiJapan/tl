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
    tl_string_trim,
    tl_string_starts_with,
    tl_string_ends_with,
    tl_string_replace,
    tl_string_substring,
    tl_string_is_empty,
    tl_string_to_uppercase,
    tl_string_to_lowercase,
    tl_string_index_of,
    tl_string_split,
    tl_string_to_f64,
    tl_string_repeat,
    tl_string_chars,
    tl_string_from_f64,
    tl_string_from_bool,
    tl_assert,
    tl_random,
    tl_random_int,
    tl_min_i64,
    tl_max_i64,
    tl_min_f64,
    tl_max_f64,
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
    tl_path_parent,
    tl_path_file_name,
    tl_path_extension,
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
    tl_file_append,
    tl_file_delete,
    tl_file_create_dir,
    tl_http_get,
    tl_http_download,
    tl_env_get,
    tl_env_set,
    tl_system_exit,
};

// string_ffi から hash_string を re-export
pub use crate::string_ffi::tl_hash_string;
