//! checkpoint スタブモジュール
//!
//! モデルチェックポイント関連のスタブ

/// チェックポイント保存（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_checkpoint_save(_path: *mut crate::string_ffi::StringStruct) {
    eprintln!("Warning: Checkpoint save not yet supported in Metal backend");
}

/// チェックポイント読み込み（スタブ）
#[unsafe(no_mangle)]
pub extern "C" fn tl_checkpoint_load(_path: *mut crate::string_ffi::StringStruct) {
    eprintln!("Warning: Checkpoint load not yet supported in Metal backend");
}

/// チェックポイント（save のエイリアス）
#[unsafe(no_mangle)]
pub extern "C" fn tl_checkpoint(path: *mut crate::string_ffi::StringStruct) {
    tl_checkpoint_save(path);
}
