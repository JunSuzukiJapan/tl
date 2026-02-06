//! registry スタブモジュール
//!
//! テンソル/パラメータ登録関連

use std::cell::RefCell;
use crate::context::TensorContext;

// memory_ffi から register_tensor を re-export
pub use crate::memory_ffi::tl_register_tensor;

// system から register_parameter を re-export  
pub use crate::system::tl_register_parameter;

// スレッドローカルなテンソルコンテキスト
thread_local! {
    static GLOBAL_CONTEXT: RefCell<TensorContext> = RefCell::new(TensorContext::new());
}

/// グローバルテンソルコンテキストを取得
pub fn get_global_context() -> TensorContext {
    GLOBAL_CONTEXT.with(|ctx| ctx.borrow().clone())
}

/// グローバルテンソルコンテキストを設定
pub fn set_global_context(ctx: TensorContext) {
    GLOBAL_CONTEXT.with(|global| *global.borrow_mut() = ctx);
}

/// グローバルテンソルコンテキストをリセット
pub fn reset_global_context() {
    GLOBAL_CONTEXT.with(|global| *global.borrow_mut() = TensorContext::new());
}

