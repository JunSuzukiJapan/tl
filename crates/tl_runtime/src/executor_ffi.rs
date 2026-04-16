use std::collections::HashMap;
use std::os::raw::c_void;
use std::sync::Mutex;
use once_cell::sync::Lazy;

/// タスクの実行状態
pub struct TaskState {
    pub handle: std::thread::JoinHandle<u64>,
}

/// グローバルタスクレジストリ
static TASK_REGISTRY: Lazy<Mutex<HashMap<i64, TaskState>>> = Lazy::new(|| {
    Mutex::new(HashMap::new())
});
static NEXT_TASK_ID: Lazy<Mutex<i64>> = Lazy::new(|| Mutex::new(1));

/// Future のポーリング関数シグネチャ。
/// 引数: state_ptr (ステートマシンへのポインタ)
/// 戻り値: u64::MAX = Poll::Pending, それ以外 = Poll::Ready(value as u64)
type PollFn = extern "C" fn(*mut c_void) -> u64;

/// poll_fn と state_ptr を受け取り、Future が完了するまでスピンループで実行する。
/// 戻り値: Ready 時の値を u64 にキャストしたもの。
///
/// @ffi_sig (*mut c_void, *mut c_void) -> u64
#[unsafe(no_mangle)]
pub extern "C" fn tl_executor_block_on(poll_fn: *mut c_void, state_ptr: *mut c_void) -> u64 {
    let f: PollFn = unsafe { std::mem::transmute(poll_fn) };
    loop {
        let result = f(state_ptr);
        if result != u64::MAX {
            // Poll::Ready
            return result;
        }
        // Poll::Pending: 次のポーリングまで CPU を譲る
        std::hint::spin_loop();
    }
}

/// バックグラウンドスレッドで Future を実行するタスクをスポーンする。
/// 戻り値: タスクID (tl_task_join で待機するために使用)
///
/// @ffi_sig (*mut c_void, *mut c_void) -> i64
#[unsafe(no_mangle)]
pub extern "C" fn tl_task_spawn(poll_fn: *mut c_void, state_ptr: *mut c_void) -> i64 {
    let id = {
        let mut id_lock = NEXT_TASK_ID.lock().unwrap();
        let current = *id_lock;
        *id_lock += 1;
        current
    };

    let poll_fn_addr = poll_fn as usize;
    let state_ptr_addr = state_ptr as usize;

    let handle = std::thread::spawn(move || {
        let f: PollFn = unsafe { std::mem::transmute(poll_fn_addr) };
        let state = state_ptr_addr as *mut c_void;
        loop {
            let result = f(state);
            if result != u64::MAX {
                return result;
            }
            std::hint::spin_loop();
        }
    });

    TASK_REGISTRY.lock().unwrap().insert(id, TaskState { handle });
    id
}

/// タスク ID で指定したタスクの完了を待ち、戻り値を返す。
/// タスクが見つからない場合は 0 を返す。
///
/// @ffi_sig (i64) -> u64
#[unsafe(no_mangle)]
pub extern "C" fn tl_task_join(task_id: i64) -> u64 {
    let handle_opt = TASK_REGISTRY.lock().unwrap().remove(&task_id);
    if let Some(task) = handle_opt {
        match task.handle.join() {
            Ok(val) => val,
            Err(e) => {
                eprintln!("[Async Error] Task {} panicked: {:?}", task_id, e);
                0
            }
        }
    } else {
        eprintln!("[Async Warning] Task {} not found", task_id);
        0
    }
}
