//! メモリリークデバッグテスト
//! N-Queens パターン（softmax + sum + backward）を模倣し、
//! make/release カウンタの不均衡を検出する。

use tl_metal::{MetalTensor, DType};
use serial_test::serial;

/// RSS (Resident Set Size) を MB 単位で取得
fn get_rss_mb() -> f64 {
    use std::mem;
    #[repr(C)]
    struct MachTaskBasicInfo {
        virtual_size: u64,
        resident_size: u64,
        resident_size_max: u64,
        user_time: [i32; 2],
        system_time: [i32; 2],
        policy: i32,
        suspend_count: i32,
    }
    const MACH_TASK_BASIC_INFO: u32 = 20;
    unsafe extern "C" {
        fn mach_task_self() -> u32;
        fn task_info(
            target_task: u32,
            flavor: u32,
            task_info_out: *mut MachTaskBasicInfo,
            task_info_out_cnt: *mut u32,
        ) -> i32;
    }
    unsafe {
        let mut info: MachTaskBasicInfo = mem::zeroed();
        let mut count = (mem::size_of::<MachTaskBasicInfo>() / mem::size_of::<u32>()) as u32;
        let kr = task_info(
            mach_task_self(),
            MACH_TASK_BASIC_INFO,
            &mut info as *mut _,
            &mut count,
        );
        if kr == 0 {
            info.resident_size as f64 / 1024.0 / 1024.0
        } else {
            0.0
        }
    }
}

// ============================================================================
// テスト 1: テンソル make/release カウンタのバランス検証
// ============================================================================

#[test]
#[serial]
fn test_tensor_refcount_balance() {
    use tl_metal::ffi_ops;

    ffi_ops::tl_metal_debug_reset_counters();

    for _ in 0..10 {
        let mut board = MetalTensor::randn(&[8, 8], DType::F32);
        board.enable_grad();
        let board_ptr = ffi_ops::make_tensor(board);

        let probs_ptr = ffi_ops::tl_metal_softmax(board_ptr, 1);
        let col_sums_ptr = ffi_ops::tl_metal_sum_dim(probs_ptr, 0, false);
        let col_diff_ptr = ffi_ops::tl_metal_sub_scalar(col_sums_ptr, 1.0);
        let two = MetalTensor::from_slice(&[2.0f32], &[1], DType::F32);
        let two_ptr = ffi_ops::make_tensor(two);
        let col_pow_ptr = ffi_ops::tl_metal_pow(col_diff_ptr, two_ptr);
        let col_loss_ptr = ffi_ops::tl_metal_sum(col_pow_ptr);

        ffi_ops::tl_metal_backward(col_loss_ptr);

        let grad_ptr = ffi_ops::tl_metal_grad(board_ptr);
        let detached_ptr = ffi_ops::tl_metal_detach(board_ptr);

        // すべて解放
        ffi_ops::release_if_live(probs_ptr);
        ffi_ops::release_if_live(col_sums_ptr);
        ffi_ops::release_if_live(col_diff_ptr);
        ffi_ops::release_if_live(two_ptr);
        ffi_ops::release_if_live(col_pow_ptr);
        ffi_ops::release_if_live(col_loss_ptr);
        if !grad_ptr.is_null() {
            ffi_ops::release_if_live(grad_ptr);
        }
        ffi_ops::release_if_live(board_ptr);
        ffi_ops::release_if_live(detached_ptr);
    }

    let make = ffi_ops::MAKE_COUNT.load(std::sync::atomic::Ordering::SeqCst);
    let release = ffi_ops::RELEASE_COUNT.load(std::sync::atomic::Ordering::SeqCst);
    let acquire = ffi_ops::ACQUIRE_COUNT.load(std::sync::atomic::Ordering::SeqCst);
    let live = (make + acquire) as i64 - release as i64;

    eprintln!("[REFCOUNT] make={}, release={}, acquire={}, live={}", make, release, acquire, live);

    // live テンソルが 10 以内なら許容（完璧なら 0）
    assert!(live.abs() <= 10,
        "テンソル参照カウント不均衡: live={} (make={}, release={}, acquire={})",
        live, make, release, acquire);
}

// ============================================================================
// テスト 2: RSS 増加量の検証（autoreleasepool リグレッション検出）
//
// CommandStream から autoreleasepool が削除されると、
// Metal の ObjC オブジェクトが蓄積して RSS が大幅に増加する。
// 500 iteration のテンソル演算で RSS が 30MB 以上増加したらリグレッション。
// ============================================================================

#[test]
#[serial]
fn test_rss_stability_under_repeated_ops() {
    use tl_metal::ffi_ops;

    // ウォームアップ: 最初の数回で Metal の初期化メモリを安定させる
    for _ in 0..5 {
        let a = MetalTensor::randn(&[8, 8], DType::F32);
        let a_ptr = ffi_ops::make_tensor(a);
        let b_ptr = ffi_ops::tl_metal_softmax(a_ptr, 1);
        ffi_ops::release_if_live(b_ptr);
        ffi_ops::release_if_live(a_ptr);
    }
    tl_metal::command_stream::sync_stream();

    let rss_before = get_rss_mb();

    // 500 iteration のテンソル演算（N-Queens の forward pass を模倣）
    for _ in 0..500 {
        let board = MetalTensor::randn(&[8, 8], DType::F32);
        let board_ptr = ffi_ops::make_tensor(board);

        let probs_ptr = ffi_ops::tl_metal_softmax(board_ptr, 1);
        let col_sums_ptr = ffi_ops::tl_metal_sum_dim(probs_ptr, 0, false);
        let col_diff_ptr = ffi_ops::tl_metal_sub_scalar(col_sums_ptr, 1.0);
        let two = MetalTensor::from_slice(&[2.0f32], &[1], DType::F32);
        let two_ptr = ffi_ops::make_tensor(two);
        let col_pow_ptr = ffi_ops::tl_metal_pow(col_diff_ptr, two_ptr);
        let loss_ptr = ffi_ops::tl_metal_sum(col_pow_ptr);

        ffi_ops::release_if_live(probs_ptr);
        ffi_ops::release_if_live(col_sums_ptr);
        ffi_ops::release_if_live(col_diff_ptr);
        ffi_ops::release_if_live(two_ptr);
        ffi_ops::release_if_live(col_pow_ptr);
        ffi_ops::release_if_live(loss_ptr);
        ffi_ops::release_if_live(board_ptr);
    }
    tl_metal::command_stream::sync_stream();

    let rss_after = get_rss_mb();
    let rss_growth = rss_after - rss_before;

    eprintln!(
        "[RSS] before={:.1}MB, after={:.1}MB, growth={:.1}MB",
        rss_before, rss_after, rss_growth
    );

    // 30MB 以上の増加はリグレッション
    // (autoreleasepool が正常なら +3MB 程度、欠落すると +60MB 以上)
    assert!(rss_growth < 30.0,
        "RSS が {:.1}MB 増加 — autoreleasepool のリグレッションの可能性 \
         (before={:.1}MB, after={:.1}MB)",
        rss_growth, rss_before, rss_after);
}
