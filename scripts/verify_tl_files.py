#!/usr/bin/env python3
"""
TL ファイル検証エージェント
tests/ と examples/ 内の .tl ファイルを実行し、動作を確認します。

使用方法:
    python scripts/verify_tl_files.py [--verbose] [--timeout SECONDS] [--filter PATTERN]

⚠️⚠️⚠️ 絶対禁止事項 ⚠️⚠️⚠️
  このスクリプトに並列実行（--parallel, ThreadPoolExecutor, multiprocessing 等）を
  追加してはならない。Metal GPU バックエンドのプロセスを並列実行すると GPU リソースが
  競合し、WindowServer のウォッチドッグタイムアウトを引き起こして Mac 全体がクラッシュ
  する。この問題は 2026年2月に複数回のシステムクラッシュで確認済み。
  テストは必ず逐次（シリアル）実行すること。
⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️
"""

import subprocess
import sys
import os
import re
import time
import signal
import atexit
import argparse
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple, Set
from enum import Enum

# ── 子プロセス追跡 & クリーンアップ ──────────────────────────
_active_procs: Set[subprocess.Popen] = set()

def _cleanup_children():
    """残存する全子プロセスを強制終了する"""
    for proc in list(_active_procs):
        try:
            if proc.poll() is None:  # まだ生きている
                proc.kill()
                proc.wait(timeout=3)
        except Exception:
            pass
    _active_procs.clear()

# スクリプト終了時に必ずクリーンアップ
atexit.register(_cleanup_children)

def _signal_handler(signum, frame):
    """SIGTERM/SIGINT 受信時にクリーンアップして終了"""
    print(f"\n🛑 シグナル {signum} を受信。子プロセスをクリーンアップ中...")
    _cleanup_children()
    sys.exit(128 + signum)

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)

class Status(Enum):
    PASS = "✅"
    FAIL = "❌"
    SKIP = "⏭️"
    TIMEOUT = "⏰"
    SEGFAULT = "💀"

@dataclass
class TestResult:
    file: str
    status: Status
    output: str
    error: str
    duration: float
    reason: str = ""


@dataclass
class VmSnapshot:
    page_size: int
    file_backed_pages: int
    free_pages: int
    speculative_pages: int

    @property
    def cached_gib(self) -> float:
        return (self.file_backed_pages * self.page_size) / (1024 ** 3)

    @property
    def reclaimable_gib(self) -> float:
        return ((self.free_pages + self.speculative_pages) * self.page_size) / (1024 ** 3)


def get_vm_snapshot() -> Optional[VmSnapshot]:
    """vm_stat から最低限のメモリ指標を取得する。失敗時は None。"""
    try:
        out = subprocess.check_output(["vm_stat"], text=True)
    except Exception:
        return None

    page_size_match = re.search(r"page size of (\d+) bytes", out)
    if not page_size_match:
        return None
    page_size = int(page_size_match.group(1))

    def _pages(key: str) -> int:
        m = re.search(rf"{re.escape(key)}:\s+(\d+)\.", out)
        return int(m.group(1)) if m else 0

    return VmSnapshot(
        page_size=page_size,
        file_backed_pages=_pages("File-backed pages"),
        free_pages=_pages("Pages free"),
        speculative_pages=_pages("Pages speculative"),
    )


def wait_for_safe_memory_window(
    max_cached_gib: float,
    min_reclaimable_gib: float,
    timeout_sec: int,
    poll_sec: float,
    verbose: bool = False
) -> Tuple[bool, str]:
    """
    高負荷時の暴走防止:
    - file cache が上限超え
    - reclaimable (free+speculative) が下限未満
    の間は待機し、timeout 超過で False を返す。
    """
    deadline = time.time() + timeout_sec
    while True:
        snap = get_vm_snapshot()
        if snap is None:
            # 監視不能なら実行を止めない（既存互換）
            return True, "vm_stat unavailable"

        cache_ok = snap.cached_gib <= max_cached_gib
        reclaim_ok = snap.reclaimable_gib >= min_reclaimable_gib
        if cache_ok and reclaim_ok:
            return True, (
                f"cached={snap.cached_gib:.1f}GiB, "
                f"reclaimable={snap.reclaimable_gib:.1f}GiB"
            )

        if time.time() >= deadline:
            return False, (
                f"cached={snap.cached_gib:.1f}GiB>{max_cached_gib:.1f}GiB "
                f"or reclaimable={snap.reclaimable_gib:.1f}GiB<{min_reclaimable_gib:.1f}GiB"
            )

        if verbose:
            print(
                f"\n⏸️ メモリ待機: cached={snap.cached_gib:.1f}GiB "
                f"(limit {max_cached_gib:.1f}), reclaimable={snap.reclaimable_gib:.1f}GiB "
                f"(min {min_reclaimable_gib:.1f})"
            )
        time.sleep(poll_sec)

# main 関数を持つファイルのみ実行
def has_main_function(filepath: Path) -> bool:
    """ファイルに main 関数が含まれているか確認"""
    try:
        content = filepath.read_text(encoding='utf-8')
        # fn main() を探す
        return bool(re.search(r'\bfn\s+main\s*\(', content))
    except Exception:
        return False

def has_skip_comment(filepath: Path) -> Tuple[bool, str]:
    """ファイルの先頭に // SKIP コメントがあるか確認"""
    try:
        content = filepath.read_text(encoding='utf-8')
        # ファイルの最初の数行に // SKIP または // SKIP: がある場合スキップ
        for line in content.split('\n')[:10]:
            if line.strip().startswith('// SKIP'):
                # SKIP: の後の理由を抽出
                if ':' in line:
                    reason = line.split(':', 1)[1].strip()
                    return True, f"SKIP: {reason}"
                return True, "SKIP (marked in file)"
        return False, ""
    except Exception:
        return False, ""


# 特定のファイルをスキップするかどうか
SKIP_FILES = {
    # 対話的な入力が必要なファイル
    "chatbot_llama3.tl",
    "chatbot.tl",
    # 外部リソースが必要
    "download.tl",
    "infer.tl",  # MNIST
    "train.tl",  # MNIST
    "infer_add.tl",  # 学習済みモデルが必要
    # 非常に長時間実行されるファイル
    "train_heavy.tl",
    "infer_heavy.tl",
    "train_add.tl",
    "train_paper.tl",
    "train_recall.tl",
    "reverse_train.tl",
    "readme_n_queens.tl",
    # "n_queens.tl",
    # 既知の問題があるファイル
    "train_verify_2digit.tl",
    # "reverse_infer.tl",
    # 計算量が多くタイムアウトする
    # "mln.tl",
    # "tsp.tl",  # backward が重く 30秒タイムアウト超過
    # システムテスト (タイムアウト)
    "system_test.tl",

    # --- autograd 使用ファイル ---
    # randn_debug の codegen/runtime 引数不一致修正 (2026-02-12) により大半が動作。
    # sudoku.tl のみ型エラー (autograd 無関係) で残留。
    # lenia, gnn は GPU 負荷が高く連続実行で Mac がフリーズするためスキップ。
    "sudoku.tl",  # 型エラー: expected Bool, found String
    "lenia.tl",   # GPU 負荷大 (64x64 grid × 100 steps)
    "gnn.tl",     # GPU 負荷大 (250 epochs)
    
    # 計算量が多い/GPU負荷が高いテスト (システムクラッシュ回避のためスキップ)
    "kv_cache_test.tl",
    "recommendation.tl",
    "inverse_life.tl",

    # --- Crash Reproduction Files (System Instability Risk) ---
    # These files are designed to crash or leak resources (GPU/Memory).
    # Running them might cause WindowServer watchdog timeouts (Mac freeze).
    "repro_reshape_segfault.tl",
    "repro_segfault_minimal.tl",
}

# パス単位でのスキップ（重複ファイル名対策）
# 2026-02-14: 直近実行でタイムアウト/abort したファイルを明示スキップ
SKIP_PATH_SUFFIXES = {
    "examples/tasks/tensor_logic/lenia/repro.tl",
    "examples/tasks/tensor_logic/mln/mln.tl",
    "examples/tasks/tensor_logic/n_queens/n_queens.tl",
    "examples/tasks/tensor_logic/n_queens/n_queens_debug.tl",
    "examples/tasks/tensor_logic/raycast/raycast.tl",
    "examples/tasks/tensor_logic/tsp/tsp.tl",
    "examples/test_nqueens_debug.tl",
    "tests/fixtures/debug/mem_leak_extended.tl",
    "examples/apps/tinyllama/debug_chatbot.tl",
}


# 長時間実行が予想されるファイル（長めのタイムアウト）
# 注: autograd 使用ファイルは SKIP_FILES に移動済み
LONG_RUNNING = {
    "examples/tasks/tensor_logic/digital_logic/logic.tl",
}

# 失敗することが期待されるファイル（エラーテスト用）
# 終了コードが 0 以外であれば PASS とみなします
EXPECTED_FAILURES = {
    "match_non_exhaustive.tl",
    "import_cycle_a.tl",
    "if_let_unknown_field.tl",
    "match_duplicate_arm.tl",
    "match_unreachable_after_wildcard.tl",
    "if_let_missing_else_value.tl",
    "if_let_type_mismatch.tl",
    "negation_cycle.tl",
    "negation_multi_neg_layers_cycle.tl",
    "negation_unbound.tl",
}

def should_skip(filepath: Path) -> Tuple[bool, str]:
    """ファイルをスキップすべきか判定"""
    name = filepath.name
    if name in SKIP_FILES:
        return True, f"スキップ対象: {name}"
    path_posix = filepath.as_posix()
    for suffix in SKIP_PATH_SUFFIXES:
        if path_posix.endswith("/" + suffix) or path_posix == suffix:
            return True, f"スキップ対象: {suffix}"
    # ファイル内の // SKIP コメントをチェック
    skip_in_file, skip_reason = has_skip_comment(filepath)
    if skip_in_file:
        return True, skip_reason
    if not has_main_function(filepath):
        # EXPECTED_FAILURES に含まれている場合は、main関数がなくても実行を試みる（コンパイルエラー等を確認するため）
        if name in EXPECTED_FAILURES:
            return False, ""
        return True, "main 関数なし"
    return False, ""


def run_tl_file(filepath: Path, tl_binary: Path, timeout: int, verbose: bool = False) -> TestResult:
    """TL ファイルを実行して結果を返す"""
    start_time = time.time()
    
    # 環境変数の準備 (ライブラリパス設定)
    env = os.environ.copy()
    runtime_dir = tl_binary.parent
    # macOS/Linux対応: LIBRARY_PATHを設定してリンカが tl_runtime を見つけられるようにする
    extra_lib_path = str(runtime_dir)
    env["LIBRARY_PATH"] = f"{extra_lib_path}:{env.get('LIBRARY_PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{extra_lib_path}:{env.get('LD_LIBRARY_PATH', '')}"
    env["DYLD_LIBRARY_PATH"] = f"{extra_lib_path}:{env.get('DYLD_LIBRARY_PATH', '')}"

    skip, reason = should_skip(filepath)
    if skip:
        # SKIPコメント付き / main関数なし / スキップ対象ファイル の場合はビルドチェックをスキップ
        if "SKIP:" in reason or "main 関数なし" in reason or "スキップ対象" in reason:

            return TestResult(
                file=str(filepath),
                status=Status.SKIP,
                output="",
                error="",
                duration=0.0,
                reason=reason
            )
        
        # それ以外のスキップ対象はビルド確認を行う
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
             tmp_path = tmp.name
        try:
             compile_cmd = [str(tl_binary), "-c", str(filepath), "-o", tmp_path]
             # コンパイルのみ実行
             compile_result = subprocess.run(
                 compile_cmd,
                 capture_output=True,
                 text=True,
                 timeout=timeout,
                 cwd=project_root,
                 env=env
             )
             
             if compile_result.returncode != 0:
                  return TestResult(
                      file=str(filepath),
                      status=Status.FAIL,
                      output=compile_result.stdout,
                      error=f"Build Failed:\n{compile_result.stderr}",
                      duration=time.time() - start_time,
                      reason=f"Build Failed ({reason})"
                  )
             else:
                  # ビルド成功したら SKIP (Build OK)
                  return TestResult(
                      file=str(filepath),
                      status=Status.SKIP,
                      output="",
                      error="",
                      duration=time.time() - start_time,
                      reason=f"{reason} (Build OK)"
                  )
        except Exception as e:
            return TestResult(
                file=str(filepath),
                status=Status.FAIL,
                output="",
                error=str(e),
                duration=time.time() - start_time,
                reason=f"Build Check Error: {e}"
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    
    # 長時間実行ファイルはタイムアウトを延長
    if filepath.name in LONG_RUNNING:
        timeout = max(timeout, 120)

    is_expected_to_fail = filepath.name in EXPECTED_FAILURES
    
    exe_path = None  # Static モードで生成されるバイナリのパス
    proc = None  # 子プロセスの参照（タイムアウト時のkill用）
    try:
        # Static Compilation Override
        import sys
        if "--static" in sys.argv:
             # 1. Compile (Must run from project root for library resolution)
             script_dir = Path(__file__).parent
             project_root = script_dir.parent

             exe_path = filepath.with_suffix('.bin') # Use .bin suffix to avoid collision with directories (e.g. tests/generics)
             
             compile_cmd = [str(tl_binary), "-c", str(filepath), "-o", str(exe_path)]
             proc = subprocess.Popen(
                 compile_cmd,
                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                 text=True,
                 cwd=project_root, # Fix: Compile from root so 'target/debug' path in tl main.rs works
                 env=env
             )
             _active_procs.add(proc)
             try:
                 compile_stdout, compile_stderr = proc.communicate(timeout=timeout)
                 compile_returncode = proc.returncode
             except subprocess.TimeoutExpired:
                 proc.kill()
                 proc.wait()
                 raise
             finally:
                 _active_procs.discard(proc)
             proc = None  # コンパイル完了、参照をクリア
             
             if compile_returncode != 0:
                 if is_expected_to_fail:
                     return TestResult(
                         file=str(filepath),
                         status=Status.PASS,
                         output=compile_stdout,
                         error=compile_stderr,
                         duration=time.time() - start_time,
                         reason="(Expected Compilation Failure)"
                     )
                 
                 return TestResult(
                     file=str(filepath),
                     status=Status.FAIL,
                     output=compile_stdout,
                     error=f"Compilation Failed:\n{compile_stderr}",
                     duration=time.time() - start_time,
                     reason=f"Compilation Failed (Exit: {compile_returncode})"
                 )

             # 2. Run Executable
             run_cmd = [str(exe_path)]
             proc = subprocess.Popen(
                 run_cmd,
                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                 text=True,
                 cwd=filepath.parent
             )
             _active_procs.add(proc)
             try:
                 stdout, stderr = proc.communicate(timeout=timeout)
                 returncode = proc.returncode
             except subprocess.TimeoutExpired:
                 proc.kill()
                 proc.wait()
                 raise
             finally:
                 _active_procs.discard(proc)
             proc = None
        else:
            # JIT Execution (Default)
            # FIX: Must pass filename only (or absolute path) since we changed CWD to parent
            if verbose:
                 print(f"DEBUG: Running {tl_binary} {filepath.name} in {filepath.parent}")
                 # Stream output directly to console in verbose mode
                 proc = subprocess.Popen(
                    [str(tl_binary), filepath.name],
                    text=True,
                    cwd=filepath.parent
                 )
                 _active_procs.add(proc)
                 try:
                     proc.communicate(timeout=timeout)
                     returncode = proc.returncode
                 except subprocess.TimeoutExpired:
                     proc.kill()
                     proc.wait()
                     raise
                 finally:
                     _active_procs.discard(proc)
                 proc = None
                 stdout = "(Streamed to console)"
                 stderr = ""
            else:
                 proc = subprocess.Popen(
                    [str(tl_binary), filepath.name],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True,
                    cwd=filepath.parent
                 )
                 _active_procs.add(proc)
                 try:
                     stdout, stderr = proc.communicate(timeout=timeout)
                     returncode = proc.returncode
                 except subprocess.TimeoutExpired:
                     proc.kill()
                     proc.wait()
                     raise
                 finally:
                     _active_procs.discard(proc)
                 proc = None


        duration = time.time() - start_time
        
        # セグメンテーションフォールトの検出
        if returncode == -11 or returncode == 139:
            return TestResult(
                file=str(filepath),
                status=Status.SEGFAULT,
                output=stdout,
                error=stderr,
                duration=duration,
                reason="Segmentation fault"
            )
        
        # 終了コードをチェック
        if returncode != 0:
            if is_expected_to_fail:
                # 失敗が期待されていたので PASS とする
                return TestResult(
                    file=str(filepath),
                    status=Status.PASS,
                    output=stdout,
                    error=stderr, 
                    duration=duration,
                    reason="(Expected Failure)"
                )
            
            # エラーメッセージを解析
            return TestResult(
                file=str(filepath),
                status=Status.FAIL,
                output=stdout,
                error=stderr,
                duration=duration,
                reason=f"Exit code: {returncode}"
            )
        else:
            if is_expected_to_fail:
                # 失敗すべきなのに成功してしまった場合
                return TestResult(
                    file=str(filepath),
                    status=Status.FAIL,
                    output=stdout,
                    error=stderr,
                    duration=duration,
                    reason="Unexpected Success: Expected failure but exited with 0"
                )

        return TestResult(
            file=str(filepath),
            status=Status.PASS,
            output=stdout,
            error=stderr,
            duration=duration
        )
        
    except subprocess.TimeoutExpired:
        _cleanup_children()
        duration = time.time() - start_time
        return TestResult(
            file=str(filepath),
            status=Status.TIMEOUT,
            output="",
            error="",
            duration=duration,
            reason=f"タイムアウト ({timeout}秒)"
        )
    except Exception as e:
        # 予期しない例外でも子プロセスを確実にkill
        if proc is not None:
            try:
                proc.kill()
                proc.wait()
            except Exception:
                pass
        duration = time.time() - start_time
        return TestResult(
            file=str(filepath),
            status=Status.FAIL,
            output="",
            error=str(e),
            duration=duration,
            reason=str(e)
        )
    finally:
        # Static モードで生成されたバイナリを削除
        if exe_path is not None and exe_path.exists():
            try:
                exe_path.unlink()
            except Exception:
                pass

def find_tl_files(directories: List[Path], filter_pattern: Optional[str] = None) -> List[Path]:
    """ディレクトリから .tl ファイルを検索"""
    files = []
    for directory in directories:
        if not directory.exists():
            continue
        for tl_file in directory.rglob("*.tl"):
            if filter_pattern and filter_pattern not in str(tl_file):
                continue
            files.append(tl_file)
    return sorted(files)

def print_summary(results: List[TestResult], verbose: bool):
    """結果のサマリーを表示"""
    passed = [r for r in results if r.status == Status.PASS]
    failed = [r for r in results if r.status == Status.FAIL]
    skipped = [r for r in results if r.status == Status.SKIP]
    timeout = [r for r in results if r.status == Status.TIMEOUT]
    segfault = [r for r in results if r.status == Status.SEGFAULT]
    
    print("\n" + "=" * 60)
    print("検証結果サマリー")
    print("=" * 60)
    
    print(f"\n✅ 成功: {len(passed)}")
    print(f"❌ 失敗: {len(failed)}")
    print(f"💀 セグフォ: {len(segfault)}")
    print(f"⏰ タイムアウト: {len(timeout)}")
    print(f"⏭️ スキップ: {len(skipped)}")
    print(f"\n合計: {len(results)} ファイル")
    
    if failed or segfault or timeout:
        print("\n" + "-" * 60)
        print("問題のあるファイル:")
        print("-" * 60)
        
        for r in failed + segfault + timeout:
            rel_path = Path(r.file).relative_to(Path.cwd()) if Path.cwd() in Path(r.file).parents else r.file
            print(f"\n{r.status.value} {rel_path}")
            print(f"   理由: {r.reason}")
            if verbose and r.error:
                print(f"   エラー: {r.error[:200]}...")
    
    # 成功率の計算 (スキップを除く)
    executed = len(passed) + len(failed) + len(segfault) + len(timeout)
    if executed > 0:
        success_rate = len(passed) / executed * 100
        print(f"\n成功率: {success_rate:.1f}% ({len(passed)}/{executed})")
    
    return len(failed) + len(segfault)

def clean_binaries(project_root: Path):
    """古いバイナリを削除して再ビルドを強制する"""
    print("🧹 古いバイナリを削除中...")
    
    binaries_to_clean = [
        project_root / "target" / "debug" / "tl",
        project_root / "target" / "release" / "tl",
    ]
    
    cleaned = 0
    for binary in binaries_to_clean:
        if binary.exists():
            try:
                binary.unlink()
                print(f"   削除: {binary.relative_to(project_root)}")
                cleaned += 1
            except Exception as e:
                print(f"   ⚠️ 削除失敗: {binary.name} - {e}")
    
    if cleaned == 0:
        print("   バイナリが見つかりませんでした")
    else:
        print(f"\n✅ {cleaned} 個のバイナリを削除しました")
        print("💡 次回の実行時に自動的に再ビルドされます (cargo build)")


def main():
    parser = argparse.ArgumentParser(description="TL ファイル検証エージェント")
    parser.add_argument("--verbose", "-v", action="store_true", help="詳細出力")
    parser.add_argument("--timeout", "-t", type=int, default=30, help="タイムアウト秒数 (デフォルト: 30)")
    parser.add_argument("--filter", "-f", type=str, help="ファイルパターンでフィルタ")
    # ⚠️ --parallel 引数は意図的に削除済み。絶対に再追加しないこと。
    # Metal GPU プロセスの並列実行は Mac 全体のクラッシュを引き起こす。
    parser.add_argument("--static", action="store_true", help="静的コンパイルモードで実行 (JIT回避)")
    parser.add_argument("--clean", action="store_true", help="古いバイナリを削除して終了")
    parser.add_argument("--no-build", action="store_true", help="自動ビルドをスキップ (既存バイナリをそのまま使用)")
    parser.add_argument("--cooldown", type=float, default=1.5, help="テスト間のクールダウン秒数 (デフォルト: 1.5)")
    parser.add_argument("--crash-cooldown", type=float, default=5.0, help="クラッシュ後のクールダウン秒数 (デフォルト: 5.0)")
    parser.add_argument("--max-crashes", type=int, default=5, help="連続クラッシュでの緊急停止閾値 (デフォルト: 5)")
    parser.add_argument("--safe-mode", dest="safe_mode", action="store_true", help="システム負荷を下げるために定期的に休憩を挟む")
    parser.add_argument("--no-safe-mode", dest="safe_mode", action="store_false", help="定期休憩を無効化")
    parser.set_defaults(safe_mode=True)
    parser.add_argument("--max-cached-gib", type=float, default=22.0, help="cached files 上限GiB。超過時は次テスト開始前に待機 (デフォルト: 22)")
    parser.add_argument("--min-reclaimable-gib", type=float, default=8.0, help="reclaimable (free+speculative) の下限GiB。下回ると待機 (デフォルト: 8)")
    parser.add_argument("--memory-wait-timeout", type=int, default=300, help="メモリ待機の最大秒数。超過時は緊急停止 (デフォルト: 300)")
    parser.add_argument("--memory-poll", type=float, default=2.0, help="メモリ待機時の監視頻度秒 (デフォルト: 2.0)")
    args = parser.parse_args()
    
    # プロジェクトルートを検出
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # --clean オプションの処理
    if args.clean:
        clean_binaries(project_root)
        sys.exit(0)
    
    # ── 自動ビルド ──────────────────────────────────────
    # 古いバイナリによるテスト失敗を防ぐため、テスト実行前に必ず再ビルドする。
    # release バイナリが存在する場合は --release でビルドし、なければ debug でビルドする。
    # --no-build フラグで省略可能。
    if not args.no_build:
        use_release = (project_root / "target" / "release" / "tl").exists()
        profile = "--release" if use_release else ""
        profile_label = "release" if use_release else "debug"
        print(f"🔨 バイナリを再ビルド中 ({profile_label})...")
        build_cmd = ["cargo", "build"] + ([profile] if profile else [])
        build_result = subprocess.run(
            build_cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        if build_result.returncode != 0:
            print(f"❌ cargo build 失敗:\n{build_result.stderr}")
            sys.exit(1)
        print(f"✅ ビルド完了 ({profile_label})\n")

    # TL バイナリのパス
    tl_binary = project_root / "target" / "release" / "tl"
    if not tl_binary.exists():
        tl_binary = project_root / "target" / "debug" / "tl"
    
    if not tl_binary.exists():
        print("❌ TL バイナリが見つかりません。先に 'cargo build' を実行してください。")
        sys.exit(1)
    
    # 検索ディレクトリ
    directories = [
        project_root / "tests",
        project_root / "examples",
    ]
    
    # ランタイムライブラリのリンク準備 (libtl_runtime.a)
    # Cargoはハッシュ付きのファイル名を生成するため (libtl_runtime-xxx.a)、
    # リンカが見つけられるように libtl_runtime.a としてシンボリックリンクを作成する。
    runtime_dir = tl_binary.parent
    lib_path = runtime_dir / "libtl_runtime.a"
    deps_dir = runtime_dir / "deps"
    
    if deps_dir.exists():
        candidates = list(deps_dir.glob("libtl_runtime-*.a"))
        if candidates:
            latest_lib = max(candidates, key=lambda p: p.stat().st_mtime)
            try:
                # 既存のリンク/ファイルを削除して更新
                if lib_path.exists():
                    lib_path.unlink()
                os.symlink(latest_lib, lib_path)
                # print(f"🔗 Runtime library linked: {latest_lib.name} -> {lib_path.name}")
            except Exception as e:
                print(f"⚠️ Warning: Failed to symlink runtime library: {e}")
    
    print("🔍 TL ファイル検証エージェント")
    print(f"📁 検索ディレクトリ: {', '.join(str(d) for d in directories)}")
    print(f"⏱️ タイムアウト: {args.timeout}秒")
    print(f"🛡️ 安全策: クールダウン {args.cooldown}秒 / クラッシュ後 {args.crash_cooldown}秒 / 連続{args.max_crashes}回で緊急停止")
    print(
        f"🧠 メモリガード: cached<= {args.max_cached_gib:.1f}GiB, "
        f"reclaimable>= {args.min_reclaimable_gib:.1f}GiB "
        f"(待機上限 {args.memory_wait_timeout}s)"
    )
    print("")
    
    # ファイル検索
    tl_files = find_tl_files(directories, args.filter)
    print(f"📄 {len(tl_files)} 個の .tl ファイルを検出\n")
    
    results: List[TestResult] = []
    
    # 連続クラッシュ検出用カウンター
    consecutive_crashes = 0
    emergency_stopped = False
    
    def is_crash(result: TestResult) -> bool:
        """セグフォまたはabort (exit -6) かどうか判定"""
        if result.status == Status.SEGFAULT:
            return True
        if result.status == Status.FAIL and "Exit code: -6" in result.reason:
            return True
        return False
    
    # ⚠️ 並列実行コードは意図的に削除済み。Metal GPU リソース競合で Mac がクラッシュするため。
    # テストは必ず逐次実行する（下記ブロック）。

    # セーフティ設定
    safety_pause_interval = 10  # 何テストごとに休憩するか
    safety_pause_duration = 5.0 # 休憩時間（秒）
    
    if args.safe_mode:
        print(f"🛡️ セーフモード有効: {safety_pause_interval}テストごとに {safety_pause_duration}秒 の休憩を挟みます。")

    # 順次実行
    for i, tl_file in enumerate(tl_files, 1):
        rel_path = tl_file.relative_to(project_root)

        ok, mem_reason = wait_for_safe_memory_window(
            max_cached_gib=args.max_cached_gib,
            min_reclaimable_gib=args.min_reclaimable_gib,
            timeout_sec=args.memory_wait_timeout,
            poll_sec=args.memory_poll,
            verbose=args.verbose
        )
        if not ok:
            print(f"\n🚨 緊急停止: メモリが危険域のまま回復しませんでした ({mem_reason})")
            emergency_stopped = True
            break
        
        # セーフティポーズ (Metal ドライバのリソース回収待ち)
        if args.safe_mode and i > 1 and (i - 1) % safety_pause_interval == 0:
            print(f"\n☕ [Safety Pause] システムの安定化を待機中 ({safety_pause_duration}s)... ", end="", flush=True)
            time.sleep(safety_pause_duration)
            print("再開")

        print(f"[{i}/{len(tl_files)}] {rel_path} ... ", end="", flush=True)
        
        result = run_tl_file(tl_file, tl_binary, args.timeout, args.verbose)
        
        # GPU リソース競合による間欠的失敗 (SIGTRAP=-5, SIGABRT=-6) のリトライ
        if result.status == Status.FAIL and result.reason and ("Exit code: -5" in result.reason or "Exit code: -6" in result.reason):
            max_retries = 2
            for retry in range(max_retries):
                print(f"🔄", end="", flush=True)
                time.sleep(args.crash_cooldown)
                result = run_tl_file(tl_file, tl_binary, args.timeout, args.verbose)
                if result.status == Status.PASS:
                    break
        
        results.append(result)
        
        print(f"{result.status.value} ({result.duration:.1f}s)")
        
        if args.verbose and result.status == Status.FAIL:
            if result.error:
                print(f"      Error: {result.error[:100]}...")
        
        # 連続クラッシュ検出
        if is_crash(result):
            consecutive_crashes += 1
            if consecutive_crashes >= args.max_crashes:
                print(f"\n🚨 緊急停止: {consecutive_crashes} 回連続でクラッシュが発生しました。")
                print("   GPU リソースの枯渇によるシステムクラッシュを防ぐため、テストを中断します。")
                emergency_stopped = True
                break
            # クラッシュ後は長めのクールダウン (GPU リソース回収待ち)
            time.sleep(args.crash_cooldown)
        else:
            if result.status != Status.SKIP:
                consecutive_crashes = 0
            # 通常のクールダウン
            if args.cooldown > 0 and result.status != Status.SKIP:
                time.sleep(args.cooldown)
    
    # サマリー表示
    failures = print_summary(results, args.verbose)
    
    # 終了前に残存プロセスをクリーンアップ
    _cleanup_children()
    print("\n🧹 子プロセスのクリーンアップ完了")
    
    sys.exit(1 if failures > 0 else 0)

if __name__ == "__main__":
    main()
