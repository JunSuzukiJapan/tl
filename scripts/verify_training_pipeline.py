#!/usr/bin/env python3
# ruff: noqa: E501
"""
学習パイプライン検証スクリプト

verify_tl_files.py でスキップされている train/infer ペアを
本格的に実行し、学習→推論のパイプラインが正しく動作するか検証する。

使用方法:
    python3 scripts/verify_training_pipeline.py [OPTIONS]

⚠️⚠️⚠️ 絶対禁止事項 ⚠️⚠️⚠️
  verify_tl_files.py と同様、並列実行は禁止。
  Metal GPU バックエンドの並列実行は Mac 全体のクラッシュを引き起こす。
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
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple, Set, Dict
from enum import Enum


# ── 子プロセス追跡 & クリーンアップ ──────────────────────────
_active_procs: Set[subprocess.Popen] = set()

def _cleanup_children():
    """残存する全子プロセスを強制終了する"""
    for proc in list(_active_procs):
        try:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=3)
        except Exception:
            pass
    _active_procs.clear()

atexit.register(_cleanup_children)

def _signal_handler(signum, frame):
    """SIGTERM/SIGINT 受信時にクリーンアップして終了"""
    print(f"\n🛑 シグナル {signum} を受信。子プロセスをクリーンアップ中...")
    _cleanup_children()
    sys.exit(128 + signum)

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


# ── メモリ監視 (verify_tl_files.py から継承) ──────────────────

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
    verbose: bool = False,
) -> Tuple[bool, str]:
    """メモリが安全な状態になるまで待機する。"""
    deadline = time.time() + timeout_sec
    while True:
        snap = get_vm_snapshot()
        if snap is None:
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


# ── ステータス & 結果 ────────────────────────────────

class StepStatus(Enum):
    PASS = "✅"
    FAIL = "❌"
    SKIP = "⏭️"
    TIMEOUT = "⏰"
    SEGFAULT = "💀"


@dataclass
class StepResult:
    name: str
    status: StepStatus
    duration: float
    output: str = ""
    error: str = ""
    reason: str = ""


@dataclass
class InferenceMetrics:
    """推論結果のメトリクス"""
    hits: int = 0           # 正解数
    misses: int = 0         # 不正解数
    total: int = 0          # テスト総数
    accuracy_pct: float = 0.0  # 正解率 (%)
    raw_summary: str = ""   # パースされた生の要約テキスト

    @property
    def has_data(self) -> bool:
        return self.total > 0

    @property
    def grade(self) -> str:
        """正解率に応じた判定"""
        if not self.has_data:
            return "N/A"
        if self.accuracy_pct >= 80.0:
            return "PASS"
        elif self.accuracy_pct >= 50.0:
            return "WARN"
        else:
            return "FAIL"

    @property
    def grade_emoji(self) -> str:
        g = self.grade
        if g == "PASS":
            return "🟢"
        elif g == "WARN":
            return "🟡"
        elif g == "FAIL":
            return "🔴"
        return "⚪"


def parse_inference_metrics(output: str, task_name: str) -> InferenceMetrics:
    """推論の stdout をパースして正解率メトリクスを抽出する"""
    metrics = InferenceMetrics()

    # 改行の正規化 (TL の print は \r\n の場合がある)
    normalized = output.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.splitlines()

    # パターン1: "Accuracy: N/M" (mnist 形式)
    m = re.search(r"Accuracy:\s*(\d+)/(\d+)", normalized)
    if m:
        metrics.hits = int(m.group(1))
        metrics.total = int(m.group(2))
        metrics.misses = metrics.total - metrics.hits
        metrics.accuracy_pct = (metrics.hits / metrics.total * 100) if metrics.total > 0 else 0.0
        metrics.raw_summary = f"{metrics.hits}/{metrics.total}"
        return metrics

    # パターン2: 正解数 + "Correct out of N" の結合パターン
    # 複数の出力形式に対応:
    #   - "Accuracy:\n1\nCorrect out of 6" (改行分割)
    #   - "Accuracy:1Correct out of 6" (改行なし結合)
    #   - "20Correct out of20" (recall 形式: Accuracy: なし)
    #   - "0\nCorrect out of\n5" (reverse 形式)
    # まず "Accuracy:" 付きを試す
    m_combined = re.search(r"Accuracy:\s*(\d+)\s*Correct out of\s*(\d+)", normalized)
    if m_combined:
        metrics.hits = int(m_combined.group(1))
        metrics.total = int(m_combined.group(2))
        metrics.misses = metrics.total - metrics.hits
        metrics.accuracy_pct = (metrics.hits / metrics.total * 100) if metrics.total > 0 else 0.0
        metrics.raw_summary = f"{metrics.hits}/{metrics.total}"
        return metrics
    # "Accuracy:" なしで 数字+Correct out of+数字 (recall)
    m_no_acc = re.search(r"(\d+)\s*Correct out of\s*(\d+)", normalized)
    if m_no_acc:
        metrics.hits = int(m_no_acc.group(1))
        metrics.total = int(m_no_acc.group(2))
        metrics.misses = metrics.total - metrics.hits
        metrics.accuracy_pct = (metrics.hits / metrics.total * 100) if metrics.total > 0 else 0.0
        metrics.raw_summary = f"{metrics.hits}/{metrics.total}"
        return metrics

    for i, line in enumerate(lines):
        m2 = re.search(r"Correct out of\s*(\d+)?", line)
        if m2:
            total_str = m2.group(1)
            if total_str:
                metrics.total = int(total_str)
            else:
                # "Correct out of" の次行に数字
                if i + 1 < len(lines):
                    m3 = re.match(r"\s*(\d+)\s*$", lines[i + 1])
                    if m3:
                        metrics.total = int(m3.group(1))
            # 正解数は直前の数字行を探す ("Accuracy:" の次行の数字)
            for j in range(i - 1, max(i - 4, -1), -1):
                m4 = re.match(r"\s*(\d+)\s*$", lines[j])
                if m4:
                    metrics.hits = int(m4.group(1))
                    break
            if metrics.total > 0:
                metrics.misses = metrics.total - metrics.hits
                metrics.accuracy_pct = (metrics.hits / metrics.total * 100) if metrics.total > 0 else 0.0
                metrics.raw_summary = f"{metrics.hits}/{metrics.total}"
            return metrics

    # パターン3: "Result: HIT" / "Result: MISS" のカウント (linear 等)
    # 全文中のマッチ数をカウント（行分割に依存しない）
    hits = len(re.findall(r"Result:\s*HIT\b", normalized))
    misses = len(re.findall(r"Result:\s*MISS\b", normalized))

    if hits + misses > 0:
        metrics.hits = hits
        metrics.misses = misses
        metrics.total = hits + misses
        metrics.accuracy_pct = (hits / metrics.total * 100) if metrics.total > 0 else 0.0
        metrics.raw_summary = f"{hits}/{metrics.total} HIT"
        return metrics

    return metrics


@dataclass
class PipelineResult:
    task_name: str
    train: Optional[StepResult] = None
    infer: Optional[StepResult] = None
    model_file_found: bool = False
    metrics: Optional[InferenceMetrics] = None

    @property
    def success(self) -> bool:
        """パイプライン全体が成功かどうか"""
        steps = [self.train, self.infer]
        return all(
            s is None or s.status == StepStatus.PASS
            for s in steps
        )


# ── パイプライン定義 ─────────────────────────────────

@dataclass
class PipelineConfig:
    """1つの学習パイプラインの構成"""
    train_file: str          # プロジェクトルートからの相対パス
    infer_file: str          # プロジェクトルートからの相対パス
    model_files: List[str]   # CWD からの相対パス (生成されるモデルファイル)
    cwd: str                 # プロジェクトルートからの相対パス (実行時のCWD)


PIPELINES: Dict[str, PipelineConfig] = {
    "reverse": PipelineConfig(
        train_file="examples/tasks/reverse/reverse_train.tl",
        infer_file="examples/tasks/reverse/reverse_infer.tl",
        model_files=["reverse_model.safetensors"],
        cwd="examples/tasks/reverse",
    ),
    "addition": PipelineConfig(
        train_file="examples/tasks/addition/train_add.tl",
        infer_file="examples/tasks/addition/infer_add.tl",
        model_files=["model_add.safetensors"],
        cwd="examples/tasks/addition",
    ),
    "paper": PipelineConfig(
        train_file="examples/tasks/paper/train_paper.tl",
        infer_file="examples/tasks/paper/infer_paper.tl",
        model_files=["model_paper.safetensors"],
        cwd="examples/tasks/paper",
    ),
    "recall": PipelineConfig(
        train_file="examples/tasks/recall/train_recall.tl",
        infer_file="examples/tasks/recall/infer_recall.tl",
        model_files=["recall_weights.safetensors"],
        cwd="examples/tasks/recall",
    ),
    "mnist": PipelineConfig(
        train_file="examples/tasks/mnist/train.tl",
        infer_file="examples/tasks/mnist/infer.tl",
        model_files=["mnist_weights.safetensors"],
        cwd="examples/tasks/mnist",
    ),
    "linear": PipelineConfig(
        train_file="examples/tasks/linear/train.tl",
        infer_file="examples/tasks/linear/infer.tl",
        model_files=["linear_model.safetensors"],
        cwd="examples/tasks/linear",
    ),
}


# ── 実行ヘルパー ─────────────────────────────────────

def run_step(
    step_name: str,
    cmd: List[str],
    cwd: Path,
    env: dict,
    timeout: int,
    verbose: bool,
    capture_stdout: bool = False,
) -> StepResult:
    """1つのステップ (学習/推論) を JIT 実行して結果を返す

    capture_stdout=True の場合、verbose モードでも stdout をキャプチャする
    （メトリクスのパースに必要な推論ステップで使用）
    """
    start_time = time.time()
    proc = None
    try:
        if verbose and not capture_stdout:
            print(f"    CMD: {' '.join(cmd)}")
            print(f"    CWD: {cwd}")
            # verbose モードでは stdout を直接表示 (学習ステップ用)
            proc = subprocess.Popen(
                cmd,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd,
                env=env,
            )
        else:
            if verbose:
                print(f"    CMD: {' '.join(cmd)}")
                print(f"    CWD: {cwd}")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd,
                env=env,
            )
        _active_procs.add(proc)

        try:
            if verbose and not capture_stdout:
                _, stderr = proc.communicate(timeout=timeout)
                stdout = "(Streamed to console)"
            else:
                stdout, stderr = proc.communicate(timeout=timeout)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            _active_procs.discard(proc)
            return StepResult(
                name=step_name,
                status=StepStatus.TIMEOUT,
                duration=time.time() - start_time,
                reason=f"タイムアウト ({timeout}秒)",
            )
        finally:
            _active_procs.discard(proc)

        duration = time.time() - start_time

        # セグメンテーションフォールト検出
        if returncode == -11 or returncode == 139:
            return StepResult(
                name=step_name,
                status=StepStatus.SEGFAULT,
                duration=duration,
                output=stdout,
                error=stderr,
                reason="Segmentation fault",
            )

        if returncode != 0:
            return StepResult(
                name=step_name,
                status=StepStatus.FAIL,
                duration=duration,
                output=stdout,
                error=stderr,
                reason=f"Exit code: {returncode}",
            )

        return StepResult(
            name=step_name,
            status=StepStatus.PASS,
            duration=duration,
            output=stdout,
            error=stderr,
        )

    except Exception as e:
        if proc is not None:
            try:
                proc.kill()
                proc.wait()
            except Exception:
                pass
            _active_procs.discard(proc)
        return StepResult(
            name=step_name,
            status=StepStatus.FAIL,
            duration=time.time() - start_time,
            reason=str(e),
        )


def run_pipeline(
    task_name: str,
    config: PipelineConfig,
    tl_binary: Path,
    project_root: Path,
    args,
    env: dict,
) -> PipelineResult:
    """1つの train/infer パイプラインを実行"""
    result = PipelineResult(task_name=task_name)
    task_cwd = project_root / config.cwd

    train_file = project_root / config.train_file
    infer_file = project_root / config.infer_file

    # ── Step 1: 学習実行 ──
    print(f"\n  🏋️ 学習実行: {config.train_file}")
    print(f"     (タイムアウト: {args.train_timeout}秒)")

    # 既存のモデルファイルを削除（テストの独立性を確保）
    for mf in config.model_files:
        model_path = task_cwd / mf
        if model_path.exists():
            model_path.unlink()
            if args.verbose:
                print(f"     🗑️ 既存モデルファイル削除: {mf}")

    result.train = run_step(
        step_name=f"train({train_file.name})",
        cmd=[str(tl_binary), train_file.name],
        cwd=task_cwd,
        env=env,
        timeout=args.train_timeout,
        verbose=args.verbose,
    )
    print(f"     {result.train.status.value} ({result.train.duration:.1f}s)")

    if result.train.status != StepStatus.PASS:
        if args.verbose and result.train.error:
            print(f"     Error: {result.train.error[:300]}")
        return result

    # モデルファイルの生成確認
    for mf in config.model_files:
        model_path = task_cwd / mf
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            result.model_file_found = True
            print(f"     📁 モデルファイル生成: {mf} ({size_mb:.2f} MB)")
        else:
            print(f"     ⚠️ モデルファイル未生成: {mf}")
            result.train = StepResult(
                name=result.train.name,
                status=StepStatus.FAIL,
                duration=result.train.duration,
                output=result.train.output,
                error=result.train.error,
                reason="モデルファイルが生成されなかった",
            )
            return result

    # クールダウン (GPU リソース回収)
    if args.cooldown > 0:
        time.sleep(args.cooldown)

    # ── Step 2: 推論実行 ──
    print(f"\n  🔮 推論実行: {config.infer_file}")

    result.infer = run_step(
        step_name=f"infer({infer_file.name})",
        cmd=[str(tl_binary), infer_file.name],
        cwd=task_cwd,
        env=env,
        timeout=args.infer_timeout,
        verbose=args.verbose,
        capture_stdout=True,  # メトリクスパース用: 常に stdout をキャプチャ
    )
    print(f"     {result.infer.status.value} ({result.infer.duration:.1f}s)")

    # 推論の出力を常に表示
    infer_output = ""
    if args.verbose and result.infer.output:
        infer_output = result.infer.output
    elif not args.verbose and result.infer.output:
        infer_output = result.infer.output
    # verbose 時は stdout がストリーム出力されているので stderr のみ表示
    if result.infer.output and result.infer.output.strip():
        if not args.verbose:
            for line in result.infer.output.strip().splitlines():
                print(f"     {line}")
    if result.infer.status != StepStatus.PASS and result.infer.error:
        for line in result.infer.error.strip().splitlines():
            print(f"     {line}")

    # 推論結果メトリクスをパース
    # verbose モードでは stdout がストリームされるため stderr に出力が混在する場合がある
    all_output = (result.infer.output or "") + "\n" + (result.infer.error or "")
    result.metrics = parse_inference_metrics(all_output, task_name)
    if result.metrics.has_data:
        print(f"     📊 正解率: {result.metrics.accuracy_pct:.1f}% ({result.metrics.raw_summary}) {result.metrics.grade_emoji} {result.metrics.grade}")
    else:
        print(f"     📊 正解率: 解析不能（出力形式未対応）")

    return result


def print_summary(results: List[PipelineResult]) -> int:
    """結果のサマリーを表示し、失敗数を返す"""
    print("\n" + "=" * 60)
    print("学習パイプライン検証結果")
    print("=" * 60)

    failures = 0
    for r in results:
        parts = []
        if r.train:
            parts.append(f"train {r.train.status.value} {r.train.duration:.1f}s")
        if r.infer:
            parts.append(f"infer {r.infer.status.value} {r.infer.duration:.1f}s")

        # 推論メトリクスを追加
        if r.metrics and r.metrics.has_data:
            parts.append(f"{r.metrics.grade_emoji} {r.metrics.accuracy_pct:.0f}% ({r.metrics.raw_summary})")

        overall = "✅" if r.success else "❌"
        if not r.success:
            failures += 1

        detail = " → ".join(parts) if parts else "未実行"
        print(f"  {overall} {r.task_name:12s}: {detail}")

        # 失敗詳細
        if not r.success:
            for step in [r.train, r.infer]:
                if step and step.status != StepStatus.PASS:
                    print(f"     └─ {step.name}: {step.reason}")

    # 推論精度サマリーテーブル
    has_any_metrics = any(r.metrics and r.metrics.has_data for r in results)
    if has_any_metrics:
        print()
        print("── 推論精度サマリー ──")
        print(f"  {'タスク':12s} {'正解':>6s} {'総数':>6s} {'正解率':>8s}  {'判定'}")
        print(f"  {'─'*12} {'─'*6} {'─'*6} {'─'*8}  {'─'*4}")
        for r in results:
            if r.metrics and r.metrics.has_data:
                print(
                    f"  {r.task_name:12s} {r.metrics.hits:6d} {r.metrics.total:6d}"
                    f" {r.metrics.accuracy_pct:7.1f}%  {r.metrics.grade_emoji} {r.metrics.grade}"
                )
            else:
                print(f"  {r.task_name:12s} {'—':>6s} {'—':>6s} {'—':>8s}  ⚪ N/A")

    print()

    passed = sum(1 for r in results if r.success)
    total = len(results)
    print(f"成功: {passed}/{total}")

    return failures


def main():
    parser = argparse.ArgumentParser(description="学習パイプライン検証スクリプト")
    parser.add_argument("--verbose", "-v", action="store_true", help="詳細出力")
    parser.add_argument(
        "--train-timeout", type=int, default=3600,
        help="学習タイムアウト秒数 (デフォルト: 3600)"
    )
    parser.add_argument(
        "--infer-timeout", type=int, default=120,
        help="推論タイムアウト秒数 (デフォルト: 120)"
    )
    parser.add_argument("--filter", "-f", type=str, help="特定タスクのみ実行")
    parser.add_argument(
        "--cooldown", type=float, default=5.0,
        help="パイプライン間クールダウン秒数 (デフォルト: 5.0)"
    )
    parser.add_argument(
        "--no-cleanup", action="store_true",
        help="テスト後にモデルファイルを削除しない"
    )
    parser.add_argument(
        "--no-build", action="store_true",
        help="自動ビルドをスキップ"
    )
    # メモリガード設定
    parser.add_argument(
        "--max-cached-gib", type=float, default=22.0,
        help="cached files 上限GiB (デフォルト: 22)"
    )
    parser.add_argument(
        "--min-reclaimable-gib", type=float, default=8.0,
        help="reclaimable 下限GiB (デフォルト: 8)"
    )
    parser.add_argument(
        "--memory-wait-timeout", type=int, default=300,
        help="メモリ待機の最大秒数 (デフォルト: 300)"
    )
    args = parser.parse_args()

    # プロジェクトルート
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # ── 自動ビルド ──
    if not args.no_build:
        use_release = (project_root / "target" / "release" / "tl").exists()
        profile = "--release" if use_release else ""
        profile_label = "release" if use_release else "debug"
        print(f"🔨 バイナリを再ビルド中 ({profile_label})...")
        build_cmd = ["cargo", "build"] + ([profile] if profile else [])
        build_result = subprocess.run(
            build_cmd, cwd=project_root, capture_output=True, text=True
        )
        if build_result.returncode != 0:
            print(f"❌ cargo build 失敗:\n{build_result.stderr}")
            sys.exit(1)
        print(f"✅ ビルド完了 ({profile_label})\n")

    # TL バイナリパス
    tl_binary = project_root / "target" / "release" / "tl"
    if not tl_binary.exists():
        tl_binary = project_root / "target" / "debug" / "tl"
    if not tl_binary.exists():
        print("❌ TL バイナリが見つかりません。先に 'cargo build' を実行してください。")
        sys.exit(1)

    # ランタイムライブラリのリンク準備
    runtime_dir = tl_binary.parent
    lib_path = runtime_dir / "libtl_runtime.a"
    deps_dir = runtime_dir / "deps"
    if deps_dir.exists():
        candidates = list(deps_dir.glob("libtl_runtime-*.a"))
        if candidates:
            latest_lib = max(candidates, key=lambda p: p.stat().st_mtime)
            try:
                if lib_path.exists():
                    lib_path.unlink()
                os.symlink(latest_lib, lib_path)
            except Exception as e:
                print(f"⚠️ Warning: Failed to symlink runtime library: {e}")

    # 環境変数
    env = os.environ.copy()
    extra_lib_path = str(runtime_dir)
    env["LIBRARY_PATH"] = f"{extra_lib_path}:{env.get('LIBRARY_PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{extra_lib_path}:{env.get('LD_LIBRARY_PATH', '')}"
    env["DYLD_LIBRARY_PATH"] = f"{extra_lib_path}:{env.get('DYLD_LIBRARY_PATH', '')}"

    # 対象パイプラインを決定
    if args.filter:
        selected = {
            k: v for k, v in PIPELINES.items()
            if args.filter.lower() in k.lower()
        }
        if not selected:
            print(f"❌ フィルタ '{args.filter}' に一致するタスクがありません。")
            print(f"   利用可能: {', '.join(PIPELINES.keys())}")
            sys.exit(1)
    else:
        selected = PIPELINES

    # ヘッダー表示
    print(f"🔍 学習パイプライン検証")
    print(f"📋 対象: {', '.join(selected.keys())}")
    print(f"⏱️ 学習タイムアウト: {args.train_timeout}秒 / 推論タイムアウト: {args.infer_timeout}秒")
    print(f"🛡️ クールダウン: {args.cooldown}秒")
    print(
        f"🧠 メモリガード: cached<= {args.max_cached_gib:.1f}GiB, "
        f"reclaimable>= {args.min_reclaimable_gib:.1f}GiB"
    )

    results: List[PipelineResult] = []

    for i, (task_name, config) in enumerate(selected.items(), 1):
        # メモリ安全確認
        ok, mem_reason = wait_for_safe_memory_window(
            max_cached_gib=args.max_cached_gib,
            min_reclaimable_gib=args.min_reclaimable_gib,
            timeout_sec=args.memory_wait_timeout,
            poll_sec=2.0,
            verbose=args.verbose,
        )
        if not ok:
            print(f"\n🚨 緊急停止: メモリが危険域のまま回復しませんでした ({mem_reason})")
            break

        print(f"\n{'='*60}")
        print(f"[{i}/{len(selected)}] 🧪 {task_name}")
        print(f"{'='*60}")

        pipeline_result = run_pipeline(
            task_name=task_name,
            config=config,
            tl_binary=tl_binary,
            project_root=project_root,
            args=args,
            env=env,
        )
        results.append(pipeline_result)

        # モデルファイルのクリーンアップ
        if not args.no_cleanup:
            task_cwd = project_root / config.cwd
            for mf in config.model_files:
                model_path = task_cwd / mf
                if model_path.exists():
                    try:
                        model_path.unlink()
                        if args.verbose:
                            print(f"  🗑️ クリーンアップ: {mf}")
                    except Exception as e:
                        print(f"  ⚠️ クリーンアップ失敗: {mf} - {e}")

        # パイプライン間クールダウン
        if i < len(selected) and args.cooldown > 0:
            time.sleep(args.cooldown)

    # サマリー表示
    failures = print_summary(results)

    # クリーンアップ
    _cleanup_children()
    print("🧹 子プロセスのクリーンアップ完了")

    sys.exit(1 if failures > 0 else 0)


if __name__ == "__main__":
    main()
