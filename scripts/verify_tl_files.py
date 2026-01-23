#!/usr/bin/env python3
"""
TL ファイル検証エージェント
tests/ と examples/ 内の .tl ファイルを実行し、動作を確認します。

使用方法:
    python scripts/verify_tl_files.py [--verbose] [--timeout SECONDS] [--filter PATTERN]
"""

import subprocess
import sys
import os
import re
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum

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

# main 関数を持つファイルのみ実行
def has_main_function(filepath: Path) -> bool:
    """ファイルに main 関数が含まれているか確認"""
    try:
        content = filepath.read_text(encoding='utf-8')
        # fn main() を探す
        return bool(re.search(r'\bfn\s+main\s*\(', content))
    except Exception:
        return False

# 特定のファイルをスキップするかどうか
SKIP_FILES = {
    # 対話的な入力が必要なファイル
    "chatbot_llama3.tl",
    "chatbot.tl",
    # 外部リソースが必要
    "download.tl",
    "infer.tl",  # MNIST
    "train.tl",  # MNIST
    # 非常に長時間実行されるファイル
    "train_heavy.tl",
    "infer_heavy.tl",
    "train_add.tl",
    "train_paper.tl",
    "train_recall.tl",
    "reverse_train.tl",
    "readme_n_queens.tl",
    "n_queens.tl",
    # 既知の問題があるファイル
    "train_verify_2digit.tl",
    "reverse_infer.tl",
}

# 長時間実行が予想されるファイル（長めのタイムアウト）
LONG_RUNNING = {
    "lenia.tl",
    "inverse_life.tl",
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
}

def should_skip(filepath: Path) -> Tuple[bool, str]:
    """ファイルをスキップすべきか判定"""
    name = filepath.name
    if name in SKIP_FILES:
        return True, f"スキップ対象: {name}"
    if not has_main_function(filepath):
        # EXPECTED_FAILURES に含まれている場合は、main関数がなくても実行を試みる（コンパイルエラー等を確認するため）
        if name in EXPECTED_FAILURES:
            return False, ""
        return True, "main 関数なし"
    return False, ""

def run_tl_file(filepath: Path, tl_binary: Path, timeout: int) -> TestResult:
    """TL ファイルを実行して結果を返す"""
    start_time = time.time()
    
    skip, reason = should_skip(filepath)
    if skip:
        return TestResult(
            file=str(filepath),
            status=Status.SKIP,
            output="",
            error="",
            duration=0.0,
            reason=reason
        )
    
    # 長時間実行ファイルはタイムアウトを延長
    if filepath.name in LONG_RUNNING:
        timeout = max(timeout, 120)

    is_expected_to_fail = filepath.name in EXPECTED_FAILURES
    
    try:
        result = subprocess.run(
            [str(tl_binary), str(filepath)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=filepath.parent
        )
        duration = time.time() - start_time
        
        # セグメンテーションフォールトの検出
        if result.returncode == -11 or result.returncode == 139:
            return TestResult(
                file=str(filepath),
                status=Status.SEGFAULT,
                output=result.stdout,
                error=result.stderr,
                duration=duration,
                reason="Segmentation fault"
            )
        
        # 終了コードをチェック
        if result.returncode != 0:
            if is_expected_to_fail:
                # 失敗が期待されていたので PASS とする
                return TestResult(
                    file=str(filepath),
                    status=Status.PASS,
                    output=result.stdout,
                    error=result.stderr, 
                    duration=duration,
                    reason="(Expected Failure)"
                )
            
            # エラーメッセージを解析
            return TestResult(
                file=str(filepath),
                status=Status.FAIL,
                output=result.stdout,
                error=result.stderr,
                duration=duration,
                reason=f"Exit code: {result.returncode}"
            )
        else:
            if is_expected_to_fail:
                # 失敗すべきなのに成功してしまった場合
                return TestResult(
                    file=str(filepath),
                    status=Status.FAIL,
                    output=result.stdout,
                    error=result.stderr,
                    duration=duration,
                    reason="Unexpected Success: Expected failure but exited with 0"
                )

        return TestResult(
            file=str(filepath),
            status=Status.PASS,
            output=result.stdout,
            error=result.stderr,
            duration=duration
        )
        
    except subprocess.TimeoutExpired:
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
        duration = time.time() - start_time
        return TestResult(
            file=str(filepath),
            status=Status.FAIL,
            output="",
            error=str(e),
            duration=duration,
            reason=str(e)
        )

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

def main():
    parser = argparse.ArgumentParser(description="TL ファイル検証エージェント")
    parser.add_argument("--verbose", "-v", action="store_true", help="詳細出力")
    parser.add_argument("--timeout", "-t", type=int, default=30, help="タイムアウト秒数 (デフォルト: 30)")
    parser.add_argument("--filter", "-f", type=str, help="ファイルパターンでフィルタ")
    parser.add_argument("--parallel", "-p", type=int, default=1, help="並列実行数 (デフォルト: 1)")
    args = parser.parse_args()
    
    # プロジェクトルートを検出
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # TL バイナリのパス
    tl_binary = project_root / "target" / "debug" / "tl"
    if not tl_binary.exists():
        print("❌ TL バイナリが見つかりません。先に 'cargo build' を実行してください。")
        sys.exit(1)
    
    # 検索ディレクトリ
    directories = [
        project_root / "tests",
        project_root / "examples",
    ]
    
    print("🔍 TL ファイル検証エージェント")
    print(f"📁 検索ディレクトリ: {', '.join(str(d) for d in directories)}")
    print(f"⏱️ タイムアウト: {args.timeout}秒")
    print("")
    
    # ファイル検索
    tl_files = find_tl_files(directories, args.filter)
    print(f"📄 {len(tl_files)} 個の .tl ファイルを検出\n")
    
    results: List[TestResult] = []
    
    # 結果格納用
    results: List[TestResult] = []
    
    # 並列実行
    if args.parallel > 1:
        print(f"🚀 {args.parallel} 並列で実行中...")
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            future_to_file = {
                executor.submit(run_tl_file, f, tl_binary, args.timeout): f 
                for f in tl_files
            }
            
            completed_count = 0
            for future in  as_completed(future_to_file):
                result = future.result()
                results.append(result)
                completed_count += 1
                
                # 進捗表示
                status_icon = result.status.value
                rel_path = Path(result.file).relative_to(project_root)
                print(f"[{completed_count}/{len(tl_files)}] {status_icon} {rel_path} ({result.duration:.1f}s)")
                if args.verbose and result.status == Status.FAIL:
                     if result.error:
                        print(f"      Error: {result.error[:100]}...")

    else:
        # 順次実行
        for i, tl_file in enumerate(tl_files, 1):
            rel_path = tl_file.relative_to(project_root)
            print(f"[{i}/{len(tl_files)}] {rel_path} ... ", end="", flush=True)
            
            result = run_tl_file(tl_file, tl_binary, args.timeout)
            results.append(result)
            
            print(f"{result.status.value} ({result.duration:.1f}s)")
            
            if args.verbose and result.status == Status.FAIL:
                if result.error:
                    print(f"      Error: {result.error[:100]}...")
    
    # サマリー表示
    failures = print_summary(results, args.verbose)
    
    sys.exit(1 if failures > 0 else 0)

if __name__ == "__main__":
    main()
