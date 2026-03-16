#!/usr/bin/env python3
"""
Arc 型安全性チェッカー

tl_runtime 内で OpaqueTensor を直接 Arc キャストしているコードを検出する。
macOS では OpaqueTensor = MetalTensor だが、--device cpu 時のテンソルは
CpuTensor<f32> なので、device_ffi 以外で直接 Arc 操作すると型不一致になる。

検出対象パターン:
1. Arc::from_raw(...OpaqueTensor...) — device_ffi.rs 以外での使用
2. Arc::increment_strong_count(...OpaqueTensor...) — 同上
3. UnsafeCell<OpaqueTensor> / UnsafeCell<crate::OpaqueTensor> — 同上

許可される場所:
- device_ffi.rs (ディスパッチ層、型ごとに分岐済み)
- lib.rs の make_tensor (GPU 専用パス、#[cfg] ガード付き)
- is_cpu() で分岐しているブロック内

使い方:
  python3 scripts/check_arc_type_safety.py
"""

import re
import sys
from pathlib import Path

# チェック対象ディレクトリ
RUNTIME_DIR = Path(__file__).parent.parent / "crates" / "tl_runtime" / "src"

# 検出パターン
DANGEROUS_PATTERNS = [
    # Arc::from_raw で OpaqueTensor にキャスト
    (r'Arc::from_raw\([^)]*OpaqueTensor', "Arc::from_raw with OpaqueTensor cast"),
    # Arc::increment_strong_count で OpaqueTensor にキャスト
    (r'Arc::increment_strong_count\([^)]*OpaqueTensor', "Arc::increment_strong_count with OpaqueTensor cast"),
    # UnsafeCell<OpaqueTensor> / UnsafeCell<crate::OpaqueTensor>
    (r'UnsafeCell<(?:crate::)?OpaqueTensor>', "UnsafeCell<OpaqueTensor> direct usage"),
]

# 除外ファイル (正当な使用がある場所)
EXCLUDED_FILES = {
    "device_ffi.rs",  # ディスパッチ層 — 型ごとに正しく分岐
    "lib.rs",         # make_tensor — #[cfg] ガード付き
}

# 安全とみなすコンテキスト (この文字列が同じ関数内にあれば許可)
SAFE_GUARDS = [
    "is_cpu()",
    "device_ffi::is_cpu()",
    "crate::device_ffi::is_cpu()",
]


def check_file(filepath: Path) -> list[tuple[int, str, str]]:
    """ファイルをチェックし、問題のある行を返す"""
    issues = []
    
    if filepath.name in EXCLUDED_FILES:
        return issues
    
    try:
        content = filepath.read_text()
        lines = content.splitlines()
    except Exception:
        return issues
    
    for line_num, line in enumerate(lines, 1):
        for pattern, desc in DANGEROUS_PATTERNS:
            if re.search(pattern, line):
                # 安全ガードが近くにあるか確認 (前後20行)
                context_start = max(0, line_num - 21)
                context_end = min(len(lines), line_num + 20)
                context = "\n".join(lines[context_start:context_end])
                
                has_guard = any(guard in context for guard in SAFE_GUARDS)
                
                if not has_guard:
                    issues.append((line_num, desc, line.strip()))
    
    return issues


def main():
    if not RUNTIME_DIR.exists():
        print(f"エラー: ディレクトリが見つかりません: {RUNTIME_DIR}")
        sys.exit(1)
    
    total_issues = 0
    checked_files = 0
    
    print("=" * 70)
    print("Arc 型安全性チェック — tl_runtime")
    print("=" * 70)
    print(f"対象: {RUNTIME_DIR}")
    print(f"除外: {', '.join(EXCLUDED_FILES)}")
    print()
    
    for rs_file in sorted(RUNTIME_DIR.glob("**/*.rs")):
        checked_files += 1
        issues = check_file(rs_file)
        
        if issues:
            rel_path = rs_file.relative_to(RUNTIME_DIR.parent.parent.parent)
            print(f"⚠️  {rel_path}")
            for line_num, desc, line_text in issues:
                print(f"   L{line_num}: {desc}")
                print(f"         {line_text}")
            print()
            total_issues += len(issues)
    
    print("-" * 70)
    if total_issues == 0:
        print(f"✅ {checked_files} ファイルをチェック — 問題なし")
    else:
        print(f"⚠️  {checked_files} ファイルをチェック — {total_issues} 件の潜在的問題")
        print()
        print("対処方法:")
        print("  1. is_cpu() で分岐し、CPU 時は CpuTensor<f32> でキャスト")
        print("  2. または device_ffi 経由のヘルパーを使用")
    
    sys.exit(1 if total_issues > 0 else 0)


if __name__ == "__main__":
    main()
