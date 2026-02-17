#!/usr/bin/env python3
"""
FFI 型安全性チェックスクリプト

tl_runtime の @ffi_sig コメントで宣言された意味的型情報と、
フロントエンド（codegen）での呼び出しパターンを照合し、
型の不整合を早期に検出する。

使い方:
    python scripts/check_ffi_type_safety.py [--verbose]
"""

import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ============================================================
# 定数
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# @ffi_sig をスキャンするランタイムソース
RUNTIME_SOURCES = [
    PROJECT_ROOT / "crates" / "tl_runtime" / "src",
]

# フロントエンド codegen ソース
FRONTEND_SOURCES = [
    PROJECT_ROOT / "src" / "compiler" / "codegen",
]

# 型の互換性マトリクス
# key: @ffi_sig 型, value: 互換 TL 型のセット
TYPE_COMPAT = {
    "Tensor*": {"Tensor", "TensorShaped"},
    "Struct*": {"Struct"},
    "String*": {"String"},
    "void*":   {"Tensor", "TensorShaped", "Struct", "String", "void", "any"},  # 汎用
    "i8*":     {"cstr"},
    "i64":     {"i64", "int"},
    "f32":     {"f32", "float"},
    "f64":     {"f64", "float"},
    "bool":    {"bool"},
    "usize":   {"usize", "u64"},
    "u32":     {"u32"},
}


# ============================================================
# データ型
# ============================================================
@dataclass
class FfiSig:
    """@ffi_sig で宣言された型情報"""
    fn_name: str
    params: list[str]       # 意味的型名リスト (e.g. ["Tensor*", "i64"])
    return_type: str        # "void" | "Tensor*" | "i64" etc.
    source_file: str = ""
    line: int = 0

    def sig_str(self) -> str:
        params = ", ".join(self.params) if self.params else ""
        return f"({params}) -> {self.return_type}"


@dataclass
class CallSite:
    """フロントエンドでの FFI 呼び出し箇所"""
    fn_name: str            # 呼び出す FFI 関数名
    context_type: str       # 呼び出しコンテキストから推定した型 (e.g. "Tensor", "Struct")
    source_file: str = ""
    line: int = 0
    arg_count: Optional[int] = None  # build_call の引数数 (抽出できた場合)
    context_snippet: str = ""  # 周辺コード


# ============================================================
# ランタイム @ffi_sig パーサー
# ============================================================
RE_FFI_SIG = re.compile(
    r'///\s*@ffi_sig\s+\(([^)]*)\)\s*->\s*(\S+)'
)
RE_FN_NAME = re.compile(
    r'pub\s+extern\s+"C"\s+fn\s+(\w+)'
)


def parse_ffi_sigs(filepath: Path) -> dict[str, FfiSig]:
    """ファイルから @ffi_sig コメントを抽出"""
    if not filepath.exists():
        return {}

    content = filepath.read_text(encoding="utf-8")
    lines = content.split("\n")
    result = {}

    pending_sig: Optional[tuple[list[str], str, int]] = None

    for i, line in enumerate(lines):
        # @ffi_sig を検出
        m = RE_FFI_SIG.search(line)
        if m:
            raw_params = m.group(1).strip()
            ret_type = m.group(2).strip()
            params = [p.strip() for p in raw_params.split(",") if p.strip()] if raw_params else []
            pending_sig = (params, ret_type, i + 1)
            continue

        # 直後の extern "C" fn を検出
        if pending_sig:
            fn_m = RE_FN_NAME.search(line)
            if fn_m:
                fn_name = fn_m.group(1)
                params, ret_type, sig_line = pending_sig
                result[fn_name] = FfiSig(
                    fn_name=fn_name,
                    params=params,
                    return_type=ret_type,
                    source_file=str(filepath.relative_to(PROJECT_ROOT)),
                    line=sig_line,
                )
                pending_sig = None
            elif line.strip() and not line.strip().startswith("///") and not line.strip().startswith("#["):
                # コメントやアトリビュート以外の行が来たら pending を破棄
                pending_sig = None

    return result


def collect_ffi_sigs() -> dict[str, FfiSig]:
    """全ランタイムソースから @ffi_sig を収集"""
    all_sigs: dict[str, FfiSig] = {}
    for src_dir in RUNTIME_SOURCES:
        if not src_dir.exists():
            continue
        for rs_file in src_dir.rglob("*.rs"):
            if ".bak" in str(rs_file) or ".orig" in str(rs_file):
                continue
            sigs = parse_ffi_sigs(rs_file)
            all_sigs.update(sigs)
    return all_sigs


# ============================================================
# フロントエンド呼び出し解析
# ============================================================

# match ty のパターン検出
RE_TYPE_MATCH = re.compile(
    r'Type::(Tensor|TensorShaped|Struct|String|Enum|Tuple|Int|Float|Bool)'
)

# get_function("xxx") の検出
RE_GET_FUNCTION = re.compile(
    r'get_function\(\s*"(\w+)"\s*\)'
)

# build_call の引数数検出 (概算)
RE_BUILD_CALL_ARGS = re.compile(
    r'build_call\([^,]+,\s*&\[([^\]]*)\]'
)


def analyze_frontend_calls(filepath: Path) -> list[CallSite]:
    """フロントエンドソースから FFI 呼び出しパターンを解析"""
    if not filepath.exists():
        return []

    content = filepath.read_text(encoding="utf-8")
    lines = content.split("\n")
    calls: list[CallSite] = []

    # コンテキスト推定: match ty => ... Type::Xxx => ... get_function("tl_yyy")
    # スコープスタックで現在の型コンテキストを追跡
    current_type_context: list[str] = []
    brace_depth = 0
    type_context_depths: list[int] = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # ブレースの深さを追跡
        brace_depth += line.count("{") - line.count("}")

        # Type::Xxx パターンの検出
        type_matches = RE_TYPE_MATCH.findall(line)
        for tm in type_matches:
            if "=>" in line or "match" in line.lower() or "if " in line:
                current_type_context.append(tm)
                type_context_depths.append(brace_depth)

        # 古いコンテキストの削除
        while type_context_depths and brace_depth < type_context_depths[-1]:
            type_context_depths.pop()
            if current_type_context:
                current_type_context.pop()

        # get_function の検出
        fn_match = RE_GET_FUNCTION.search(line)
        if fn_match:
            fn_name = fn_match.group(1)
            ctx = current_type_context[-1] if current_type_context else "unknown"

            # 前後5行のコンテキストをスニペットとして保存
            start = max(0, i - 3)
            end = min(len(lines), i + 4)
            snippet = "\n".join(lines[start:end])

            calls.append(CallSite(
                fn_name=fn_name,
                context_type=ctx,
                source_file=str(filepath.relative_to(PROJECT_ROOT)),
                line=i + 1,
                context_snippet=snippet,
            ))

    return calls


def collect_frontend_calls() -> list[CallSite]:
    """全フロントエンドソースから FFI 呼び出しを収集"""
    all_calls: list[CallSite] = []
    for src_dir in FRONTEND_SOURCES:
        if not src_dir.exists():
            continue
        for rs_file in src_dir.rglob("*.rs"):
            calls = analyze_frontend_calls(rs_file)
            all_calls.extend(calls)
    return all_calls


# ============================================================
# 整合性チェック
# ============================================================

def check_type_compat(ffi_sig: FfiSig, call: CallSite) -> Optional[str]:
    """
    @ffi_sig の型とフロントエンドの呼び出しコンテキスト型の互換性チェック。
    
    戻り値: エラーメッセージ (問題なければ None)
    """
    # 引数のない関数はコンテキスト不問
    if not ffi_sig.params:
        return None

    ctx = call.context_type
    if ctx == "unknown":
        return None  # コンテキスト不明はスキップ

    # 第1引数の型チェック (最も重要)
    first_param = ffi_sig.params[0]

    # パイプ記法対応: "Struct*|String*" → ["Struct*", "String*"]
    param_alternatives = [p.strip() for p in first_param.split("|")]

    # 各代替型の互換型を集約
    compat_types: set[str] = set()
    for alt in param_alternatives:
        compat_types.update(TYPE_COMPAT.get(alt, set()))

    if not compat_types:
        return None  # 未知の型はスキップ

    # void* はすべてと互換
    if "void*" in param_alternatives:
        return None

    # TL型が期待される型と互換かチェック
    if ctx not in compat_types:
        return (
            f"型不整合: {call.fn_name}\n"
            f"   @ffi_sig 第1引数: {first_param} (期待: {compat_types})\n"
            f"   呼び出しコンテキスト: Type::{ctx}\n"
            f"   場所: {call.source_file}:{call.line}"
        )

    return None


# ============================================================
# レポート: @ffi_sig カバレッジ
# ============================================================

RE_EXTERN_FN = re.compile(
    r'(?:pub\s+)?(?:#\[[\w()\s]*\]\s*)*(?:pub\s+)?extern\s+"C"\s+fn\s+(\w+)',
)


def find_uncovered_fns() -> list[tuple[str, str, int]]:
    """@ffi_sig コメントがない extern "C" fn を検出"""
    uncovered = []
    for src_dir in RUNTIME_SOURCES:
        if not src_dir.exists():
            continue
        for rs_file in src_dir.rglob("*.rs"):
            if ".bak" in str(rs_file) or ".orig" in str(rs_file):
                continue
            content = rs_file.read_text(encoding="utf-8")
            lines = content.split("\n")
            for i, line in enumerate(lines):
                fn_m = RE_EXTERN_FN.search(line)
                if fn_m and 'extern "C" fn' in line:
                    fn_name = fn_m.group(1)
                    # 直前の行に @ffi_sig があるか確認 (最大10行遡る)
                    has_sig = False
                    for j in range(max(0, i - 10), i):
                        if "@ffi_sig" in lines[j]:
                            has_sig = True
                            break
                    if not has_sig:
                        rel_path = str(rs_file.relative_to(PROJECT_ROOT))
                        uncovered.append((fn_name, rel_path, i + 1))
    return uncovered


# ============================================================
# メイン
# ============================================================

def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    print("🔍 FFI 型安全性チェックツール")
    print(f"   プロジェクト: {PROJECT_ROOT}")
    print()

    # --- 1. @ffi_sig 収集 ---
    ffi_sigs = collect_ffi_sigs()
    print(f"📋 @ffi_sig 定義:    {len(ffi_sigs)} 個")

    if verbose:
        for name, sig in sorted(ffi_sigs.items()):
            print(f"   {name}: {sig.sig_str()}  ({sig.source_file}:{sig.line})")
        print()

    # --- 2. フロントエンド呼び出し収集 ---
    calls = collect_frontend_calls()
    print(f"📞 FFI 呼び出し箇所: {len(calls)} 個")
    print()

    # --- 3. 型整合性チェック ---
    issues: list[str] = []
    checked = 0
    skipped = 0

    for call in calls:
        if call.fn_name in ffi_sigs:
            sig = ffi_sigs[call.fn_name]
            error = check_type_compat(sig, call)
            if error:
                issues.append(error)
            checked += 1
            if verbose and not error:
                print(f"   ✅ {call.fn_name} ({call.context_type}) @ {call.source_file}:{call.line}")
        else:
            skipped += 1

    # --- 4. カバレッジチェック ---
    uncovered = find_uncovered_fns()

    # --- 5. 結果表示 ---
    print("=" * 60)
    print("検査結果")
    print("=" * 60)

    if issues:
        print(f"\n🚨 型不整合: {len(issues)} 件\n")
        for issue in issues:
            print(f"❌ {issue}")
            print()
    else:
        print("\n✅ 型不整合は検出されませんでした。\n")

    # カバレッジレポート
    if uncovered:
        print(f"📊 @ffi_sig 未定義の関数: {len(uncovered)} 個\n")
        # ファイルごとにグループ化
        by_file: dict[str, list[tuple[str, int]]] = {}
        for fn_name, path, line in uncovered:
            by_file.setdefault(path, []).append((fn_name, line))
        for path, fns in sorted(by_file.items()):
            print(f"   📁 {path}:")
            for fn_name, line in fns:
                print(f"      - {fn_name} (L{line})")
        print()

    # --- サマリー ---
    print("-" * 60)
    print(f"📊 サマリー:")
    print(f"   @ffi_sig 定義済み:          {len(ffi_sigs)}")
    print(f"   フロントエンド FFI 呼び出し: {len(calls)}")
    print(f"   チェック済み:               {checked}")
    print(f"   @ffi_sig 未対応 (スキップ): {skipped}")
    print(f"   型不整合:                   {len(issues)}")
    print(f"   @ffi_sig 未定義関数:        {len(uncovered)}")

    ffi_sig_coverage = len(ffi_sigs) / (len(ffi_sigs) + len(uncovered)) * 100 if (len(ffi_sigs) + len(uncovered)) > 0 else 0
    print(f"   カバレッジ:                 {ffi_sig_coverage:.1f}%")

    if issues:
        print(f"\n💡 型不整合が検出されました。修正が必要です。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
