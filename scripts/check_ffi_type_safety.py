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

# コンテキスト推論の限界による偽陽性ホワイトリスト
# key: (関数名, コンテキスト型)
# スクリプトは「呼び出し元の match ty コンテキスト = 第1引数の型」と仮定するが、
# 変換関数やユーティリティ関数ではこの仮定が成り立たないケースがある。
FALSE_POSITIVE_WHITELIST: dict[tuple[str, str], str] = {
    # 文字コード(i64) → String 変換関数。第1引数は i64 で正しい。
    ("tl_string_from_char", "String"): "i64 の文字コードを受け取って String* を返す変換関数",

    # arena のサイズ(i64)を指定する初期化関数。Enum コンテキストとは無関係。
    ("tl_arena_init", "Enum"): "arena サイズ(i64)の初期化。Enum ブロック内で呼ばれるが引数は Enum と無関係",

    # Tuple 内の String 要素を表示する関数。第1引数は String* で正しい。
    ("tl_display_string", "Tuple"): "Tuple 要素の表示処理内で呼ばれる。第1引数は Tuple ではなく String*",
    ("tl_print_string", "Tuple"): "Tuple 要素の表示処理内で呼ばれる。第1引数は Tuple ではなく String*",

    # Tensor→String 変換パス内で C文字列(i8*)から StringStruct を生成。
    ("tl_string_new", "Tensor"): "Tensor の文字列表現を生成するパスで呼ばれる。第1引数は i8* (C文字列)",
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
class RuntimeFn:
    """ランタイム側の extern "C" fn の実際の引数情報"""
    fn_name: str
    param_count: int        # 実際の引数数
    param_types: list[str]  # Rust の型名 (e.g. ["*mut c_void", "i64"])
    return_type: str        # Rust の戻り値型
    source_file: str = ""
    line: int = 0


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
# ランタイム extern "C" fn の実引数パーサー
# ============================================================

RE_EXTERN_C_FN = re.compile(
    r'pub\s+extern\s+"C"\s+fn\s+(\w+)\s*\('
)

def parse_runtime_fns(filepath: Path) -> dict[str, RuntimeFn]:
    """ファイルから extern "C" fn の実際の引数情報を抽出"""
    if not filepath.exists():
        return {}

    content = filepath.read_text(encoding="utf-8")
    lines = content.split("\n")
    result = {}

    i = 0
    while i < len(lines):
        line = lines[i]
        m = RE_EXTERN_C_FN.search(line)
        if m:
            fn_name = m.group(1)
            # 関数シグネチャを複数行にわたって収集
            sig_text = line[m.start():]
            # 閉じ括弧と戻り値型まで読む
            brace_count = sig_text.count("(") - sig_text.count(")")
            j = i + 1
            while brace_count > 0 and j < len(lines):
                sig_text += " " + lines[j].strip()
                brace_count += lines[j].count("(") - lines[j].count(")")
                j += 1

            # 引数部分を抽出: fn name(args) -> RetType {
            paren_match = re.search(r'\(([^)]*)\)', sig_text[sig_text.index("("):])
            if paren_match:
                args_str = paren_match.group(1).strip()
                if not args_str:
                    param_count = 0
                    param_types = []
                else:
                    # 引数をカンマで分割（ジェネリクス内のカンマを考慮）
                    params = split_params(args_str)
                    param_types = []
                    for p in params:
                        p = p.strip()
                        if not p:
                            continue
                        # "name: Type" からType部分を抽出
                        if ":" in p:
                            type_part = p.split(":", 1)[1].strip()
                            param_types.append(type_part)
                        else:
                            param_types.append(p)
                    param_count = len(param_types)

                # 戻り値型を抽出
                ret_match = re.search(r'\)\s*->\s*([^{]+)', sig_text)
                ret_type = ret_match.group(1).strip() if ret_match else "void"

                result[fn_name] = RuntimeFn(
                    fn_name=fn_name,
                    param_count=param_count,
                    param_types=param_types,
                    return_type=ret_type,
                    source_file=str(filepath.relative_to(PROJECT_ROOT)),
                    line=i + 1,
                )
        i += 1

    return result


def split_params(s: str) -> list[str]:
    """ジェネリクス内のカンマを考慮して引数を分割"""
    parts = []
    depth = 0
    current = []
    for ch in s:
        if ch in "<([":
            depth += 1
            current.append(ch)
        elif ch in ">)]":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def collect_runtime_fns() -> dict[str, RuntimeFn]:
    """全ランタイムソースから extern "C" fn を収集"""
    all_fns: dict[str, RuntimeFn] = {}
    for src_dir in RUNTIME_SOURCES:
        if not src_dir.exists():
            continue
        for rs_file in src_dir.rglob("*.rs"):
            if ".bak" in str(rs_file) or ".orig" in str(rs_file):
                continue
            fns = parse_runtime_fns(rs_file)
            all_fns.update(fns)
    return all_fns


# ============================================================
# codegen 側の build_call 引数数抽出
# ============================================================

# build_call の引数配列パターン検出（複数行対応）
RE_BUILD_CALL = re.compile(
    r'build_call\s*\(\s*(\w+)\s*,\s*&\[([^\]]*)\]'
)

# get_function("xxx") の検出
RE_GET_FUNCTION = re.compile(
    r'get_function\(\s*"(\w+)"\s*\)'
)

def extract_build_call_arg_counts(filepath: Path) -> dict[str, list[tuple[int, int, str]]]:
    """
    codegen ソースから get_function("name") と対応する build_call の引数数を抽出。
    戻り値: {fn_name: [(arg_count, line, snippet), ...]}
    """
    if not filepath.exists():
        return {}

    content = filepath.read_text(encoding="utf-8")
    lines = content.split("\n")
    result: dict[str, list[tuple[int, int, str]]] = {}

    # get_function で取得した関数名をトラッキング
    # 変数名 -> FFI関数名 のマッピング
    fn_vars: dict[str, str] = {}

    for i, line in enumerate(lines):
        # get_function("tl_xxx") のパターン
        gf_match = RE_GET_FUNCTION.search(line)
        if gf_match:
            fn_name = gf_match.group(1)
            # let xxx = ... get_function("tl_yyy") or if let Some(xxx) = ... get_function
            # 変数名を抽出
            var_match = re.search(r'(?:let\s+(?:mut\s+)?|Some\()(\w+)', line)
            if var_match:
                var_name = var_match.group(1)
                fn_vars[var_name] = fn_name

        # build_call パターン検出
        # build_call(fn_val, &[arg1, arg2, ...], "name")
        bc_match = re.search(r'build_call\s*\(\s*(\w+)\s*,\s*&\[', line)
        if bc_match:
            call_var = bc_match.group(1)
            fn_name = fn_vars.get(call_var)
            if not fn_name:
                # 直接 fn_val が使われるケース: 同行に get_function がある場合
                gf_same = RE_GET_FUNCTION.search(line)
                if gf_same:
                    fn_name = gf_same.group(1)

            if fn_name:
                # &[ から対応する ] まで全テキストを結合
                # まず &[ の位置を特定
                amp_bracket_pos = line.index("&[")
                full_text = line[amp_bracket_pos + 2:]  # &[ の後から開始
                bracket_depth = 1
                j = i
                
                # ] が見つかるまでテキストを結合
                while bracket_depth > 0:
                    for ci, ch in enumerate(full_text if j == i else lines[j]):
                        if ch == "[":
                            bracket_depth += 1
                        elif ch == "]":
                            bracket_depth -= 1
                            if bracket_depth == 0:
                                # ] の手前まで取得
                                if j == i:
                                    full_text = full_text[:ci]
                                else:
                                    full_text += " " + lines[j][:ci]
                                break
                    else:
                        if j > i:
                            full_text += " " + lines[j]
                        j += 1
                        if j >= len(lines):
                            break
                        continue
                    break  # bracket_depth == 0 で break

                # 引数数をカウント: .into() の出現回数が最も信頼性が高い
                full_text = full_text.strip()
                if not full_text:
                    arg_count = 0
                else:
                    into_count = full_text.count(".into()")
                    if into_count > 0:
                        arg_count = into_count
                    else:
                        # .into() がない場合はカンマ+1 でフォールバック
                        arg_count = len(split_params(full_text))

                snippet = lines[i].strip()
                result.setdefault(fn_name, []).append((arg_count, i + 1, snippet))

    return result


def collect_build_call_arg_counts() -> dict[str, list[tuple[int, int, str]]]:
    """全フロントエンドソースからbuild_callの引数数を収集"""
    all_counts: dict[str, list[tuple[int, int, str]]] = {}
    for src_dir in FRONTEND_SOURCES:
        if not src_dir.exists():
            continue
        for rs_file in src_dir.rglob("*.rs"):
            counts = extract_build_call_arg_counts(rs_file)
            for fn_name, entries in counts.items():
                all_counts.setdefault(fn_name, []).extend(entries)
    return all_counts


# ============================================================
# フロントエンド呼び出し解析
# ============================================================

# match ty のパターン検出
RE_TYPE_MATCH = re.compile(
    r'Type::(Tensor|TensorShaped|Struct|String|Enum|Tuple|Int|Float|Bool)'
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

    # ホワイトリストに該当する場合はスキップ
    if (call.fn_name, ctx) in FALSE_POSITIVE_WHITELIST:
        return None

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


def check_arg_count_mismatch(
    runtime_fns: dict[str, RuntimeFn],
    build_call_counts: dict[str, list[tuple[int, int, str]]],
) -> list[str]:
    """
    ランタイム側の実引数数と codegen 側の build_call 引数数を比較。
    
    device_ffi のディスパッチ関数は tl_device_ プレフィックスを持つので、
    codegen 側の tl_tensor_ / tl_ → ランタイム側の tl_device_ への
    名前マッピングも考慮する。
    """
    issues = []

    for fn_name, call_entries in build_call_counts.items():
        # ランタイム側の関数を探す
        runtime_fn = runtime_fns.get(fn_name)
        if not runtime_fn:
            # tl_tensor_xxx -> tl_device_tensor_xxx マッピング
            device_name = fn_name.replace("tl_tensor_", "tl_device_tensor_", 1)
            if device_name == fn_name:
                device_name = "tl_device_" + fn_name[3:] if fn_name.startswith("tl_") else None
            runtime_fn = runtime_fns.get(device_name)
        if not runtime_fn:
            continue

        for arg_count, line, snippet in call_entries:
            if arg_count != runtime_fn.param_count:
                issues.append(
                    f"引数数不一致: {fn_name}\n"
                    f"   codegen 側: {arg_count} 引数 (L{line})\n"
                    f"   ランタイム側: {runtime_fn.param_count} 引数 "
                    f"({runtime_fn.source_file}:L{runtime_fn.line})\n"
                    f"   ランタイム引数: ({', '.join(runtime_fn.param_types)})\n"
                    f"   codegen: {snippet}"
                )

    return issues


# ============================================================
# レポート: @ffi_sig カバレッジ
# ============================================================

RE_EXTERN_FN = re.compile(
    r'(?:pub\s+)?(?:#[\w()\s]*\]\s*)*(?:pub\s+)?extern\s+"C"\s+fn\s+(\w+)',
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

    # --- 2. ランタイム extern "C" fn 収集 ---
    runtime_fns = collect_runtime_fns()
    print(f"📋 ランタイム関数:   {len(runtime_fns)} 個")

    # --- 3. フロントエンド呼び出し収集 ---
    calls = collect_frontend_calls()
    print(f"📞 FFI 呼び出し箇所: {len(calls)} 個")

    # --- 4. build_call 引数数収集 ---
    build_call_counts = collect_build_call_arg_counts()
    print(f"🔧 build_call 検出:  {len(build_call_counts)} 関数")
    print()

    # --- 5. 型整合性チェック ---
    type_issues: list[str] = []
    checked = 0
    skipped = 0

    for call in calls:
        if call.fn_name in ffi_sigs:
            sig = ffi_sigs[call.fn_name]
            error = check_type_compat(sig, call)
            if error:
                type_issues.append(error)
            checked += 1
            if verbose and not error:
                print(f"   ✅ {call.fn_name} ({call.context_type}) @ {call.source_file}:{call.line}")
        else:
            skipped += 1

    # --- 6. 引数数整合性チェック ---
    arg_issues = check_arg_count_mismatch(runtime_fns, build_call_counts)

    # --- 7. カバレッジチェック ---
    uncovered = find_uncovered_fns()

    # --- 8. 結果表示 ---
    print("=" * 60)
    print("検査結果")
    print("=" * 60)

    all_issues = type_issues + arg_issues

    if type_issues:
        print(f"\n🚨 型不整合: {len(type_issues)} 件\n")
        for issue in type_issues:
            print(f"❌ {issue}")
            print()

    if arg_issues:
        print(f"\n🚨 引数数不一致: {len(arg_issues)} 件\n")
        for issue in arg_issues:
            print(f"❌ {issue}")
            print()

    if not all_issues:
        print("\n✅ 型不整合・引数数不一致は検出されませんでした。\n")

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
    print(f"   ランタイム関数:             {len(runtime_fns)}")
    print(f"   フロントエンド FFI 呼び出し: {len(calls)}")
    print(f"   build_call 検出:            {len(build_call_counts)} 関数")
    print(f"   チェック済み:               {checked}")
    print(f"   @ffi_sig 未対応 (スキップ): {skipped}")
    print(f"   型不整合:                   {len(type_issues)}")
    print(f"   引数数不一致:               {len(arg_issues)}")
    print(f"   @ffi_sig 未定義関数:        {len(uncovered)}")

    ffi_sig_coverage = len(ffi_sigs) / (len(ffi_sigs) + len(uncovered)) * 100 if (len(ffi_sigs) + len(uncovered)) > 0 else 0
    print(f"   カバレッジ:                 {ffi_sig_coverage:.1f}%")

    if all_issues:
        print(f"\n💡 不整合が検出されました。修正が必要です。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
