#!/usr/bin/env python3
"""
FFI シグネチャ不一致検出スクリプト

builtins.rs 内の map_tensor_fn! マクロおよび add_global_mapping で
紐付けられた runtime (GPU) 関数と CPU FFI 関数のシグネチャを比較し、
引数の数・型・戻り値の不一致を検出します。

使い方:
    python scripts/check_ffi_signatures.py [--verbose]
"""

import re
import sys
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# 定数 — プロジェクト内のファイルパス
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

BUILTINS_RS = PROJECT_ROOT / "src" / "compiler" / "codegen" / "builtins.rs"

# Runtime FFI が定義されているファイル群
RUNTIME_SOURCES = [
    PROJECT_ROOT / "crates" / "tl_runtime" / "src" / "lib.rs",
    PROJECT_ROOT / "crates" / "tl_runtime" / "src" / "tensor_ops_ext.rs",
    PROJECT_ROOT / "crates" / "tl_runtime" / "src" / "memory_ffi.rs",
    PROJECT_ROOT / "crates" / "tl_runtime" / "src" / "registry.rs",
    PROJECT_ROOT / "crates" / "tl_runtime" / "src" / "io_ffi.rs",
]

# CPU FFI
CPU_FFI_RS = PROJECT_ROOT / "crates" / "tl_cpu" / "src" / "ffi.rs"


# ============================================================
# データ型
# ============================================================
@dataclass
class FnSig:
    """extern "C" 関数のシグネチャ"""
    name: str
    params: list[str]          # 型名のリスト (変数名は除去)
    return_type: str           # "void" | "*mut OpaqueTensor" etc.
    source_file: str = ""
    line: int = 0

    @property
    def arity(self) -> int:
        return len(self.params)

    def sig_str(self) -> str:
        params = ", ".join(self.params) if self.params else "(none)"
        return f"({params}) -> {self.return_type}"


@dataclass
class Mapping:
    """builtins.rs 内のマッピング情報"""
    ffi_name: str              # LLVM 側の名前 (例: "tl_tensor_get")
    runtime_path: str          # Rust パス (例: "runtime::tl_tensor_get")
    cpu_path: Optional[str]    # CPU パス (例: "cpu_ffi::tl_cpu_tensor_get") or None
    line: int = 0
    source: str = ""           # "map_tensor_fn!" | "add_global_mapping"

    @property
    def runtime_fn(self) -> str:
        """Rust パスから関数名を抽出"""
        return self.runtime_path.rsplit("::", 1)[-1]

    @property
    def cpu_fn(self) -> Optional[str]:
        if self.cpu_path is None:
            return None
        return self.cpu_path.rsplit("::", 1)[-1]


# ============================================================
# パーサー
# ============================================================

# extern "C" fn の正規表現 — 複数行にまたがるケースにも対応
RE_EXTERN_FN = re.compile(
    r'pub\s+extern\s+"C"\s+fn\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*(.+?))?\s*\{',
    re.DOTALL,
)

# 型名の抽出 — パラメータの "名前: 型" から型だけ取り出す
RE_PARAM = re.compile(r'(\w+)\s*:\s*(.+)')


def normalize_type(ty: str) -> str:
    """型名を正規化して比較可能にする"""
    ty = ty.strip().rstrip(",")
    # ポインタ系はすべて "ptr" にまとめる
    if ty.startswith("*mut ") or ty.startswith("*const "):
        return "ptr"
    return ty


def parse_extern_fns(filepath: Path) -> dict[str, FnSig]:
    """ファイルから pub extern "C" fn をすべて抽出"""
    if not filepath.exists():
        return {}
    content = filepath.read_text(encoding="utf-8")

    result = {}
    for m in RE_EXTERN_FN.finditer(content):
        fn_name = m.group(1)
        raw_params = m.group(2).strip()
        raw_return = (m.group(3) or "").strip()

        # パラメータ解析
        params = []
        if raw_params:
            # カンマで分割するが、ジェネリクス内のカンマは無視
            depth = 0
            current = []
            for ch in raw_params:
                if ch in ("<", "("):
                    depth += 1
                elif ch in (">", ")"):
                    depth -= 1
                if ch == "," and depth == 0:
                    current_str = "".join(current).strip()
                    if current_str:
                        params.append(current_str)
                    current = []
                else:
                    current.append(ch)
            last = "".join(current).strip()
            if last:
                params.append(last)

        # 型だけ抽出
        param_types = []
        for p in params:
            m2 = RE_PARAM.match(p.strip())
            if m2:
                param_types.append(normalize_type(m2.group(2)))
            else:
                param_types.append(normalize_type(p.strip()))

        ret = normalize_type(raw_return) if raw_return else "void"

        # 行番号
        line_no = content[:m.start()].count("\n") + 1

        result[fn_name] = FnSig(
            name=fn_name,
            params=param_types,
            return_type=ret,
            source_file=str(filepath.relative_to(PROJECT_ROOT)),
            line=line_no,
        )
    return result


# map_tensor_fn!("name", gpu_path, cpu_path);
RE_MAP_TENSOR = re.compile(
    r'map_tensor_fn!\(\s*"(\w+)"\s*,'
    r'\s*([a-zA-Z_][\w:]*)\s*,'
    r'\s*([a-zA-Z_][\w:]*)\s*\)',
)

# add_global_mapping(&f, runtime::xxx as usize);  ← ペアは直前の get_function("name")
RE_GET_FUNCTION = re.compile(
    r'module\.get_function\(\s*"(\w+)"\s*\)'
)
RE_ADD_MAPPING = re.compile(
    r'execution_engine\.add_global_mapping\(\s*&f\s*,\s*([a-zA-Z_][\w:]*)\s+as\s+usize\s*\)'
)


def parse_builtins(filepath: Path) -> list[Mapping]:
    """builtins.rs のマッピング情報を抽出"""
    content = filepath.read_text(encoding="utf-8")
    mappings: list[Mapping] = []
    seen_ffi: set[str] = set()

    # 1. map_tensor_fn! — CPU/GPU 両方の情報がある
    for m in RE_MAP_TENSOR.finditer(content):
        ffi_name = m.group(1)
        gpu_path = m.group(2)
        cpu_path = m.group(3)
        line = content[:m.start()].count("\n") + 1
        mappings.append(Mapping(
            ffi_name=ffi_name,
            runtime_path=gpu_path,
            cpu_path=cpu_path,
            line=line,
            source="map_tensor_fn!",
        ))
        seen_ffi.add(ffi_name)

    # 2. 直接 add_global_mapping — get_function の直後にマッピング
    lines = content.split("\n")
    current_fn_name = None
    for i, line in enumerate(lines, 1):
        gf = RE_GET_FUNCTION.search(line)
        if gf:
            current_fn_name = gf.group(1)
            continue

        am = RE_ADD_MAPPING.search(line)
        if am and current_fn_name and current_fn_name not in seen_ffi:
            runtime_path = am.group(1)
            mappings.append(Mapping(
                ffi_name=current_fn_name,
                runtime_path=runtime_path,
                cpu_path=None,
                line=i,
                source="add_global_mapping",
            ))
            seen_ffi.add(current_fn_name)
            current_fn_name = None

    return mappings


# ============================================================
# メインロジック
# ============================================================

def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    print("🔍 FFI シグネチャ不一致検出ツール")
    print(f"   プロジェクト: {PROJECT_ROOT}")
    print()

    # --- 1. 関数シグネチャの収集 ---
    runtime_fns: dict[str, FnSig] = {}
    for src in RUNTIME_SOURCES:
        runtime_fns.update(parse_extern_fns(src))

    cpu_fns = parse_extern_fns(CPU_FFI_RS)

    print(f"📦 Runtime FFI 関数: {len(runtime_fns)} 個")
    print(f"📦 CPU FFI 関数:     {len(cpu_fns)} 個")

    # --- 2. builtins.rs マッピングの解析 ---
    mappings = parse_builtins(BUILTINS_RS)
    print(f"🔗 マッピング:       {len(mappings)} 個")
    print()

    # --- 3. 不一致検出 ---
    issues: list[str] = []
    warnings: list[str] = []
    info: list[str] = []

    for mapping in mappings:
        rt_fn_name = mapping.runtime_fn
        cpu_fn_name = mapping.cpu_fn

        # Runtime 関数のシグネチャ取得
        rt_sig = runtime_fns.get(rt_fn_name)
        if rt_sig is None:
            # runtime パスにサブモジュールがある場合は関数名だけで探す
            for name, sig in runtime_fns.items():
                if name == rt_fn_name:
                    rt_sig = sig
                    break

        if rt_sig is None:
            warnings.append(
                f"⚠️  Runtime 関数が見つかりません: {mapping.runtime_path}\n"
                f"   FFI名: {mapping.ffi_name}  (builtins.rs L{mapping.line})"
            )
            continue

        # CPU マッピングがない場合 (runtime のみ) — CPU モードでの問題リスク
        if cpu_fn_name is None:
            info.append(
                f"ℹ️  CPU マッピングなし (runtime のみ): {mapping.ffi_name}\n"
                f"   → {mapping.runtime_path}  (builtins.rs L{mapping.line})"
            )
            continue

        # CPU 関数のシグネチャ取得
        cpu_sig = cpu_fns.get(cpu_fn_name)
        if cpu_sig is None:
            issues.append(
                f"❌ CPU 関数が見つかりません: {mapping.cpu_path}\n"
                f"   FFI名: {mapping.ffi_name}  (builtins.rs L{mapping.line})\n"
                f"   Runtime: {rt_sig.name}{rt_sig.sig_str()}"
            )
            continue

        # --- シグネチャ比較 ---
        mismatches = []

        # 引数の数
        if rt_sig.arity != cpu_sig.arity:
            mismatches.append(
                f"   引数の数: runtime={rt_sig.arity}, cpu={cpu_sig.arity}"
            )

        # 引数の型 (位置ごとに比較)
        min_arity = min(rt_sig.arity, cpu_sig.arity)
        for j in range(min_arity):
            if rt_sig.params[j] != cpu_sig.params[j]:
                mismatches.append(
                    f"   引数[{j}]: runtime={rt_sig.params[j]}, cpu={cpu_sig.params[j]}"
                )

        # 戻り値の型
        if rt_sig.return_type != cpu_sig.return_type:
            mismatches.append(
                f"   戻り値: runtime={rt_sig.return_type}, cpu={cpu_sig.return_type}"
            )

        if mismatches:
            detail = "\n".join(mismatches)
            issues.append(
                f"❌ シグネチャ不一致: {mapping.ffi_name}  (builtins.rs L{mapping.line})\n"
                f"   Runtime: {rt_sig.name}{rt_sig.sig_str()}\n"
                f"            ({rt_sig.source_file}:{rt_sig.line})\n"
                f"   CPU:     {cpu_sig.name}{cpu_sig.sig_str()}\n"
                f"            ({cpu_sig.source_file}:{cpu_sig.line})\n"
                f"{detail}"
            )
        elif verbose:
            print(f"   ✅ {mapping.ffi_name}: OK ({rt_sig.arity} args → {rt_sig.return_type})")

    # --- 4. 結果表示 ---
    print("=" * 60)
    print("検査結果")
    print("=" * 60)

    if issues:
        print(f"\n🚨 シグネチャ不一致: {len(issues)} 件\n")
        for issue in issues:
            print(issue)
            print()
    else:
        print("\n✅ シグネチャ不一致は検出されませんでした。\n")

    if warnings:
        print(f"⚠️  警告: {len(warnings)} 件\n")
        for w in warnings:
            print(w)
            print()

    if verbose and info:
        print(f"ℹ️  CPU マッピングなし (runtime のみ): {len(info)} 件\n")
        for i_msg in info:
            print(i_msg)
            print()

    # --- サマリー ---
    print("-" * 60)
    total_map_tensor = sum(1 for m in mappings if m.source == "map_tensor_fn!")
    total_direct = sum(1 for m in mappings if m.source == "add_global_mapping")
    total_no_cpu = sum(1 for m in mappings if m.cpu_path is None)
    print(f"📊 サマリー:")
    print(f"   map_tensor_fn! マッピング: {total_map_tensor}")
    print(f"   直接 add_global_mapping:   {total_direct}")
    print(f"   CPU マッピングなし:        {total_no_cpu}")
    print(f"   不一致: {len(issues)} / 警告: {len(warnings)}")

    if issues:
        print(f"\n💡 不一致が検出されました。CPU モードで上記の関数を")
        print(f"   呼び出すとセグフォやデータ破損の原因になります。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
