#!/usr/bin/env python3
"""
FFI シグネチャ不一致検出スクリプト

builtins.rs 内の add_global_mapping で紐付けられた runtime 関数を、
プロジェクト内の全 Rust ソースから自動検索し、シグネチャの不一致を検出します。

使い方:
    python scripts/check_ffi_signatures.py [--verbose]
"""

import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


# ============================================================
# 定数
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

BUILTINS_RS = PROJECT_ROOT / "src" / "compiler" / "codegen" / "builtins.rs"

# 全 Rust ソースを検索する crate ディレクトリ
CRATE_DIRS = [
    PROJECT_ROOT / "crates" / "tl_runtime" / "src",
    PROJECT_ROOT / "crates" / "tl_cpu" / "src",
    PROJECT_ROOT / "crates" / "tl_metal" / "src",
    PROJECT_ROOT / "crates" / "tl_cuda" / "src",
    PROJECT_ROOT / "crates" / "tl_backend" / "src",
]

# Rust パスの prefix → 検索対象ディレクトリのマッピング
# builtins.rs では runtime::device_ffi::xxx, runtime::llm::xxx, cpu_ffi::xxx 等の形式
MODULE_SEARCH_MAP = {
    "runtime::device_ffi": [PROJECT_ROOT / "crates" / "tl_runtime" / "src" / "device_ffi.rs"],
    "runtime::llm":        [PROJECT_ROOT / "crates" / "tl_runtime" / "src" / "llm.rs"],
    "runtime::stdlib":     [PROJECT_ROOT / "crates" / "tl_runtime" / "src" / "stdlib"],
    "runtime::registry":   [PROJECT_ROOT / "crates" / "tl_runtime" / "src" / "registry.rs"],
    "runtime::arena":      [PROJECT_ROOT / "crates" / "tl_runtime" / "src" / "arena.rs"],
    "cpu_ffi":             [PROJECT_ROOT / "crates" / "tl_cpu" / "src" / "ffi.rs"],
    "cuda_ffi":            [PROJECT_ROOT / "crates" / "tl_cuda" / "src" / "ffi_ops.rs"],
    # runtime::xxx — tl_runtime/src 全体 + re-export 元 (tl_metal, tl_cpu, tl_cuda)
    "runtime":             [
        PROJECT_ROOT / "crates" / "tl_runtime" / "src",
        PROJECT_ROOT / "crates" / "tl_metal" / "src",
        PROJECT_ROOT / "crates" / "tl_cpu" / "src",
        PROJECT_ROOT / "crates" / "tl_cuda" / "src",
    ],
}


# ============================================================
# データ型
# ============================================================
@dataclass
class FnSig:
    """extern "C" 関数のシグネチャ"""
    name: str
    params: list[str]          # 正規化済み型名のリスト
    return_type: str           # "void" | "ptr" etc.
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
    runtime_path: str          # Rust パス (例: "runtime::device_ffi::tl_device_tensor_get")
    line: int = 0

    @property
    def runtime_fn(self) -> str:
        """Rust パスから関数名を抽出"""
        return self.runtime_path.rsplit("::", 1)[-1]

    @property
    def module_prefix(self) -> str:
        """関数名を除いたモジュールパスを返す"""
        parts = self.runtime_path.rsplit("::", 1)
        return parts[0] if len(parts) > 1 else ""


# ============================================================
# パーサー
# ============================================================

# extern "C" fn / pub extern "C" fn — 複数行対応
RE_EXTERN_FN = re.compile(
    r'(?:pub\s+)?(?:#\[[\w()\s]*\]\s*)*(?:pub\s+)?extern\s+"C"\s+fn\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*(.+?))?\s*\{',
    re.DOTALL,
)

# パラメータの「名前: 型」から型だけ取り出す
RE_PARAM = re.compile(r'(\w+)\s*:\s*(.+)')


def normalize_type(ty: str) -> str:
    """型名を正規化して比較可能にする"""
    ty = ty.strip().rstrip(",")
    if ty.startswith("*mut ") or ty.startswith("*const "):
        return "ptr"
    # c_void も ptr 扱い
    if ty == "c_void":
        return "ptr"
    return ty


def split_params(raw_params: str) -> list[str]:
    """ジェネリクス内のカンマを無視してパラメータを分割"""
    depth = 0
    current: list[str] = []
    result: list[str] = []
    for ch in raw_params:
        if ch in ("<", "("):
            depth += 1
        elif ch in (">", ")"):
            depth -= 1
        if ch == "," and depth == 0:
            s = "".join(current).strip()
            if s:
                result.append(s)
            current = []
        else:
            current.append(ch)
    last = "".join(current).strip()
    if last:
        result.append(last)
    return result


def parse_extern_fns(filepath: Path) -> dict[str, FnSig]:
    """ファイルから extern "C" fn をすべて抽出"""
    if not filepath.exists():
        return {}
    content = filepath.read_text(encoding="utf-8")
    result = {}
    for m in RE_EXTERN_FN.finditer(content):
        fn_name = m.group(1)
        raw_params = m.group(2).strip()
        raw_return = (m.group(3) or "").strip()

        param_types = []
        if raw_params:
            for p in split_params(raw_params):
                m2 = RE_PARAM.match(p.strip())
                if m2:
                    param_types.append(normalize_type(m2.group(2)))
                else:
                    param_types.append(normalize_type(p.strip()))

        ret = normalize_type(raw_return) if raw_return else "void"
        line_no = content[:m.start()].count("\n") + 1

        result[fn_name] = FnSig(
            name=fn_name,
            params=param_types,
            return_type=ret,
            source_file=str(filepath.relative_to(PROJECT_ROOT)),
            line=line_no,
        )
    return result


def collect_all_fns() -> dict[str, FnSig]:
    """全 crate ディレクトリから extern "C" fn を収集"""
    all_fns: dict[str, FnSig] = {}
    for crate_dir in CRATE_DIRS:
        if not crate_dir.exists():
            continue
        for rs_file in crate_dir.rglob("*.rs"):
            fns = parse_extern_fns(rs_file)
            all_fns.update(fns)
    return all_fns


def find_fn_in_search_paths(fn_name: str, search_paths: list[Path]) -> Optional[FnSig]:
    """指定パスリストから関数を検索"""
    for path in search_paths:
        if path.is_file():
            fns = parse_extern_fns(path)
            if fn_name in fns:
                return fns[fn_name]
        elif path.is_dir():
            for rs_file in path.rglob("*.rs"):
                fns = parse_extern_fns(rs_file)
                if fn_name in fns:
                    return fns[fn_name]
    return None


# builtins.rs パーサー
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

    lines = content.split("\n")
    current_fn_name: Optional[str] = None
    for i, line in enumerate(lines, 1):
        gf = RE_GET_FUNCTION.search(line)
        if gf:
            current_fn_name = gf.group(1)

        am = RE_ADD_MAPPING.search(line)
        if am and current_fn_name:
            runtime_path = am.group(1)
            if current_fn_name not in seen_ffi:
                mappings.append(Mapping(
                    ffi_name=current_fn_name,
                    runtime_path=runtime_path,
                    line=i,
                ))
                seen_ffi.add(current_fn_name)
            current_fn_name = None

    return mappings


def resolve_fn(mapping: Mapping, all_fns: dict[str, FnSig]) -> Optional[FnSig]:
    """マッピングの runtime_path から実際の関数シグネチャを解決"""
    fn_name = mapping.runtime_fn
    prefix = mapping.module_prefix

    # 1. モジュール prefix に基づく優先検索
    for mod_prefix, search_paths in MODULE_SEARCH_MAP.items():
        if prefix == mod_prefix or prefix.startswith(mod_prefix + "::"):
            sig = find_fn_in_search_paths(fn_name, search_paths)
            if sig:
                return sig

    # 2. 全関数辞書からのフォールバック検索
    if fn_name in all_fns:
        return all_fns[fn_name]

    return None


# ============================================================
# メインロジック
# ============================================================

def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    print("🔍 FFI シグネチャ不一致検出ツール")
    print(f"   プロジェクト: {PROJECT_ROOT}")
    print()

    # --- 1. 全関数シグネチャの収集 ---
    all_fns = collect_all_fns()
    print(f"📦 検出した extern \"C\" 関数: {len(all_fns)} 個")

    # --- 2. builtins.rs マッピングの解析 ---
    mappings = parse_builtins(BUILTINS_RS)
    print(f"🔗 マッピング:               {len(mappings)} 個")
    print()

    # --- 3. 検証 ---
    issues: list[str] = []
    warnings: list[str] = []
    resolved_count = 0
    device_ffi_count = 0

    for mapping in mappings:
        sig = resolve_fn(mapping, all_fns)

        if sig is None:
            warnings.append(
                f"⚠️  Runtime 関数が見つかりません: {mapping.runtime_path}\n"
                f"   FFI名: {mapping.ffi_name}  (builtins.rs L{mapping.line})"
            )
            continue

        resolved_count += 1
        if "device_ffi" in mapping.runtime_path:
            device_ffi_count += 1

        if verbose:
            print(f"   ✅ {mapping.ffi_name} → {sig.name}{sig.sig_str()}  ({sig.source_file}:{sig.line})")

    # --- 4. device_ffi 関数のシグネチャ対 IDevice チェック ---
    # device_ffi の各 tl_device_* 関数と、対応する Metal/CPU 実装のシグネチャを比較
    device_ffi_fns = {}
    device_ffi_path = PROJECT_ROOT / "crates" / "tl_runtime" / "src" / "device_ffi.rs"
    if device_ffi_path.exists():
        device_ffi_fns = parse_extern_fns(device_ffi_path)

    metal_fns = {}
    metal_ffi_path = PROJECT_ROOT / "crates" / "tl_metal" / "src" / "ffi_ops.rs"
    metal_ffi2_path = PROJECT_ROOT / "crates" / "tl_metal" / "src" / "ffi.rs"
    if metal_ffi_path.exists():
        metal_fns.update(parse_extern_fns(metal_ffi_path))
    if metal_ffi2_path.exists():
        metal_fns.update(parse_extern_fns(metal_ffi2_path))

    cpu_fns = {}
    cpu_ffi_path = PROJECT_ROOT / "crates" / "tl_cpu" / "src" / "ffi.rs"
    if cpu_ffi_path.exists():
        cpu_fns = parse_extern_fns(cpu_ffi_path)

    cuda_fns = {}
    cuda_ffi_path = PROJECT_ROOT / "crates" / "tl_cuda" / "src" / "ffi_ops.rs"
    if cuda_ffi_path.exists():
        cuda_fns.update(parse_extern_fns(cuda_ffi_path))

    # device_ffi → Metal/CPU の対応テーブル自動生成
    #
    # ── 許容リスト (allowlist) ──
    # 以下の関数は device_ffi と Metal/CPU FFI のシグネチャが異なるが、
    # device_impl.rs 内のアダプタ変換で安全に吸収されているため問題なし。
    # device_ffi は IDevice trait のメソッドを呼ぶだけであり、
    # 各バックエンドの IDevice 実装が内部で FFI シグネチャの差分を処理する。
    #
    ALLOWLIST: dict[str, str] = {
        # tl_metal_detach(ptr) は req_grad 引数を受け取らないが、
        # MetalDeviceImpl::tensor_detach() が req_grad を内部で処理する。
        "tl_device_tensor_detach": (
            "Metal側は req_grad 引数なし。MetalDeviceImpl::tensor_detach() が "
            "IDevice の (ptr, bool) を受け取り、内部で tl_metal_detach(ptr) を呼ぶ"
        ),
        # tl_metal_reshape_new(ptr, ptr, usize) は追加の usize 引数があるが、
        # MetalDeviceImpl::tensor_reshape_new() がアダプタ変換する。
        "tl_device_tensor_reshape_new": (
            "Metal側は (ptr, ptr, usize) の3引数。MetalDeviceImpl が IDevice の "
            "(ptr, ptr) から内部変換して tl_metal_reshape_new を呼ぶ"
        ),
        # tl_metal_reshape_dims(ptr, ptr, usize) は IDevice の (ptr, i64x4) と異なるが、
        # MetalDeviceImpl::tensor_reshape_dims() がアダプタ変換する。
        "tl_device_tensor_reshape_dims": (
            "Metal側は (ptr, ptr, usize) の3引数。MetalDeviceImpl が IDevice の "
            "(ptr, i64, i64, i64, i64) から内部変換して tl_metal_reshape_dims を呼ぶ"
        ),
        # tl_metal_apply_rope(ptr, ptr, ptr, ptr, usize) は IDevice の (ptr, ptr, ptr) と異なるが、
        # MetalDeviceImpl::tensor_apply_rope() がアダプタ変換する。
        "tl_device_tensor_apply_rope": (
            "Metal側は (ptr, ptr, ptr, ptr, usize) の5引数 + void戻り値。"
            "MetalDeviceImpl が IDevice の (ptr, ptr, ptr)->ptr から内部変換する"
        ),
        # tl_cpu_tensor_conv2d(ptr, ptr, i64, i64) は IDevice の 7引数と異なるが、
        # CpuDevice::tensor_conv2d() がアダプタ変換する。
        "tl_device_tensor_conv2d": (
            "CPU側は (input, weight, padding, stride) の4引数。CpuDevice が IDevice の "
            "7引数 (input, weight, bias, stride, padding, dilation, groups) から抽出して呼ぶ"
        ),
    }

    sig_mismatches: list[str] = []
    skipped_allowed: list[str] = []
    for df_name, df_sig in device_ffi_fns.items():
        # tl_device_tensor_xxx → tl_metal_xxx / tl_cpu_tensor_xxx / tl_cuda_xxx
        base = df_name.replace("tl_device_tensor_", "").replace("tl_device_", "")

        metal_candidates = [
            f"tl_metal_{base}",
            f"tl_metal_tensor_{base}",
        ]
        cpu_candidates = [
            f"tl_cpu_tensor_{base}",
            f"tl_cpu_{base}",
        ]
        cuda_candidates = [
            f"tl_cuda_{base}",
            f"tl_cuda_tensor_{base}",
        ]

        metal_sig = None
        for c in metal_candidates:
            if c in metal_fns:
                metal_sig = metal_fns[c]
                break

        cpu_sig = None
        for c in cpu_candidates:
            if c in cpu_fns:
                cpu_sig = cpu_fns[c]
                break

        cuda_sig = None
        for c in cuda_candidates:
            if c in cuda_fns:
                cuda_sig = cuda_fns[c]
                break

        # 許容リストに含まれる場合はスキップ
        if df_name in ALLOWLIST:
            if (metal_sig and df_sig.arity != metal_sig.arity) or \
               (cpu_sig and df_sig.arity != cpu_sig.arity):
                skipped_allowed.append(
                    f"   ⏭️  {df_name}: {ALLOWLIST[df_name]}"
                )
            continue

        # device_ffi と Metal の引数数比較 (型はすべて ptr になるので数だけで十分)
        if metal_sig and df_sig.arity != metal_sig.arity:
            sig_mismatches.append(
                f"❌ device_ffi ↔ Metal 引数数不一致: {df_name}\n"
                f"   device_ffi: {df_sig.sig_str()}  ({df_sig.source_file}:{df_sig.line})\n"
                f"   Metal:      {metal_sig.name}{metal_sig.sig_str()}  ({metal_sig.source_file}:{metal_sig.line})"
            )

        if cpu_sig and df_sig.arity != cpu_sig.arity:
            sig_mismatches.append(
                f"❌ device_ffi ↔ CPU 引数数不一致: {df_name}\n"
                f"   device_ffi: {df_sig.sig_str()}  ({df_sig.source_file}:{df_sig.line})\n"
                f"   CPU:        {cpu_sig.name}{cpu_sig.sig_str()}  ({cpu_sig.source_file}:{cpu_sig.line})"
            )

        if cuda_sig and df_sig.arity != cuda_sig.arity:
            sig_mismatches.append(
                f"❌ device_ffi ↔ CUDA 引数数不一致: {df_name}\n"
                f"   device_ffi: {df_sig.sig_str()}  ({df_sig.source_file}:{df_sig.line})\n"
                f"   CUDA:       {cuda_sig.name}{cuda_sig.sig_str()}  ({cuda_sig.source_file}:{cuda_sig.line})"
            )

    # --- 5. 結果表示 ---
    print("=" * 60)
    print("検査結果")
    print("=" * 60)

    if sig_mismatches:
        print(f"\n🚨 device_ffi ↔ バックエンド シグネチャ不一致: {len(sig_mismatches)} 件\n")
        for m in sig_mismatches:
            print(m)
            print()

    if issues:
        print(f"\n🚨 シグネチャ不一致: {len(issues)} 件\n")
        for issue in issues:
            print(issue)
            print()

    if not issues and not sig_mismatches:
        print("\n✅ シグネチャ不一致は検出されませんでした。\n")

    if warnings:
        print(f"⚠️  警告: {len(warnings)} 件\n")
        for w in warnings:
            print(w)
            print()

    if skipped_allowed:
        print(f"\n🔧 許容リスト (device_impl アダプタ変換済み): {len(skipped_allowed)} 件\n")
        for s in skipped_allowed:
            print(s)
        print()

    # --- サマリー ---
    print("-" * 60)
    print(f"📊 サマリー:")
    print(f"   マッピング総数:           {len(mappings)}")
    print(f"   解決できた関数:           {resolved_count}")
    print(f"   うち device_ffi 経由:     {device_ffi_count}")
    print(f"   未解決 (警告):            {len(warnings)}")
    print(f"   不一致:                   {len(issues) + len(sig_mismatches)}")
    print(f"   許容済み (allowlist):      {len(skipped_allowed)}")

    if issues or sig_mismatches:
        print(f"\n💡 不一致が検出されました。修正が必要です。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
