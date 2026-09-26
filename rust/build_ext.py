"""
Rust 版ルールエンジンを CPython 拡張モジュールとしてビルドし、
リポジトリ直下に quoridor_rs.pyd（Linux/macOS では quoridor_rs.so）を置く。

    python rust/build_ext.py

必要なもの: Rust（cargo）。依存クレートやビルドスクリプトは使わない。
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def find_cargo():
    cargo = shutil.which("cargo")
    if cargo:
        return cargo
    cand = Path.home() / ".cargo" / "bin" / ("cargo.exe" if os.name == "nt" else "cargo")
    if cand.exists():
        return str(cand)
    sys.exit("cargo が見つかりません。Rust をインストールしてください: https://rustup.rs/")


def main():
    if os.name != "nt":
        sys.exit("現在の C API 実装は Windows 専用です（関数を python3XX.dll から解決するため）。")
    cargo = find_cargo()
    subprocess.run([cargo, "build", "--release", "--lib"], cwd=HERE, check=True)
    src = HERE / "target" / "release" / "quoridor_rs.dll"
    dst = ROOT / "quoridor_rs.pyd"
    shutil.copyfile(src, dst)
    print(f"built: {dst}")


if __name__ == "__main__":
    main()
