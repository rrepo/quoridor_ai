"""
PGO（プロファイルに基づく最適化）でビルドする。コードを変えたら実行し直すこと。

    python rust/pgo.py            # quoridor.exe と quoridor_rs.pyd を PGO でビルド
    python rust/pgo.py --compare  # あわせて PGO なしの版と速度を比べる

必要なもの（Windows）:
  - Visual Studio Build Tools（「C++ によるデスクトップ開発」）
  - rustup toolchain install stable-x86_64-pc-windows-msvc --profile minimal
  - rustup component add llvm-tools --toolchain stable-x86_64-pc-windows-msvc

出力:
  rust/target/pgo-use/x86_64-pc-windows-msvc/release/quoridor.exe
  リポジトリ直下の quoridor_rs.pyd
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TARGET = "x86_64-pc-windows-msvc"
TOOLCHAIN = f"stable-{TARGET}"
PGO_DIR = HERE / "target" / "pgo"

# プロファイルを集めるために実行する処理（探索・合法手生成・並列の自己対局）
WORKLOAD = [
    ["bench", "--depth", "9"],
    ["bench", "--depth", "8", "--threads", "4"],
    ["perft", "4", "--bulk"],
    ["perft", "3"],
    ["selfplay", "--games", "16", "--jobs", "8", "--a", "depth=6", "--b", "time=30,threads=2"],
]


def tool(name):
    p = shutil.which(name)
    if p:
        return p
    cand = Path.home() / ".cargo" / "bin" / f"{name}.exe"
    if cand.exists():
        return str(cand)
    sys.exit(f"{name} が見つかりません。Rust をインストールしてください: https://rustup.rs/")


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def build(cargo, rustflags, target_dir, *extra):
    env = dict(os.environ)
    # RUSTFLAGS を設定すると .cargo/config.toml の rustflags は使われないので、ここでも native を指定する
    env["RUSTFLAGS"] = " ".join(["-C", "target-cpu=native"] + rustflags)
    run([cargo, f"+{TOOLCHAIN}", "build", "--release", "--target", TARGET, "--target-dir", str(target_dir), *extra],
        cwd=HERE, env=env)
    return Path(target_dir) / TARGET / "release"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true", help="PGO なしの版と速度を比べる")
    args = ap.parse_args()
    if os.name != "nt":
        sys.exit("このスクリプトは Windows（MSVC ツールチェーン）用です。")

    cargo, rustup = tool("cargo"), tool("rustup")
    installed = subprocess.run([rustup, "toolchain", "list"], capture_output=True, text=True).stdout
    if TOOLCHAIN not in installed:
        sys.exit(f"{TOOLCHAIN} がありません: rustup toolchain install {TOOLCHAIN} --profile minimal")
    profdata = Path.home() / ".rustup" / "toolchains" / TOOLCHAIN / "lib" / "rustlib" / TARGET / "bin" / "llvm-profdata.exe"
    if not profdata.exists():
        sys.exit(f"llvm-profdata がありません: rustup component add llvm-tools --toolchain {TOOLCHAIN}")

    raw = PGO_DIR / "raw"
    shutil.rmtree(PGO_DIR, ignore_errors=True)
    raw.mkdir(parents=True)
    merged = PGO_DIR / "merged.profdata"

    # 1. 計測用ビルド → 典型的な処理を実行してプロファイルを集める
    gen = build(cargo, [f"-Cprofile-generate={raw.as_posix()}"], HERE / "target" / "pgo-gen", "--bin", "quoridor")
    for w in WORKLOAD:
        run([str(gen / "quoridor.exe"), *w], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 2. プロファイルをまとめる
    run([str(profdata), "merge", "-o", str(merged), str(raw)])

    # 3. プロファイルを使ってビルド（ライブラリ = Python 拡張 と コマンドライン）
    out = build(cargo, [f"-Cprofile-use={merged.as_posix()}"], HERE / "target" / "pgo-use")
    shutil.copyfile(out / "quoridor_rs.dll", ROOT / "quoridor_rs.pyd")
    print(f"built: {out / 'quoridor.exe'}")
    print(f"built: {ROOT / 'quoridor_rs.pyd'}")

    if args.compare:
        plain = build(cargo, [], HERE / "target" / "plain-msvc", "--bin", "quoridor")
        for label, exe in (("PGO なし", plain / "quoridor.exe"), ("PGO あり", out / "quoridor.exe")):
            for w in (["bench", "--depth", "9"], ["perft", "4", "--bulk"]):
                r = subprocess.run([str(exe), *w], capture_output=True, text=True, encoding="utf-8").stdout
                print(f"  {label}: {r.strip().splitlines()[-1]}")


if __name__ == "__main__":
    main()
