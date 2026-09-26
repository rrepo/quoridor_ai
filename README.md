# quoridor_rs — コリドールのルールエンジンと探索（Rust）

コリドールのルールエンジンと AI（αβ 探索）。Rust で実装している。
以前の Python 版は [legacy/](legacy/) に参照用として残している（Rust 版の正しさの検証にも使う）。

- ルール: 盤面・指し手の実行と取消・全合法手・最短経路・Zobrist ハッシュ・perft
- 探索: 反復深化 + aspiration window、PVS、LMR、ヌルムーブ、置換表、キラー手・history
  （`legacy/ai/search.py` の移植）。**複数スレッド（Lazy SMP）** と **時間制限** に対応
- 使い方: コマンドライン（`quoridor`）/ Rust ライブラリ / Python 拡張 `quoridor_rs`

## ビルドと実行

必要なもの: Rust（`rustup` の `x86_64-pc-windows-gnu` ツールチェーン）。依存クレートはない。
`.cargo/config.toml` で `target-cpu=native`（この PC の CPU 向け最適化）を有効にしている。

```sh
cargo build --release

# perft（全合法手の木の葉の数）
target/release/quoridor perft 4 --bulk --threads 8

# 探索: 局面は初期局面からの手（m13 = コマを 13 へ、h3,3 / v5,5 = 水平壁 / 垂直壁）
target/release/quoridor search --depth 10 --threads 8 m13 m67 h3,3 v5,5
target/release/quoridor search --time 500 --threads 8 m13 m67

# 自己対局: 2 つの設定を先後入れ替えで対局（--jobs で対局自体を並列化）
target/release/quoridor selfplay --games 40 --jobs 8 --a depth=6 --b time=50
target/release/quoridor selfplay --games 40 --a time=50,threads=8 --b time=50,threads=1

# 探索速度のベンチマーク
target/release/quoridor bench --depth 9 --threads 1
```

### テスト

```sh
cargo test --release                          # 参照実装との突き合わせ・perft の既知値・探索
python scripts/build_ext.py                   # Python 拡張をビルド（リポジトリ直下に quoridor_rs.pyd。PGO 版は scripts/pgo.py）
python legacy/compare/test_python.py          # Python 版 legacy/game/ との突き合わせ（生成順・経路まで一致）
python legacy/compare/match_python.py 40 4 4  # Rust 版 AI と Python 版 AI の対局（同じ深さ）
```

計測用のカウンタ（BFS の回数や処理ごとの CPU サイクル）は `--features stats` で有効になる:
`cargo build --release --bin quoridor --features stats && target/release/quoridor bench`

## Python API

```python
import quoridor_rs as q

b = q.Board()
b.make_move(("move", 13)); b.make_move(67); b.make_move(("hwall", 3, 3))
b.undo_move()
b.legal_actions(); b.legal_moves(); b.count_legal_actions(); b.action_mask(); b.to_planes()
b.shortest_path(0); b.shortest_path_nodes(1); b.winner(); b.is_terminal(); b.repetitions()

mv, score, depth, nodes = b.search(depth=6)                 # 深さ指定
mv, score, depth, nodes = b.search(time_ms=200, threads=8)  # 時間制限 + 並列
q.clear_search()                                            # 置換表などを消す（対局の開始時）

b.perft(4, bulk=True, threads=8)
```

`search` の置換表・キラー手・history は呼び出し間で持ち越す（Python 版 `legacy/ai/search.py` と同じ）。
探索中と perft 中は GIL を解放する。

## ディレクトリ構成

```
Cargo.toml, .cargo/config.toml   Rust プロジェクト（target-cpu=native）
src/
  board.rs    盤面・make/undo・合法手・最短経路（ビットボード）
  consts.rs   定数と事前計算テーブル
  path.rs     ビットボード BFS
  zobrist.rs  Zobrist ハッシュ
  eval.rs     評価関数・壁の手順付け用スコア
  search.rs   αβ 探索（Lazy SMP・時間制限）・並列 perft
  tt.rs       置換表（ロックなし・スレッド共有）
  capi.rs     Python 拡張モジュール（CPython の安定 ABI を直接呼ぶ）
  stats.rs    計測用カウンタ（--features stats）
  bin/quoridor.rs  コマンドライン
tests/reference.rs   独立した参照実装との突き合わせ・perft・探索のテスト、ベンチマーク
scripts/
  build_ext.py  Python 拡張（quoridor_rs.pyd）のビルド
  pgo.py        PGO ビルド（MSVC ツールチェーン）
legacy/         Python 版（参照用）と、Rust 版との突き合わせスクリプト
```

## 実装の要点

- 通行可能な辺を方向別ビットボード `open_d` / `open_r`（u128）で持ち、BFS の 1 層をシフト演算 4 回で展開する。
- 壁は u64 のマスク。重なり・交差の判定は 64 通りまとめてビット演算で行う。
- 壁の格子点の連結成分を塗りつぶしで求め、閉路を作る壁だけを経路チェックの対象にする。
- 各プレイヤーの「ゴールへ通じる経路」を make/undo で差分管理し、その経路を切る壁だけ BFS で確認する。
- 探索: 置換表はロックなし（key ^ data 方式）で全スレッドが共有。どのスレッドでも最も深く探索し終えた結果を採用する。
  最短距離と最短経路の切断マスクはスレッドごとの小さなキャッシュ（64KB 程度）で使い回す。

### Python 版からの変更点（探索）

- 評価値は Python 版の 100 倍の整数（探索窓の幅 1 が正確な意味を持つ）。
- 壁の手順付けで向きを取り違えていたバグを修正（`wall_score` に `"hwall"` を渡して `'h'` と比較していたため、
  すべて垂直壁として評価していた。ルートでは候補集合のキーも一致せず、壁がすべて最低点になっていた）。
- 同じ深さでの対局は Python 版とほぼ互角（深さ 3〜5 で計 240 局、49.6%）。

## 速度（AMD Ryzen 7 3700X、8 コア 16 スレッド）

| 処理 | Python 版 | Rust |
|---|---|---|
| 全合法手の生成（実戦局面の平均） | 53 µs | 0.2 µs |
| perft(4) 初期局面（247,569,030、bulk） | — | 0.14 s（1 スレッド） |
| 探索速度 | 約 5 万ノード/秒 | 約 700 万ノード/秒（1 スレッド） |
| 深さ 10 の探索（中盤局面） | — | 5.6 s（1 スレッド）/ 2.2 s（8 スレッド） |
| 同じ思考時間での対局 | Python 深さ 4 | Rust 1 手 25 ms で 83% 勝ち（思考時間は約 1/3） |

## Smart App Control について

この PC では Smart App Control が有効で、ビルド中に実行される依存クレートのビルドスクリプトなどが
ブロックされることがある。そのため PyO3 は使わず、CPython の安定 ABI を直接呼ぶ実装にしている（`src/capi.rs`）。
関数はインポート時に読み込み済みの `python3XX.dll` から取得するので、import ライブラリもビルドスクリプトも不要。
作り直した実行ファイルや `.pyd` が将来ブロックされる可能性はある（判定はファイルごと）。

## PGO（プロファイルに基づく最適化）

```sh
python scripts/pgo.py            # quoridor.exe と quoridor_rs.pyd を PGO でビルド（コードを変えたら実行し直す）
python scripts/pgo.py --compare  # PGO なしの版と速度を比べる
```

計測用ビルドでベンチ・perft・自己対局を実行してプロファイルを集め、それを使って
`target/pgo-use/x86_64-pc-windows-msvc/release/quoridor.exe` と、リポジトリ直下の `quoridor_rs.pyd` を作る。
効果はこの PC で探索・perft とも 3〜5% 程度（1 ノードあたりの処理が既に小さいため控えめ）。

`x86_64-pc-windows-gnu` には PGO の計測用ランタイム（`profiler_builtins`）が含まれないため、MSVC 版を使う:

- Visual Studio Build Tools（「C++ によるデスクトップ開発」）
  `winget install --id Microsoft.VisualStudio.2022.BuildTools --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"`
- `rustup toolchain install stable-x86_64-pc-windows-msvc --profile minimal`
- `rustup component add llvm-tools --toolchain stable-x86_64-pc-windows-msvc`

`rustup` / `cargo` は `%USERPROFILE%\.cargo\bin` にある（PATH に追加していない場合はフルパスで呼ぶ）。
