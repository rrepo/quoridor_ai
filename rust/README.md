# quoridor_rs — コリドールのルールエンジン（Rust）

`game/`（Python 版）と同じアルゴリズムを Rust で実装したもの。
盤面・指し手の実行と取消・全合法手の生成・最短経路・Zobrist ハッシュ・perft を提供する。
Python からは拡張モジュール `quoridor_rs` として使える。

## ビルド

必要なもの: Rust（`rustup` の `x86_64-pc-windows-gnu` ツールチェーン）。依存クレートはない。

```sh
# Python 拡張（リポジトリ直下に quoridor_rs.pyd を作る）
python rust/build_ext.py

# Rust のテスト（参照実装との突き合わせ・perft の既知値）
cd rust && cargo test --release

# Python 版 game/ との突き合わせ（生成順・経路まで一致するか）
python rust/test_python.py

# ベンチマーク（positions.txt は 1 行 1 局面、初期局面からの手番号をスペース区切り）
cd rust && QUORIDOR_POSITIONS=positions.txt cargo test --release --test reference bench -- --ignored --nocapture
```

`cargo` に PATH が通っていない場合、`build_ext.py` は `%USERPROFILE%\.cargo\bin` を自動で探す。

### Smart App Control について

この PC では Smart App Control が有効で、ビルド中に実行される依存クレートのビルドスクリプトや
新しい実行ファイルがブロックされることがある。そのため:

- PyO3 は使わず、CPython の安定 ABI（Limited API）を直接呼ぶ実装にしている（`src/capi.rs`）。
  関数はインポート時に読み込み済みの `python3XX.dll` から取得するので、import ライブラリも
  ビルドスクリプトも不要。
- ベンチマークは examples ではなく `tests/reference.rs` 内の `#[ignore]` テストにしている。

作り直した `.pyd` / 実行ファイルが将来ブロックされる可能性はある（判定はファイルごと）。

## Python API

```python
import quoridor_rs as q

b = q.Board()                      # Board(seed=...) で Zobrist テーブルを seed から作る
b.make_move(("move", 13))          # タプル形式（既存 Python 版と同じ）
b.make_move(67)                    # 整数の手番号: 0〜80 コマ / 81〜144 水平壁 / 145〜208 垂直壁
b.make_move(("hwall", 3, 3))       # 非合法手は ValueError（check=False で検査を省略）
b.undo_move()

b.legal_actions()                  # 全合法手（手番号のリスト）
b.legal_moves()                    # 全合法手（タプルのリスト）
b.count_legal_actions()
b.action_mask()                    # 209 バイトの bytes（np.frombuffer(..., np.uint8)）
b.to_planes()                      # (4, 9, 9) float32 の bytes（自分・相手・横壁・縦壁）
b.shortest_path(0), b.shortest_path_nodes(1)
b.pawn_dest_mask(), b.legal_wall_masks(), b.valid_wall_masks(), b.path_cut_masks(0)
b.winner(), b.is_terminal(), b.repetitions()
b.make_pass()                      # ヌルムーブ（undo_move で取り消せる）
b.perft(4, bulk=True)              # 計算中は GIL を解放する
b.turn, b.positions, b.walls_left, b.h_walls, b.v_walls, b.hmask, b.vmask, b.zobrist, b.ply
b.clone(); copy.deepcopy(b)
q.Board.from_state((40, 49), walls_left=(3, 0), turn=1, hmask=0, vmask=0)
q.encode_move(("vwall", 7, 7)), q.decode_move(208)
```

Python からの呼び出しには 1 回あたり数十 ns のオーバーヘッドがある。探索のように細かい呼び出しを
大量に繰り返す処理は、Python のループではなく Rust 側に置くと最も速い。

## 実装の要点

- 通行可能な辺を方向別ビットボード `open_d` / `open_r`（u128）で持ち、BFS の 1 層をシフト演算 4 回で展開する。
- 壁は u64 のマスク。重なり・交差の判定は 64 通りまとめてビット演算で行う。
- 壁の格子点のつながりをビットボードの塗りつぶしで求め、**閉路を作る壁**（同じ成分に 2 点以上で接する壁）だけを
  経路チェックの対象にする。閉路を作らない壁は誰も孤立させられない。
- 各プレイヤーの「ゴールへ通じる経路」を make/undo で差分管理し、その経路を切る壁だけ BFS で確認する。

## 速度（この PC、実戦局面の平均）

| 処理 | Python 版 (`game/`) | Rust |
|---|---|---|
| 全合法手の生成 | 53 µs | 0.25 µs |
| make + undo（壁） | 1.35 µs | 11 ns |
| 最短距離 ×2 | 13.3 µs | 95 ns |
| perft(3) 初期局面（2,062,264 ノード） | 4.65 s | 0.027 s |
| perft(4) 初期局面（247,569,030、bulk） | — | 0.16 s |
