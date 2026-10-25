# legacy — Python 版（参照用）

Rust 版（リポジトリ直下）に置き換える前の Python 実装。新しい開発は Rust 版で行う。
ここは参照用として残しており、Rust 版の正しさの検証（突き合わせ・対局）にも使っている。

- `game/` — ルールエンジン（ビットボード版）
- `ai/` — αβ 探索・評価関数（Rust 版 `src/search.rs` / `src/eval.rs` の移植元）
- `simulate_ai_match.py` — AI 同士の対局
- `test_board.py`, `test_game.py` — Python 版のテスト
- `mcts.py` — MCTS の試作（未完成。import が壊れていて動かない）
- `compare/` — Rust 版との比較
  - `test_python.py` — ランダム対局で Rust 版と Python 版の合法手・経路などを突き合わせる
  - `match_python.py` — Rust 版 AI と Python 版 AI を対局させる

```sh
python legacy/test_board.py
python legacy/test_game.py                     # 「壁ボーナス」の 1 件は元から失敗（評価の係数とテストの期待値の食い違い）
python legacy/simulate_ai_match.py --depth 3

python scripts/build_ext.py                    # 比較の前に Rust 版の Python 拡張をビルド
python legacy/compare/test_python.py
python legacy/compare/match_python.py 40 4 4
```

既知の問題（Python 版のみ。Rust 版では修正済み）:

- `ai/search.py` の壁の手順付けで、`wall_score` に `"hwall"` を渡して `'h'` と比較しているため、
  すべて垂直壁として評価している。ルートでは候補集合のキーも一致せず、壁がすべて最低点になる。
- `simulate_ai_match.py` の `--p0eval` / `--p1eval` は探索に反映されない（常に `ai/eval_func.py` を使う）。
