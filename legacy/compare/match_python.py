"""
Rust 版 AI（quoridor_rs.Board.search）と Python 版 AI（legacy/ai/search.py）を対局させる。

    python scripts/build_ext.py
    python legacy/compare/match_python.py <序盤の数> <Rust の深さ> <Python の深さ>
    （Rust の深さに負の値 -N を渡すと「1 手 N ミリ秒」の時間制限になる）

序盤 4 手をランダムに指した局面から、先後を入れ替えて 2 局ずつ対局する。
"""
import sys, random, time, multiprocessing as mp
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]     # リポジトリ直下（quoridor_rs.pyd）
sys.path[:0] = [str(_ROOT), str(_ROOT / "legacy")]

def play(args):
    opening_seed, rust_first, depth_rs, depth_py = args
    import quoridor_rs as q
    from game.board import Board
    from game.movegen import pawn_moves
    from ai.search import best_move, clear_tt
    from ai.cache import clear_dist_cache
    rng = random.Random(opening_seed)
    pb, rb = Board(), q.Board()
    for _ in range(4):
        m = rng.choice(pawn_moves(pb)); pb.make_move(m); rb.make_move(m)
    clear_tt(); q.clear_search()
    t_rs = t_py = 0.0
    for ply in range(200):
        if rb.winner() is not None:
            w = rb.winner()
            return ((w == 0) == rust_first, t_rs, t_py, ply)
        rust_turn = (rb.turn == 0) == rust_first
        t = time.perf_counter()
        if rust_turn:
            m = (rb.search(time_ms=-depth_rs)[0] if depth_rs < 0 else rb.search(depth_rs)[0])
            t_rs += time.perf_counter() - t
        else:
            clear_dist_cache()
            m = best_move(pb, depth_py)
            t_py += time.perf_counter() - t
        pb.make_move(m); rb.make_move(m)
    return (None, t_rs, t_py, 200)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    n, d_rs, d_py = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    tasks = [(1000 + i, rf, d_rs, d_py) for i in range(n) for rf in (True, False)]
    with mp.Pool(14) as pool:
        res = pool.map(play, tasks)
    rs = sum(1.0 if r[0] else 0.5 if r[0] is None else 0.0 for r in res)
    t_rs = sum(r[1] for r in res); t_py = sum(r[2] for r in res)
    print(f"Rust {'time=%dms' % -d_rs if d_rs < 0 else 'd=%d' % d_rs} vs Python d={d_py}: Rust {rs} - Python {len(res) - rs}  ({100*rs/len(res):.1f}%)  "
          f"think: Rust {t_rs:.1f}s  Python {t_py:.1f}s  ({t_py/max(t_rs,1e-9):.0f}x)")
