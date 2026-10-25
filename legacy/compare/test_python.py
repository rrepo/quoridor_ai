"""
Rust 版（quoridor_rs）と Python 版（legacy/game/）をランダム対局で突き合わせる。

    python scripts/build_ext.py
    python legacy/compare/test_python.py
"""
import copy
import ctypes
import gc
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # リポジトリ直下（quoridor_rs.pyd）
LEGACY = ROOT / "legacy"                        # Python 版（game/, ai/）
sys.path[:0] = [str(ROOT), str(LEGACY)]

import quoridor_rs as q                                   # noqa: E402
from game.board import Board as PyBoard                   # noqa: E402
from game.movegen import all_legal_moves, pawn_dest_mask  # noqa: E402
from game.pathfinding import shortest_path, shortest_path_nodes  # noqa: E402
from game.wall import legal_wall_masks, path_cut_masks, valid_wall_masks  # noqa: E402


def check_same(r, p):
    assert r.positions == (p.players[0].pos, p.players[1].pos)
    assert r.walls_left == (p.players[0].walls, p.players[1].walls)
    assert r.turn == p.turn
    assert r.hmask == p.hmask and r.vmask == p.vmask
    assert sorted(r.h_walls) == sorted(p.h_walls) and sorted(r.v_walls) == sorted(p.v_walls)
    moves = all_legal_moves(p)
    assert r.legal_moves() == moves                       # 生成順も同じ
    assert r.legal_actions() == [PyBoard.encode_move(m) for m in moves]
    assert r.count_legal_actions() == len(moves)
    assert r.pawn_dest_mask() == pawn_dest_mask(p)
    assert r.legal_wall_masks() == legal_wall_masks(p)
    assert r.valid_wall_masks() == valid_wall_masks(p)
    mask = bytes(1 if a in set(r.legal_actions()) else 0 for a in range(209))
    assert r.action_mask() == mask
    for pl in (0, 1):
        assert r.shortest_path(pl) == shortest_path(p, pl)
        assert r.shortest_path_nodes(pl) == shortest_path_nodes(p, pl)
        assert r.path_cut_masks(pl) == path_cut_masks(p, pl)
    term = (p.players[0].pos // 9 == 8, 0) if p.players[0].pos // 9 == 8 else \
           (True, 1) if p.players[1].pos // 9 == 0 else (False, -1)
    assert r.is_terminal() == term
    return moves


def random_games(n_games=300, seed=11):
    rng = random.Random(seed)
    positions = 0
    for g in range(n_games):
        r, p = q.Board(), PyBoard()
        for ply in range(rng.randint(0, 70)):
            moves = check_same(r, p)
            # 非合法手は拒否され、盤面は変わらない
            legal = set(r.legal_actions())
            illegal = [a for a in range(209) if a not in legal]
            if illegal:
                a = rng.choice(illegal)
                try:
                    r.make_move(a)
                    raise AssertionError(f"illegal move accepted: {a}")
                except ValueError:
                    pass
            for a in range(209):
                assert r.is_legal(a) == (a in legal)
            walls = [m for m in moves if m[0] != "move"]
            m = rng.choice(walls) if walls and rng.random() < 0.45 else rng.choice(moves)
            r.make_move(m if rng.random() < 0.5 else q.encode_move(m))
            p.make_move(m)
            positions += 1
            if r.winner() is not None:
                break
        # 取り消しで一致したまま初期局面へ戻る
        while p.move_stack:
            assert r.undo_move()
            p.undo_move()
            if rng.random() < 0.1:
                check_same(r, p)
        assert not r.undo_move() and r.ply == 0
        assert r.zobrist == q.Board().zobrist
    return positions


def test_misc():
    b = q.Board()
    assert b.perft(1) == 131 and b.perft(2) == 16677 and b.perft(3, bulk=True) == 2062264
    for a in (13, 67, ("hwall", 3, 3), ("vwall", 5, 5), ("hwall", 4, 1), ("move", 58)):
        b.make_move(a)
    assert b.perft(3) == 1551223
    c, d, e = b.clone(), copy.copy(b), copy.deepcopy(b)
    for x in (c, d, e):
        assert x.zobrist == b.zobrist and x.legal_actions() == b.legal_actions()
    c.make_move(c.legal_actions()[0])
    assert c.zobrist != b.zobrist and b.ply == 6
    z, t = b.zobrist, b.turn
    b.make_pass()
    assert b.turn == 1 - t and b.zobrist != z
    assert b.undo_move() and b.zobrist == z and b.turn == t
    r = q.Board()
    for mv in (13, 67, 4, 76):
        r.make_move(mv)
    assert r.zobrist == q.Board().zobrist and r.repetitions() == 1
    for mv in (13, 67, 4, 76):
        r.make_move(mv)
    assert r.repetitions() == 2
    s = q.Board.from_state((40, 49), (3, 0), 1, hmask=1 << 10, vmask=1 << 20)
    assert s.positions == (40, 49) and s.walls_left == (3, 0) and s.turn == 1
    assert s.h_walls == [(2, 1)] and s.v_walls == [(4, 2)]
    assert all(a < 81 for a in s.legal_actions())          # P1 は壁 0 枚
    for bad in [((81, 0),), ((1, 1),), ((1, 2), (11, 0)), ((1, 2), (1, 1), 2)]:
        try:
            q.Board.from_state(*bad)
            raise AssertionError("bad state accepted")
        except ValueError:
            pass
    for bad in ("x", ("move",), ("hwall", 8, 0), ("jump", 1), 209, -1, 1.5):
        try:
            q.Board().make_move(bad)
            raise AssertionError(f"bad move accepted: {bad}")
        except (ValueError, TypeError):
            pass
    try:
        q.Board().shortest_path(2)
        raise AssertionError
    except IndexError:
        pass
    assert q.decode_move(0) == ("move", 0) and q.decode_move(208) == ("vwall", 7, 7)
    assert q.encode_move(("hwall", 7, 7)) == 144
    assert q.Board(seed=1).zobrist == q.Board(1).zobrist != q.Board().zobrist
    assert "Turn: Player 0" in str(q.Board()) and "Board(" in repr(q.Board())


def working_set_mb():
    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + \
                   [(n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                                  "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                                  "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
    ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return pmc.PagefileUsage / 2**20


def test_no_leak():
    def churn():
        for _ in range(20000):
            b = q.Board()
            for a in (13, 67, 108, 190):
                b.make_move(a)
            b.legal_moves(); b.legal_actions(); b.h_walls; b.positions; b.shortest_path_nodes(0)
            b.path_cut_masks(1); b.pawn_dest_mask(); b.action_mask(); b.to_planes(); b.clone()
            try:
                b.make_move(108)
            except ValueError:
                pass
    churn(); gc.collect()
    before = working_set_mb()
    for _ in range(5):
        churn()
    gc.collect()
    after = working_set_mb()
    assert after - before < 5, f"memory grew {before:.1f} -> {after:.1f} MB"
    t = q.decode_move(5)
    rc = sys.getrefcount(t)
    for _ in range(1000):
        q.Board().legal_moves()
    assert sys.getrefcount(t) == rc


if __name__ == "__main__":
    n = random_games()
    print(f"random games: {n} positions matched game/ (Python)")
    test_misc()
    print("misc: ok")
    test_no_leak()
    print("leak check: ok")
