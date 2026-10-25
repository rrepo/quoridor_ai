# ------------------------------------------------------------------
# test_game.py — 修正版
# ------------------------------------------------------------------

import sys
import traceback
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from game.board import Board, BOARD_SIZE, NODE_COUNT
from game.pathfinding import shortest_path, shortest_path_nodes
from game.wall import is_valid_wall_placement, place_hwall, place_vwall
from game.movegen import pawn_moves, wall_moves, legal_moves
from ai.evaluation import evaluate
from game.search import is_terminal
from ai.search import alphabeta, best_move, clear_tt
from game.game import QuoridorGame

# ------------------------------------------------------------------ helpers --
section_times = {}

def node(col, row):
    return row * BOARD_SIZE + col

def fresh():
    return Board()

def board_with(p0, p1):
    b = Board()
    b.players[0].pos = p0
    b.players[1].pos = p1
    return b

_passed = 0
_failed = 0
_failures = []

def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}", flush=True)
    else:
        _failed += 1
        msg = f"{name}" + (f"  [{detail}]" if detail else "")
        _failures.append(msg)
        print(f"  FAIL  {msg}", flush=True)

def run(name, fn):
    global _passed, _failed
    start = time.time()
    try:
        fn()
        _passed += 1
        print(f"  PASS  {name}", flush=True)
    except Exception as e:
        _failed += 1
        msg = f"{name}  [{type(e).__name__}: {e}]"
        _failures.append(msg)
        print(f"  FAIL  {msg}", flush=True)
    finally:
        end = time.time()
        section_times[name] = end - start

def section(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}", flush=True)

# ==================================================================
# ここから既存のテストコードをそのままコピー
# （省略、質問の内容と同じコード）
# ==================================================================


# ==================================================================
# 1. Board — 初期化
# ==================================================================
section("1. Board — 初期化")

b = fresh()
check("P0 初期位置 = node(4,0) = 4",  b.players[0].pos == 4)
check("P1 初期位置 = node(4,8) = 76", b.players[1].pos == 76)
check("P0 壁数 = 10", b.players[0].walls == 10)
check("P1 壁数 = 10", b.players[1].walls == 10)
check("初期ターン = 0", b.turn == 0)
check("角ノード(0,0) のエッジ数 = 2", len(b.edges[0]) == 2)
check("辺ノード(1,0) のエッジ数 = 3", len(b.edges[1]) == 3)
check("中央ノード(4,4) のエッジ数 = 4", len(b.edges[node(4, 4)]) == 4)
check("Zobrist 初期値 != 0", b.zobrist != 0)

sym_ok = all(n in b.edges[nb] for n in range(NODE_COUNT) for nb in b.edges[n])
check("全エッジが対称", sym_ok)


# ==================================================================
# 2. Board — make_move / undo_move
# ==================================================================
section("2. Board — make_move / undo_move")

b = fresh()
z0 = b.zobrist
b.make_move(("move", 13))
check("コマ移動後 P0.pos == 13", b.players[0].pos == 13)
check("コマ移動後 turn == 1",    b.turn == 1)
check("コマ移動で Zobrist が変化", b.zobrist != z0)
b.undo_move()
check("undo後 P0.pos == 4",      b.players[0].pos == 4)
check("undo後 turn == 0",        b.turn == 0)
check("undo後 Zobrist が元に戻る", b.zobrist == z0)

b = fresh()
z0 = b.zobrist
b.make_move(("hwall", 3, 3))
check("hwall後 walls == 9",       b.players[0].walls == 9)
check("hwall後 h_walls に追加",   (3, 3) in b.h_walls)
check("hwall で Zobrist が変化",  b.zobrist != z0)
b.undo_move()
check("undo後 walls == 10",       b.players[0].walls == 10)
check("undo後 h_walls から削除",  (3, 3) not in b.h_walls)
check("undo後 Zobrist が元に戻る", b.zobrist == z0)

b = fresh()
z0 = b.zobrist
b.make_move(("vwall", 2, 2))
check("vwall後 walls == 9",       b.players[0].walls == 9)
check("vwall後 v_walls に追加",   (2, 2) in b.v_walls)
check("vwall で Zobrist が変化",  b.zobrist != z0)
b.undo_move()
check("undo後 walls == 10",       b.players[0].walls == 10)
check("undo後 v_walls から削除",  (2, 2) not in b.v_walls)
check("undo後 Zobrist が元に戻る", b.zobrist == z0)

b1, b2 = fresh(), fresh()
b1.make_move(("hwall", 1, 1))
b2.make_move(("hwall", 2, 2))
check("異なる壁設置 → 異なる Zobrist", b1.zobrist != b2.zobrist)

b = fresh()
b.make_move(("move", 13))
b.make_move(("move", 67))
check("2手後 turn == 0", b.turn == 0)

b = fresh()
z0 = b.zobrist
seq = [
    ("move", 13), ("move", 67),
    ("hwall", 2, 2), ("move", 22),
    ("vwall", 5, 5), ("move", 58),
    ("move", 31), ("move", 49),
    ("hwall", 1, 6), ("move", 40),
]
for mv in seq:
    b.make_move(mv)
for _ in seq:
    b.undo_move()
check("10手 make/undo後 P0.pos が復元",   b.players[0].pos == 4)
check("10手 make/undo後 P1.pos が復元",   b.players[1].pos == 76)
check("10手 make/undo後 h_walls が空",    len(b.h_walls) == 0)
check("10手 make/undo後 v_walls が空",    len(b.v_walls) == 0)
check("10手 make/undo後 Zobrist が復元",  b.zobrist == z0)


# ==================================================================
# 3. Board — 壁エッジ apply / undo
# ==================================================================
section("3. Board — 壁エッジ apply / undo")

b = fresh()
tl, tr = node(3, 3), node(4, 3)
bl, br = node(3, 4), node(4, 4)

b.apply_hwall(3, 3)
check("hwall: tl↔bl が切断", bl not in b.edges[tl])
check("hwall: tr↔br が切断", br not in b.edges[tr])
b.undo_hwall(3, 3)
check("hwall undo: tl↔bl が復元", bl in b.edges[tl])
check("hwall undo: tr↔br が復元", br in b.edges[tr])

b.apply_vwall(3, 3)
check("vwall: tl↔tr が切断", tr not in b.edges[tl])
check("vwall: bl↔br が切断", br not in b.edges[bl])
b.undo_vwall(3, 3)
check("vwall undo: tl↔tr が復元", tr in b.edges[tl])
check("vwall undo: bl↔br が復元", br in b.edges[bl])


# ==================================================================
# 4. pathfinding
# ==================================================================
section("4. pathfinding")

b = fresh()
check("P0 初期最短距離 = 8", shortest_path(b, 0) == 8)
check("P1 初期最短距離 = 8", shortest_path(b, 1) == 8)

b.players[0].pos = node(4, 8)
check("P0 ゴール到達時 距離 = 0", shortest_path(b, 0) == 0)

b = fresh()
path = shortest_path_nodes(b, 0)
check("path_nodes: リスト型",             isinstance(path, list))
check("path_nodes: 長さ = 9 (8ステップ)", len(path) == 9)
check("path_nodes: 先頭 = P0.pos",        path[0] == b.players[0].pos)
check("path_nodes: 末尾 = goal row 8",    path[-1] // BOARD_SIZE == 8)
connected = all(path[i + 1] in b.edges[path[i]] for i in range(len(path) - 1))
check("path_nodes: 全ノードがエッジで連結", connected)

b = fresh()
for col in range(BOARD_SIZE):
    b.remove_edge(node(col, 0), node(col, 1))
check("完全封鎖時 shortest_path = None", shortest_path(b, 0) is None)

b = fresh()
b.apply_hwall(0, 0)
b.apply_hwall(2, 0)
b.apply_hwall(4, 0)
b.apply_hwall(6, 0)
dist = shortest_path(b, 0)
check("壁4枚で迂回 → 距離が 8 より大きい", dist is not None and dist > 8)


# ==================================================================
# 5. wall — is_valid_wall_placement
# ==================================================================
section("5. wall — is_valid_wall_placement")

b = fresh()
check("(3,3) hwall: 有効",             is_valid_wall_placement(b, 3, 3, 'h'))
check("(3,3) vwall: 有効",             is_valid_wall_placement(b, 3, 3, 'v'))
check("(7,7) hwall 境界: 有効",        is_valid_wall_placement(b, 7, 7, 'h'))
check("(8,3) hwall 範囲外: 無効",      not is_valid_wall_placement(b, 8, 3, 'h'))
check("(-1,3) hwall 範囲外: 無効",     not is_valid_wall_placement(b, -1, 3, 'h'))

b.h_walls.add((3, 3))
check("同座標 hwall 重複: 無効",        not is_valid_wall_placement(b, 3, 3, 'h'))
check("x+1 隣接 hwall 重複: 無効",     not is_valid_wall_placement(b, 4, 3, 'h'))
check("x-1 隣接 hwall 重複: 無効",     not is_valid_wall_placement(b, 2, 3, 'h'))
check("同座標 hwall/vwall 交差: 無効", not is_valid_wall_placement(b, 3, 3, 'v'))
b.h_walls.discard((3, 3))

b.v_walls.add((3, 3))
check("同座標 vwall 重複: 無効",        not is_valid_wall_placement(b, 3, 3, 'v'))
check("y+1 隣接 vwall 重複: 無効",     not is_valid_wall_placement(b, 3, 4, 'v'))
check("y-1 隣接 vwall 重複: 無効",     not is_valid_wall_placement(b, 3, 2, 'v'))
check("同座標 vwall/hwall 交差: 無効", not is_valid_wall_placement(b, 3, 3, 'h'))
b.v_walls.discard((3, 3))


# ==================================================================
# 6. wall — place_hwall / place_vwall
# ==================================================================
section("6. wall — place_hwall / place_vwall")

b = fresh()
check("place_hwall(3,3): 成功",         place_hwall(b, 3, 3))
check("place_hwall後 h_walls に追加",   (3, 3) in b.h_walls)
check("place_hwall後 walls == 9",       b.players[0].walls == 9)
check("place_hwall後 エッジ切断済み",   node(3, 4) not in b.edges[node(3, 3)])
b.turn ^= 1
check("同座標 hwall 再設置: 失敗",      not place_hwall(b, 3, 3))

b = fresh()
check("place_vwall(3,3): 成功",         place_vwall(b, 3, 3))
check("place_vwall後 v_walls に追加",   (3, 3) in b.v_walls)
check("place_vwall後 walls == 9",       b.players[0].walls == 9)
check("place_vwall後 エッジ切断済み",   node(4, 3) not in b.edges[node(3, 3)])

b = fresh()
b.players[0].walls = 0
check("壁残数0 で place_hwall: 失敗",   not place_hwall(b, 3, 3))
check("壁残数0 で place_vwall: 失敗",   not place_vwall(b, 3, 3))

b = fresh()
for col in range(BOARD_SIZE):
    b.remove_edge(node(col, 0), node(col, 1))
result = place_hwall(b, 0, 0)
check("封鎖済み盤面への place_hwall: 設置後も経路保証",
      not result or shortest_path(b, 0) is not None)


# ==================================================================
# 7. movegen — 通常移動
# ==================================================================
section("7. movegen — 通常移動")

b = fresh()
dests = {m[1] for m in pawn_moves(b)}
check("P0 初期位置(4,0) の移動先 = {3,5,13}", dests == {3, 5, 13}, f"got {dests}")

b = board_with(node(4, 4), node(4, 8))
dests = {m[1] for m in pawn_moves(b)}
expected = {node(3, 4), node(5, 4), node(4, 3), node(4, 5)}
check("中央(4,4) の移動先 = 4方向", dests == expected, f"got {dests}")

b = fresh()
b.apply_hwall(3, 0)
b.h_walls.add((3, 0))
dests = {m[1] for m in pawn_moves(b)}
check("hwall でブロック → node(4,1) に移動不可", node(4, 1) not in dests)


# ==================================================================
# 8. movegen — 直進ジャンプ
# ==================================================================
section("8. movegen — 直進ジャンプ")

b = board_with(node(4, 3), node(4, 4))
dests = {m[1] for m in pawn_moves(b)}
check("直進ジャンプ(下): (4,5) に跳べる",    node(4, 5) in dests, f"got {dests}")
check("直進ジャンプ: 相手マス(4,4) は不可",  node(4, 4) not in dests)

b = board_with(node(4, 5), node(4, 4))
dests = {m[1] for m in pawn_moves(b)}
check("直進ジャンプ(上): (4,3) に跳べる", node(4, 3) in dests)

b = board_with(node(3, 4), node(4, 4))
dests = {m[1] for m in pawn_moves(b)}
check("直進ジャンプ(右): (5,4) に跳べる", node(5, 4) in dests)

b = board_with(node(5, 4), node(4, 4))
dests = {m[1] for m in pawn_moves(b)}
check("直進ジャンプ(左): (3,4) に跳べる", node(3, 4) in dests)

b = board_with(node(4, 7), node(4, 8))
check("ボード端: row=9 への直進ジャンプ不可",
      all(m[1] // BOARD_SIZE < BOARD_SIZE for m in pawn_moves(b)))


# ==================================================================
# 9. movegen — 側面ジャンプ（壁絡み）
# ==================================================================
section("9. movegen — 側面ジャンプ（壁絡み）")

b = board_with(node(4, 3), node(4, 4))
b.apply_hwall(3, 4)
b.h_walls.add((3, 4))
dests = {m[1] for m in pawn_moves(b)}
check("直進先に壁 → (4,5) には跳べない",          node(4, 5) not in dests)
check("直進先に壁 → 側面(3,4) or (5,4) に跳べる",
      node(3, 4) in dests or node(5, 4) in dests, f"got {dests}")
check("側面ジャンプ: 自マス(4,3) は不可",          node(4, 3) not in dests)
check("両側面ともエッジあり → 両方候補",
      node(3, 4) in dests and node(5, 4) in dests, f"got {dests}")

b2 = board_with(node(4, 3), node(4, 4))
b2.apply_hwall(3, 4);  b2.h_walls.add((3, 4))
b2.apply_vwall(4, 4);  b2.v_walls.add((4, 4))
dests2 = {m[1] for m in pawn_moves(b2)}
check("右側面を vwall 封鎖 → (5,4) に跳べない", node(5, 4) not in dests2)
check("右側面を vwall 封鎖 → (3,4) には跳べる", node(3, 4) in dests2)

b3 = board_with(node(0, 7), node(0, 8))
dests3 = {m[1] for m in pawn_moves(b3)}
check("端コーナーの相手 → (1,8) にのみ側面ジャンプ", node(1, 8) in dests3)

b4 = board_with(node(4, 3), node(4, 4))
b4.apply_hwall(3, 3)
b4.h_walls.add((3, 3))
dests4 = {m[1] for m in pawn_moves(b4)}
check("壁で隣接が消えた相手マスへは移動不可", node(4, 4) not in dests4)


# ==================================================================
# 10. movegen — wall_moves
# ==================================================================
section("10. movegen — wall_moves")

b = fresh()
wm = wall_moves(b)
check("初期局面で wall_moves が空でない", len(wm) > 0)
check("wall_moves の手は hwall or vwall",
      all(m[0] in ("hwall", "vwall") for m in wm))

b = fresh()
b.players[0].walls = 0
check("壁残数0 → wall_moves が空", wall_moves(b) == [])

b = fresh()
path_block = False
for mv in wall_moves(b):
    b.make_move(mv)
    if shortest_path(b, 0) is None or shortest_path(b, 1) is None:
        path_block = True
    b.undo_move()
check("wall_moves の全手が経路を保持", not path_block)

b = fresh()
types = {m[0] for m in legal_moves(b)}
check("legal_moves に 'move' が含まれる",              "move"  in types)
check("legal_moves に 'hwall' or 'vwall' が含まれる",
      "hwall" in types or "vwall" in types)


# ==================================================================
# 11. evaluation
# ==================================================================
section("11. evaluation")

b = fresh()
check("開始時 evaluate == 0.0 (対称局面)", evaluate(b) == 0.0)

b = fresh()
b.players[0].pos = node(4, 7)
check("P0 がゴール近く → 正のスコア", evaluate(b) > 0)

b = fresh()
b.players[0].walls = 10
b.players[1].walls = 5
score = evaluate(b)
check("壁ボーナス: walls差5 → score == 2.5", abs(score - 2.5) < 1e-9, f"got {score}")


# ==================================================================
# 12. search — is_terminal
# ==================================================================
section("12. search — is_terminal")

b = fresh()
t, _ = is_terminal(b)
check("開始時 is_terminal == False", not t)

b = fresh()
b.players[0].pos = node(4, 8)
t, w = is_terminal(b)
check("P0 が row=8 → terminal, winner=0", t and w == 0)

b = fresh()
b.players[1].pos = node(4, 0)
t, w = is_terminal(b)
check("P1 が row=0 → terminal, winner=1", t and w == 1)

b = fresh()
b.players[0].pos = node(4, 5)
t, _ = is_terminal(b)
check("P0 が row=5 → not terminal", not t)


# ==================================================================
# 13. search — alphabeta / best_move
# ==================================================================
section("13. search — alphabeta / best_move")

b = fresh()
mv = best_move(b, depth=2)
check("best_move が None でない", mv is not None)
check("best_move が legal_moves の中にある", mv in set(legal_moves(b)))

b = fresh()
b.players[0].pos = node(4, 7)
b.players[1].pos = node(4, 0)
mv = best_move(b, depth=1)
check("1手勝ち局面 → best_move が (4,8) への移動",
      mv == ("move", node(4, 8)), f"got {mv}")

b = fresh()
b.players[0].pos = node(4, 8)
score = alphabeta(b, depth=0, alpha=-99999, beta=99999)
check("P0 ゴール到達時 alphabeta > 0", score > 0)

b = fresh()
b.players[1].pos = node(4, 0)
score = alphabeta(b, depth=0, alpha=-99999, beta=99999)
check("P1 ゴール到達時 alphabeta < 0 (P0視点)", score < 0)

b = fresh()
z0 = b.zobrist
p0, p1 = b.players[0].pos, b.players[1].pos
best_move(b, depth=3)
check("best_move後 Zobrist が不変",  b.zobrist == z0)
check("best_move後 P0.pos が不変",   b.players[0].pos == p0)
check("best_move後 P1.pos が不変",   b.players[1].pos == p1)


# ==================================================================
# 14. QuoridorGame
# ==================================================================
section("14. QuoridorGame")

def t_game_init():
    g = QuoridorGame()
    assert g.board.players[0].pos == 4
    assert g.board.players[1].pos == 76

def t_game_str():
    g = QuoridorGame()
    s = str(g)
    assert "0" in s and "1" in s

def t_ai_move():
    g = QuoridorGame()
    mv = g.ai_move(depth=2)
    assert mv is not None
    assert g.board.turn == 1

def t_full_game():
    g = QuoridorGame()
    for _ in range(300):
        t, _ = is_terminal(g.board)
        if t:
            break
        g.ai_move(depth=1)

run("QuoridorGame 初期化",              t_game_init)
run("QuoridorGame.__str__",             t_game_str)
run("ai_move depth=2",                  t_ai_move)
run("フルゲーム (depth=1, 最大300手)", t_full_game)


# ==================================================================
# 15. 複合シナリオ
# ==================================================================
section("15. 複合シナリオ")

b = board_with(node(4, 3), node(4, 4))
b.make_move(("hwall", 3, 4))
dests = {m[1] for m in pawn_moves(b)}
check("壁設置後 P1 が P0 を飛び越せる (4,2)",
      node(4, 2) in dests, f"got {dests}")

b = board_with(node(4, 3), node(4, 4))
b.apply_hwall(3, 3)
b.h_walls.add((3, 3))
check("壁で隣接が消えた相手への移動は不可",
      node(4, 4) not in {m[1] for m in pawn_moves(b)})

b = fresh()
b.players[0].walls = 0
check("壁0枚 → legal_moves が全て 'move'",
      all(m[0] == "move" for m in legal_moves(b)))

b1, b2 = fresh(), fresh()
b1.make_move(("move", 13))
b2.make_move(("move", 3))
check("異なる局面 → 異なる Zobrist", b1.zobrist != b2.zobrist)

b = fresh()
b.apply_hwall(0, 0)
b.apply_hwall(2, 0)
b.apply_hwall(4, 0)
b.apply_hwall(6, 0)
dist = shortest_path(b, 0)
check("左8列封鎖 → 迂回で距離 > 8", dist is not None and dist > 8)


# ==================================================================
# 結果サマリ
# ==================================================================
total = _passed + _failed
print("\n" + "=" * 60)
print(f"  結果: {_passed} / {total} passed  " + ("OK" if _failed == 0 else "FAIL"))
if _failures:
    print(f"\n  失敗したテスト ({_failed}件):")
    for f in _failures:
        print(f"    - {f}")
print("=" * 60)


# ------------------------------------------------------------------
# 結果サマリ
# ------------------------------------------------------------------
total = _passed + _failed
print("\n" + "=" * 60, flush=True)
print(f"  結果: {_passed} / {total} passed  " + ("OK" if _failed == 0 else "FAIL"), flush=True)
if _failures:
    print(f"\n  失敗したテスト ({_failed}件):", flush=True)
    for f in _failures:
        print(f"    - {f}", flush=True)
print("=" * 60, flush=True)

print("\n" + "=" * 60, flush=True)
print("  ベンチマーク結果（各 run() の実行時間）", flush=True)
for name, t in section_times.items():
    print(f"    {name:40s}: {t:.6f} 秒", flush=True)
print("=" * 60, flush=True)

sys.exit(0 if _failed == 0 else 1)