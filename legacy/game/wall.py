# wall.py
#
# 壁の合法性判定（すべてビットボード）
#
# 壁座標 w = y*8 + x（0〜63）。盤面の board.hmask / board.vmask を参照する。
#
# 合法性 = (1) 既存の壁と重ならない・交差しない
#          (2) 置いた後も両プレイヤーがゴールに到達できる
#
# (1) は 64 通りまとめてシフト演算で求まる（valid_wall_masks）。
# (2) は次の 2 段階の枝刈りの後、残ったものだけ BFS で確認する（legal_wall_masks）。
#   a. 壁の 3 つの格子点（両端・中点）のうち、盤端か既存の壁に接する点が 2 つ未満
#      なら閉路を作れない → 誰も孤立しない。これも 64 通りまとめて計算できる。
#      格子点は内部の 8x8 点を壁座標と同じ番号で表す（壁 w の中点 = 点 w）。
#   b. 現在の最短経路の辺を切らない壁は、そのプレイヤーの経路を断たない。
#
# 前提: 盤面は合法な局面（両者ともゴールに到達可能）で、辺の遮断は壁の集合と
#       一致していること（remove_edge で直接辺を消した盤面では (2) の枝刈りが成り立たない）。
from game.board import (
    BOARD_SIZE, ALL_WALLS, W_COL0, W_COL7, W_ROW0, W_ROW7, _GOAL_MASK,
)
from game.pathfinding import shortest_path, reachable, _bfs_layers, _backtrack

_NOT_COL0 = ALL_WALLS & ~W_COL0
_NOT_COL7 = ALL_WALLS & ~W_COL7


def _build_tables():
    h_conf = [0] * 64   # 水平壁 w と重なる水平壁の集合
    v_conf = [0] * 64   # 垂直壁 w と重なる垂直壁の集合
    hw_d   = [0] * 64   # 水平壁 w が塞ぐ下方向の辺（open_d のビット）
    vw_r   = [0] * 64   # 垂直壁 w が塞ぐ右方向の辺（open_r のビット）
    for w in range(64):
        x, y = w % 8, w // 8
        tl = y * 9 + x
        h_conf[w] = (1 << w) | ((1 << (w - 1)) if x > 0 else 0) | ((1 << (w + 1)) if x < 7 else 0)
        v_conf[w] = (1 << w) | ((1 << (w - 8)) if y > 0 else 0) | ((1 << (w + 8)) if y < 7 else 0)
        hw_d[w] = 3 << tl
        vw_r[w] = 0x201 << tl

    # 経路の辺 → その辺を切る壁の集合
    #   下方向の辺（マス a と a+9）を切る水平壁: (c, r), (c-1, r)
    #   右方向の辺（マス a と a+1）を切る垂直壁: (c, r), (c, r-1)
    hcut_d = [0] * 81
    vcut_r = [0] * 81
    for a in range(81):
        c, r = a % 9, a // 9
        if r < 8:
            m = 0
            if c < 8: m |= 1 << (r * 8 + c)
            if c > 0: m |= 1 << (r * 8 + c - 1)
            hcut_d[a] = m
        if c < 8:
            m = 0
            if r < 8: m |= 1 << (r * 8 + c)
            if r > 0: m |= 1 << ((r - 1) * 8 + c)
            vcut_r[a] = m
    return h_conf, v_conf, hw_d, vw_r, hcut_d, vcut_r


_H_CONF, _V_CONF, _HW_D, _VW_R, _HCUT_D, _VCUT_R = _build_tables()


# ----------------------------------------------------------------------
# (1) 重なり・交差
# ----------------------------------------------------------------------
def is_valid_wall_placement(board, x, y, orientation):
    """壁が盤内にあり、既存の壁と重ならず交差もしないか（経路は見ない）。"""
    if not (0 <= x < BOARD_SIZE - 1 and 0 <= y < BOARD_SIZE - 1):
        return False
    w = y * 8 + x
    if orientation == 'h':
        return not (board.hmask & _H_CONF[w]) and not ((board.vmask >> w) & 1)
    return not (board.vmask & _V_CONF[w]) and not ((board.hmask >> w) & 1)


def valid_wall_masks(board):
    """重なり・交差のない壁の集合 (水平, 垂直) を 64bit マスクで返す。"""
    hw = board.hmask
    vw = board.vmask
    vh = ALL_WALLS & ~(hw | ((hw << 1) & _NOT_COL0) | ((hw >> 1) & _NOT_COL7) | vw)
    vv = ALL_WALLS & ~(vw | (vw << 8) | (vw >> 8) | hw)
    return vh, vv


# ----------------------------------------------------------------------
# (2) 経路
# ----------------------------------------------------------------------
def _risky_masks(hw, vw):
    """閉路を作りうる（接点が 2 つ以上ある）壁の集合 (水平, 垂直)。"""
    # 既存の壁が占める格子点（各壁の両端と中点）
    occ = (hw | ((hw << 1) & _NOT_COL0) | ((hw >> 1) & _NOT_COL7) |
           vw | ((vw << 8) & ALL_WALLS) | (vw >> 8))
    # 水平壁: 左端 = 点 w-1（x=0 なら盤端）、中点 = 点 w、右端 = 点 w+1（x=7 なら盤端）
    tl = ((occ << 1) & _NOT_COL0) | W_COL0
    tr = ((occ >> 1) & _NOT_COL7) | W_COL7
    rh = (tl & occ) | (tl & tr) | (occ & tr)
    # 垂直壁: 上端 = 点 w-8（y=0 なら盤端）、下端 = 点 w+8（y=7 なら盤端）
    tt = ((occ << 8) & ALL_WALLS) | W_ROW0
    tb = (occ >> 8) | W_ROW7
    rv = (tt & occ) | (tt & tb) | (occ & tb)
    return rh, rv


def path_cut_masks(board, player):
    """
    player の最短経路（shortest_path_nodes と同じ経路）を切る壁の集合
    (水平, 垂直) を返す。到達不能なら None。
    """
    layers, hit = _bfs_layers(board, player)
    if layers is None:
        return None
    path = _backtrack(layers, hit, board.open_d, board.open_r)
    hcut = vcut = 0
    cur = path[0]
    for p in path[1:]:
        diff = cur - p
        if diff == 9:
            hcut |= _HCUT_D[p]
        elif diff == -9:
            hcut |= _HCUT_D[cur]
        elif diff == 1:
            vcut |= _VCUT_R[p]
        else:
            vcut |= _VCUT_R[cur]
        cur = p
    return hcut, vcut


def legal_wall_masks(board, restrict_h=ALL_WALLS, restrict_v=ALL_WALLS, cuts=None):
    """
    合法な壁の集合 (水平, 垂直) を 64bit マスクで返す（壁の残数は見ない）。

    restrict_h / restrict_v: 調べる壁をこの集合に限定する
    cuts: 計算済みの (path_cut_masks(board, 0), path_cut_masks(board, 1))
    """
    hw = board.hmask
    vw = board.vmask
    ch = restrict_h & ALL_WALLS & ~(hw | ((hw << 1) & _NOT_COL0) | ((hw >> 1) & _NOT_COL7) | vw)
    cv = restrict_v & ALL_WALLS & ~(vw | (vw << 8) | (vw >> 8) | hw)
    if not (ch | cv):
        return 0, 0

    rh, rv = _risky_masks(hw, vw)
    need_h = ch & rh
    need_v = cv & rv
    if not (need_h | need_v):
        return ch, cv

    if cuts is None:
        cuts = (path_cut_masks(board, 0), path_cut_masks(board, 1))
    c0, c1 = cuts
    if c0 is None or c1 is None:
        return 0, 0
    h0, v0 = c0
    h1, v1 = c1
    need_h &= h0 | h1
    need_v &= v0 | v1

    D = board.open_d
    R = board.open_r
    s0 = 1 << board.players[0].pos
    s1 = 1 << board.players[1].pos
    g0, g1 = _GOAL_MASK

    m = need_h
    while m:
        lsb = m & -m
        m ^= lsb
        D2 = D & ~_HW_D[lsb.bit_length() - 1]
        if ((h0 & lsb and not reachable(s0, g0, D2, R)) or
                (h1 & lsb and not reachable(s1, g1, D2, R))):
            ch ^= lsb

    m = need_v
    while m:
        lsb = m & -m
        m ^= lsb
        R2 = R & ~_VW_R[lsb.bit_length() - 1]
        if ((v0 & lsb and not reachable(s0, g0, D, R2)) or
                (v1 & lsb and not reachable(s1, g1, D, R2))):
            cv ^= lsb

    return ch, cv


def is_legal_wall(board, x, y, orientation):
    """壁 1 枚の完全な合法性（重なり・交差・経路。壁の残数は見ない）。"""
    if not is_valid_wall_placement(board, x, y, orientation):
        return False
    bit = 1 << (y * 8 + x)
    if orientation == 'h':
        h, _ = legal_wall_masks(board, bit, 0)
        return bool(h)
    _, v = legal_wall_masks(board, 0, bit)
    return bool(v)


# ----------------------------------------------------------------------
# 手番側が壁を置く（成功すれば make_move と同じく手番・ハッシュを更新）
# ----------------------------------------------------------------------
def _both_paths_exist(board):
    return (shortest_path(board, 0) is not None and
            shortest_path(board, 1) is not None)


def _place(board, x, y, orientation):
    if board.players[board.turn].walls == 0:
        return False
    if not is_valid_wall_placement(board, x, y, orientation):
        return False

    # 辺を直接いじった盤面でも安全なよう、ここは実際に置いて BFS で確認する
    apply = board.apply_hwall if orientation == 'h' else board.apply_vwall
    undo  = board.undo_hwall  if orientation == 'h' else board.undo_vwall
    apply(x, y)
    ok = _both_paths_exist(board)
    undo(x, y)
    if not ok:
        return False

    board.make_move(("hwall" if orientation == 'h' else "vwall", x, y))
    return True


def place_hwall(board, x, y):
    return _place(board, x, y, 'h')


def place_vwall(board, x, y):
    return _place(board, x, y, 'v')
