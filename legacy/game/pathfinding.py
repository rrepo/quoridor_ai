# pathfinding.py
#
# 方向別ビットボード（board.open_d / board.open_r）による BFS。
# 1 層の展開をシフト演算 4 回で行うため、ノード単位のループが不要。
#
#   下へ: (f & D) << 9     上へ: (f >> 9) & D
#   右へ: (f & R) << 1     左へ: (f >> 1) & R
#
# 未訪問マスク U を XOR で減らしていくことで、1 層あたりの演算を最小化している。
from game.board import _GOAL_MASK, ALL_NODES


def _static_neighbors():
    nb = []
    for n in range(81):
        c, r = n % 9, n // 9
        m = 0
        if r > 0: m |= 1 << (n - 9)
        if r < 8: m |= 1 << (n + 9)
        if c > 0: m |= 1 << (n - 1)
        if c < 8: m |= 1 << (n + 1)
        nb.append(m)
    return nb


# 壁を無視した盤上の隣接マス（経路復元で候補を絞るのに使う）
_STATIC_NB = _static_neighbors()


def shortest_path(board, player):
    """player のゴール行までの最短距離。到達不能なら None。"""
    f = 1 << board.players[player].pos
    goal = _GOAL_MASK[player]
    if f & goal:
        return 0

    D = board.open_d
    R = board.open_r
    U = ALL_NODES ^ f
    d = 0
    while True:
        d += 1
        f = (((f & D) << 9) | ((f >> 9) & D) | ((f & R) << 1) | ((f >> 1) & R)) & U
        if not f:
            return None
        if f & goal:
            return d
        U ^= f


def reachable(start_bit, goal, D, R):
    """
    盤面を変更せずに到達可能性だけを調べる低レベル関数。
    D / R に「仮に壁を置いた」マスクを渡せば、make/undo なしで合法性判定ができる。
    """
    if start_bit & goal:
        return True
    f = start_bit
    U = ALL_NODES ^ f
    while True:
        f = (((f & D) << 9) | ((f >> 9) & D) | ((f & R) << 1) | ((f >> 1) & R)) & U
        if not f:
            return False
        if f & goal:
            return True
        U ^= f


def _bfs_layers(board, player):
    """
    BFS の各層（距離ごとのノード集合）を返す。
    戻り値: (layers, hit)  layers[k] = 距離 k の集合、hit = ゴール行で最初に届いた集合。
    到達不能なら (None, 0)。始点がゴール行なら ([], 始点)。
    """
    f = 1 << board.players[player].pos
    goal = _GOAL_MASK[player]
    if f & goal:
        return [], f

    D = board.open_d
    R = board.open_r
    U = ALL_NODES ^ f
    layers = [f]
    while True:
        f = (((f & D) << 9) | ((f >> 9) & D) | ((f & R) << 1) | ((f >> 1) & R)) & U
        if not f:
            return None, 0
        hit = f & goal
        if hit:
            return layers, hit
        U ^= f
        layers.append(f)


def _backtrack(layers, hit, D, R):
    """
    BFS の層から最短経路を復元する（ゴール → 始点の順のノード列）。
    ゴールは hit の最小番号、各層では辺が通れる隣接ノードのうち最小番号を選ぶ。
    """
    cur = (hit & -hit).bit_length() - 1
    path = [cur]
    for layer in reversed(layers):
        cand = layer & _STATIC_NB[cur]
        while True:
            lsb = cand & -cand
            p = lsb.bit_length() - 1
            d = cur - p
            if d == 9:
                if (D >> p) & 1: break
            elif d == -9:
                if (D >> cur) & 1: break
            elif d == 1:
                if (R >> p) & 1: break
            elif (R >> cur) & 1:
                break
            cand ^= lsb
        cur = p
        path.append(cur)
    return path


def shortest_path_nodes(board, player):
    """
    最短経路のノード列（始点〜ゴール行）を返す。到達不能なら None。
    同距離の候補が複数ある場合は番号の小さいノードを選ぶ。
    """
    layers, hit = _bfs_layers(board, player)
    if layers is None:
        return None
    path = _backtrack(layers, hit, board.open_d, board.open_r)
    path.reverse()
    return path
