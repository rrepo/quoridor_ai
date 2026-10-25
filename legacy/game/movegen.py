# game/movegen.py
#
# 合法手生成（ビットボード版）
#
#   all_legal_moves(board) : 全合法手（コマ移動 + 全合法壁）
#   pawn_moves(board)      : コマ移動のみ
#   all_wall_moves(board)  : 全合法壁
#   wall_moves(board)      : 両者の最短経路付近・前方に限定した壁（探索用の候補絞り込み）
#   legal_moves(board)     : pawn_moves + wall_moves（従来互換。全合法手ではない）
#
# 指し手タプルは board.PAWN_MOVE / HWALL_MOVE / VWALL_MOVE の共有オブジェクトを返す。
# 生成順はビット番号の昇順（コマ移動 → 水平壁 → 垂直壁）。

from game.board import BOARD_SIZE, PAWN_MOVE, HWALL_MOVE, VWALL_MOVE
from game.wall import legal_wall_masks, path_cut_masks, valid_wall_masks  # noqa: F401 (再エクスポート)

_BS = BOARD_SIZE


def _bits_to_moves(mask, table, out):
    while mask:
        lsb = mask & -mask
        out.append(table[lsb.bit_length() - 1])
        mask ^= lsb
    return out


# 壁マスク（64bit）→ 指し手リストを 8bit ずつの表引きで作る。
# 壁は大半が合法なため、1 ビットずつ走査するより大幅に速い。
#   _BYTE_MOVES[table][k][b] = バイト k の値が b のときの指し手タプル群
def _build_byte_moves(table):
    return [tuple(tuple(table[8 * k + i] for i in range(8) if (b >> i) & 1) for b in range(256))
            for k in range(8)]


_H_BYTES = _build_byte_moves(HWALL_MOVE)
_V_BYTES = _build_byte_moves(VWALL_MOVE)


def _wall_bits_to_moves(mask, byte_table, out):
    k = 0
    while mask:
        b = mask & 0xFF
        if b:
            out.extend(byte_table[k][b])
        mask >>= 8
        k += 1
    return out


# ----------------------------------------------------------------------
# コマ移動
# ----------------------------------------------------------------------
def pawn_dest_mask(board):
    """手番側のコマの移動先をビットマスクで返す（ジャンプ・斜め移動を含む）。"""
    t = board.turn
    players = board.players
    pos = players[t].pos
    opp = players[t ^ 1].pos
    D = board.open_d
    R = board.open_r
    c = 1 << pos
    m = ((c & D) << 9) | ((c >> 9) & D) | ((c & R) << 1) | ((c >> 1) & R)
    ob = 1 << opp
    if m & ob:
        # 相手と隣接 → 直進ジャンプ、塞がれていれば相手の左右（斜め）
        m ^= ob
        ao = ((ob & D) << 9) | ((ob >> 9) & D) | ((ob & R) << 1) | ((ob >> 1) & R)
        jump = opp + opp - pos
        if jump >= 0 and (ao >> jump) & 1:
            m |= 1 << jump
        else:
            m |= ao & ~c
    return m


def pawn_moves(board):
    return _bits_to_moves(pawn_dest_mask(board), PAWN_MOVE, [])


# ----------------------------------------------------------------------
# 壁
# ----------------------------------------------------------------------
def all_wall_moves(board):
    """手番側が置ける全ての合法な壁。"""
    if board.players[board.turn].walls == 0:
        return []
    h, v = legal_wall_masks(board)
    out = _wall_bits_to_moves(h, _H_BYTES, [])
    return _wall_bits_to_moves(v, _V_BYTES, out)


def all_legal_moves(board):
    """全合法手（コマ移動 + 全合法壁）。"""
    out = _bits_to_moves(pawn_dest_mask(board), PAWN_MOVE, [])
    if board.players[board.turn].walls:
        h, v = legal_wall_masks(board)
        _wall_bits_to_moves(h, _H_BYTES, out)
        _wall_bits_to_moves(v, _V_BYTES, out)
    return out


# 前方の壁（行マスク）: _FORWARD[player][自分の行]
#   P0: y >= row - 1   P1: y <= row
_FORWARD = [
    [sum(0xFF << (8 * y) for y in range(8) if y >= row - 1) for row in range(_BS)],
    [sum(0xFF << (8 * y) for y in range(8) if y <= row) for row in range(_BS)],
]


def wall_moves(board):
    """
    探索用に絞り込んだ壁: 両者の最短経路（shortest_path_nodes の経路）の辺を
    切る壁のうち、手番側から見て前方にあるもの。
    """
    my = board.turn
    if board.players[my].walls == 0:
        return []

    c0 = path_cut_masks(board, 0)
    if c0 is None:
        return []
    c1 = path_cut_masks(board, 1)
    if c1 is None:
        return []

    fwd = _FORWARD[my][board.players[my].pos // _BS]
    h, v = legal_wall_masks(board, (c0[0] | c1[0]) & fwd, (c0[1] | c1[1]) & fwd, (c0, c1))
    out = _wall_bits_to_moves(h, _H_BYTES, [])
    return _wall_bits_to_moves(v, _V_BYTES, out)


def legal_moves(board):
    """従来互換: コマ移動 + 絞り込んだ壁（全合法手が必要なら all_legal_moves）。"""
    return pawn_moves(board) + wall_moves(board)


# ----------------------------------------------------------------------
# 互換用ヘルパ（経路から壁候補を列挙する旧実装）
# ----------------------------------------------------------------------
def _wall_candidates_from_path(path):
    """経路上の各エッジに対して直交する壁候補を列挙する。"""
    candidate = set()
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        ax, ay = a % _BS, a // _BS
        bx, by = b % _BS, b // _BS

        if ay == by:
            wx = min(ax, bx)
            for wy in (ay - 1, ay):
                if 0 <= wx < _BS - 1 and 0 <= wy < _BS - 1:
                    candidate.add(('v', wx, wy))
        else:
            wy = min(ay, by)
            for wx in (ax - 1, ax):
                if 0 <= wx < _BS - 1 and 0 <= wy < _BS - 1:
                    candidate.add(('h', wx, wy))
    return candidate


def _is_forward_wall(y: int, my_row: int, player: int) -> bool:
    if player == 0:
        return y >= my_row - 1
    else:
        return y <= my_row


def _pawn_move_score(turn, dest):
    if turn == 0:
        return dest // _BS
    else:
        return _BS - 1 - dest // _BS


def order_moves(board, moves):
    turn = board.turn
    pawn = []
    wall = []
    for m in moves:
        if m[0] == "move":
            pawn.append(m)
        else:
            wall.append(m)
    pawn.sort(key=lambda m: _pawn_move_score(turn, m[1]), reverse=True)
    return pawn + wall
