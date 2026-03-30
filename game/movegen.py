# game/movegen.py
#
# board.edges 廃止対応:
#   board.edges[node] → board.adj ビットマスクのイテレーション に統一
#   隣接ノードの列挙は _iter_adj(adj, node) で行う

from game.board import BOARD_SIZE
from game.pathfinding import shortest_path, shortest_path_nodes
from game.wall import is_valid_wall_placement

_BS = BOARD_SIZE


def _iter_adj(adj: list, node: int):
    """adj[node] のビットマスクから隣接ノードを順に yield する。"""
    mask = adj[node]
    while mask:
        lsb = mask & (-mask)
        yield lsb.bit_length() - 1
        mask ^= lsb


def pawn_moves(board):
    moves = []
    turn  = board.turn
    pos   = board.players[turn].pos
    opp   = board.players[1 - turn].pos
    adj   = board.adj  # edges は廃止、adj のみ使用

    for nxt in _iter_adj(adj, pos):
        if nxt != opp:
            moves.append(nxt)
            continue

        # 相手と隣接 → ジャンプ or 側面移動
        dy = (opp // _BS) - (pos // _BS)
        dx = (opp %  _BS) - (pos %  _BS)

        jy = (opp // _BS) + dy
        jx = (opp %  _BS) + dx
        if 0 <= jy < _BS and 0 <= jx < _BS:
            jump = jy * _BS + jx
            if (adj[opp] >> jump) & 1:
                moves.append(jump)
                continue

        for side in _iter_adj(adj, opp):
            if side == pos:
                continue
            if (side // _BS - opp // _BS) == dy and (side % _BS - opp % _BS) == dx:
                continue
            moves.append(side)

    return [("move", d) for d in moves]


def _wall_candidates_from_path(path):
    """経路上の各エッジに対して直交する壁候補を列挙する。"""
    candidate = set()
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        ax, ay = a % _BS, a // _BS
        bx, by = b % _BS, b // _BS

        if ay == by:
            # 水平移動 → 垂直壁候補
            wx = min(ax, bx)
            for wy in (ay - 1, ay):
                if 0 <= wx < _BS - 1 and 0 <= wy < _BS - 1:
                    candidate.add(('v', wx, wy))
        else:
            # 垂直移動 → 水平壁候補
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


def wall_moves(board):
    my    = board.turn
    enemy = 1 - my
    player = board.players[my]

    if player.walls == 0:
        return []

    my_row = board.players[my].pos // _BS

    enemy_path = shortest_path_nodes(board, enemy)
    if enemy_path is None:
        return []
    candidates = _wall_candidates_from_path(enemy_path)

    my_path = shortest_path_nodes(board, my)
    if my_path is not None:
        candidates |= _wall_candidates_from_path(my_path)

    candidates = {
        (kind, x, y) for (kind, x, y) in candidates
        if _is_forward_wall(y, my_row, my)
    }

    moves   = []
    h_walls = board.h_walls
    v_walls = board.v_walls

    for kind, x, y in candidates:
        if kind == 'h':
            if not is_valid_wall_placement(board, x, y, 'h'):
                continue
            board.apply_hwall(x, y)
            h_walls.add((x, y))
            if (shortest_path(board, enemy) is not None and
                    shortest_path(board, my) is not None):
                moves.append(("hwall", x, y))
            board.undo_hwall(x, y)
            h_walls.discard((x, y))
        else:
            if not is_valid_wall_placement(board, x, y, 'v'):
                continue
            board.apply_vwall(x, y)
            v_walls.add((x, y))
            if (shortest_path(board, enemy) is not None and
                    shortest_path(board, my) is not None):
                moves.append(("vwall", x, y))
            board.undo_vwall(x, y)
            v_walls.discard((x, y))

    return moves


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


def legal_moves(board):
    return pawn_moves(board) + wall_moves(board)