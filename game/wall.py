# wall.py
# 修正点(#1): 封鎖チェックを両プレイヤーに拡張。
#             壁は2マス幅を持つため、稀に自分側も封鎖される可能性がある。
from game.board import BOARD_SIZE
from game.pathfinding import shortest_path

def is_valid_wall_placement(board, x, y, orientation):
    if not (0 <= x < BOARD_SIZE - 1 and 0 <= y < BOARD_SIZE - 1):
        return False

    # basic collision / adjacency checks
    if orientation == 'h':
        if (x, y) in board.h_walls:
            return False
        if (x - 1, y) in board.h_walls or (x + 1, y) in board.h_walls:
            return False
        # crossing with vertical wall at same coords
        if (x, y) in board.v_walls:
            return False
    else:
        if (x, y) in board.v_walls:
            return False
        if (x, y - 1) in board.v_walls or (x, y + 1) in board.v_walls:
            return False
        if (x, y) in board.h_walls:
            return False

    return True


def _both_paths_exist(board):
    return (shortest_path(board, 0) is not None and
            shortest_path(board, 1) is not None)


def place_hwall(board, x, y):
    if board.players[board.turn].walls == 0:
        return False
    if not is_valid_wall_placement(board, x, y, 'h'):
        return False

    # temporarily apply (without touching zobrist here)
    board.apply_hwall(x, y)
    board.h_walls.add((x, y))

    if not _both_paths_exist(board):
        board.undo_hwall(x, y)
        board.h_walls.remove((x, y))
        return False

    # finalize: update zobrist and move stack and walls count
    board.players[board.turn].walls -= 1
    board.zobrist ^= board.z_hwall[y][x]
    board.move_stack.append(("hwall", x, y))
    board.turn ^= 1
    board.zobrist ^= board.z_turn
    return True


def place_vwall(board, x, y):
    if board.players[board.turn].walls == 0:
        return False
    if not is_valid_wall_placement(board, x, y, 'v'):
        return False

    board.apply_vwall(x, y)
    board.v_walls.add((x, y))

    if not _both_paths_exist(board):
        board.undo_vwall(x, y)
        board.v_walls.remove((x, y))
        return False

    board.players[board.turn].walls -= 1
    board.zobrist ^= board.z_vwall[y][x]
    board.move_stack.append(("vwall", x, y))
    board.turn ^= 1
    board.zobrist ^= board.z_turn
    return True