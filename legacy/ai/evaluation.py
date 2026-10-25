# ai/evaluation.py
# シンプルな評価関数（ベースライン）
# 最短距離差 + 残り壁数のみで評価する軽量版

from game.pathfinding import shortest_path
from ai.cache import get_dist, set_dist


def evaluate(board) -> float:
    """
    基本評価関数：最短経路差と壁残数のみで評価。
    軽量で安定しており、比較ベースラインとして使用する。

    Returns:
        float: 正の値 = 現在のプレイヤーに有利
    """
    key = board.zobrist
    cached = get_dist(key)

    if cached:
        my_dist, enemy_dist = cached
    else:
        my_dist = shortest_path(board, board.turn)
        enemy_dist = shortest_path(board, 1 - board.turn)
        set_dist(key, my_dist, enemy_dist)

    if my_dist is None:
        return -10000.0
    if enemy_dist is None:
        return 10000.0

    my_walls = board.players[board.turn].walls
    enemy_walls = board.players[1 - board.turn].walls

    score = (enemy_dist - my_dist) * 10.0
    score += (my_walls - enemy_walls) * 1.5

    return score