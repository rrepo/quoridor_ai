# ai/eval_func.py（修正版）

from game.pathfinding import shortest_path
from game.board import BOARD_SIZE
from ai.cache import get_dist, set_dist

_BS  = BOARD_SIZE
_BS1 = BOARD_SIZE - 1  # 8

W_DIST_DIFF       = 10.0
W_PROGRESS        =  5.0
W_ENEMY_PROGRESS  =  1.5
W_WALL_DIFF       =  0.3
W_RACE_BOOST      = 15.0
W_FUNNEL          =  2.0
W_CENTER          =  0.4
W_NO_WALL_BONUS   =  5.0
W_WALL_POS        =  0.3

_RACE_WALL_THRESH = 3
_RACE_DIST_THRESH = 5
_FUNNEL_THRESH    = 4


def _wall_position_value_diff(board, me: int) -> float:
    """
    【バグ修正版】
    壁位置価値の差分（me側 - enemy側）を一度のループで計算する。

    従来実装は h_walls/v_walls を両プレイヤーで同一集計していたため
    常に差分=0になっていた。
    本実装では「壁の行位置がどちらのゴールに近いか」で差分を直接算出する。

    P0のゴール = row8（y が大きいほど P0 に有利な壁）
    P1のゴール = row0（y が小さいほど P1 に有利な壁）

    差分 = Σ(2*y/7 - 1) を me==0 視点で計算。
      y=7: +1.0（P0 ゴール寄り）
      y=3: ~-0.14（中央付近）
      y=0: -1.0（P1 ゴール寄り）
    me==1 のときは符号反転。
    """
    total = 0.0
    for (_, y) in board.h_walls:
        total += (2.0 * y / 7.0) - 1.0
    for (_, y) in board.v_walls:
        total += (2.0 * y / 7.0) - 1.0

    # me==1 なら P1 視点に反転
    return total if me == 0 else -total


def evaluate(board) -> float:
    key = board.zobrist
    cached = get_dist(key)

    if cached:
        my_dist, enemy_dist = cached
    else:
        my_dist    = shortest_path(board, board.turn)
        enemy_dist = shortest_path(board, 1 - board.turn)
        set_dist(key, my_dist, enemy_dist)

    if my_dist is None:
        return -10000.0
    if enemy_dist is None:
        return 10000.0

    me    = board.turn
    enemy = 1 - me

    my_walls    = board.players[me].walls
    enemy_walls = board.players[enemy].walls
    my_pos      = board.players[me].pos
    enemy_pos   = board.players[enemy].pos
    my_row      = my_pos    // _BS
    enemy_row   = enemy_pos // _BS
    my_col      = my_pos    %  _BS
    enemy_col   = enemy_pos %  _BS

    # 1. 距離差
    dist_weight = W_DIST_DIFF + (W_NO_WALL_BONUS if enemy_walls == 0 else 0.0)
    dist_diff   = enemy_dist - my_dist
    score       = dist_diff * dist_weight

    # 2. 進行度
    if me == 0:
        my_progress    =         my_row  / _BS1
        enemy_progress = (_BS1 - enemy_row) / _BS1
    else:
        my_progress    = (_BS1 - my_row)    / _BS1
        enemy_progress =         enemy_row  / _BS1

    score += my_progress    * W_PROGRESS
    score -= enemy_progress * W_ENEMY_PROGRESS

    # 3. 壁残数差
    score += (my_walls - enemy_walls) * W_WALL_DIFF

    # 4. 終盤競走検出
    if my_walls <= _RACE_WALL_THRESH and enemy_walls <= _RACE_WALL_THRESH:
        if abs(dist_diff) <= _RACE_DIST_THRESH:
            score += dist_diff * W_RACE_BOOST

    # 5. 閉じ込め評価
    if dist_diff > _FUNNEL_THRESH:
        score += (dist_diff - _FUNNEL_THRESH) * W_FUNNEL

    # 6. センター支配（インライン化）
    score += (1.0 - abs(my_col    - 4) / 4.0) * W_CENTER
    score -= (1.0 - abs(enemy_col - 4) / 4.0) * W_CENTER * 0.5

    # 7. 壁位置価値（バグ修正: 差分を一度に計算）
    if board.h_walls or board.v_walls:
        score += _wall_position_value_diff(board, me) * W_WALL_POS

    return score