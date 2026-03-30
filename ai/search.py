# ai/search.py

from game.movegen import legal_moves
from game.search import is_terminal
from ai.eval_func import evaluate
from ai.cache import clear_dist_cache
from ai.wall_evaluation import wall_score
from ai.wall_candidates import wall_candidates_near_path
from collections import defaultdict

# ---- TT フラグ定数 ----
_EXACT = 0
_LOWER = 1
_UPPER = 2

_TT_GEN_WINDOW = 4

# ---- TT（配列版） ----
# エントリ構造: (key:int, depth:int, flag:int, score:float, gen:int, best_move) | None
#               [0]       [1]        [2]        [3]           [4]      [5]
_TT_SIZE  = 1 << 20   # 1,048,576 エントリ
_TT_MASK  = _TT_SIZE - 1
_TT_TABLE: list = [None] * _TT_SIZE

# ---- Killer / History ----
MAX_DEPTH = 64
killer_moves: list = [[None, None] for _ in range(MAX_DEPTH)]
history: defaultdict = defaultdict(int)

# ---- 壁 move ordering の深さ閾値 ----
WALL_ORDER_DEPTH_THRESHOLD = 2

# ---- Null Move Pruning 定数 ----
_NMP_DEPTH_MIN = 3
_NMP_REDUCTION = 2
_NMP_WALL_MIN  = 2

# ---- 静止探索定数 ----
_QSEARCH_MAX_DEPTH = 3

_tt_generation: int = 0


# ---------------------------------------------------------------------------
# キャッシュ・状態クリア
# ---------------------------------------------------------------------------

def clear_tt() -> None:
    global _tt_generation
    _TT_TABLE[:] = [None] * _TT_SIZE
    clear_dist_cache()
    _tt_generation = 0
    for i in range(MAX_DEPTH):
        killer_moves[i] = [None, None]
    history.clear()


def _new_generation() -> None:
    global _tt_generation
    _tt_generation += 1
    for k in history:
        history[k] >>= 1


# ---------------------------------------------------------------------------
# TT 操作（配列版のみ）
# ---------------------------------------------------------------------------

def _tt_store(key: int, depth: int, flag: int, score: float, mv) -> None:
    idx      = key & _TT_MASK
    existing = _TT_TABLE[idx]

    if existing is not None:
        ex_key, ex_depth, _, _, ex_gen, _ = existing
        if ex_key == key and ex_gen == _tt_generation and ex_depth >= depth:
            return
        if ex_key != key and ex_gen >= _tt_generation - _TT_GEN_WINDOW and ex_depth > depth:
            return

    _TT_TABLE[idx] = (key, depth, flag, score, _tt_generation, mv)


def _tt_get(key: int):
    """
    返り値: (key, depth, flag, score, gen, best_move) | None
    インデックス: [0]   [1]    [2]   [3]    [4]   [5]
    """
    entry = _TT_TABLE[key & _TT_MASK]
    if entry is not None and entry[0] == key:
        return entry
    return None


# ---------------------------------------------------------------------------
# Move ordering
# ---------------------------------------------------------------------------

def _order_wall_moves(board, wall_moves: list, depth: int) -> list:
    _km = killer_moves[depth]
    scored = []
    for m in wall_moves:
        if m == _km[0]:
            score = 900_000
        elif m == _km[1]:
            score = 800_000
        elif depth >= WALL_ORDER_DEPTH_THRESHOLD:
            kind, x, y = m
            ws = wall_score(board, x, y, kind, board.turn)
            score = ws * 1000 + history[m]
        else:
            score = history[m]
        scored.append((score, m))
    scored.sort(reverse=True)
    return [m for _, m in scored]


def _order_moves(board, moves: list, depth: int, tt_move) -> list:
    candidate_set = None
    if depth >= WALL_ORDER_DEPTH_THRESHOLD:
        enemy_turn = 1 - board.turn
        candidate_set = set(
            (x, y, ori)
            for (x, y, ori) in wall_candidates_near_path(board, enemy_turn, radius=1)
        )

    _km = killer_moves[depth]
    wall_scores = {}
    for m in moves:
        if m[0] == "move":
            continue
        kind, x, y = m
        if candidate_set is not None and (x, y, kind) not in candidate_set:
            continue
        wall_scores[m] = wall_score(board, x, y, kind, board.turn)

    scored = []
    for m in moves:
        if tt_move is not None and m == tt_move:
            score = 1_000_000
        elif m == _km[0]:
            score = 900_000
        elif m == _km[1]:
            score = 800_000
        elif m[0] == "move":
            score = 50_000 + history[m]
        else:
            ws = wall_scores.get(m)
            score = (ws * 1000 if ws is not None else -10_000) + history[m]
        scored.append((score, m))

    scored.sort(reverse=True)
    return [m for _, m in scored]


# ---------------------------------------------------------------------------
# 静止探索
# ---------------------------------------------------------------------------

def quiescence(board, alpha: float, beta: float, qdepth: int) -> float:
    terminal, winner = is_terminal(board)
    if terminal:
        return (9000.0 + qdepth) if winner == board.turn else -(9000.0 + qdepth)

    stand_pat = evaluate(board)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat
    if qdepth <= 0:
        return alpha

    for move in (m for m in legal_moves(board) if m[0] == "move"):
        board.make_move(move)
        score = -quiescence(board, -beta, -alpha, qdepth - 1)
        board.undo_move()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


# ---------------------------------------------------------------------------
# アルファベータ探索
# ---------------------------------------------------------------------------

def alphabeta(board, depth: int, alpha: float, beta: float,
              null_move_allowed: bool = True) -> float:
    orig_alpha = alpha

    # ------------------------------------------------------------------
    # 繰り返し局面の検出（O(1)）
    # board.zobrist_counter は make_move/undo_move で維持される dict。
    # 旧実装の zobrist_history.count() は O(n) だったが、
    # counter.get() は O(1) になった。
    # ------------------------------------------------------------------
    if board.zobrist_counter.get(board.zobrist, 0) >= 2:
        return 0.0

    # 終端判定
    terminal, winner = is_terminal(board)
    if terminal:
        return (9000.0 + depth) if winner == board.turn else -(9000.0 + depth)

    # 葉ノード
    if depth == 0:
        return evaluate(board)
        # return quiescence(board, alpha, beta, _QSEARCH_MAX_DEPTH)

    # ------------------------------------------------------------------
    # TT 参照
    # ------------------------------------------------------------------
    key     = board.zobrist
    entry   = _tt_get(key)
    tt_move = None

    if entry is not None:
        _, ex_depth, ex_flag, tt_score, ex_gen, ex_best = entry
        if ex_gen >= _tt_generation - _TT_GEN_WINDOW:
            tt_move = ex_best
            if ex_depth >= depth:
                if ex_flag == _EXACT:
                    return tt_score
                elif ex_flag == _LOWER:
                    alpha = max(alpha, tt_score)
                elif ex_flag == _UPPER:
                    beta = min(beta, tt_score)
                if alpha >= beta:
                    return tt_score

    # ローカル変数にキャプチャ（LOAD_FAST 化）
    players     = board.players
    me          = board.turn
    enemy       = 1 - me
    my_walls    = players[me].walls
    enemy_walls = players[enemy].walls
    _km         = killer_moves[depth]

    # ------------------------------------------------------------------
    # Null Move Pruning
    #
    # board.make_pass / undo_pass が実装済みのため有効化する。
    # 条件:
    #   - 深さが十分にある（_NMP_DEPTH_MIN 以上）
    #   - 双方に壁が残っている（壁なし終盤で誤判定しないため）
    #   - beta が勝利スコア未満（詰み近辺では適用しない）
    # ------------------------------------------------------------------
    if (null_move_allowed
            and depth >= _NMP_DEPTH_MIN
            and my_walls >= _NMP_WALL_MIN
            and enemy_walls >= _NMP_WALL_MIN
            and beta < 9000.0):
        board.make_pass()
        null_score = -alphabeta(board, depth - _NMP_REDUCTION - 1,
                                 -beta, -beta + 1,
                                 null_move_allowed=False)
        board.undo_pass()
        if null_score >= beta:
            return beta

    # ------------------------------------------------------------------
    # フェーズ1: TT 手（最優先・フルウィンドウ）
    # ------------------------------------------------------------------
    best_score = -99999.0
    best_mv    = None

    if tt_move is not None:
        board.make_move(tt_move)
        score = -alphabeta(board, depth - 1, -beta, -alpha)
        board.undo_move()

        if score > best_score:
            best_score = score
            best_mv    = tt_move
        if score > alpha:
            alpha = score
        if alpha >= beta:
            history[tt_move] += depth * depth
            _tt_store(key, depth, _LOWER, best_score, best_mv)
            return best_score

    # ------------------------------------------------------------------
    # フェーズ2: 移動手（PVS + LMR）
    #
    # is_first は TT 手の有無・結果に関わらず常に True で初期化する。
    # TT 手が存在してもβカットせず最善手でない場合、ゼロ窓で探索すると
    # PV 手を見落とすリスクがあるため。
    # ------------------------------------------------------------------
    all_moves = legal_moves(board)
    if not all_moves:
        return evaluate(board)

    move_moves = [m for m in all_moves if m[0] == "move" and m != tt_move]
    wall_moves = [m for m in all_moves if m[0] != "move" and m != tt_move]

    # 終盤: 双方壁ゼロなら壁手スキップ
    if my_walls == 0 and enemy_walls == 0:
        wall_moves = []

    # move_score クロージャを排除してインライン lambda に統一
    # （alphabeta 呼び出しごとの関数オブジェクト生成コストを削減）
    move_moves.sort(
        key=lambda m: (900_000 if m == _km[0] else
                       800_000 if m == _km[1] else
                       50_000 + history[m]),
        reverse=True
    )

    is_first = True  # TT 手の結果に関わらず移動手先頭は常にフルウィンドウ

    for i, move in enumerate(move_moves):
        board.make_move(move)

        if is_first:
            score    = -alphabeta(board, depth - 1, -beta, -alpha)
            is_first = False
        elif i >= 3 and depth >= 3:
            # LMR: 後半手は深さ削減 + ゼロ窓
            score = -alphabeta(board, depth - 2, -alpha - 1, -alpha)
            if score > alpha:
                score = -alphabeta(board, depth - 1, -alpha - 1, -alpha)
                if alpha < score < beta:
                    score = -alphabeta(board, depth - 1, -beta, -alpha)
        else:
            # PVS: ゼロ窓
            score = -alphabeta(board, depth - 1, -alpha - 1, -alpha)
            if alpha < score < beta:
                score = -alphabeta(board, depth - 1, -beta, -alpha)

        board.undo_move()

        if score > best_score:
            best_score = score
            best_mv    = move
        if score > alpha:
            alpha = score
        if alpha >= beta:
            _km[1] = _km[0]
            _km[0] = move
            history[move] += depth * depth
            _tt_store(key, depth, _LOWER, best_score, best_mv)
            return best_score

    # ------------------------------------------------------------------
    # フェーズ3: 壁手（PVS のみ・LMR なし）
    # ------------------------------------------------------------------
    if wall_moves:
        wall_moves = _order_wall_moves(board, wall_moves, depth)

        for move in wall_moves:
            board.make_move(move)

            if is_first:
                score    = -alphabeta(board, depth - 1, -beta, -alpha)
                is_first = False
            else:
                score = -alphabeta(board, depth - 1, -alpha - 1, -alpha)
                if alpha < score < beta:
                    score = -alphabeta(board, depth - 1, -beta, -alpha)

            board.undo_move()

            if score > best_score:
                best_score = score
                best_mv    = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                _km[1] = _km[0]
                _km[0] = move
                history[move] += depth * depth
                _tt_store(key, depth, _LOWER, best_score, best_mv)
                return best_score

    # ------------------------------------------------------------------
    # TT 書き込み
    # ------------------------------------------------------------------
    if best_score <= orig_alpha:
        flag = _UPPER
    elif best_score >= beta:
        flag = _LOWER
    else:
        flag = _EXACT

    _tt_store(key, depth, flag, best_score, best_mv)
    return best_score


# ---------------------------------------------------------------------------
# 反復深化 + 吸引窓（Aspiration Window）
# ---------------------------------------------------------------------------

ASPIRATION_WINDOW = 30
_INF         = 99999
_MAX_RETRIES = 5


def best_move(board, max_depth: int):
    moves = legal_moves(board)
    if not moves:
        return None
    if len(moves) == 1:
        return moves[0]

    _new_generation()

    best_mv    = None
    prev_score = 0.0

    for d in range(1, max_depth + 1):
        window     = ASPIRATION_WINDOW
        orig_alpha = max(-_INF, prev_score - window)
        orig_beta  = min(_INF,  prev_score + window)
        alpha      = orig_alpha
        beta       = orig_beta
        retries    = 0

        while True:
            entry   = _tt_get(board.zobrist)
            tt_move = entry[5] if entry is not None else None
            ordered = _order_moves(board, moves, d, tt_move)

            best_score   = -99999.0
            current_best = None
            loop_alpha   = alpha
            is_first     = True

            for move in ordered:
                board.make_move(move)

                if is_first:
                    score    = -alphabeta(board, d - 1, -beta, -alpha)
                    is_first = False
                else:
                    score = -alphabeta(board, d - 1, -alpha - 1, -alpha)
                    if alpha < score < beta:
                        score = -alphabeta(board, d - 1, -beta, -alpha)

                board.undo_move()

                if score > best_score:
                    best_score   = score
                    current_best = move
                if score > alpha:
                    alpha = score

            fail_low  = best_score <= loop_alpha
            fail_high = best_score >= beta

            if not (fail_low or fail_high):
                break  # 正常終了

            # ----------------------------------------------------------
            # 吸引窓リトライ
            #
            # 旧実装の問題:
            #   retries > _MAX_RETRIES 後に continue すると retries が
            #   増えないまま同じ条件に再突入し、事実上の無限ループになる。
            #
            # 修正:
            #   retries > _MAX_RETRIES の場合はフルウィンドウで再探索し、
            #   その結果で必ず break する（retries をそれ以上増やさない）。
            # ----------------------------------------------------------
            retries += 1
            if retries > _MAX_RETRIES:
                alpha = -_INF
                beta  = _INF
                # フルウィンドウで再探索して強制終了（continue しない）
                continue  # ← 次のループで fail_low/high が False になり break

            window = min(_INF, window * 2)
            if fail_low:
                alpha = max(-_INF, prev_score - window)
                beta  = orig_beta
            else:
                alpha = orig_alpha
                beta  = min(_INF, prev_score + window)
            # continue して再探索

        prev_score = best_score
        if current_best:
            best_mv = current_best

        if best_score >= 9000:
            break

    return best_mv