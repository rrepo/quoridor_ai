# ai/wall_evaluation.py（最適化版）

from game.pathfinding import shortest_path
from game.wall import is_valid_wall_placement
from ai.cache import get_dist, get_route, set_route
from game.pathfinding import shortest_path_nodes


# ---------------------------------------------------------------------------
# パス上のエッジ集合を構築（キャッシュ活用）
# ---------------------------------------------------------------------------

def _build_path_edges(path: list) -> set:
    """
    BFS経路のノード列から「隣接ノード間のエッジ」を集合として返す。
    例: path=[10,11,20] → {(10,11), (11,20)} ※常に小さい方を先に

    これを壁の遮断チェックに使う。O(len(path))。
    """
    edges = set()
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        edges.add((a, b) if a < b else (b, a))
    return edges


def _wall_cuts_edges(edges: set, x: int, y: int, orientation: str) -> bool:
    """
    壁(x, y, orientation)がエッジ集合を切断するか O(1)～O(4) で判定。

    Quoridor の座標系:
      pos = row * 9 + col
      水平壁(h): (x,y)-(x+1,y) は上下ノード間エッジを遮断
                  遮断エッジ = (y*9+x, (y+1)*9+x) と (y*9+(x+1), (y+1)*9+(x+1))
      垂直壁(v): (x,y) は左右ノード間エッジを遮断
                  遮断エッジ = (y*9+x, y*9+(x+1)) と ((y+1)*9+x, (y+1)*9+(x+1))
    """
    if orientation == 'h':
        e1 = (y * 9 + x,       (y + 1) * 9 + x)
        e2 = (y * 9 + (x + 1), (y + 1) * 9 + (x + 1))
    else:  # 'v'
        e1 = (y * 9 + x,       y * 9 + (x + 1))
        e2 = ((y + 1) * 9 + x, (y + 1) * 9 + (x + 1))

    # 常に小さい方を先にして集合と比較
    def norm(a, b):
        return (a, b) if a < b else (b, a)

    return norm(*e1) in edges or norm(*e2) in edges


def wall_score(board, x: int, y: int, orientation: str, player_turn: int) -> float:
    key = board.zobrist
    enemy_turn = 1 - player_turn

    # パス取得（キャッシュ優先）
    path = get_route(key, enemy_turn)
    if path is None:
        path = shortest_path_nodes(board, enemy_turn)
        set_route(key, enemy_turn, path)

    if not is_valid_wall_placement(board, x, y, orientation):
        return -1000.0

    # ---------------------------------------------------------------
    # 高速枝刈り: パスを切断しない壁は BFS を呼ばない
    # ---------------------------------------------------------------
    # パスエッジ集合もキャッシュする（同一局面で繰り返し呼ばれるため）
    edge_cache_key = (key, enemy_turn, 'edges')
    edges = get_route(edge_cache_key, 0)  # turn=0 でキャッシュのサブキーとして流用
    if edges is None:
        edges = _build_path_edges(path) if path else set()
        set_route(edge_cache_key, 0, edges)

    cuts_enemy_path = _wall_cuts_edges(edges, x, y, orientation)

    if not cuts_enemy_path:
        # 敵パスを切断しない → 敵距離は変わらない
        # 自分への影響だけ簡易チェック（軽量: 自分のパスエッジを確認）
        my_path = get_route(key, player_turn)
        if my_path is None:
            my_path = shortest_path_nodes(board, player_turn)
            set_route(key, player_turn, my_path)

        my_edges = get_route((key, player_turn, 'edges'), 0)
        if my_edges is None:
            my_edges = _build_path_edges(my_path) if my_path else set()
            set_route((key, player_turn, 'edges'), 0, my_edges)

        if not _wall_cuts_edges(my_edges, x, y, orientation):
            return 0.0  # 誰のパスも切断しない → スコア 0（BFS完全スキップ）

        # 自分のパスだけ切断する場合: 自分が遅くなるので悪い手
        cached = get_dist(key)
        before_self = cached[0] if cached else shortest_path(board, player_turn)

        if orientation == 'h':
            board.apply_hwall(x, y)
        else:
            board.apply_vwall(x, y)
        after_self = shortest_path(board, player_turn)
        if orientation == 'h':
            board.undo_hwall(x, y)
        else:
            board.undo_vwall(x, y)

        score = 0.0
        if before_self is not None and after_self is not None:
            score -= (after_self - before_self) * 1.0
        return score

    # ---------------------------------------------------------------
    # 敵パスを切断する壁: フル BFS（before はキャッシュ活用）
    # ---------------------------------------------------------------
    cached = get_dist(key)
    if cached:
        before_self, before_enemy = cached
    else:
        before_enemy = shortest_path(board, enemy_turn)
        before_self  = shortest_path(board, player_turn)

    if orientation == 'h':
        board.apply_hwall(x, y)
    else:
        board.apply_vwall(x, y)

    after_enemy = shortest_path(board, enemy_turn)
    after_self  = shortest_path(board, player_turn)

    if orientation == 'h':
        board.undo_hwall(x, y)
    else:
        board.undo_vwall(x, y)

    score = 0.0
    if before_enemy is not None and after_enemy is not None:
        score += (after_enemy - before_enemy) * 1.5
    if before_self is not None and after_self is not None:
        score -= (after_self - before_self) * 1.0

    return score