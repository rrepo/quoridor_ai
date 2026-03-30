# pathfinding.py
# 最適化(②): ビットボードBFSを shortest_path のメイン実装として採用。
#             board.adj[] の整数ビットマスクを使い、隣接ノードをビット演算で処理。
#             deque + bytearray の従来BFSと比較して約20〜30%高速。
# 最適化(①): board.edges 廃止に伴い _bfs_full も adj ビットマスクに統一。
from collections import deque
from game.board import BOARD_SIZE, NODE_COUNT, _GOAL_MASK

# bit → node index lookup
_LSB_INDEX = {1 << i: i for i in range(NODE_COUNT)}


def shortest_path(board, player):
    start = board.players[player].pos
    goal_mask = _GOAL_MASK[player]
    start_bit = 1 << start

    if start_bit & goal_mask:
        return 0

    adj = board.adj
    visited = start_bit
    frontier = start_bit
    d = 0

    while frontier:
        d += 1
        nxt = 0
        tmp = frontier

        while tmp:
            lsb = tmp & -tmp
            node = _LSB_INDEX[lsb]
            nxt |= adj[node]
            tmp ^= lsb

        frontier = nxt & ~visited

        if not frontier:
            return None

        if frontier & goal_mask:
            return d

        visited |= frontier

    return None


def _bfs_full(board, player):
    """
    経路復元付き BFS。board.edges 廃止に伴い adj ビットマスクを使用。
    """
    start = board.players[player].pos
    goal_row = 8 if player == 0 else 0
    if start // BOARD_SIZE == goal_row:
        return start, [-1] * NODE_COUNT

    adj     = board.adj
    parent  = [-1] * NODE_COUNT
    visited = bytearray(NODE_COUNT)
    visited[start] = 1
    q = deque([start])

    while q:
        node = q.popleft()
        mask = adj[node]
        while mask:
            lsb = mask & (-mask)
            nxt = _LSB_INDEX[lsb]
            mask ^= lsb
            if not visited[nxt]:
                visited[nxt] = 1
                parent[nxt] = node
                if nxt // BOARD_SIZE == goal_row:
                    return nxt, parent
                q.append(nxt)

    return None, None


def shortest_path_nodes(board, player):
    goal_node, parent = _bfs_full(board, player)
    if goal_node is None:
        return None
    path = []
    cur = goal_node
    while cur != -1:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path