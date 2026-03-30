# wall_candidates.py

from game.pathfinding import shortest_path_nodes  # ← shortest_path_route を変更

def wall_candidates_near_path(board, enemy_turn, radius=1):
    path = shortest_path_nodes(board, enemy_turn)  # ← 関数名を変更
    
    if path is None:
        return []

    candidates = set()

    for pos in path:
        x = pos % 9
        y = pos // 9

        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                wx = x + dx
                wy = y + dy

                if 0 <= wx < 8 and 0 <= wy < 8:
                    candidates.add((wx, wy, 'h'))
                    candidates.add((wx, wy, 'v'))

    return list(candidates)